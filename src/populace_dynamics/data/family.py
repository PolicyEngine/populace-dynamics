"""Head and spouse labor income from the per-wave family files.

The cross-year individual file carries no uniform individual labor
income, so real earnings histories come from the family files: each
wave reports the family's head/reference-person and wife/spouse labor
income for the *prior* calendar year, and the individual file's
per-wave interview number, relationship, and sequence attach those
amounts to persons.

Scope: waves 1994-2023 (income reference years 1993-2022), where the
labels resolve uniquely under one pattern family (verified against
every staged file on 2026-07-05):

* interview: ``"<wave> INTERVIEW #"`` (1994-1997) or
  ``"<wave> FAMILY INTERVIEW (ID) NUMBER"`` (1999+);
* head labor income: ``"LABOR INCOME OF HEAD-<yyyy>"``,
  ``"LABOR INCOME-HEAD"`` (1997/1999), or
  ``"LABOR INCOME OF REF PERSON-<yyyy>"`` (2017+);
* spouse labor income: the WIFE/SPOUSE forms of the same.

Waves 1968-1993 use era-specific abbreviations (``"HDS LABOR
INCOME"``, ``"HEAD LABOR Y"``, ``"WIFE 84 LABOR/WAGE"``) and are a
documented extension, not silently included.

Interviews are annual through 1997 and biennial from 1999, so income
reference years run 1993-1996 annually and then 1998, 2000, ...,
2022.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from populace_dynamics.data import panels, psid

__all__ = [
    "FAMILY_WAVES",
    "read_family_labor",
    "family_earnings_panel",
]

#: Waves whose labels the loader resolves; annual to 1997, biennial after.
FAMILY_WAVES: tuple[int, ...] = (
    1994,
    1995,
    1996,
    1997,
    *range(1999, 2024, 2),
)

_HEAD_LABOR = r"^LABOR INCOME( OF)?[ -](HEAD|REF PERSON)"
_SPOUSE_LABOR = r"^LABOR INCOME( OF)?[ -](WIFE|SPOUSE)"

#: Individual-file relationship codes attaching family amounts to
#: persons: 10 = head/reference person; 20 = legal wife/spouse;
#: 22 = cohabiting partner ("wife").
_RELATIONSHIP_HEAD = (10,)
_RELATIONSHIP_SPOUSE = (20, 22)

#: Defensive missing sentinel; family labor income is edited and far
#: below it, which the integration tests assert.
_MISSING = 9_999_998


def _family_paths(wave: int, data_dir: Path | None) -> tuple[Path, Path]:
    base = psid._resolve_data_dir(data_dir) / "family" / str(wave)
    if not base.is_dir():
        raise FileNotFoundError(
            f"Family wave directory not found: {base} "
            f"({psid._README_POINTER})"
        )
    sps = sorted(
        p
        for p in base.glob("*.sps")
        if not p.name.lower().endswith("_formats.sps")
    )
    txt = sorted(base.glob("*.txt"))
    if len(sps) != 1 or len(txt) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .sps and one .txt in {base}; "
            f"found {len(sps)} and {len(txt)}."
        )
    return sps[0], txt[0]


def _single(labels: dict[str, str], pattern: str, wave: int, what: str) -> str:
    regex = re.compile(pattern, re.IGNORECASE)
    hits = {n: v for n, v in labels.items() if regex.search(v)}
    if len(hits) != 1:
        raise ValueError(
            f"Wave {wave}: pattern for {what} matched {len(hits)} "
            f"variables ({dict(list(hits.items())[:4])}); expected "
            "exactly one."
        )
    return next(iter(hits))


def _check_reference_year(label: str, wave: int) -> None:
    """Labels that carry a 4-digit year must carry ``wave - 1``."""
    match = re.search(r"(19|20)\d{2}\s*$", label)
    if match and int(match.group(0)) != wave - 1:
        raise ValueError(
            f"Wave {wave}: label {label!r} carries reference year "
            f"{match.group(0)}, expected {wave - 1}. The release "
            "layout may have changed."
        )


def read_family_labor(
    wave: int,
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read one wave's family-level labor income.

    Returns a frame with ``interview``, ``head_labor``, and
    ``spouse_labor`` for every responding family, with every variable
    resolved from the file's own labels and reference years verified
    where the label carries one.
    """
    if wave not in FAMILY_WAVES:
        raise ValueError(
            f"Wave {wave} is outside the resolved range "
            f"{FAMILY_WAVES[0]}-{FAMILY_WAVES[-1]}; pre-1994 labels "
            "are era-specific and not yet mapped."
        )
    sps_path, txt_path = _family_paths(wave, data_dir)
    labels = psid.parse_sps_labels(sps_path)
    interview = _single(
        labels,
        rf"^{wave} (INTERVIEW #|FAMILY INTERVIEW \(ID\) NUMBER)$",
        wave,
        "interview number",
    )
    head = _single(labels, _HEAD_LABOR, wave, "head labor income")
    spouse = _single(labels, _SPOUSE_LABOR, wave, "spouse labor income")
    _check_reference_year(labels[head], wave)
    _check_reference_year(labels[spouse], wave)

    layout = psid.parse_sps_layout(sps_path)
    layout_by_name = layout.set_index("name")
    colspecs = [
        (
            int(layout_by_name.loc[name, "start"]) - 1,
            int(layout_by_name.loc[name, "end"]),
        )
        for name in (interview, head, spouse)
    ]
    frame = pd.read_fwf(
        txt_path,
        colspecs=colspecs,
        names=["interview", "head_labor", "spouse_labor"],
        header=None,
        nrows=nrows,
    )
    return frame


def family_earnings_panel(
    *,
    waves: tuple[int, ...] | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Person-level labor-income histories from the family files.

    Merges each wave's family head/spouse labor income onto persons
    through the individual file's interview number, keeping persons
    present in a responding family (sequence 1-20) whose relationship
    is head/reference person or wife/spouse/partner. ``period`` is
    the income reference year, ``wave - 1``.

    Columns: ``person_id``, ``period``, ``earnings``, ``role``
    (``"head"``/``"spouse"``), ``age``, ``weight`` (age and weight
    are measured at the collection wave).
    """
    use_waves = tuple(waves) if waves is not None else FAMILY_WAVES
    demo = panels.demographic_panel(data_dir=data_dir)
    demo = demo[demo.period.isin(use_waves)]

    frames = []
    for wave in use_waves:
        labor = read_family_labor(wave, data_dir=data_dir)
        wave_people = demo[demo.period == wave]
        merged = wave_people.merge(
            labor, left_on="interview", right_on="interview", how="inner"
        )
        is_head = merged.relationship.isin(_RELATIONSHIP_HEAD)
        is_spouse = merged.relationship.isin(_RELATIONSHIP_SPOUSE)
        merged = merged[is_head | is_spouse].copy()
        merged["earnings"] = merged.head_labor.where(
            merged.relationship.isin(_RELATIONSHIP_HEAD),
            merged.spouse_labor,
        ).astype("float64")
        merged["role"] = "spouse"
        merged.loc[merged.relationship.isin(_RELATIONSHIP_HEAD), "role"] = (
            "head"
        )
        merged["period"] = wave - 1
        frames.append(
            merged[
                ["person_id", "period", "earnings", "role", "age", "weight"]
            ]
        )
    panel = pd.concat(frames, ignore_index=True)
    keep = (panel.earnings < _MISSING) & (panel.weight > 0)
    panel = panel.loc[keep]
    return panel.sort_values(["person_id", "period"]).reset_index(drop=True)
