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

Interviews are annual through 1997 and biennial from 1999. The
usable panel starts at reference year 1968: the merge requires the
individual file's sequence numbers, which begin in 1969, so the 1968
wave's 1967 income is not attachable to persons. Around the
1992/1993 reference-year seam the head series shows the expected
concept dip (about 7 percent in the raw median), from the pre-1994
totals including farm/business labor parts that the ER era carries
separately.
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
    *range(1968, 1998),
    *range(1999, 2024, 2),
)

#: Pre-1994 waves use era-specific variable names and labels; this
#: table is the adjudicated map (wave -> interview, head labor, wife
#: labor variables with their exact labels), built by reading every
#: staged file's own label block (2026-07-05). Labels verify at read
#: time under whitespace normalization. wife_concept flags 1976-1978,
#: where no edited wife labor-income total exists and the wife series
#: is her wage item only — a documented series limitation, not a
#: silent substitution. Note the 1992->1993 seam: pre-1994 head
#: totals include the labor part of farm and business income, while
#: the ER-era series carries those parts in separate variables, so
#: windows straddling reference years 1992/1993 mix concepts.
_PRE94: dict[int, dict[str, tuple[str, str] | str]] = {
    1968: {
        "interview": ("V2", "INTERVIEW NUMBER 68"),
        "head": ("V74", "HDS LABOR INCOME"),
        "wife": ("V75", "WIFE LBR INCOME"),
        "wife_concept": "total",
    },
    1969: {
        "interview": ("V442", "1969 INT NUMBER"),
        "head": ("V514", "LABOR INC HEAD"),
        "wife": ("V516", "LABOR INC WIFE"),
        "wife_concept": "total",
    },
    1970: {
        "interview": ("V1102", "1970 INT #"),
        "head": ("V1196", "LABOR INC HEAD"),
        "wife": ("V1198", "LABOR INC WIFE"),
        "wife_concept": "total",
    },
    1971: {
        "interview": ("V1802", "71 ID NO."),
        "head": ("V1897", "LABOR INC HEAD"),
        "wife": ("V1899", "LABOR INC WIFE"),
        "wife_concept": "total",
    },
    1972: {
        "interview": ("V2402", "1972 INT #"),
        "head": ("V2498", "LABOR INC HEAD"),
        "wife": ("V2500", "LABOR INC WIFE"),
        "wife_concept": "total",
    },
    1973: {
        "interview": ("V3002", "1973 INT #"),
        "head": ("V3051", "HDS TOT LABOR Y"),
        "wife": ("V3053", "WFS LABOR INC"),
        "wife_concept": "total",
    },
    1974: {
        "interview": ("V3402", "1974 ID NUMBER"),
        "head": ("V3463", "TOT LABOR INC-HD"),
        "wife": ("V3465", "TOT LABOR INC-WF"),
        "wife_concept": "total",
    },
    1975: {
        "interview": ("V3802", "1975 INT #"),
        "head": ("V3863", "HEAD LABOR Y"),
        "wife": ("V3865", "WIFE LABOR Y"),
        "wife_concept": "total",
    },
    1976: {
        "interview": ("V4302", "1976 ID NUMBER"),
        "head": ("V5031", "HEAD TOTAL LABOR Y"),
        "wife": ("V4379", "WIFES ANNUAL WAGE H25"),
        "wife_concept": "wages_only",
    },
    1977: {
        "interview": ("V5202", "1977 ID"),
        "head": ("V5627", "TOT 1976 LABOR INCM HEAD"),
        "wife": ("V5289", "WIFE 1976 WAGES"),
        "wife_concept": "wages_only",
    },
    1978: {
        "interview": ("V5702", "1978 ID"),
        "head": ("V6174", "TOT 1977 HEAD LABOR Y"),
        "wife": ("V5788", "WIFE 1977 WAGE"),
        "wife_concept": "wages_only",
    },
    1979: {
        "interview": ("V6302", "1979 ID"),
        "head": ("V6767", "TOT 1978 HEAD LABOR Y"),
        "wife": ("V6398", "WIFE 1978 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1980: {
        "interview": ("V6902", "1980 INTERVIEW NUMBER"),
        "head": ("V7413", "TOT HD LABOR $ Y 79"),
        "wife": ("V6988", "WIFE 1979 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1981: {
        "interview": ("V7502", "1981 INTERVIEW NUMBER"),
        "head": ("V8066", "TOT HD LABOR $ $ Y 80"),
        "wife": ("V7580", "WIFE 1980 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1982: {
        "interview": ("V8202", "1982 INTERVIEW NUMBER"),
        "head": ("V8690", "TOT HD LABOR $ $ Y 81"),
        "wife": ("V8273", "WIFE 1981 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1983: {
        "interview": ("V8802", "1983 INTERVIEW NUMBER"),
        "head": ("V9376", "TOTAL HEAD LABOR Y 82"),
        "wife": ("V8881", "WIFE 1982 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1984: {
        "interview": ("V10002", "1984 INTERVIEW NUMBER"),
        "head": ("V11023", "TOTAL HEAD LABOR Y 83"),
        "wife": ("V10263", "WIFE 1983 LABOR/Y"),
        "wife_concept": "total",
    },
    1985: {
        "interview": ("V11102", "1985 INTERVIEW NUMBER"),
        "head": ("V12372", "TOTAL HEAD LABOR Y 84"),
        "wife": ("V11404", "WIFE 84 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1986: {
        "interview": ("V12502", "1986 INTERVIEW NUMBER"),
        "head": ("V13624", "TOTAL HEAD LABOR Y 85"),
        "wife": ("V12803", "WIFE 85 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1987: {
        "interview": ("V13702", "1987 INTERVIEW NUMBER"),
        "head": ("V14671", "TOTAL HEAD LABOR Y 86"),
        "wife": ("V13905", "WIFE 86 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1988: {
        "interview": ("V14802", "1988 INTERVIEW NUMBER"),
        "head": ("V16145", "TOTAL HEAD LABOR Y 87"),
        "wife": ("V14920", "WIFE 87 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1989: {
        "interview": ("V16302", "1989 INTERVIEW NUMBER"),
        "head": ("V17534", "TOTAL HEAD LABOR Y 88"),
        "wife": ("V16420", "WIFE 88 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1990: {
        # "INTERVEW" is the source file's own typo.
        "interview": ("V17702", "1990 INTERVEW NUMBER"),
        "head": ("V18878", "TOTAL HEAD LABOR Y 89"),
        "wife": ("V17836", "WIFE 89 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1991: {
        "interview": ("V19002", "1991 INTERVIEW NUMBER"),
        "head": ("V20178", "TOTAL HEAD LABOR Y 90"),
        "wife": ("V19136", "WIFE 90 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1992: {
        "interview": ("V20302", "1992 INTERVIEW NUMBER"),
        "head": ("V21484", "TOTAL HEAD LABOR Y 91"),
        "wife": ("V20436", "WIFE 91 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1993: {
        "interview": ("V21602", "1993 INTERVIEW NUMBER"),
        "head": ("V23323", "HD 1992 TOTAL LABOR INCOME"),
        "wife": ("V23324", "WF 1992 TOTAL LABOR INCOME"),
        "wife_concept": "total",
    },
}

_HEAD_LABOR = r"^LABOR INCOME( OF)?[ -](HEAD|REF PERSON)"
_SPOUSE_LABOR = r"^LABOR INCOME( OF)?[ -](WIFE|SPOUSE)"

#: Individual-file relationship codes attaching family amounts to
#: persons. The coding changed in 1983 (verified empirically on the
#: cross-year file): single digits through 1982 (1 = head,
#: 2 = wife), two digits from 1983 (10 = head/reference person,
#: 20 = legal wife/spouse, 22 = cohabiting partner).
_RELATIONSHIP_ERA_BREAK = 1983
_RELATIONSHIP_HEAD_PRE83 = (1,)
_RELATIONSHIP_SPOUSE_PRE83 = (2,)
_RELATIONSHIP_HEAD = (10,)
_RELATIONSHIP_SPOUSE = (20, 22)


def _relationship_codes(wave: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if wave < _RELATIONSHIP_ERA_BREAK:
        return _RELATIONSHIP_HEAD_PRE83, _RELATIONSHIP_SPOUSE_PRE83
    return _RELATIONSHIP_HEAD, _RELATIONSHIP_SPOUSE


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


def _verified(
    labels: dict[str, str], var: str, expected: str, wave: int
) -> str:
    """Return ``var`` after checking its label matches the table.

    Comparison normalizes runs of whitespace, because PSID labels pad
    columns with variable spacing. A mismatch means the release
    layout changed and the adjudicated table must be re-verified.
    """
    actual = " ".join(labels.get(var, "").split())
    wanted = " ".join(expected.split())
    if actual != wanted:
        raise ValueError(
            f"Wave {wave}: variable {var} label {actual!r} does not "
            f"match the adjudicated table ({wanted!r}). The release "
            "layout may have changed."
        )
    return var


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
            f"{FAMILY_WAVES[0]}-{FAMILY_WAVES[-1]}."
        )
    sps_path, txt_path = _family_paths(wave, data_dir)
    labels = psid.parse_sps_labels(sps_path)
    if wave in _PRE94:
        entry = _PRE94[wave]
        interview = _verified(labels, *entry["interview"], wave)
        head = _verified(labels, *entry["head"], wave)
        spouse = _verified(labels, *entry["wife"], wave)
    else:
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
        head_codes, spouse_codes = _relationship_codes(wave)
        is_head = merged.relationship.isin(head_codes)
        is_spouse = merged.relationship.isin(spouse_codes)
        merged = merged[is_head | is_spouse].copy()
        merged["earnings"] = merged.head_labor.where(
            merged.relationship.isin(head_codes),
            merged.spouse_labor,
        ).astype("float64")
        merged["role"] = "spouse"
        merged.loc[merged.relationship.isin(head_codes), "role"] = "head"
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
