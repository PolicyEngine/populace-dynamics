"""Build the SSA claim-age reference from Statistical Supplement 6.B5.1.

The Social Security Administration's *Annual Statistical Supplement,
2023*, Table 6.B5.1 gives the **percentage distribution of
retired-worker awardees by age at month of entitlement, by sex and
year of entitlement, 1998-2022**, together with the number of awardees
(thousands) and their average age. It is the external anchor for the
claiming-age component (task B2 of the replication program, #74).

``ssa.gov`` returns HTTP 403 to programmatic fetches, so the table was
read from an authenticated browser session on 2026-07-07 and
transcribed verbatim to
``~/PolicyEngine/dynasim-refs/ssa_supplement_2023_6b.txt`` (provenance:
``ssa_supplement_2023_6b_provenance.md``). The verbatim rows are
embedded below as :data:`RAW_TABLE` so this build is fully reproducible
with no network and no external file; every embedded cell was
cross-validated against a live DOM extraction of the same page in the
same session (the header structure, footnotes, and all 25x2 data rows
matched to the digit).

The one subtlety this script handles honestly is the **era-dependent
column structure**. The published table has a fixed union header of
twelve age columns and marks inapplicable cells ``. . .``; which columns
apply moves with the full-retirement-age (FRA) transition:

* ``62``, ``63``, ``64`` -- always a single column each;
* ``65`` -- split *Before FRA / At FRA / After FRA*; the At/After
  columns apply through entitlement year 2008 (when a 65-year-old could
  still be at or past an FRA of 65-and-months) and the Before column
  from 2003 (when FRA first exceeds 65);
* ``66`` -- split *Before FRA / At FRA / After FRA*; the At column
  applies from 2009 (FRA reaches 66) and the Before column from 2021
  (FRA first exceeds 66); the After column applies in every year;
* ``Disability conversions`` -- a disabled-worker benefit that
  auto-converts to a retired-worker benefit at FRA (footnote b); a
  column in every year, **not a claiming choice**;
* ``67-69`` and ``70 or older`` -- always a single column each.

The build **preserves every published value exactly** in a twelve-key
``raw`` block (``null`` for ``. . .``), and additionally exposes a
collapsed eight-category partition (``age65``/``age66`` are the sum of
their populated sub-columns) plus ``fra_at`` -- the single populated
"At FRA" column each year, a distinguishable subset of ``age65``/
``age66``, exposed for the FRA-kink behaviour and explicitly **not** an
additional partition member. The per-year mapping is recorded so no
category is silently merged across eras.

Run from the repository root::

    .venv/bin/python scripts/build_ssa_claim_ages.py
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external"
OUT_PATH = OUT_DIR / "ssa_claim_ages_2023supplement.json"

SCHEMA_VERSION = "ssa_claim_ages.v1"
SUPPLEMENT_YEAR = 2023
TABLE = "6.B5.1"

SOURCE_URL = (
    "https://www.ssa.gov/policy/docs/statcomps/supplement/2023/6b.html"
)
PROVENANCE_FILE = (
    "~/PolicyEngine/dynasim-refs/ssa_supplement_2023_6b_provenance.md"
)
TRANSCRIPTION_FILE = "~/PolicyEngine/dynasim-refs/ssa_supplement_2023_6b.txt"

#: The twelve published age columns, left to right, after "Total, all
#: ages". This is the exact union header of Table 6.B5.1 (verified
#: against the live page thead: "65 a" and "66 a" each span
#: Before/At/After FRA; "Disability conversions b" and the flanking
#: bands are single columns).
RAW_COLUMNS = (
    "age62",
    "age63",
    "age64",
    "age65_before_fra",
    "age65_at_fra",
    "age65_after_fra",
    "age66_before_fra",
    "age66_at_fra",
    "age66_after_fra",
    "disability_conversion",
    "age67_69",
    "age70plus",
)

#: The collapsed partition used by the module and the held-out check.
#: A stable eight-way split that exists in every year (unlike the raw
#: FRA sub-columns, which have era gaps).
COLLAPSED_CATEGORIES = (
    "age62",
    "age63",
    "age64",
    "age65",
    "age66",
    "disability_conversion",
    "age67_69",
    "age70plus",
)

#: Full retirement age in months, keyed by the year the worker attains
#: age 65 (footnote a's own framing). Birth year = attain-65 year - 65,
#: which reproduces the 416(l) schedule policyengine-us stores by birth
#: year (checked in the tests). 780 = 65y0m, 792 = 66y0m, 804 = 67y0m.
FRA_MONTHS_BY_ATTAIN_65_YEAR: list[tuple[int, int]] = [
    (1900, 780),
    (2003, 782),
    (2004, 784),
    (2005, 786),
    (2006, 788),
    (2007, 790),
    (2008, 792),
    (2020, 794),
    (2021, 796),
    (2022, 798),
    (2023, 800),
    (2024, 802),
    (2025, 804),
]

#: Verbatim transcription of Table 6.B5.1 (Men then Women). Columns:
#: year, number (thousands), average age, total, then the twelve
#: RAW_COLUMNS. ``. . .`` = not applicable. Cross-validated cell for
#: cell against the live DOM on 2026-07-07.
RAW_TABLE = """
Men
1998	902	63.4	100.0	50.8	6.7	10.6	. . .	12.1	2.5	. . .	. . .	1.4	12.7	2.1	1.1
1999	964	63.5	100.0	49.0	6.8	10.8	. . .	12.3	3.2	. . .	. . .	1.8	12.3	2.7	1.2
2000	1,092	63.6	100.0	44.8	6.1	9.7	. . .	15.5	4.5	. . .	. . .	2.8	11.6	4.2	0.8
2001	977	63.4	100.0	48.3	6.6	12.3	. . .	16.2	1.3	. . .	. . .	0.7	12.9	1.1	0.7
2002	998	63.4	100.0	47.7	6.8	13.6	. . .	15.8	1.2	. . .	. . .	0.6	12.6	0.9	0.6
2003	973	63.3	100.0	49.6	6.9	13.1	3.8	11.5	1.2	. . .	. . .	0.6	11.7	0.9	0.6
2004	1,012	63.3	100.0	50.3	7.2	11.1	5.3	11.5	1.1	. . .	. . .	0.7	11.3	0.9	0.7
2005	1,058	63.4	100.0	49.6	7.1	9.5	7.3	11.4	1.0	. . .	. . .	0.9	10.9	1.2	1.0
2006	1,042	63.5	100.0	45.7	7.8	9.3	9.9	11.3	0.8	. . .	. . .	1.0	12.3	1.1	0.8
2007	1,069	63.6	100.0	42.6	7.5	9.4	12.2	11.8	0.6	. . .	. . .	1.4	12.5	1.3	0.8
2008	1,191	63.6	100.0	42.6	7.1	8.4	13.5	12.0	0.3	. . .	. . .	1.7	12.4	1.4	0.7
2009	1,454	63.8	100.0	44.0	7.2	7.3	11.7	. . .	. . .	. . .	13.5	1.9	12.1	1.5	0.8
2010	1,384	63.9	100.0	42.6	8.3	6.9	10.6	. . .	. . .	. . .	13.5	2.2	13.0	2.0	0.9
2011	1,348	64.0	100.0	41.3	7.1	7.5	10.3	. . .	. . .	. . .	14.3	2.4	13.4	2.4	1.2
2012	1,422	64.2	100.0	37.0	6.4	6.6	11.4	. . .	. . .	. . .	16.5	2.6	15.3	2.7	1.4
2013	1,459	64.3	100.0	34.7	6.0	6.1	10.7	. . .	. . .	. . .	17.7	3.2	16.8	3.0	1.7
2014	1,443	64.4	100.0	34.2	6.1	5.9	10.2	. . .	. . .	. . .	17.0	3.5	16.9	4.0	2.2
2015	1,488	64.6	100.0	31.6	6.1	5.8	10.0	. . .	. . .	. . .	17.6	4.3	16.7	5.4	2.5
2016	1,508	64.6	100.0	30.7	6.1	6.0	9.9	. . .	. . .	. . .	18.1	4.1	16.5	5.3	3.3
2017	1,539	64.7	100.0	28.9	5.9	6.0	10.4	. . .	. . .	. . .	18.5	4.1	16.8	5.2	4.2
2018	1,599	64.8	100.0	27.2	5.9	6.0	10.3	. . .	. . .	. . .	19.1	4.4	16.6	6.0	4.6
2019	1,631	64.9	100.0	25.6	5.7	6.0	10.5	. . .	. . .	. . .	19.4	4.5	16.6	6.6	5.1
2020	1,700	65.0	100.0	23.5	5.8	6.1	10.6	. . .	. . .	. . .	19.5	5.0	16.4	7.0	6.1
2021	1,626	65.1	100.0	24.0	6.3	6.6	10.8	. . .	. . .	1.1	15.8	5.6	14.2	7.8	7.8
2022	1,580	65.1	100.0	23.8	6.6	6.8	11.7	. . .	. . .	2.1	14.5	4.9	14.5	7.0	8.1
Women
1998	727	63.5	100.0	55.9	6.0	9.6	. . .	9.7	1.9	. . .	. . .	1.2	9.7	2.6	3.4
1999	755	63.3	100.0	55.4	6.2	10.0	. . .	10.3	2.2	. . .	. . .	1.4	9.9	2.5	2.1
2000	837	63.5	100.0	52.7	6.0	9.5	. . .	11.6	3.1	. . .	. . .	1.8	9.8	3.3	2.1
2001	785	63.3	100.0	54.6	6.1	11.4	. . .	11.4	1.1	. . .	. . .	0.7	11.0	1.8	1.8
2002	817	63.4	100.0	53.3	6.2	12.5	. . .	11.2	1.2	. . .	. . .	0.6	11.1	1.6	2.2
2003	823	63.3	100.0	54.5	6.5	12.3	3.2	7.7	1.1	. . .	. . .	0.7	10.3	1.8	2.0
2004	879	63.3	100.0	54.9	6.7	10.6	4.5	7.5	1.0	. . .	. . .	0.8	10.0	2.0	1.9
2005	939	63.4	100.0	54.1	6.8	9.5	6.2	7.5	0.8	. . .	. . .	0.9	9.7	2.1	2.3
2006	938	63.5	100.0	50.4	7.6	9.5	8.5	7.6	0.7	. . .	. . .	1.0	10.6	2.2	2.0
2007	965	63.6	100.0	47.5	7.3	10.0	10.6	7.5	0.5	. . .	. . .	1.2	11.1	2.0	2.2
2008	1,077	63.6	100.0	48.0	6.8	8.8	11.6	7.5	0.2	. . .	. . .	1.6	11.2	1.9	2.4
2009	1,280	63.7	100.0	49.9	6.5	7.2	10.2	. . .	. . .	. . .	9.3	1.6	11.5	2.0	1.6
2010	1,246	63.7	100.0	47.9	8.1	7.0	9.8	. . .	. . .	. . .	9.3	1.7	12.3	2.1	1.8
2011	1,245	63.8	100.0	46.5	7.2	8.0	9.8	. . .	. . .	. . .	9.9	1.9	12.3	2.4	2.1
2012	1,323	64.0	100.0	42.1	6.8	7.4	11.3	. . .	. . .	. . .	11.3	2.0	14.0	2.6	2.5
2013	1,353	64.1	100.0	40.2	6.4	6.9	10.7	. . .	. . .	. . .	12.4	2.4	15.4	2.7	2.9
2014	1,357	64.3	100.0	39.2	6.6	6.8	10.2	. . .	. . .	. . .	11.9	2.5	15.4	3.3	4.1
2015	1,361	64.3	100.0	37.3	6.6	6.8	10.1	. . .	. . .	. . .	12.3	2.8	15.8	4.1	4.1
2016	1,393	64.5	100.0	35.6	6.5	6.8	9.9	. . .	. . .	. . .	12.8	2.9	15.8	4.5	5.2
2017	1,449	64.6	100.0	32.7	6.3	6.8	10.2	. . .	. . .	. . .	13.7	3.1	16.1	4.8	6.2
2018	1,519	64.7	100.0	30.7	6.3	6.9	10.2	. . .	. . .	. . .	14.4	3.4	16.1	5.3	6.6
2019	1,563	64.8	100.0	28.8	6.2	7.1	10.6	. . .	. . .	. . .	14.6	3.5	16.2	5.6	7.4
2020	1,660	65.0	100.0	25.7	6.0	6.9	10.8	. . .	. . .	. . .	17.2	4.1	15.9	5.8	7.6
2021	1,602	65.0	100.0	25.9	6.4	7.2	11.0	. . .	. . .	1.0	14.1	4.9	13.8	6.8	9.0
2022	1,563	65.0	100.0	25.6	6.8	7.4	12.0	. . .	. . .	2.0	13.0	4.2	14.1	6.2	8.8
"""

#: SSA's own footnotes and notes (verbatim from the live page tfoot).
SOURCE_NOTE = (
    "Social Security Administration, Master Beneficiary Record, "
    "100 percent data."
)
TABLE_NOTES = (
    "The data in this table differ from those in table 6.B5 because "
    "awards are summarized here by the year in which entitlement began "
    "rather than by the year the award action was posted on the Master "
    "Beneficiary Record. The year of entitlement often precedes the "
    "year of action because of retroactive entitlements, delayed award "
    "actions, and other reasons. Because entitlements can be awarded "
    "retroactively, data for current and prior years are subject to "
    "revision with each annual update of this table. Because of "
    "differences in data sources and calculation methods, statistics "
    "reported in this table may differ from those reported by the "
    "Office of the Chief Actuary. Totals do not necessarily equal the "
    "sum of rounded components. FRA = full retirement age; "
    ". . . = not applicable."
)
FOOTNOTE_A_FRA = (
    "FRA is 65 for workers attaining age 65 before 2003. It increases "
    "in 2-month increments for workers attaining age 65 in each of the "
    "years 2003 through 2008. It is 66 for workers attaining age 65 in "
    "the years 2008 through 2019. It again increases in 2-month "
    "increments for workers attaining age 65 in the years 2020 through "
    "2025. It is 67 for workers attaining age 65 in 2025 or later."
)
FOOTNOTE_B_DISABILITY = (
    "Disabled-worker benefit automatically converts to a retired-worker "
    "benefit in the month the worker attains FRA."
)

#: Rounding-residual tolerance. SSA warns "Totals do not necessarily
#: equal the sum of rounded components"; each row's twelve published
#: cells sum to 100 only up to one-decimal rounding.
SUM_TOLERANCE = 0.5

_NA_TOKEN = "·NA·"


def _parse_rows() -> dict[str, dict[int, dict[str, Any]]]:
    """Parse :data:`RAW_TABLE` into ``{sex: {year: row}}``.

    Each row records the number of awardees (thousands), the average
    age, the published total, and the twelve :data:`RAW_COLUMNS` values
    (``None`` for ``. . .``).
    """
    out: dict[str, dict[int, dict[str, Any]]] = {"male": {}, "female": {}}
    sex: str | None = None
    for line in RAW_TABLE.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "Men":
            sex = "male"
            continue
        if stripped == "Women":
            sex = "female"
            continue
        if sex is None:
            raise ValueError(f"Data row before any sex header: {stripped!r}")
        # Collapse the ". . ." not-applicable marker to a single token
        # so a plain whitespace split yields fixed-width fields.
        tokenized = re.sub(r"\.\s*\.\s*\.", f" {_NA_TOKEN} ", stripped)
        fields = tokenized.split()
        if len(fields) != 4 + len(RAW_COLUMNS):
            raise ValueError(
                f"Row {stripped!r} parsed to {len(fields)} fields, "
                f"expected {4 + len(RAW_COLUMNS)}."
            )
        year = int(fields[0])
        number = int(fields[1].replace(",", ""))
        average_age = float(fields[2])
        total = float(fields[3])
        raw: dict[str, float | None] = {}
        for name, token in zip(RAW_COLUMNS, fields[4:], strict=True):
            raw[name] = None if token == _NA_TOKEN else float(token)
        out[sex][year] = {
            "number_thousands": number,
            "average_age": average_age,
            "published_total": total,
            "raw": raw,
        }
    return out


def _collapse(raw: dict[str, float | None]) -> dict[str, float]:
    """Collapse the twelve raw columns to the eight-way partition."""

    def s(*keys: str) -> float:
        return round(sum(raw[k] for k in keys if raw[k] is not None), 1)

    return {
        "age62": raw["age62"] or 0.0,
        "age63": raw["age63"] or 0.0,
        "age64": raw["age64"] or 0.0,
        "age65": s("age65_before_fra", "age65_at_fra", "age65_after_fra"),
        "age66": s("age66_before_fra", "age66_at_fra", "age66_after_fra"),
        "disability_conversion": raw["disability_conversion"] or 0.0,
        "age67_69": raw["age67_69"] or 0.0,
        "age70plus": raw["age70plus"] or 0.0,
    }


def _fra_at(raw: dict[str, float | None]) -> dict[str, Any]:
    """The single populated 'At FRA' column for the year.

    Exactly one of ``age65_at_fra`` / ``age66_at_fra`` is populated in
    every year (65 through entitlement year 2008, 66 from 2009), so the
    at-FRA spike is always distinguishable. It is a **subset** of
    ``age65`` / ``age66`` -- exposed, not added.
    """
    if raw["age65_at_fra"] is not None:
        return {"share": raw["age65_at_fra"], "at_age": 65}
    if raw["age66_at_fra"] is not None:
        return {"share": raw["age66_at_fra"], "at_age": 66}
    raise ValueError("No populated 'At FRA' column; unexpected layout.")


def _applicable(raw: dict[str, float | None]) -> list[str]:
    return [k for k in RAW_COLUMNS if raw[k] is not None]


def _era_mapping() -> list[dict[str, Any]]:
    """Per-era record of which raw columns apply and how they collapse."""
    return [
        {
            "entitlement_years": "1998-2002",
            "fra_regime": "FRA = 65 (attain-65 years <= 2002)",
            "applicable_65": ["age65_at_fra", "age65_after_fra"],
            "applicable_66": ["age66_after_fra"],
            "at_fra_column": "age65_at_fra",
            "note": (
                "FRA is exactly 65y0m; a 65-year-old is at FRA (spike) or "
                "just past it. No 65-before-FRA (nobody's FRA exceeds 65) "
                "and no 66-at/before-FRA (FRA has not reached 66)."
            ),
        },
        {
            "entitlement_years": "2003-2008",
            "fra_regime": "FRA = 65y2m..66y0m (attain-65 years 2003-2008)",
            "applicable_65": [
                "age65_before_fra",
                "age65_at_fra",
                "age65_after_fra",
            ],
            "applicable_66": ["age66_after_fra"],
            "at_fra_column": "age65_at_fra",
            "note": (
                "FRA crosses 65-and-months, so age 65 splits three ways. "
                "66 is still beyond FRA for every cohort (66-after-FRA "
                "only)."
            ),
        },
        {
            "entitlement_years": "2009-2020",
            "fra_regime": "FRA = 66y0m (attain-65 years 2008-2019)",
            "applicable_65": ["age65_before_fra"],
            "applicable_66": ["age66_at_fra", "age66_after_fra"],
            "at_fra_column": "age66_at_fra",
            "note": (
                "FRA is 66y0m: age 65 is entirely before FRA; age 66 is "
                "at FRA (spike) or just past. No 65-at/after-FRA and no "
                "66-before-FRA."
            ),
        },
        {
            "entitlement_years": "2021-2022",
            "fra_regime": "FRA = 66y2m.. (attain-65 years 2020+)",
            "applicable_65": ["age65_before_fra"],
            "applicable_66": [
                "age66_before_fra",
                "age66_at_fra",
                "age66_after_fra",
            ],
            "at_fra_column": "age66_at_fra",
            "note": (
                "FRA crosses 66-and-months, so age 66 splits three ways; "
                "age 65 remains entirely before FRA."
            ),
        },
    ]


def build() -> dict[str, Any]:
    """Parse, validate, and assemble the reference artifact dict."""
    parsed = _parse_rows()

    data: dict[str, dict[str, Any]] = {"male": {}, "female": {}}
    residuals: list[float] = []
    years_by_sex: dict[str, list[int]] = {}
    for sex, rows in parsed.items():
        years_by_sex[sex] = sorted(rows)
        for year in years_by_sex[sex]:
            row = rows[year]
            raw = row["raw"]
            categories = _collapse(raw)
            component_sum = round(sum(categories.values()), 1)
            residual = round(component_sum - 100.0, 1)
            residuals.append(residual)
            data[sex][str(year)] = {
                "number_thousands": row["number_thousands"],
                "average_age": row["average_age"],
                "published_total": row["published_total"],
                "raw": raw,
                "categories": categories,
                "fra_at": _fra_at(raw),
                "applicable_raw_columns": _applicable(raw),
                "component_sum": component_sum,
                "residual": residual,
            }

    # Validation: coverage, exactly one at-FRA column per row, residuals
    # inside the rounding tolerance.
    if years_by_sex["male"] != list(range(1998, 2023)):
        raise ValueError(f"Men years not 1998-2022: {years_by_sex['male']}")
    if years_by_sex["female"] != list(range(1998, 2023)):
        raise ValueError(
            f"Women years not 1998-2022: {years_by_sex['female']}"
        )
    max_abs_residual = max(abs(r) for r in residuals)
    if max_abs_residual > SUM_TOLERANCE:
        raise ValueError(
            f"A row's components deviate {max_abs_residual} from 100, "
            f"beyond the {SUM_TOLERANCE} rounding tolerance."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "table": TABLE,
        "title": (
            "Number and average age of retired-worker awardees, and "
            "percentage distribution by age at entitlement: By sex and "
            "year of entitlement, 1998-2022"
        ),
        "supplement_year": SUPPLEMENT_YEAR,
        "provenance": {
            "source": (
                "SSA Annual Statistical Supplement 2023, "
                f"Table {TABLE} (section 6.B)"
            ),
            "source_url": SOURCE_URL,
            "fetch_method": (
                "authenticated browser session 2026-07-07; ssa.gov blocks "
                "programmatic fetches (HTTP 403). Values transcribed "
                "verbatim and embedded in scripts/build_ssa_claim_ages.py; "
                "every cell cross-validated against a live DOM extraction "
                "of the same page in the same session (header structure, "
                "footnotes, and all 25x2 data rows matched to the digit)."
            ),
            "provenance_file": PROVENANCE_FILE,
            "transcription_file": TRANSCRIPTION_FILE,
            "data_basis": SOURCE_NOTE,
            "mbr_100_percent_note": (
                "Master Beneficiary Record, 100 percent data (not a "
                "sample) for these entitlement years 1998-2022; the "
                "1985-2001 rows of the sister table 6.B5 are a 1 percent "
                "sample, but 6.B5.1 as published is 100 percent data."
            ),
            "table_notes": TABLE_NOTES,
            "footnote_a_fra_schedule": FOOTNOTE_A_FRA,
            "footnote_b_disability_conversion": FOOTNOTE_B_DISABILITY,
            "not_applicable_marker": ". . .",
            "contact": "statistics@ssa.gov",
        },
        "column_schema": {
            "raw_columns": list(RAW_COLUMNS),
            "collapsed_categories": list(COLLAPSED_CATEGORIES),
            "fra_at_note": (
                "fra_at is the single populated 'At FRA' column each year "
                "(age65_at_fra for entitlement years 1998-2008, "
                "age66_at_fra for 2009-2022). It is a distinguishable "
                "SUBSET of age65/age66 exposed for the FRA-kink behaviour "
                "(no 402(q) reduction, no delayed credit) -- NOT an extra "
                "partition member; adding it to the collapsed categories "
                "would double-count."
            ),
            "disability_conversion_note": (
                "Disability conversions (footnote b) are disabled-worker "
                "benefits that auto-convert at FRA. They are a published "
                "column but NOT a claiming choice; the module excludes "
                "them from the claim-age sampler and exposes their share "
                "separately."
            ),
            "era_map": _era_mapping(),
        },
        "fra_schedule": {
            "unit": "months",
            "keyed_by": "year worker attains age 65",
            "birth_year_equivalent": "attain_65_year - 65 (reproduces 416(l))",
            "schedule": [
                {"attain_65_year_from": y, "fra_months": m}
                for y, m in FRA_MONTHS_BY_ATTAIN_65_YEAR
            ],
            "source": "Table 6.B5.1 footnote a; 42 USC 416(l)",
        },
        "validation": {
            "sum_tolerance": SUM_TOLERANCE,
            "max_abs_residual": max_abs_residual,
            "n_rows": len(residuals),
            "residual_note": (
                "SSA: 'Totals do not necessarily equal the sum of rounded "
                "components.' Each row's residual (component_sum - 100) is "
                "recorded per row."
            ),
            "years": "1998-2022",
            "sexes": ["male", "female"],
        },
        "build": {
            "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "built_by": "scripts/build_ssa_claim_ages.py",
            "reproducible": (
                "verbatim table embedded in the build script; no network, "
                "no external file required"
            ),
        },
        "data": data,
    }


def main() -> None:
    artifact = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        f"wrote {OUT_PATH} "
        f"({artifact['validation']['n_rows']} rows, "
        f"max|residual|={artifact['validation']['max_abs_residual']})"
    )


if __name__ == "__main__":
    main()
