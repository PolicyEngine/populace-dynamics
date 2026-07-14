"""Extract the SSA claim-age reference from the *2014* Supplement 6.B5.1.

This is the ``supplement_year = 2014`` sibling of
``scripts/build_ssa_claim_ages.py`` (which builds the 2023 edition). It is
the ``<= T*`` claiming binding the M6 run lane's second designed stop names
as its unblock (design amendment 3d, ``docs/design/m6_projection_engine.md``
§2.8.10.1): the 2014 edition is the **maximal edition with
``supplement_year <= 2014``**, so ``validate_external_vintage`` admits it,
and its information is entirely 1998-2013 claiming behaviour (leakage-safe on
the §2.7.6.3 publication-lag principle -- only its publication date, not its
information, is post-``T*``).

Unlike the 2023 build (which embeds a hand-transcribed ``RAW_TABLE``), this
extractor **parses the committed raw HTML source** and is therefore
**byte-reproducible from that source with no network and no transcription
step**:

    data/external/ssa_supplement_2014_6b.source.html
        -- sha256 e2247c602428f69d09669456bae86a0fd3a67025b0fd1b981df6888b691dc088

The source was acquired by the orchestrator from an authenticated browser
session (ssa.gov returns HTTP 403 to programmatic fetches); build-time
provenance is in ``…source.provenance.json``. The extractor verifies the
source hash before parsing, so any drift in the committed source fails the
build loudly.

Structural contract (§2.8.10.1). The emitted JSON is **structurally
identical** to ``ssa_claim_ages_2023supplement.json`` at every field
``populace_dynamics.claiming._load`` parses and its consumers read
(``claiming.py:147-166``): ``schema_version = "ssa_claim_ages.v1"``,
``table = "6.B5.1"``, ``supplement_year = 2014``, ``column_schema`` (the same
twelve ``raw_columns`` and eight ``collapsed_categories``), ``provenance``,
``validation``, ``fra_schedule``, and ``data.{male,female}.{"YYYY":
{number_thousands, average_age, raw, categories, fra_at, …}}``. Only its
``provenance`` additionally carries ``source_sha256`` and ``retrieval_date``.

The one structural subtlety the 2014 edition adds. Its published table has
**eleven** age columns, not twelve: full retirement age never exceeds 66
through entitlement year 2013, so the edition **has no "66 Before FRA"
column**. The twelve-key ``raw`` block is preserved for structural identity
with the 2023 file, with ``age66_before_fra = null`` in **every** row (the
era-inapplicable ``. . .`` convention). The "66 Before FRA" column first
appears in the 2021 data of later editions.

Run from the repository root::

    .venv/bin/python scripts/extract_ssa_claim_ages_2014.py
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
SOURCE_PATH = EXTERNAL / "ssa_supplement_2014_6b.source.html"
OUT_PATH = EXTERNAL / "ssa_claim_ages_2014supplement.json"

SCHEMA_VERSION = "ssa_claim_ages.v1"
SUPPLEMENT_YEAR = 2014
TABLE = "6.B5.1"

#: sha256 of the committed raw HTML source. Verified before parsing so a
#: tampered or re-fetched source cannot silently change the extracted JSON.
#: This is BUILD-TIME provenance; the M6 factory's load-time tamper gate is
#: the sha256 of the extracted JSON, not this hash (§2.8.10.1/§2.8.10.4).
SOURCE_SHA256 = (
    "e2247c602428f69d09669456bae86a0fd3a67025b0fd1b981df6888b691dc088"
)
SOURCE_URL = (
    "https://www.ssa.gov/policy/docs/statcomps/supplement/2014/6b.html"
)
#: UTC retrieval instant, from the committed source provenance sidecar.
RETRIEVAL_DATE = "2026-07-14T10:06:04.312118+00:00"
FETCH_METHOD = (
    "browser (in-app Browser pane) on 2026-07-14; ssa.gov blocks "
    "programmatic fetches (HTTP 403). The fetched page is committed verbatim "
    "at data/external/ssa_supplement_2014_6b.source.html and this extractor "
    "parses it directly, verifying its sha256 first, so the build is "
    "byte-reproducible with no network and no manual transcription."
)

#: The twelve published age columns of the 6.B5.1 union header, left to
#: right. Identical to scripts/build_ssa_claim_ages.py so the two editions
#: share one raw schema. The 2014 edition has no populated "66 Before FRA"
#: column (FRA never exceeds 66 through 2013); it is null in every 2014 row.
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

#: The collapsed eight-way partition the module and held-out check consume.
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

#: Maps each published ``<td>``'s leaf column id (the last token of its
#: ``headers`` attribute in the source table) to a raw-column name. The 2014
#: thead assigns: c2 Number, c3 Average age, c5 Total, c6/c7/c8 ages 62/63/64,
#: c9={c14,c15,c16} = 65 Before/At/After FRA, c10={c17,c18} = 66 At/After FRA,
#: c11 Disability conversions, c12 = 67-69, c13 = 70 or older. There is no
#: leaf for "66 Before FRA": the column does not exist in this edition.
_LEAF_TO_FIELD = {
    "c2": "number_thousands",
    "c3": "average_age",
    "c5": "published_total",
    "c6": "age62",
    "c7": "age63",
    "c8": "age64",
    "c14": "age65_before_fra",
    "c15": "age65_at_fra",
    "c16": "age65_after_fra",
    "c17": "age66_at_fra",
    "c18": "age66_after_fra",
    "c11": "disability_conversion",
    "c12": "age67_69",
    "c13": "age70plus",
}

#: FRA in months, keyed by year of attaining age 65, transcribed from the
#: 2014 edition's footnote a: "FRA is 65 for workers attaining age 65 before
#: 2003, and increases by 2 months per year for workers attaining age 65 in a
#: year after 2002, until it becomes 66 for workers attaining age 65 between
#: 2008 and 2019." (The later-edition 2020-2025 increase toward 67 is absent:
#: the 2014 edition's data ends at entitlement year 2013.) 780 = 65y0m,
#: 792 = 66y0m. birth_year = attain_65_year - 65 reproduces 416(l).
FRA_MONTHS_BY_ATTAIN_65_YEAR: list[tuple[int, int]] = [
    (1900, 780),
    (2003, 782),
    (2004, 784),
    (2005, 786),
    (2006, 788),
    (2007, 790),
    (2008, 792),
]

FOOTNOTE_A_FRA = (
    "FRA is 65 for workers attaining age 65 before 2003, and increases by 2 "
    "months per year for workers attaining age 65 in a year after 2002, "
    "until it becomes 66 for workers attaining age 65 between 2008 and 2019."
)
FOOTNOTE_B_DISABILITY = (
    "Disabled worker benefit automatically converts to a retired worker "
    "benefit in the month the worker attains FRA."
)
SOURCE_NOTE = (
    "Social Security Administration, Master Beneficiary Record, "
    "100 percent data."
)
TABLE_NOTES = (
    "The data in this table differ from those in table 6.B5 because awards "
    "are summarized here by the year in which entitlement began rather than "
    "by the year the award action was posted on the Master Beneficiary "
    "Record. The year of entitlement often precedes the year of action "
    "because of retroactive entitlements, delayed award actions, and other "
    "reasons. Because entitlements can be awarded retroactively, data for "
    "current and prior years are subject to revision with each annual update "
    "of this table. Totals do not necessarily equal the sum of rounded "
    "components. . . . = not applicable; FRA = full retirement age."
)

#: SSA warns "Totals do not necessarily equal the sum of rounded
#: components"; each row's components sum to 100 only up to one-decimal
#: rounding.
SUM_TOLERANCE = 0.5

_FIRST_ENTITLEMENT_YEAR = 1998
#: The latest entitlement year the 2014 edition tabulates. Each edition Y
#: tabulates 1998..Y-1; the 2014 edition (published early 2015) carries
#: 1998-2013. Asserted against the parsed source in :func:`build`.
_LAST_ENTITLEMENT_YEAR = 2013


def read_source() -> str:
    """Return the committed raw HTML, asserting its sha256 first."""
    raw = SOURCE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"{SOURCE_PATH} sha256 {digest} != pinned {SOURCE_SHA256}; the "
            "committed raw source has changed. Re-verify provenance before "
            "rebuilding."
        )
    return raw.decode("utf-8")


def _table_block(html: str) -> str:
    """Return the ``<table>…</table>`` for the anchor ``table6.b5.1``."""
    anchor = html.find('id="table6.b5.1"')
    if anchor < 0:
        raise ValueError("table6.b5.1 anchor not found in source")
    start = html.find("<table>", anchor)
    end = html.find("</table>", start)
    if start < 0 or end < 0:
        raise ValueError("6.B5.1 <table> boundaries not found in source")
    return html[start : end + len("</table>")]


def _cell_value(text: str) -> str | None:
    """Decode a published cell; ``. . .`` (not applicable) becomes ``None``."""
    decoded = html_lib.unescape(text).replace("\xa0", " ").strip()
    # A not-applicable cell is the dot leader ". . ." (no digits).
    if not any(character.isdigit() for character in decoded):
        return None
    return decoded


def parse_rows(html: str) -> dict[str, dict[int, dict[str, Any]]]:
    """Parse Table 6.B5.1 into ``{sex: {year: row}}`` from the raw HTML.

    Cells are mapped to columns by the **leaf column id** (the last token of
    each ``<td headers="…">``), not by position, so the parse is robust to
    the era-varying set of applicable columns. Every row records the number
    of awardees (thousands), the average age, the published total, and the
    twelve :data:`RAW_COLUMNS` (``None`` for ``. . .`` and always ``None`` for
    the edition-absent ``age66_before_fra``).
    """
    tbody = _table_block(html)
    body = tbody[tbody.find("<tbody>") : tbody.find("</tbody>")]
    out: dict[str, dict[int, dict[str, Any]]] = {"male": {}, "female": {}}
    sex: str | None = None
    # Split on any <tr ...> open tag: every fifth data row carries a
    # class="topPad1" spacing attribute, so a bare "<tr>" split would merge
    # those rows into their predecessor.
    for chunk in re.split(r"<tr[^>]*>", body):
        panel = re.search(r'class="panel"[^>]*>([^<]+)</th>', chunk)
        if panel:
            label = panel.group(1).strip()
            sex = "male" if label == "Men" else "female"
            continue
        year_match = re.search(r'class="stub0"[^>]*>(\d{4})</th>', chunk)
        if not year_match:
            continue
        if sex is None:
            raise ValueError("data row encountered before any Men/Women panel")
        year = int(year_match.group(1))
        fields: dict[str, str | None] = {}
        for headers, raw_text in re.findall(
            r'<td headers="([^"]*)">(.*?)</td>', chunk
        ):
            field = _LEAF_TO_FIELD.get(headers.split()[-1])
            if field is not None:
                fields[field] = _cell_value(raw_text)
        raw: dict[str, float | None] = {}
        for column in RAW_COLUMNS:
            token = fields.get(column)
            raw[column] = None if token is None else float(token)
        out[sex][year] = {
            "number_thousands": int(
                fields["number_thousands"].replace(",", "")
            ),
            "average_age": float(fields["average_age"]),
            "published_total": float(fields["published_total"]),
            "raw": raw,
        }
    return out


def _collapse(raw: dict[str, float | None]) -> dict[str, float]:
    """Collapse the twelve raw columns to the eight-way partition.

    Verbatim mirror of ``scripts/build_ssa_claim_ages.py._collapse``.
    """

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

    Verbatim mirror of ``scripts/build_ssa_claim_ages.py._fra_at``: exactly
    one of ``age65_at_fra`` / ``age66_at_fra`` is populated every year (65
    through entitlement year 2008, 66 from 2009).
    """
    if raw["age65_at_fra"] is not None:
        return {"share": raw["age65_at_fra"], "at_age": 65}
    if raw["age66_at_fra"] is not None:
        return {"share": raw["age66_at_fra"], "at_age": 66}
    raise ValueError("No populated 'At FRA' column; unexpected layout.")


def _applicable(raw: dict[str, float | None]) -> list[str]:
    return [k for k in RAW_COLUMNS if raw[k] is not None]


def _era_mapping(
    parsed: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Derive per-era applicable-column records from the parsed data.

    Consecutive years that share an applicable-column signature form one era.
    Derived from the source rather than hardcoded, so the record can never
    disagree with the data. The men and women panels share the same era
    structure in the 2014 edition -- an empirical property of this source,
    asserted nowhere (the era records are provenance-only and read by no
    consumer); men's panel drives the derivation.
    """
    men = parsed["male"]
    eras: list[dict[str, Any]] = []
    for year in sorted(men):
        raw = men[year]["raw"]
        applicable = tuple(_applicable(raw))
        fra = _fra_at(raw)
        if eras and eras[-1]["_signature"] == applicable:
            eras[-1]["_years"].append(year)
            continue
        eras.append(
            {
                "_signature": applicable,
                "_years": [year],
                "applicable_65": [
                    k for k in applicable if k.startswith("age65")
                ],
                "applicable_66": [
                    k for k in applicable if k.startswith("age66")
                ],
                "at_fra_column": (
                    "age65_at_fra" if fra["at_age"] == 65 else "age66_at_fra"
                ),
            }
        )
    records: list[dict[str, Any]] = []
    for era in eras:
        years = era.pop("_years")
        era.pop("_signature")
        low, high = years[0], years[-1]
        era["entitlement_years"] = f"{low}-{high}" if low != high else str(low)
        records.append(
            {
                "entitlement_years": era["entitlement_years"],
                "applicable_65": era["applicable_65"],
                "applicable_66": era["applicable_66"],
                "at_fra_column": era["at_fra_column"],
            }
        )
    return records


def build() -> dict[str, Any]:
    """Parse, validate, and assemble the 2014-edition reference artifact."""
    html = read_source()
    parsed = parse_rows(html)

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

    expected_years = list(
        range(_FIRST_ENTITLEMENT_YEAR, _LAST_ENTITLEMENT_YEAR + 1)
    )
    for sex in ("male", "female"):
        if years_by_sex[sex] != expected_years:
            raise ValueError(
                f"{sex} years {years_by_sex[sex]} != expected "
                f"{_FIRST_ENTITLEMENT_YEAR}-{_LAST_ENTITLEMENT_YEAR}"
            )
    # The edition has no "66 Before FRA" column: null in every row.
    for sex in ("male", "female"):
        for year in expected_years:
            if data[sex][str(year)]["raw"]["age66_before_fra"] is not None:
                raise ValueError(
                    f"age66_before_fra unexpectedly populated for {sex} {year}"
                )
    max_abs_residual = max(abs(r) for r in residuals)
    if max_abs_residual > SUM_TOLERANCE:
        raise ValueError(
            f"A row's components deviate {max_abs_residual} from 100, beyond "
            f"the {SUM_TOLERANCE} rounding tolerance."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "table": TABLE,
        "title": (
            "Number and average age of retired-worker awardees, and "
            "percentage distribution by age at entitlement: By sex and year "
            f"of entitlement, {_FIRST_ENTITLEMENT_YEAR}-"
            f"{_LAST_ENTITLEMENT_YEAR}"
        ),
        "supplement_year": SUPPLEMENT_YEAR,
        "provenance": {
            "source": (
                "SSA Annual Statistical Supplement 2014, "
                f"Table {TABLE} (section 6.B)"
            ),
            "source_url": SOURCE_URL,
            "source_file": (
                "data/external/ssa_supplement_2014_6b.source.html"
            ),
            "source_sha256": SOURCE_SHA256,
            "retrieval_date": RETRIEVAL_DATE,
            "fetch_method": FETCH_METHOD,
            "data_basis": SOURCE_NOTE,
            "mbr_100_percent_note": (
                "Master Beneficiary Record, 100 percent data (not a sample) "
                "for entitlement years "
                f"{_FIRST_ENTITLEMENT_YEAR}-{_LAST_ENTITLEMENT_YEAR}."
            ),
            "table_notes": TABLE_NOTES,
            "footnote_a_fra_schedule": FOOTNOTE_A_FRA,
            "footnote_b_disability_conversion": FOOTNOTE_B_DISABILITY,
            "not_applicable_marker": ". . .",
            "contact": "statistics@ssa.gov",
            "edition_note": (
                "The 2014 edition is the maximal edition with supplement_year "
                "<= 2014; it tabulates only 1998-2013 claiming behaviour and "
                "so has no populated '66 Before FRA' column (FRA never "
                "exceeds 66 through 2013). Values for a given entitlement "
                "year differ from later editions because SSA revises prior "
                "years retroactively with each annual update."
            ),
        },
        "column_schema": {
            "raw_columns": list(RAW_COLUMNS),
            "collapsed_categories": list(COLLAPSED_CATEGORIES),
            "fra_at_note": (
                "fra_at is the single populated 'At FRA' column each year "
                "(age65_at_fra for entitlement years 1998-2008, age66_at_fra "
                "for 2009-2013). It is a distinguishable SUBSET of "
                "age65/age66 exposed for the FRA-kink behaviour -- NOT an "
                "extra partition member."
            ),
            "disability_conversion_note": (
                "Disability conversions (footnote b) auto-convert at FRA. A "
                "published column but NOT a claiming choice; the module "
                "excludes them from the sampler and exposes their share "
                "separately."
            ),
            "age66_before_fra_note": (
                "Present in the raw schema for structural identity with the "
                "2023 edition, but null in every 2014 row: the 2014 edition's "
                "published table has no '66 Before FRA' column (FRA does not "
                "exceed 66 through entitlement year 2013)."
            ),
            "era_map": _era_mapping(parsed),
        },
        "fra_schedule": {
            "unit": "months",
            "keyed_by": "year worker attains age 65",
            "birth_year_equivalent": (
                "attain_65_year - 65 (reproduces 416(l))"
            ),
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
            "years": (f"{_FIRST_ENTITLEMENT_YEAR}-{_LAST_ENTITLEMENT_YEAR}"),
            "sexes": ["male", "female"],
        },
        "build": {
            "built_by": "scripts/extract_ssa_claim_ages_2014.py",
            "source_file": (
                "data/external/ssa_supplement_2014_6b.source.html"
            ),
            "source_sha256": SOURCE_SHA256,
            "reproducible": (
                "parsed deterministically from the committed raw HTML source "
                "(sha256-verified); byte-identical on re-run -- no wall-clock "
                "timestamp, no network, no manual transcription."
            ),
        },
        "data": data,
    }


def main() -> None:
    artifact = build()
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    digest = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(
        f"wrote {OUT_PATH} "
        f"({artifact['validation']['n_rows']} rows, "
        f"max|residual|={artifact['validation']['max_abs_residual']})"
    )
    print(f"json sha256: {digest}")


if __name__ == "__main__":
    main()
