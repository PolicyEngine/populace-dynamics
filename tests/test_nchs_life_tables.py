"""Tests for the committed NCHS life-table reference.

``data/external/nchs_life_tables_2023.json`` is external truth for the
differential-mortality component, fetched and parsed by
``scripts/fetch_nchs_life_tables.py``. These tests touch only the
committed JSON (always runnable, no network): the schema and provenance
fields are present, and the parsed tables pass the standard life-table
sanity conditions (radix, ex(0) headline, adult qx monotonicity, open
terminal interval).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "external" / "nchs_life_tables_2023.json"

POPULATIONS = ("total", "male", "female")
PUBLISHED_EX0 = {"total": 78.4, "male": 75.8, "female": 81.1}


def _reference() -> dict:
    return json.loads(REFERENCE.read_text())


def test_schema_and_vintage():
    ref = _reference()
    assert ref["schema_version"] == "nchs_life_tables.v1"
    assert ref["vintage_year"] == 2023
    assert set(ref["tables"]) == set(POPULATIONS)


def test_report_and_fetch_provenance():
    ref = _reference()
    report = ref["report"]
    assert report["title"] == "United States Life Tables, 2023"
    assert "Vol. 74, No. 6" in report["nvsr_citation"]
    assert report["report_pdf_url"].startswith("https://www.cdc.gov/")
    assert report["report_pdf_url"].endswith("nvsr74-06.pdf")

    fetch = ref["fetch"]
    assert fetch["fetched_by"] == "scripts/fetch_nchs_life_tables.py"
    assert fetch["fetched_utc"]
    assert set(fetch["source_files"]) == set(POPULATIONS)
    for meta in fetch["source_files"].values():
        assert meta["url"].startswith("https://ftp.cdc.gov/")
        assert meta["url"].endswith(".xlsx")
        assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256"])
        assert meta["n_bytes"] > 0
    assert ref["parse_notes"].strip()


@pytest.mark.parametrize("population", POPULATIONS)
def test_radix_and_ex0_headline(population):
    ref = _reference()
    rows = ref["tables"][population]
    ages = [r["age"] for r in rows]
    # Contiguous single years of age from 0 to the open terminal age.
    assert ages == list(range(0, ref["terminal_age"] + 1))
    assert ages[0] == 0

    row0 = rows[0]
    assert round(row0["lx"]) == 100000
    assert round(row0["ex"], 1) == PUBLISHED_EX0[population]
    # The validation block agrees with the raw table.
    val = ref["validation"][population]
    assert round(val["ex0"], 1) == PUBLISHED_EX0[population]
    assert val["ex0_published_headline"] == PUBLISHED_EX0[population]


@pytest.mark.parametrize("population", POPULATIONS)
def test_adult_qx_monotone_and_terminal(population):
    ref = _reference()
    rows = ref["tables"][population]
    by_age = {r["age"]: r for r in rows}
    terminal = ref["terminal_age"]
    # qx non-decreasing across adult ages (40..terminal-1).
    for age in range(40, terminal):
        assert by_age[age]["qx"] <= by_age[age + 1]["qx"] + 1e-9, age
    # Open terminal interval: qx == 1.
    assert round(by_age[terminal]["qx"], 6) == 1.0


@pytest.mark.parametrize("population", POPULATIONS)
def test_lx_and_ex_decrease_with_age(population):
    ref = _reference()
    rows = ref["tables"][population]
    lx = [r["lx"] for r in rows]
    ex = [r["ex"] for r in rows]
    # Survivorship is strictly decreasing; life expectancy is decreasing.
    assert all(lx[i] > lx[i + 1] for i in range(len(lx) - 1))
    assert all(ex[i] >= ex[i + 1] for i in range(len(ex) - 1))
