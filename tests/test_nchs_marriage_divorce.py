"""Tests for the committed NCHS marriage/divorce-rate reference.

``data/external/nchs_marriage_divorce_rates_2023.json`` is the external
marriage/divorce anchor for the gate-2 family-transition floors, fetched
and parsed by ``scripts/fetch_nchs_marriage_divorce.py`` from the NCHS/NVSS
national trend xlsx. These tests touch only the committed JSON (always
runnable, no network): schema and provenance fields are present, the
parsed crude rates match the CDC FastStats headline, and the marriage rate
exceeds the divorce rate every year.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT / "data" / "external" / "nchs_marriage_divorce_rates_2023.json"
)

PUBLISHED = {
    "2023": {"marriage": 6.1, "divorce": 2.4},
    "2022": {"marriage": 6.2, "divorce": 2.4},
    "2021": {"marriage": 6.0, "divorce": 2.5},
}


def _reference() -> dict:
    return json.loads(REFERENCE.read_text())


def test_schema_and_measure():
    ref = _reference()
    assert ref["schema_version"] == "nchs_marriage_divorce.v1"
    assert ref["latest_year"] == 2023
    assert ref["measure"] == "crude_rate_per_1000_total_population"
    assert set(ref["tables"]) == {"marriage", "divorce"}


def test_report_and_fetch_provenance():
    ref = _reference()
    report = ref["report"]
    assert "National Marriage and Divorce Rate Trends" in report["title"]
    assert report["nvss_page"].startswith("https://www.cdc.gov/")
    fetch = ref["fetch"]
    assert fetch["fetched_by"] == "scripts/fetch_nchs_marriage_divorce.py"
    assert fetch["fetched_utc"]
    assert fetch["xlsx_url"].endswith(".xlsx")
    assert re.fullmatch(r"[0-9a-f]{64}", fetch["xlsx_sha256"])
    assert fetch["n_bytes"] > 0
    # The five states excluded from the national divorce denominator are
    # recorded, not silently dropped.
    assert len(ref["divorce_excluded_states"]) == 5
    assert "California" in ref["divorce_excluded_states"]
    assert ref["concept_note"].strip()


@pytest.mark.parametrize("year", PUBLISHED)
def test_rates_match_faststats_headline(year):
    ref = _reference()
    for series, rate in PUBLISHED[year].items():
        assert round(ref["tables"][series][year]["rate_per_1000"], 1) == rate


def test_marriage_exceeds_divorce_every_year():
    ref = _reference()
    years = set(ref["tables"]["marriage"]) & set(ref["tables"]["divorce"])
    assert years
    for year in years:
        m = ref["tables"]["marriage"][year]["rate_per_1000"]
        d = ref["tables"]["divorce"][year]["rate_per_1000"]
        assert m > d, year
    assert ref["validation"]["marriage_gt_divorce_all_years"] is True


def test_span_covered():
    ref = _reference()
    assert ref["validation"]["years_covered"] == [2000, 2023]
