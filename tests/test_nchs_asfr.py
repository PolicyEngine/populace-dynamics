"""Tests for the committed NCHS age-specific fertility-rate reference.

``data/external/nchs_asfr_2024.json`` is the external fertility anchor for
the gate-2 family-transition floors, fetched and parsed by
``scripts/fetch_nchs_asfr.py`` from the NCHS DQS Socrata API. These tests
touch only the committed JSON (always runnable, no network): schema and
provenance fields are present, the parsed ASFRs match the published NVSR
headline, the TFR reproduces from the bins, and the rate is a unimodal age
hump.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "external" / "nchs_asfr_2024.json"

YEARS = ("2024", "2023")
PUBLISHED = {
    "2024": {
        "15-19": 12.6,
        "20-24": 55.8,
        "25-29": 89.5,
        "30-34": 93.7,
        "35-39": 54.3,
        "40-44": 12.7,
        "45-49": 1.1,
        "10-14": 0.2,
    },
    "2023": {
        "15-19": 13.1,
        "20-24": 57.7,
        "25-29": 91.0,
        "30-34": 94.3,
        "35-39": 54.3,
        "40-44": 12.5,
        "45-49": 1.1,
        "10-14": 0.2,
    },
}
PUBLISHED_TFR = {"2024": 1599.5, "2023": 1621.0}
GATE2_BANDS = ("15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49")


def _reference() -> dict:
    return json.loads(REFERENCE.read_text())


def test_schema_and_vintage():
    ref = _reference()
    assert ref["schema_version"] == "nchs_asfr.v1"
    assert ref["vintage_year"] == 2024
    assert ref["measure"] == "age_specific_fertility_rate_per_1000_women"
    assert set(ref["tables"]) == set(YEARS)
    assert list(ref["gate2_bands"]) == list(GATE2_BANDS)


def test_report_and_fetch_provenance():
    ref = _reference()
    report = ref["report"]
    assert report["title"] == "Births: Final Data for 2024"
    assert "Vol. 75, No. 2" in report["nvsr_citation"]
    assert report["report_pdf_url"].startswith("https://www.cdc.gov/")
    fetch = ref["fetch"]
    assert fetch["fetched_by"] == "scripts/fetch_nchs_asfr.py"
    assert fetch["fetched_utc"]
    assert fetch["socrata_resource"] == "daba-4vfq"
    assert fetch["query_url"].startswith("https://data.cdc.gov/resource/")
    assert re.fullmatch(r"[0-9a-f]{64}", fetch["response_sha256"])
    assert fetch["n_bytes"] > 0
    assert ref["band_note"].strip()


@pytest.mark.parametrize("year", YEARS)
def test_asfr_matches_published_headline(year):
    ref = _reference()
    table = ref["tables"][year]
    for band, value in PUBLISHED[year].items():
        assert round(table[band], 1) == value, (year, band)


@pytest.mark.parametrize("year", YEARS)
def test_tfr_reproduces_from_bins(year):
    ref = _reference()
    table = ref["tables"][year]
    tfr = 5.0 * sum(table[b] for b in PUBLISHED[year])
    assert round(tfr, 1) == PUBLISHED_TFR[year]
    assert ref["total_fertility_rate"][year] == PUBLISHED_TFR[year]
    assert ref["validation"][year]["tfr_computed"] == PUBLISHED_TFR[year]


@pytest.mark.parametrize("year", YEARS)
def test_asfr_is_unimodal_hump(year):
    ref = _reference()
    table = ref["tables"][year]
    series = [table[b] for b in GATE2_BANDS]
    peak = series.index(max(series))
    assert all(series[i] <= series[i + 1] for i in range(peak))
    assert all(
        series[i] >= series[i + 1] for i in range(peak, len(series) - 1)
    )
    assert ref["validation"][year]["unimodal_hump"] is True
