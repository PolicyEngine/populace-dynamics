"""Reader-free tests for 2014 Trustees Report ultimate assumptions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXTERNAL = ROOT / "data" / "external"
SOURCE = EXTERNAL / "ssa_tr_2014_ii_c1.source.html"
OUTPUT = EXTERNAL / "ssa_tr_ultimate_assumptions_2014.json"
SIDECAR = EXTERNAL / "ssa_tr_2014_ii_c1.source.provenance.json"
OUTPUT_SHA256 = (
    "860794e51274a01b93c7eebe19cea60e87cd32da70d8ca8b7f45c7bf291473ea"
)

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import extract_ssa_tr_ultimate_assumptions_2014 as extractor  # noqa: E402


def _committed() -> dict:
    return json.loads(OUTPUT.read_text())


def test__ultimate_assumptions_source__has_the_pinned_sha256():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == extractor.SOURCE_SHA256


def test__ultimate_assumptions_json__has_the_reviewed_sha256():
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    assert digest == OUTPUT_SHA256


def test__ultimate_assumptions_build__reproduces_json_byte_for_byte():
    rendered = json.dumps(extractor.build(), indent=2) + "\n"
    assert rendered == OUTPUT.read_text()


def test__ultimate_assumptions__binds_report_scenario_and_vintage():
    doc = _committed()
    assert doc["schema_version"] == "ssa_tr_ultimate_assumptions.v1"
    assert doc["trustees_report_year"] == 2014
    assert doc["vintage_year"] == 2014
    assert doc["publication_date"] == "2014-07-28"
    assert doc["scenario"] == "intermediate"
    assert doc["table"] == "II.C1"
    crosswalk = doc["provenance"]["design_crosswalk"]
    assert "Table V.B1" in crosswalk
    assert "complete five-assumption bundle" in crosswalk


def test__ultimate_assumptions__pins_all_five_intermediate_values():
    assumptions = _committed()["assumptions"]
    assert {key: record["value"] for key, record in assumptions.items()} == {
        "ultimate_total_fertility_rate": 2.0,
        "average_mortality_improvement_rate": 0.79,
        "ultimate_cpi_w_growth_rate": 2.7,
        "ultimate_real_wage_differential": 1.13,
        "ultimate_real_interest_rate": 2.9,
    }


def test__ultimate_assumptions__preserves_labels_horizons_and_units():
    assumptions = _committed()["assumptions"]
    mortality = assumptions["average_mortality_improvement_rate"]
    assert mortality["published_value"] == ".79"
    assert mortality["unit"] == "percent_per_year"
    assert mortality["horizon"] == "2013-2088 average annual reduction"
    assert "not a terminal-year" in mortality["interpretation_note"]
    assert assumptions["ultimate_total_fertility_rate"]["horizon"] == (
        "2038 and later"
    )
    assert assumptions["ultimate_cpi_w_growth_rate"]["source_label"] == (
        "Consumer Price Index (CPI-W), for 2020 and later"
    )


def test__ultimate_assumptions__verifies_all_scenario_columns():
    validation = _committed()["validation"]
    assert validation["expected_headers"] == list(extractor.TABLE_HEADERS)
    assert validation["all_scenario_values_label_checked"] is True
    assert validation["published_scenario_values"][
        "ultimate_real_wage_differential"
    ] == {"Intermediate": "1.13", "Low-cost": "1.76", "High-cost": ".52"}


def test__ultimate_assumptions_sidecar__pins_source_provenance():
    sidecar = json.loads(SIDECAR.read_text())
    assert sidecar["schema_version"] == "external_source_provenance.v1"
    assert sidecar["committed_source_file"].endswith(
        "ssa_tr_2014_ii_c1.source.html"
    )
    assert sidecar["source_sha256"] == extractor.SOURCE_SHA256
    assert sidecar["source_length_bytes_utf8"] == len(SOURCE.read_bytes())
    assert sidecar["source_url"] == extractor.SOURCE_URL


def test__ultimate_assumptions_reader__rejects_source_drift(
    tmp_path, monkeypatch
):
    tampered = tmp_path / "tampered.html"
    tampered.write_bytes(SOURCE.read_bytes() + b"<!-- drift -->")
    monkeypatch.setattr(extractor, "SOURCE_PATH", tampered)
    with pytest.raises(ValueError, match="sha256"):
        extractor.read_source()
