"""Reader-free tests for the 2014-TR opening-reserve binding."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXTERNAL = ROOT / "data" / "external"
SOURCE = EXTERNAL / "ssa_tr_2014_vi_g8.source.html"
OUTPUT = EXTERNAL / "ssa_trust_fund_opening_reserve_2014.json"
SIDECAR = EXTERNAL / "ssa_tr_2014_vi_g8.source.provenance.json"
OUTPUT_SHA256 = (
    "6ece324e331c959d814292195cff333aa8dde6de7d309c7959afc7bffe7b09b9"
)

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import extract_ssa_trust_fund_opening_reserve_2014 as extractor  # noqa: E402


def _committed() -> dict:
    return json.loads(OUTPUT.read_text())


def test__opening_reserve_source__has_the_pinned_sha256():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == extractor.SOURCE_SHA256


def test__opening_reserve_json__has_the_reviewed_sha256():
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    assert digest == OUTPUT_SHA256


def test__opening_reserve_build__reproduces_json_byte_for_byte():
    rendered = json.dumps(extractor.build(), indent=2) + "\n"
    assert rendered == OUTPUT.read_text()


def test__opening_reserve__binds_report_scenario_and_vintage():
    doc = _committed()
    assert doc["schema_version"] == "ssa_trust_fund_opening_reserve.v1"
    assert doc["trustees_report_year"] == 2014
    assert doc["vintage_year"] == 2014
    assert doc["publication_date"] == "2014-07-28"
    assert doc["scenario"] == "intermediate"
    assert doc["table"] == "VI.G8"


def test__opening_reserve__pins_amount_date_and_fund_combination():
    reserve = _committed()["opening_reserve"]
    assert reserve["as_of_date"] == "2014-12-31"
    assert reserve["opening_year"] == 2015
    assert reserve["funds"] == ["OASI", "DI"]
    assert reserve["amount_billions_usd"] == 2783.7
    assert reserve["amount_millions_usd"] == 2783700
    assert reserve["published_value"] == "2,783.7"


def test__opening_reserve__is_loudly_an_estimate_not_an_actual():
    doc = _committed()
    assert doc["opening_reserve"]["estimate_status"] == (
        "projected_intermediate_assumptions"
    )
    timing_note = doc["provenance"]["timing_note"]
    assert "intermediate-assumptions estimate" in timing_note
    assert "2015" in timing_note


def test__opening_reserve__binds_exact_row_and_column_labels():
    validation = _committed()["validation"]
    assert validation["expected_headers"] == list(extractor.TABLE_HEADERS)
    assert validation["published_intermediate_2014_row"] == list(
        extractor.EXPECTED_INTERMEDIATE_2014_ROW
    )
    assert validation["value_status_label_checked"] is True


def test__opening_reserve_sidecar__pins_source_provenance():
    sidecar = json.loads(SIDECAR.read_text())
    assert sidecar["schema_version"] == "external_source_provenance.v1"
    assert sidecar["committed_source_file"].endswith(
        "ssa_tr_2014_vi_g8.source.html"
    )
    assert sidecar["source_sha256"] == extractor.SOURCE_SHA256
    assert sidecar["source_length_bytes_utf8"] == len(SOURCE.read_bytes())
    assert sidecar["source_url"] == extractor.SOURCE_URL


def test__opening_reserve_reader__rejects_source_drift(tmp_path, monkeypatch):
    tampered = tmp_path / "tampered.html"
    tampered.write_bytes(SOURCE.read_bytes() + b"<!-- drift -->")
    monkeypatch.setattr(extractor, "SOURCE_PATH", tampered)
    with pytest.raises(ValueError, match="sha256"):
        extractor.read_source()
