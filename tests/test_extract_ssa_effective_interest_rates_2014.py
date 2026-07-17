"""Reader-free tests for the <=2014 SSA effective-rate binding."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXTERNAL = ROOT / "data" / "external"
SOURCE = EXTERNAL / "ssa_effective_interest_rates_2014.source.html"
OUTPUT = EXTERNAL / "ssa_effective_interest_rates_2014.json"
SIDECAR = EXTERNAL / "ssa_effective_interest_rates_2014.source.provenance.json"
OUTPUT_SHA256 = (
    "fe73bc555925c627d9a0cdd10e80a24b8ce9f1ec3c23c5387bdff82c8f267d8b"
)

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import extract_ssa_effective_interest_rates_2014 as extractor  # noqa: E402


def _committed() -> dict:
    return json.loads(OUTPUT.read_text())


def test__effective_rates_source__has_the_pinned_sha256():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == extractor.SOURCE_SHA256


def test__effective_rates_json__has_the_reviewed_sha256():
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    assert digest == OUTPUT_SHA256


def test__effective_rates_build__reproduces_json_byte_for_byte():
    rendered = json.dumps(extractor.build(), indent=2) + "\n"
    assert rendered == OUTPUT.read_text()


def test__effective_rates__binds_exact_labels_and_vintage():
    doc = _committed()
    assert doc["schema_version"] == "ssa_effective_interest_rates.v1"
    assert doc["vintage_year"] == 2014
    assert doc["latest_observation_year"] == 2013
    assert doc["unit"] == "percent"
    assert doc["rate_basis"] == "estimated_effective_rate_earned_by_assets"
    assert "distinct from" in doc["interpretation_note"]
    assert "real interest" in doc["interpretation_note"]
    assert doc["table"] == extractor.TABLE_TITLE
    assert doc["validation"]["expected_headers"] == list(
        extractor.TABLE_HEADERS
    )


def test__effective_rates__are_continuous_and_stop_before_boundary():
    doc = _committed()
    years = [int(year) for year in doc["data"]]
    assert years == list(range(1980, 2014))
    assert doc["validation"]["n_observations"] == 34
    assert not any(year > 2014 for year in years)


def test__effective_rates__pin_first_and_latest_rows():
    data = _committed()["data"]
    assert data["1980"] == {"oasi": 8.5, "di": 8.8, "oasdi": 8.6}
    assert data["2012"] == {"oasi": 4.1, "di": 4.7, "oasdi": 4.1}
    assert data["2013"] == {"oasi": 3.8, "di": 4.5, "oasdi": 3.8}


def test__effective_rates_sidecar__pins_source_file_hash_and_length():
    sidecar = json.loads(SIDECAR.read_text())
    assert sidecar["schema_version"] == "external_source_provenance.v1"
    assert sidecar["committed_source_file"].endswith(
        "ssa_effective_interest_rates_2014.source.html"
    )
    assert sidecar["source_sha256"] == extractor.SOURCE_SHA256
    assert sidecar["source_length_bytes_utf8"] == len(SOURCE.read_bytes())
    assert sidecar["source_url"] == extractor.SOURCE_URL


def test__effective_rates_reader__rejects_source_drift(tmp_path, monkeypatch):
    tampered = tmp_path / "tampered.html"
    tampered.write_bytes(SOURCE.read_bytes() + b"<!-- drift -->")
    monkeypatch.setattr(extractor, "SOURCE_PATH", tampered)
    with pytest.raises(ValueError, match="sha256"):
        extractor.read_source()
