"""Tests for the 2014-Supplement claim-age extraction (§2.8.10.1).

Always runnable: they touch only the committed raw HTML source, the
committed extracted JSON, the 2023-edition JSON, and the extraction script.
No PSID, no policyengine-us. Two guarantees:

* **byte-reproducibility** -- the committed JSON is exactly what the script
  emits from the committed, sha256-verified raw source; and
* **structural identity** -- the JSON exposes every field
  ``populace_dynamics.claiming._load`` parses and its consumers read, so
  ``claiming_pmfs_from_reference`` admits it at the M6 boundary.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from populace_dynamics import claiming
from populace_dynamics.engine.refit import (
    claiming_pmfs_from_reference,
    validate_external_vintage,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXTERNAL = ROOT / "data" / "external"
SOURCE = EXTERNAL / "ssa_supplement_2014_6b.source.html"
JSON_2014 = EXTERNAL / "ssa_claim_ages_2014supplement.json"
JSON_2023 = EXTERNAL / "ssa_claim_ages_2023supplement.json"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import extract_ssa_claim_ages_2014 as extractor  # noqa: E402


def _committed() -> dict:
    return json.loads(JSON_2014.read_text())


def test__committed_source__has_the_pinned_sha256():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == extractor.SOURCE_SHA256


def test__build__reproduces_the_committed_json_byte_for_byte():
    rendered = json.dumps(extractor.build(), indent=2) + "\n"
    assert rendered == JSON_2014.read_text()


def test__build__is_deterministic_across_runs():
    first = json.dumps(extractor.build(), indent=2)
    second = json.dumps(extractor.build(), indent=2)
    assert first == second


def test__read_source__rejects_a_tampered_source(tmp_path, monkeypatch):
    tampered = tmp_path / "tampered.html"
    tampered.write_bytes(SOURCE.read_bytes() + b"<!-- drift -->")
    monkeypatch.setattr(extractor, "SOURCE_PATH", tampered)
    try:
        extractor.read_source()
    except ValueError as error:
        assert "sha256" in str(error)
    else:  # pragma: no cover - guard must fire
        raise AssertionError("tampered source did not raise")


def test__committed_json__is_the_maximal_pre_boundary_edition():
    doc = _committed()
    assert doc["schema_version"] == "ssa_claim_ages.v1"
    assert doc["table"] == "6.B5.1"
    assert doc["supplement_year"] == 2014


def test__latest_entitlement_year__is_2013():
    reference = claiming.load_claim_age_reference(JSON_2014)
    years = reference.years()
    assert years == list(range(1998, 2014))
    assert max(years) == 2013  # confirmed at extraction (2013, as expected)


def test__structural_identity__with_the_2023_reference():
    doc14, doc23 = _committed(), json.loads(JSON_2023.read_text())
    assert sorted(doc14) == sorted(doc23)
    assert (
        doc14["column_schema"]["raw_columns"]
        == doc23["column_schema"]["raw_columns"]
    )
    assert (
        doc14["column_schema"]["collapsed_categories"]
        == doc23["column_schema"]["collapsed_categories"]
    )
    row14 = doc14["data"]["male"]["2005"]
    row23 = doc23["data"]["male"]["2005"]
    assert sorted(row14) == sorted(row23)
    assert sorted(row14["raw"]) == sorted(row23["raw"])
    assert sorted(row14["categories"]) == sorted(row23["categories"])
    assert sorted(row14["fra_at"]) == sorted(row23["fra_at"])


def test__reference__loads_and_is_admitted_at_the_boundary():
    reference = claiming.load_claim_age_reference(JSON_2014)
    assert reference.supplement_year == 2014
    validate_external_vintage(
        "claiming reference", reference.supplement_year, boundary_year=2014
    )
    pmfs = claiming_pmfs_from_reference(reference, boundary_year=2014)
    # two sexes x sixteen entitlement years (1998-2013).
    assert len(pmfs) == 32
    for pmf in pmfs.values():
        assert abs(sum(pmf.values()) - 1.0) < 1e-9


def test__age66_before_fra__is_null_in_every_row():
    doc = _committed()
    for sex in ("male", "female"):
        for row in doc["data"][sex].values():
            assert row["raw"]["age66_before_fra"] is None


def test__every_row__has_exactly_one_populated_at_fra_column():
    doc = _committed()
    for sex in ("male", "female"):
        for row in doc["data"][sex].values():
            raw = row["raw"]
            populated = [
                raw["age65_at_fra"] is not None,
                raw["age66_at_fra"] is not None,
            ]
            assert sum(populated) == 1
            assert row["fra_at"]["at_age"] in (65, 66)


def test__each_row__collapses_to_eight_categories_summing_near_100():
    doc = _committed()
    tolerance = doc["validation"]["sum_tolerance"]
    for sex in ("male", "female"):
        for row in doc["data"][sex].values():
            categories = row["categories"]
            assert sorted(categories) == [
                "age62",
                "age63",
                "age64",
                "age65",
                "age66",
                "age67_69",
                "age70plus",
                "disability_conversion",
            ]
            assert abs(sum(categories.values()) - 100.0) <= tolerance


def test__provenance__carries_source_sha256_and_retrieval_date():
    provenance = _committed()["provenance"]
    assert provenance["source_sha256"] == extractor.SOURCE_SHA256
    assert provenance["retrieval_date"]
    assert provenance["source_url"].endswith("2014/6b.html")


def test__collapsed_values__equal_the_sum_of_their_raw_subcolumns():
    doc = _committed()
    row = doc["data"]["female"]["2013"]
    raw = row["raw"]
    expected_age65 = round(
        sum(
            raw[key]
            for key in ("age65_before_fra", "age65_at_fra", "age65_after_fra")
            if raw[key] is not None
        ),
        1,
    )
    expected_age66 = round(
        sum(
            raw[key]
            for key in ("age66_before_fra", "age66_at_fra", "age66_after_fra")
            if raw[key] is not None
        ),
        1,
    )
    assert row["categories"]["age65"] == expected_age65
    assert row["categories"]["age66"] == expected_age66
