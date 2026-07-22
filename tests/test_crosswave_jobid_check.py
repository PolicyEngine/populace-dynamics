"""Pin the cross-wave job-ID check artifact (#230 §6 pre-lock)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT = Path(__file__).resolve().parents[1] / (
    "runs/crosswave_jobid_check_draft_v0.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_disclosure_language(artifact):
    # The status must carry the disclosed-re-analysis framing, never
    # a pre-registration claim (review on #235).
    assert "DISCLOSED RE-ANALYSIS" in artifact["status"]
    assert "UNRATIFIED" in artifact["status"]
    assert "pre-registered" not in artifact["status"]


def test_both_populations_reported(artifact):
    bounds = artifact["bounds"]
    assert bounds["excess_rekey_share_ee_population"] == 0.1512
    assert bounds["excess_rekey_share_all_separations"] == 0.0936
    verdicts = artifact["verdict_by_population"]
    assert verdicts["ee_population"] == "PASS_WITH_CORRECTION_BAND"
    assert verdicts["all_separations"] == "PASS"
    assert verdicts["operative"] == "REFEREE"


def test_identity_is_labelled(artifact):
    identity = artifact["bounds"]["gross_id_survival_identity"]
    assert identity["value"] == pytest.approx(
        1 - artifact["across_wave_seam"]["sep_rate"]
    )
    assert "NOT evidence" in identity["note"]


def test_uncertainty_and_strict_variant(artifact):
    bounds = artifact["bounds"]
    assert (
        bounds["one_sided_95_upper_ee_population"]
        > bounds["excess_rekey_share_ee_population"]
    )
    strict = artifact["strict_nan_variant"]
    assert strict["excess_ee_population"] == 0.1232
    assert "MISMATCH" in strict["note"]


def test_inputs_pinned(artifact):
    assert set(artifact["inputs"]) == {"pu2022.csv.gz", "pu2023.csv"}
    for pin in artifact["inputs"].values():
        assert len(pin["sha256"]) == 64
        assert pin["bytes"] > 10_000_000
