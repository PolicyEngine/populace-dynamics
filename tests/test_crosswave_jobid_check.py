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


def test_derived_fields_recompute_from_counts(artifact):
    # Referee S3 (#230 round 1): the committed JSON's derived fields
    # and verdicts must recompute exactly from its own counts, so a
    # hand-edited verdict cannot pass unnoticed.
    within = artifact["within_wave_baseline"]
    seam = artifact["across_wave_seam"]
    assert seam["sep_rate"] == pytest.approx(
        seam["separations"] / seam["jobs_held"], abs=5e-5
    )
    ee_excess = max(
        0.0,
        seam["rekey_signature"] / seam["to_employment"]
        - within["rekey_signature"] / within["to_employment"],
    )
    assert artifact["bounds"][
        "excess_rekey_share_ee_population"
    ] == pytest.approx(ee_excess, abs=5e-5)
    all_seps = ee_excess * seam["to_employment"] / seam["separations"]
    assert artifact["bounds"][
        "excess_rekey_share_all_separations"
    ] == pytest.approx(all_seps, abs=5e-5)

    def band(x):
        if x < 0.15:
            return "PASS"
        if x <= 0.30:
            return "PASS_WITH_CORRECTION_BAND"
        return "REFER_BACK"

    verdicts = artifact["verdict_by_population"]
    assert verdicts["ee_population"] == band(
        artifact["bounds"]["excess_rekey_share_ee_population"]
    )
    assert verdicts["all_separations"] == band(
        artifact["bounds"]["excess_rekey_share_all_separations"]
    )


def test_ee_conditional_baseline_stated(artifact):
    assert "E->E-CONDITIONAL" in artifact["rekey_signature_definition"]
    assert len(artifact["sipp_jobs_reader_commit"]) == 40


def test_inputs_pinned(artifact):
    assert set(artifact["inputs"]) == {"pu2022.csv.gz", "pu2023.csv"}
    for pin in artifact["inputs"].values():
        assert len(pin["sha256"]) == 64
        assert pin["bytes"] > 10_000_000
