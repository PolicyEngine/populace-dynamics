"""Pin the three Workstream A floor artifacts (#212, pre-C3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RUNS = Path(__file__).resolve().parents[1] / "runs"


@pytest.fixture(scope="module")
def spells() -> dict:
    return json.loads((RUNS / "sipp_spell_floors_draft_v0.json").read_text())


@pytest.fixture(scope="module")
def tenure() -> dict:
    return json.loads((RUNS / "tenure_floors_draft_v0.json").read_text())


@pytest.fixture(scope="module")
def e8e9() -> dict:
    return json.loads((RUNS / "sipp_e8_e9_floors_draft_v0.json").read_text())


def test_all_carry_draft_status_and_scale_gap(spells, tenure, e8e9):
    for artifact in (spells, tenure, e8e9):
        assert "DRAFT" in artifact["status"]
        assert "RECORDED GAP" in artifact["deployment_scale_note"]
        assert "ctx20" in artifact["deployment_scale_note"]


def test_e4_e5_pinned_values(spells):
    e4 = spells["e4_retention_by_age_sex"]["16_24|sex1"]
    assert e4["rate"] == 0.9897
    assert e4["floor_abs_log_ratio_mean"] == 0.00182
    e5 = spells["e5_runs_by_age"]["45_54"]
    assert e5["full_year_run_share"] == 0.8671
    assert spells["seam_caveat"]


def test_tenure_pinned_values_and_heaping(tenure):
    cell = tenure["by_year"]["2024"]["35_44"]
    assert cell["p50"] == 5.0
    assert cell["floor_abs_gap_years"]["p50"]["mean"] == 0.0
    assert cell["floor_ecdf_max_gap"]["mean"] > 0
    assert "exactly zero" in tenure["heaping_caveat"]


def test_e8_e9_pinned_values(e8e9):
    assert "seeds 0-19" in e8e9["method"]
    assert "ESTIMAND NOTE" in e8e9["source"]
    mix = e8e9["e9_transitions"]["transition_rates"]
    assert mix["stay"] == 0.977
    assert mix["j2j"] == 0.0035
    stay = e8e9["e9_transitions"]["earnings_change"]["stay"]
    assert stay["median_log_change"] == 0.0
    assert "heaps at exactly 0" in e8e9["stay_median_heaping_caveat"]
    e8 = e8e9["e8_nonemployment_by_age"]["16_24"]
    assert e8["any_nonemp_share"] == pytest.approx(0.4145, abs=0.001)
