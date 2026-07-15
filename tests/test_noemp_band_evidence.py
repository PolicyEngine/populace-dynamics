"""Pin the NOEMP band-label evidence artifact (issue #192).

The committed ``runs/noemp_band_evidence_v1.json`` records the
discontinuity test behind the C2 decision to read ASEC NOEMP codes
2/3 as 10-49 / 50-99 in every year. These tests pin the artifact's
internal consistency, and — when the ASEC files are staged —
reproduce it from the raw data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT = Path(__file__).resolve().parents[1] / (
    "runs/noemp_band_evidence_v1.json"
)

REAL_DATA = Path("~/PolicyEngine/asec-data").expanduser()
needs_real_asec = pytest.mark.skipif(
    not (REAL_DATA / "pppub24.csv").exists(),
    reason="ASEC person files not staged",
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_artifact_shape(artifact):
    assert artifact["artifact"] == "noemp_band_evidence"
    assert artifact["version"] == 1
    assert set(artifact["share_pct_by_year"]) == {"2017", "2019", "2024"}
    for year_block in artifact["share_pct_by_year"].values():
        shares = year_block["share_pct_by_code"]
        assert set(shares) == {"1", "2", "3", "4", "5", "6"}
        assert sum(shares.values()) == pytest.approx(100.0, abs=0.1)
        assert year_block["workers_unweighted"] > 50_000


def test_continuity_is_the_finding(artifact):
    # The discriminating fact: a genuine 10-49 -> 10-24 re-bin
    # would move codes 2/3 by ~7 pp; the committed shares move by
    # under one point across seven years.
    assert artifact["max_adjacent_delta_pct_codes_2_3"] < 1.0
    shares = artifact["share_pct_by_year"]
    for year in ("2017", "2019", "2024"):
        code3 = shares[year]["share_pct_by_code"]["3"]
        # Code 3 tracks SUSB's 50-99 employment share (~7.5%), not
        # a 25-99 band (~15%).
        assert 5.0 < code3 < 9.0


def test_dictionary_conflict_is_recorded(artifact):
    labels = artifact["dictionary_labels_by_year"]
    assert labels["2017"]["2"] == "10-49"
    assert labels["2019"]["2"] == "10-24"
    assert "documentary" in artifact["finding"]


@needs_real_asec
def test_reproduces_from_staged_files(artifact):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_noemp_band_evidence import build

    rebuilt = build()
    assert rebuilt["share_pct_by_year"] == artifact["share_pct_by_year"]
    assert (
        rebuilt["max_adjacent_delta_pct_codes_2_3"]
        == artifact["max_adjacent_delta_pct_codes_2_3"]
    )
