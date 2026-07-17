"""Pin the DRAFT employer-firm aggregate noise-floor artifact (#192).

``runs/employer_firm_floors_draft_v0.json`` is a reported anchor
(workstream B counterpart to the #212 battery): DRAFT, NOT RATIFIED,
no thresholds — it commits the floor-building method for the E1/E2/
E6/E7/E11 aggregate references, and the E11/E12 deferral findings,
before C3 locks. These tests pin its internal consistency and — since
the source extracts are committed — always reproduce it in full.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from populace_dynamics.firms import banding

ARTIFACT = Path(__file__).resolve().parents[1] / (
    "runs/employer_firm_floors_draft_v0.json"
)

CANONICAL_NAMES = {band.name for band in banding.CANONICAL_BANDS}


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_artifact_is_a_draft_with_no_thresholds(artifact):
    assert artifact["artifact"] == "employer_firm_floors"
    assert artifact["version"] == "draft_v0"
    assert "DRAFT" in artifact["status"]
    assert "NOT RATIFIED" in artifact["status"]

    def keys_of(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys_of(value)
        elif isinstance(node, list):
            for value in node:
                yield from keys_of(value)

    assert not any("threshold" in key.lower() for key in keys_of(artifact))


def test_unit_rules_are_carried(artifact):
    rules = " ".join(artifact["unit_rules"])
    assert "jobs, not persons" in rules
    assert "MEAN monthly earnings" in rules
    assert "oslp" in rules


def test_e1_susb_cells(artifact):
    by_sector = artifact["e1"]["susb_2022_share_by_sector_band"]
    assert len(by_sector) == 20  # 19 NAICS sectors + 99 unclassified
    for bands in by_sector.values():
        assert set(bands) <= CANONICAL_NAMES
        total = sum(cell["share"] for cell in bands.values())
        assert total == pytest.approx(1.0, abs=1e-3)
        for cell in bands.values():
            assert cell["noise_flag_worst"] in {"G", "H", "J"}
            if cell["noise_flag_worst"] == "J":
                assert cell["cv_upper_bound"] is None
            else:
                assert 0 < cell["cv_upper_bound"] <= 0.05


def test_e1_bds_margin_carries_the_straddle(artifact):
    groups = artifact["e1"]["bds_size_margin_yoy_stability"]["groups"]
    assert set(groups) == {"1_9", "10_19", "20_99", "100_499", "500_plus"}
    straddle = groups["20_99"]
    assert straddle["exact"] is False
    assert set(straddle["canonical_bands"]) == {"B10_49", "B50_99"}
    shares = [g["share_2022"] for g in groups.values()]
    assert sum(shares) == pytest.approx(1.0, abs=1e-3)
    for group in groups.values():
        assert group["floor_abs_log_ratio_mean"] > 0
        assert group["n_pairs_ex_pandemic"] < group["n_pairs"]


def test_lehd_blocks_shape(artifact):
    for block, rates in (
        (
            artifact["e6_e7"],
            ("e6_hire_rate", "e6_separation_rate", "e7_earns_mean"),
        ),
        (
            artifact["e2"],
            ("hire_rate", "j2j_hire_rate", "ee_separation_rate"),
        ),
    ):
        detail = block["by_firmsize_all_industry"]
        assert set(detail) == {f"firmsize{i}" for i in range(1, 6)}
        for cell in detail.values():
            assert set(cell["canonical_bands"]) <= CANONICAL_NAMES
            assert cell["thin"] is False
            for rate in rates:
                floor = cell[rate]
                assert floor["floor_abs_log_ratio_mean"] > 0
                assert floor["n_pairs_ex_pandemic"] < floor["n_pairs"]
        summary = block["sector_cells"]
        assert summary["n_cells"] == 95  # 19 sectors x 5 sizes
        for rate in rates:
            assert (
                summary[rate]["cell_floor_median"]
                <= summary[rate]["cell_floor_p90"]
                <= summary[rate]["cell_floor_max"]
            )


def test_e7_relative_floor_is_tighter_than_nominal(artifact):
    # The aggregate-relative EarnS floor strips the shared nominal
    # wage trend, so it must come in below the raw nominal floor.
    for cell in artifact["e6_e7"]["by_firmsize_all_industry"].values():
        raw = cell["e7_earns_mean"]["floor_abs_log_ratio_mean"]
        rel = cell["e7_earns_rel_to_aggregate"]["floor_abs_log_ratio_mean"]
        assert rel < raw


def test_method_findings_are_recorded(artifact):
    findings = artifact["method_findings"]
    assert "sector axis has no stability floor" in (
        findings["e1_no_sector_replicate"]
    )
    assert "straddles the canonical 50 edge" in findings["e1_bds_straddle"]
    assert "demographic-free" in findings["e2_no_age_sex_axis"]
    assert "not committed" in findings["e11_no_od_extract"]
    assert "must not lock with C3" in findings["e12_deferred"]
    assert "business-cycle" in findings["cycle_signal_in_floors"]
    assert "nominal wage growth" in findings["e7_nominal_trend"]
    assert artifact["e11"]["status"].startswith("floor not derivable")
    assert artifact["e12"]["status"].startswith("deferred")


def test_reproduces_from_committed_extracts(artifact):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_employer_firm_floors import build

    assert json.loads(json.dumps(build())) == artifact
