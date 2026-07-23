"""Pin the v1 employer-firm aggregate noise-floor artifact (#192).

``runs/employer_firm_floors_v1.json`` is a reported anchor
(workstream B counterpart to the #212 battery): PRE-LOCK, NOT
RATIFIED, no thresholds — it commits the floor-building method for
the E1/E2/E6/E7/E11 aggregate references, and the E11/E12 deferral
findings, before C3 locks.

**v1 is a pinning event, not a ratification** (#230 section 12.2
item 2). Three digests are pinned, and each catches a different way
the artifact could drift out from under the C3 record:

* the artifact's own bytes — an edited artifact;
* the builder's bytes — a changed method that happens to land on
  the same numbers, or a reproduction test quietly rewritten to
  agree with a new build;
* every input extract's bytes — a re-fetched source. This is the
  one a reproduction test alone cannot catch: rebuild from a
  silently changed extract and the artifact and the rebuild agree
  with each other while both differ from what C3 was shown.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from populace_dynamics.firms import banding

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs/employer_firm_floors_v1.json"
BUILDER = ROOT / "scripts/build_employer_firm_floors.py"

ARTIFACT_SHA256 = (
    "c9c50b7521b1df3ee0c9ffc942a3aa8593fc89c0eec8d1cc1682ce3a426716ed"
)
BUILDER_SHA256 = (
    "b85c2234289c99e166f9343a69a1bb14417b98bb77351b9ba9ca4131f3625677"
)

CANONICAL_NAMES = {band.name for band in banding.CANONICAL_BANDS}


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_artifact_is_a_draft_with_no_thresholds(artifact):
    assert artifact["artifact"] == "employer_firm_floors"
    assert artifact["version"] == "v1"
    # "DRAFT" gave way to "PRE-LOCK REFERENCE" at v1: the artifact
    # is pinned now, so calling it a draft would misdescribe it. What
    # must not weaken is the ratification status.
    assert "PRE-LOCK REFERENCE" in artifact["status"]
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
    # The two findings draft_v0 deferred, now superseded rather than
    # silently dropped: #228 landed the extracts each one blamed.
    assert "e2_no_age_sex_axis" not in findings
    assert "e11_no_od_extract" not in findings
    assert "SUPERSEDES" in findings["e2_sex_age_axis_built"]
    assert "'sa'" in findings["e2_sex_age_axis_built"]
    assert "sex x EDUCATION" in findings["e2_sex_age_axis_built"]
    assert "SUPERSEDES" in (
        findings["e11_extract_committed_but_no_temporal_replicate"]
    )
    assert "ONE pair per detail cell" in (
        findings["e11_extract_committed_but_no_temporal_replicate"]
    )
    assert "revision" in findings["release_revision_noise_unfloored"]
    assert "not evidence that it is zero" in (
        findings["release_revision_noise_unfloored"]
    )
    assert "trend, not noise" in findings["e11_margin_trend"]
    assert "must not lock with C3" in findings["e12_deferred"]
    assert "business-cycle" in findings["cycle_signal_in_floors"]
    assert "nominal wage growth" in findings["e7_nominal_trend"]
    assert artifact["e11"]["status"].startswith("detail floor NOT")
    assert artifact["e12"]["status"].startswith("deferred")


def test_reproduces_from_committed_extracts(artifact):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_employer_firm_floors import build

    assert json.loads(json.dumps(build())) == artifact


def test_e2_sex_age_axis_is_built(artifact):
    """E2's registered gate axis has a floor, not a deferral."""
    block = artifact["e2"]["by_sex_age"]
    # Full 3 sexes x 9 age groups, margins included.
    assert len(block["cells"]) == 27
    for cell in block["cells"].values():
        for name in (
            "hire_rate",
            "separation_rate",
            "j2j_hire_rate",
            "j2j_separation_rate",
        ):
            floor = cell[name]
            assert floor["n_pairs"] > 0
            assert floor["floor_abs_log_ratio_mean"] > 0
            assert floor["floor_abs_log_ratio_sd"] is not None
            # Ex-pandemic is a strict subset of the full sample.
            assert floor["n_pairs_ex_pandemic"] < floor["n_pairs"]


def test_e2_sex_age_cross_cell_excludes_the_margins(artifact):
    """Pooling margins with the cells they aggregate would double count."""
    summary = artifact["e2"]["by_sex_age"]["cross_cell"]
    # 2 sexes x 8 age groups; the all-sexes / all-ages rows are out.
    assert summary["n_cells"] == 16
    assert "non-margin cells only" in summary["note"]


def test_e2_sex_age_floors_are_not_monotone_in_disaggregation(artifact):
    """A margin's floor does not bound the floors beneath it.

    The intuition that disaggregating can only add noise is wrong
    here, and the threshold policy depends on knowing that: the
    45-99 age cells are more temporally stable than the all-sexes
    all-ages cell, which carries compositional shift they do not.
    So a floor measured on a margin cannot stand in as a
    conservative bound for its constituent cells.
    """
    cells = artifact["e2"]["by_sex_age"]["cells"]
    tighter = {
        family: sum(
            1
            for key, cell in cells.items()
            if key != "sex0_A00"
            and cell[family]["ex_pandemic_mean"]
            < cells["sex0_A00"][family]["ex_pandemic_mean"]
        )
        for family in (
            "hire_rate",
            "separation_rate",
            "j2j_hire_rate",
            "j2j_separation_rate",
        )
    }
    assert tighter == {
        "hire_rate": 13,
        "separation_rate": 10,
        "j2j_hire_rate": 6,
        "j2j_separation_rate": 7,
    }
    # And the direction of the pattern: oldest tighter than youngest.
    assert (
        cells["sex1_A08"]["separation_rate"]["ex_pandemic_mean"]
        < cells["sex1_A04"]["separation_rate"]["ex_pandemic_mean"]
    )


def test_non_monotonicity_is_recorded_as_a_finding(artifact):
    finding = artifact["method_findings"][
        "floors_not_monotone_in_disaggregation"
    ]
    assert "CANNOT be used as a conservative bound" in finding


def test_e11_detail_window_gives_one_pair_per_cell(artifact):
    """The reason the E11 cross has no floor, pinned as a number.

    Not availability -- the extract is committed (#228). The national
    origin x destination detail is published for 2015Q1-2016Q1 only,
    so same-quarter YoY pairing yields one pair per cell: a gap with
    no dispersion, hence no mean + k*sd floor.
    """
    window = artifact["e11"]["detail_window"]
    assert window["n_quarters"] == 5
    assert window["observed_quarters"][0] == "2015Q1"
    assert window["observed_quarters"][-1] == "2016Q1"
    assert window["max_yoy_pairs_per_cell"] == 1


def test_e11_margin_relative_floor_is_tighter_than_raw(artifact):
    """EE counts carry aggregate flow growth, as EarnS carries wages."""
    for cell in artifact["e11"]["destination_size_margin"].values():
        assert cell["ee_rel"]["floor_abs_log_ratio_mean"] < (
            cell["ee"]["floor_abs_log_ratio_mean"]
        )


def test_e11_records_the_cross_source_margin_disagreement(artifact):
    """The margins-only bound that survives the missing detail."""
    note = artifact["e11"]["cross_source_margin_disagreement"]
    assert note["all_size_ee_tool_above_flat_file"] == "37 of 41 quarters"
    lo, hi = note["per_size_deviation_range_pct"]
    assert lo < 0 < hi
    assert "margins-only" in note["note"]


# ---------------------------------------------------------------
# v1 pinning (#230 section 12.2 item 2)
# ---------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_artifact_sha256_is_pinned():
    assert _sha256(ARTIFACT) == ARTIFACT_SHA256


def test_builder_sha256_is_pinned():
    """A changed method must not slip past the reproduction test.

    ``test_reproduces_from_committed_extracts`` compares the builder
    against the artifact, so editing both together passes. Pinning
    the builder makes that edit visible.
    """
    assert _sha256(BUILDER) == BUILDER_SHA256


def test_input_extract_digests_match_the_committed_files(artifact):
    """The drift a reproduction test structurally cannot catch.

    If an extract is re-fetched, the artifact and a rebuild from it
    agree with each other while both differ from what the C3 record
    was shown. Only a digest recorded *at build time* and compared
    against the file *now* separates those.
    """
    recorded = artifact["input_extract_sha256"]
    for name, digest in recorded.items():
        path = ROOT / "data" / "external" / name
        assert path.exists(), f"{name} is recorded but not committed"
        assert _sha256(path) == digest, (
            f"{name} has changed since the floors were built; rebuild "
            "the artifact and re-pin deliberately, and say so in the "
            "C3 record — the floors move with it"
        )


def test_every_consumed_extract_is_digest_recorded(artifact):
    """No source may be consumed without appearing in the pin."""
    recorded = set(artifact["input_extract_sha256"])
    sources = {
        Path(value).name
        for key, value in artifact["sources"].items()
        if key != "provenance"
    }
    assert sources == recorded


def test_v1_is_pinned_but_not_ratified(artifact):
    # The distinction the whole ceremony rests on: pinning makes the
    # numbers immovable, not binding. Thresholds arrive only with the
    # C3 amendment PR.
    status = artifact["status"]
    assert "NOT RATIFIED" in status
    assert "no thresholds" in status
    assert "not a ratification" in status
