"""Pin the paper figures to the committed evidence artifacts.

The figures are deterministic string renderings of committed
``runs/`` artifacts and ``gates.yaml``: a rebuild on any checkout
must reproduce the committed SVGs byte for byte. These tests run
everywhere — they touch only committed files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIGURES = ROOT / "paper" / "figures"


def _builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_paper_figures as builder

    return builder


def test_every_committed_figure_rebuilds_byte_identically():
    builder = _builder()
    artifacts = builder.load_artifacts()
    rebuilt = {
        "autocorr_ladder.svg": builder.build_ladder(artifacts),
        "c2st_noise.svg": builder.build_c2st_noise(),
        "gate_scorecard.svg": builder.build_scorecard(artifacts),
    }
    for name, content in rebuilt.items():
        committed = (FIGURES / name).read_text()
        assert content == committed, (
            f"{name} does not rebuild from the committed artifacts; "
            "rerun scripts/build_paper_figures.py and commit the result"
        )


def test_ladder_uses_all_twelve_registered_runs():
    builder = _builder()
    assert len(builder.RUNS) == 12
    for run, _ in builder.RUNS:
        assert (ROOT / "runs" / f"{run}.json").is_file(), run


def test_ladder_values_recompute_from_artifacts():
    """Spot-pin: each highlighted run's plotted means equal the artifact."""
    builder = _builder()
    svg = (FIGURES / "autocorr_ladder.svg").read_text()
    for run in ("gate1_qrf_baseline_v1", "gate1_rank_knn_v4"):
        art = json.loads((ROOT / "runs" / f"{run}.json").read_text())
        per = art["per_seed"]
        for h in builder.HORIZONS:
            mean = sum(
                s["battery_values"][f"autocorr_log_{h}yr"] for s in per
            ) / len(per)
            # The plotted y-coordinate of this mean appears in the SVG.
            bands = builder.load_bands()
            del bands  # bands only shape the axes; y-map is fixed
            y_lo, y_hi, top, height = 0.28, 0.84, 28, 430 - 28 - 46
            y = top + (y_hi - mean) / (y_hi - y_lo) * height
            assert f"cy='{y:.1f}'" in svg, (run, h)


def test_noise_figure_marks_threshold_and_both_means():
    art = json.loads((ROOT / "runs" / "c10_diagnostics_v1.json").read_text())
    ext = art["diagnostic_1_seed_extension"]
    svg = (FIGURES / "c2st_noise.svg").read_text()
    mean20 = ext["candidate_c2st_distribution"]["mean"]
    locked = [
        r["candidate_pairs_c2st"]
        for r in ext["per_seed"]
        if r["source"].startswith("committed")
    ]
    assert len(locked) == 5
    assert f"{mean20:.4f}" in svg
    assert f"{sum(locked) / 5:.4f}" in svg
    assert str(ext["pairs_c2st_threshold"]) in svg


def test_scorecard_verdicts_match_artifacts():
    builder = _builder()
    svg = (FIGURES / "gate_scorecard.svg").read_text()
    n_fail = svg.count(">fail</text>")
    verdicts = []
    for run, _ in builder.RUNS:
        art = json.loads((ROOT / "runs" / f"{run}.json").read_text())
        verdicts.append(art["verdict"]["gate_1_pass"])
    assert n_fail == sum(1 for v in verdicts if not v)
    assert svg.count(">pass</text>") == sum(1 for v in verdicts if v)


def test_figures_use_only_palette_colors():
    """Every fill/stroke in the SVGs comes from the declared palette."""
    builder = _builder()
    import re

    allowed = {
        builder.BLUE,
        builder.ORANGE,
        builder.GOOD,
        builder.BAD,
        builder.INK,
        builder.MUTED,
        builder.CONTEXT,
        builder.GRID,
        "#ffffff",
        "none",
    }
    for path in sorted(FIGURES.glob("*.svg")):
        used = set(re.findall(r"(?:fill|stroke)='([^']+)'", path.read_text()))
        used |= set(re.findall(r"background:(#[0-9a-f]{6})", path.read_text()))
        assert used <= allowed, (path.name, used - allowed)


def test_builder_reads_thresholds_not_literals():
    """The band geometry comes from gates.yaml + the committed reference."""
    bands = _builder().load_bands()
    assert set(bands) == {2, 4, 10}
    for ref, tol in bands.values():
        assert 0 < tol < 0.1
        assert 0.3 < ref < 0.8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
