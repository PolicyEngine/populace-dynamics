"""Tests for the gate-2 family-transition floors (runs/gate2_floors_v1.json).

The artifact is PRE-LOCK EVIDENCE (reads no gate, changes no gate): the
committed reference moments, the person-disjoint half-split noise floor
the DRAFT gate-2 thresholds derive from, and the honest NCHS external
anchors. It is pinned like the other ``runs/`` floors.

Two tiers, mirroring ``tests/test_mortality_floors.py``:

* Always-runnable internal-consistency tests touching only committed
  files: the schema is a reported anchor whose thresholds are draft, not
  ratified; the NCHS references it cites are pinned by sha256; every
  pooled floor statistic recomputes from the per-seed cells; every DRAFT
  threshold recomputes as round(floor mean + k*sd); every external ratio
  recomputes from its parts; and the gate-eligible/report-only partition
  matches the stored stability data.
* A seed-0 + anchor reproduction pin (skipped when the PSID history files
  are absent) that rebuilds the panel and reruns the seed-0 half-split and
  the ASFR anchor, matching the committed numbers -- with populace.fit
  never imported.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2_floors_v1.json"
ASFR = ROOT / "data" / "external" / "nchs_asfr_2024.json"
MD = ROOT / "data" / "external" / "nchs_marriage_divorce_rates_2023.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23" / "MH85_23.txt").is_file()
    or not (REAL_DATA / "cah85_23" / "CAH85_23.txt").is_file()
    or not (REAL_DATA / "ind2023er").is_dir(),
    reason="PSID history files not staged",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_gate2_floors as builder

    return builder


# --------------------------------------------------------------------------
# Schema and pre-lock framing (always runnable)
# --------------------------------------------------------------------------
def test_schema_and_reported_anchor():
    art = _artifact()
    assert art["schema_version"] == "gate2_floors.v1"
    assert art["run"] == "gate2_floors_v1"
    assert art["reported_anchor_not_gated"] is True
    assert "changes no gate" in art["purpose"]
    assert "family transitions" in art["component"]
    assert art["holdout_basis"] == ["mh85_23", "cah85_23", "MX23REL"]
    note = art["draft_thresholds_note"]
    assert "NOT RATIFIED" in note
    assert "referee round" in note
    assert "draft_pending_referee_round" in note


def test_no_reform_scored():
    art = _artifact()
    assert art["reported_anchor_not_gated"] is True
    assert "reform" not in art
    assert "gate_result" not in art


def test_nchs_references_pinned_by_sha256():
    art = _artifact()
    anchor = art["external_anchor"]
    for ref_path, block in (
        (ASFR, anchor["asfr"]),
        (MD, anchor["marriage_divorce"]),
    ):
        committed = hashlib.sha256(ref_path.read_bytes()).hexdigest()
        assert block["nchs_reference_sha256"] == committed
    pins = art["revision_pins"]
    assert (
        pins["nchs_asfr_sha256"]
        == hashlib.sha256(ASFR.read_bytes()).hexdigest()
    )
    assert (
        pins["nchs_marriage_divorce_sha256"]
        == hashlib.sha256(MD.read_bytes()).hexdigest()
    )


# --------------------------------------------------------------------------
# Pooled floor recomputes from per-seed (always runnable)
# --------------------------------------------------------------------------
def test_pooled_floor_recomputes_from_per_seed():
    art = _artifact()
    per_seed = art["noise_floor_per_seed"]
    assert [s["seed"] for s in per_seed] == list(
        art["internal_noise_floor"]["seeds"]
    )
    for key, block in art["noise_floor_seeds_0_4"].items():
        log_ratios = [s["cells"][key]["log_ratio_abs"] for s in per_seed]
        assert all(v is not None for v in log_ratios), key
        assert block["values"] == pytest.approx(log_ratios)
        assert block["mean"] == pytest.approx(np.mean(log_ratios))
        assert block["sd"] == pytest.approx(np.std(log_ratios, ddof=1))
        pct = [s["cells"][key]["pct_diff_abs"] for s in per_seed]
        assert block["pct_diff_abs"]["values"] == pytest.approx(pct)
        assert block["pct_diff_abs"]["mean"] == pytest.approx(np.mean(pct))


def test_per_seed_cells_are_internally_consistent():
    art = _artifact()
    for seed in art["noise_floor_per_seed"]:
        for key, cell in seed["cells"].items():
            if cell["log_ratio_abs"] is None:
                assert cell["rate_a"] == 0 or cell["rate_b"] == 0, key
                continue
            expected = abs(math.log(cell["rate_a"] / cell["rate_b"]))
            assert cell["log_ratio_abs"] == pytest.approx(expected), key


# --------------------------------------------------------------------------
# DRAFT thresholds recompute (always runnable)
# --------------------------------------------------------------------------
def test_draft_thresholds_recompute_from_floor():
    """Every DRAFT tolerance == round(floor mean + k*sd, rounding)."""
    art = _artifact()
    k = art["draft_thresholds"]["k"]
    rounding = art["draft_thresholds"]["rounding"]
    floor = art["noise_floor_seeds_0_4"]
    for key, spec in art["draft_thresholds"]["cells"].items():
        mean = floor[key]["mean"]
        sd = floor[key]["sd"]
        assert spec["derivation"]["floor_mean"] == pytest.approx(mean)
        assert spec["derivation"]["floor_sd"] == pytest.approx(sd)
        derived = round(mean + k * sd, rounding)
        assert spec["log_ratio_abs_max"] == pytest.approx(derived), key


def test_draft_thresholds_only_on_gate_eligible_cells():
    art = _artifact()
    drafts = set(art["draft_thresholds"]["cells"])
    gate_eligible = {
        k for k, v in art["cell_stability"].items() if v["gate_eligible"]
    }
    report_only = {
        k for k, v in art["cell_stability"].items() if not v["gate_eligible"]
    }
    assert drafts == gate_eligible
    assert drafts.isdisjoint(report_only)
    # Reference moments cover every cell (gated + report-only).
    assert set(art["reference_moments"]) == gate_eligible | report_only


# --------------------------------------------------------------------------
# External anchors recompute (always runnable)
# --------------------------------------------------------------------------
def test_asfr_ratios_recompute_from_parts():
    art = _artifact()
    for window in art["external_anchor"]["asfr"]["windows"].values():
        for band, cell in window["by_band"].items():
            if cell["ratio"] is None or cell["nchs_asfr_per_1000"] == 0:
                continue
            assert cell["ratio"] == pytest.approx(
                cell["psid_asfr_per_1000"] / cell["nchs_asfr_per_1000"]
            ), band


def test_asfr_recent_window_tracks_reality():
    """The concept-aligned recent-window ASFR sits near the NCHS level.

    Not a gate -- a sanity check that the honest anchor lands where the
    docstring says (recent-window median PSID/NCHS ratio ~1), evidence the
    fertility construction is not wildly off despite the coverage caveats.
    """
    art = _artifact()
    recent = art["external_anchor"]["asfr"]["windows"]["recent"]
    assert 0.7 < recent["ratio_summary"]["median_ratio"] < 1.4


def test_marriage_divorce_ratios_recompute_and_ordering():
    art = _artifact()
    md = art["external_anchor"]["marriage_divorce"]
    for window in md["windows"].values():
        m = window["psid_marriage_rate_per_1000_py15plus"]
        d = window["psid_divorce_rate_per_1000_py15plus"]
        assert window["marriage_ratio"] == pytest.approx(
            m / window["nchs_marriage_rate_per_1000_totalpop"]
        )
        assert window["divorce_ratio"] == pytest.approx(
            d / window["nchs_divorce_rate_per_1000_totalpop"]
        )
        # PSID crude-equivalent exceeds the national crude rate (the 15+
        # denominator concept delta), reported not tuned.
        assert window["marriage_ratio"] > 1.0
        assert m > d  # marriage exceeds divorce, as nationally


# --------------------------------------------------------------------------
# Stability partition (always runnable)
# --------------------------------------------------------------------------
def test_stability_partition_matches_rule():
    art = _artifact()
    stability = art["cell_stability"]
    n_seeds = len(art["internal_noise_floor"]["seeds"])
    min_events = art["internal_noise_floor"]["min_events_for_gate"]

    gate_eligible, report_only = set(), set()
    for key, v in stability.items():
        expected = v["defined_seeds"] == n_seeds and (
            v["min_events_either_half"] >= min_events
        )
        assert v["gate_eligible"] is expected, key
        (gate_eligible if v["gate_eligible"] else report_only).add(key)

    # A real, non-degenerate partition covering every cell.
    assert gate_eligible and report_only
    assert gate_eligible.isdisjoint(report_only)
    assert len(gate_eligible) + len(report_only) == len(
        art["reference_moments"]
    )
    # Every gate-eligible cell carries a noise-floor sd basis.
    for key in gate_eligible:
        assert key in art["noise_floor_seeds_0_4"], key


# --------------------------------------------------------------------------
# Seed-0 + anchor reproduction (needs PSID; NO populace-fit)
# --------------------------------------------------------------------------
@needs_real
def test_seed0_and_anchor_reproduce_without_populace_fit():
    assert "populace.fit" not in sys.modules
    builder = _import_builder()
    assert (
        "populace.fit" not in sys.modules
    ), "importing the gate-2 builder pulled populace.fit"

    panel, fert, _ = builder.load_panels()
    art = _artifact()

    got0 = builder.measure_seed_halfsplit(0, panel, fert)
    ref0 = next(s for s in art["noise_floor_per_seed"] if s["seed"] == 0)
    assert got0["n_persons_side_a"] == ref0["n_persons_side_a"]
    assert got0["n_persons_side_b"] == ref0["n_persons_side_b"]
    for key, ref_cell in ref0["cells"].items():
        got_cell = got0["cells"][key]
        assert got_cell["rate_a"] == pytest.approx(
            ref_cell["rate_a"], abs=1e-12
        ), key
        assert got_cell["n_events_a"] == ref_cell["n_events_a"], key
        if ref_cell["log_ratio_abs"] is None:
            assert got_cell["log_ratio_abs"] is None, key
        else:
            assert got_cell["log_ratio_abs"] == pytest.approx(
                ref_cell["log_ratio_abs"], abs=1e-12
            ), key

    got_asfr = builder.asfr_anchor(fert)
    ref_asfr = art["external_anchor"]["asfr"]["windows"]["recent"]["by_band"]
    got_recent = got_asfr["windows"]["recent"]["by_band"]
    for band, ref_cell in ref_asfr.items():
        assert got_recent[band]["psid_asfr_per_1000"] == pytest.approx(
            ref_cell["psid_asfr_per_1000"], abs=1e-9
        ), band

    assert "populace.fit" not in sys.modules
