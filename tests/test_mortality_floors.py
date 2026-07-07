"""Tests for the mortality foundation floors (runs/mortality_floors_v1.json).

The artifact is a REPORTED ANCHOR (reads no gate, changes no gate): the
PSID-vs-NCHS differential-mortality external anchor plus the person-
disjoint half-split noise floor a future gate would derive thresholds
from. It is pinned like the other ``runs/`` floors.

Two tiers, mirroring ``tests/test_pia_proxy_floor.py``:

* Always-runnable internal-consistency tests touching only committed
  files: the schema is a reported anchor with a NOT-RATIFIED proposal;
  the NCHS reference it cites is pinned by sha256; every pooled floor
  statistic recomputes from the per-seed cells; every PSID/NCHS ratio
  recomputes from its parts; each NCHS band rate recomputes from the
  committed life table; and the gate-eligible/report-only partition
  matches the stored stability data. No reform is scored here.
* A seed-0 + anchor reproduction pin (skipped when the PSID individual
  file is absent) that rebuilds the exposure slices and reruns the
  seed-0 half-split and the all-window anchor, matching the committed
  numbers to float precision -- with populace.fit never imported.
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
ARTIFACT = ROOT / "runs" / "mortality_floors_v1.json"
NCHS = ROOT / "data" / "external" / "nchs_life_tables_2023.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_ind = pytest.mark.skipif(
    not (REAL_DATA / "ind2023er").is_dir(),
    reason="PSID ind2023er not staged",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_mortality_floors as builder

    return builder


# --------------------------------------------------------------------------
# Schema and reported-anchor framing (always runnable)
# --------------------------------------------------------------------------
def test_schema_and_reported_anchor():
    art = _artifact()
    assert art["schema_version"] == "mortality_floors.v1"
    assert art["run"] == "mortality_floors_v1"
    assert art["reported_anchor_not_gated"] is True
    assert "changes no gate" in art["purpose"]
    assert "differential mortality" in art["component"]
    # The proposed standard exists but is explicitly NOT ratified.
    note = art["proposed_thresholds_note"]
    assert "NOT RATIFIED" in note
    assert "referee round" in note


def test_no_reform_scored():
    """This is a pure data/anchor artifact -- it scores no reform."""
    art = _artifact()
    assert art["reported_anchor_not_gated"] is True
    assert "reform" not in art
    assert "gate_result" not in art


def test_nchs_reference_pinned_by_sha256():
    """The anchor pins the exact committed NCHS reference it aggregated."""
    art = _artifact()
    committed_sha = hashlib.sha256(NCHS.read_bytes()).hexdigest()
    assert art["external_anchor"]["nchs_reference_sha256"] == committed_sha
    assert art["revision_pins"]["nchs_reference_sha256"] == committed_sha
    assert art["external_anchor"]["nchs_vintage_year"] == 2023
    # The NCHS source-file hashes are carried through from the reference.
    ref = json.loads(NCHS.read_text())
    carried = art["external_anchor"]["nchs_source_file_sha256"]
    for pop, meta in ref["fetch"]["source_files"].items():
        assert carried[pop] == meta["sha256"]


# --------------------------------------------------------------------------
# Pooled floor recomputes from per-seed (always runnable)
# --------------------------------------------------------------------------
def test_pooled_floor_recomputes_from_per_seed():
    art = _artifact()
    inf = art["internal_noise_floor"]
    per_seed = inf["per_seed"]
    assert [s["seed"] for s in per_seed] == list(inf["seeds"])

    for key, block in inf["noise_floor_seeds_0_4"].items():
        log_ratios = [s["cells"][key]["log_ratio_abs"] for s in per_seed]
        assert all(v is not None for v in log_ratios), key
        assert block["values"] == pytest.approx(log_ratios)
        assert block["mean"] == pytest.approx(np.mean(log_ratios))
        assert block["sd"] == pytest.approx(np.std(log_ratios, ddof=1))
        # Nested percent-gap sub-block recomputes too.
        pct = [s["cells"][key]["pct_diff_abs"] for s in per_seed]
        assert block["pct_diff_abs"]["values"] == pytest.approx(pct)
        assert block["pct_diff_abs"]["mean"] == pytest.approx(np.mean(pct))


def test_per_seed_cells_are_internally_consistent():
    """Each per-seed cell's log-ratio equals |ln(m_a/m_b)| from its rates."""
    art = _artifact()
    for seed in art["internal_noise_floor"]["per_seed"]:
        for key, cell in seed["cells"].items():
            if cell["log_ratio_abs"] is None:
                assert cell["m_a"] == 0 or cell["m_b"] == 0, key
                continue
            expected = abs(math.log(cell["m_a"] / cell["m_b"]))
            assert cell["log_ratio_abs"] == pytest.approx(expected), key


# --------------------------------------------------------------------------
# External anchor recomputes (always runnable)
# --------------------------------------------------------------------------
def test_external_ratios_recompute_from_parts():
    art = _artifact()
    for window in art["external_anchor"]["windows"].values():
        for key, cell in window["by_band_sex"].items():
            if cell["ratio"] is None:
                continue
            assert cell["psid_m"] == pytest.approx(
                cell["psid_deaths_wt"] / cell["psid_exposure_py"]
            ), key
            assert cell["ratio"] == pytest.approx(
                cell["psid_m"] / cell["nchs_M"]
            ), key


def test_nchs_band_rates_recompute_from_reference():
    """Each cell's nchs_M equals the band central rate from the committed
    life table: (l_a - l_{b+1}) / (T_a - T_{b+1})."""
    art = _artifact()
    ref = json.loads(NCHS.read_text())

    def band_bounds(band: str) -> tuple[int, int]:
        if band.endswith("+"):
            return int(band[:-1]), 120
        lo, hi = band.split("-")
        return int(lo), int(hi)

    all_window = art["external_anchor"]["windows"]["all"]["by_band_sex"]
    for key, cell in all_window.items():
        band, sex = key.split("|")
        lo, hi = band_bounds(band)
        rows = {r["age"]: r for r in ref["tables"][sex]}
        lx = {a: rows[a]["lx"] for a in rows}
        tx = {a: rows[a]["Tx"] for a in rows}
        deaths_b = lx[lo] - lx.get(hi + 1, 0.0)
        py_b = tx[lo] - tx.get(hi + 1, 0.0)
        assert cell["nchs_M"] == pytest.approx(deaths_b / py_b), key


def test_undercount_reported_not_calibrated():
    """The undercount is reported honestly, not calibrated to 1.

    Every estimable all-window ratio is below 1 (PSID observes fewer
    deaths per person-year than the NCHS period population), and the
    artifact is a reported anchor -- no ratio is scaled toward 1.
    """
    art = _artifact()
    all_window = art["external_anchor"]["windows"]["all"]
    ratios = [
        c["ratio"]
        for c in all_window["by_band_sex"].values()
        if c["ratio"] is not None
    ]
    assert ratios
    assert all(r < 1.0 for r in ratios)
    assert all_window["ratio_summary"]["median_ratio"] < 1.0
    assert art["external_anchor"]["undercount_note"]


# --------------------------------------------------------------------------
# Stability partition (always runnable)
# --------------------------------------------------------------------------
def test_stability_partition_matches_rule():
    """gate_eligible <=> defined on all seeds AND >= 20 deaths per half."""
    art = _artifact()
    inf = art["internal_noise_floor"]
    stability = inf["band_sex_stability"]
    n_seeds = len(inf["seeds"])

    gate_eligible, report_only = set(), set()
    for key, v in stability.items():
        expected = v["defined_seeds"] == n_seeds and (
            v["min_deaths_either_half"] >= 20
        )
        assert v["gate_eligible"] is expected, key
        (gate_eligible if v["gate_eligible"] else report_only).add(key)

    # The partition covers every band x sex and the two sides are disjoint.
    n_cells = len(art["age_bands"]) * len(art["sexes"])
    assert len(gate_eligible) + len(report_only) == n_cells
    assert gate_eligible.isdisjoint(report_only)
    # There is at least one of each (a real, non-degenerate partition).
    assert report_only and gate_eligible
    # Every gate-eligible cell carries a noise-floor sd basis.
    for key in gate_eligible:
        assert key in inf["noise_floor_seeds_0_4"], key


# --------------------------------------------------------------------------
# Seed-0 + anchor reproduction (needs PSID; NO populace-fit)
# --------------------------------------------------------------------------
@needs_real_ind
def test_seed0_and_anchor_reproduce_without_populace_fit():
    assert "populace.fit" not in sys.modules
    builder = _import_builder()
    assert (
        "populace.fit" not in sys.modules
    ), "importing the mortality builder pulled populace.fit"

    from populace_dynamics.data import deaths, panels

    demo = panels.demographic_panel()
    death_records = deaths.read_death_records()
    slices = builder.build_exposure_slices(demo, death_records)

    art = _artifact()

    # Seed-0 half-split cells reproduce to float precision.
    got0 = builder.measure_seed_halfsplit(0, slices)
    ref0 = next(
        s for s in art["internal_noise_floor"]["per_seed"] if s["seed"] == 0
    )
    assert got0["n_persons_side_a"] == ref0["n_persons_side_a"]
    assert got0["n_persons_side_b"] == ref0["n_persons_side_b"]
    for key, ref_cell in ref0["cells"].items():
        got_cell = got0["cells"][key]
        assert got_cell["m_a"] == pytest.approx(
            ref_cell["m_a"], abs=1e-12
        ), key
        assert got_cell["m_b"] == pytest.approx(
            ref_cell["m_b"], abs=1e-12
        ), key
        assert got_cell["n_death_a"] == ref_cell["n_death_a"], key
        if ref_cell["log_ratio_abs"] is None:
            assert got_cell["log_ratio_abs"] is None, key
        else:
            assert got_cell["log_ratio_abs"] == pytest.approx(
                ref_cell["log_ratio_abs"], abs=1e-12
            ), key

    # All-window anchor reproduces (rates and ratios).
    nchs_rates = builder.nchs_band_rates(json.loads(NCHS.read_text()))
    got_anchor = builder.external_anchor(
        slices, nchs_rates, start_year_min=None
    )
    ref_anchor = art["external_anchor"]["windows"]["all"]
    assert got_anchor["total_death_events_unwt"] == (
        ref_anchor["total_death_events_unwt"]
    )
    for key, ref_cell in ref_anchor["by_band_sex"].items():
        got_cell = got_anchor["by_band_sex"][key]
        if ref_cell["ratio"] is None:
            assert got_cell["ratio"] is None, key
            continue
        assert got_cell["psid_m"] == pytest.approx(
            ref_cell["psid_m"], abs=1e-12
        ), key
        assert got_cell["ratio"] == pytest.approx(
            ref_cell["ratio"], abs=1e-12
        ), key

    assert "populace.fit" not in sys.modules
