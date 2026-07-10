"""Tests for the M4 disability floors (runs/m4_disability_v1.json).

The artifact is a REPORTED ANCHOR (reads no gate, changes no gate): the
PSID work-limitation incidence/recovery/prevalence hazards, the DI->
retirement conversion validation vs the archived SSA 6.B5.1 column, and
the person-disjoint half-split noise floor a future DI gate would derive
thresholds from. It is pinned like the other ``runs/`` floors.

Two tiers, mirroring ``tests/test_mortality_floors.py``:

* Always-runnable internal-consistency tests touching only committed
  files: the reported-anchor schema with a NOT-RATIFIED proposal; every
  reference-moment rate recomputes from its parts; every pooled floor
  statistic recomputes from the per-seed cells; the gate-eligible/report-
  only partition matches the stored stability; the conversion ratios
  recompute and the 6.B5.1 targets match the committed claim-age
  reference; and the concept deltas and wanted-table list are present.
  No reform is scored.
* A reference-moment + seed-0 + conversion reproduction pin (skipped when
  the PSID individual file is absent) that rebuilds the panel and reruns
  the moments, the seed-0 half-split and the conversion validation,
  matching the committed numbers to float precision -- with populace.fit
  never imported.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "m4_disability_v1.json"
CLAIM_REF = ROOT / "data" / "external" / "ssa_claim_ages_2023supplement.json"
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
    import build_disability_floors as builder

    return builder


# --------------------------------------------------------------------------
# Schema and reported-anchor framing (always runnable)
# --------------------------------------------------------------------------
def test_schema_and_reported_anchor():
    art = _artifact()
    assert art["schema_version"] == "m4_disability.v1"
    assert art["run"] == "m4_disability_v1"
    assert art["reported_anchor_not_gated"] is True
    assert "changes no gate" in art["purpose"]
    assert "disability" in art["component"]
    note = art["proposed_thresholds_note"]
    assert "NOT RATIFIED" in note
    assert "referee round" in note


def test_no_reform_scored():
    art = _artifact()
    assert art["reported_anchor_not_gated"] is True
    assert "reform" not in art
    assert "gate_result" not in art


def test_wave_coverage_recorded_honestly():
    art = _artifact()
    waves = art["data"]["waves_covered"]
    assert waves[0] == 1982 and waves[-1] == 2023
    assert len(waves) == 20
    assert "1984-1992" in art["data"]["wave_coverage_note"]
    # The value-code verification is recorded.
    ver = art["data"]["employment_status_code_verification"]
    assert ver["n_waves_verified"] == 20
    assert "disabl" in ver["example_2023"]["code5_label"].lower()


# --------------------------------------------------------------------------
# Reference moments recompute (always runnable)
# --------------------------------------------------------------------------
def test_reference_moments_recompute_from_parts():
    art = _artifact()
    moments = art["reference_moments"]
    # 5 bands x 2 sexes x {incidence, recovery, prevalence} + 2 conversion.
    assert len(moments) == 5 * 2 * 3 + 2
    for key, cell in moments.items():
        if cell["den_wt"] > 0:
            assert cell["rate"] == pytest.approx(
                cell["num_wt"] / cell["den_wt"]
            ), key
        assert cell["n_events"] <= cell["n_at_risk"]


def test_incidence_and_prevalence_rise_with_age():
    """The documented shape: incidence and prevalence climb with age."""
    art = _artifact()
    m = art["reference_moments"]
    for sex in ("male", "female"):
        assert (
            m[f"prevalence.20-29|{sex}"]["rate"]
            < m[f"prevalence.50-59|{sex}"]["rate"]
        )
        assert (
            m[f"incidence.20-29|{sex}"]["rate"]
            < m[f"incidence.50-59|{sex}"]["rate"]
        )


# --------------------------------------------------------------------------
# Noise floor recomputes (always runnable)
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


def test_per_seed_cells_internally_consistent():
    art = _artifact()
    for seed in art["internal_noise_floor"]["per_seed"]:
        for key, cell in seed["cells"].items():
            if cell["log_ratio_abs"] is None:
                assert cell["rate_a"] == 0 or cell["rate_b"] == 0, key
                continue
            expected = abs(math.log(cell["rate_a"] / cell["rate_b"]))
            assert cell["log_ratio_abs"] == pytest.approx(expected), key


def test_stability_partition_matches_rule():
    art = _artifact()
    inf = art["internal_noise_floor"]
    stability = inf["cell_stability"]
    n_seeds = len(inf["seeds"])
    floor_min = inf["min_events_for_gate"]

    gate_eligible = set()
    for key, v in stability.items():
        expected = v["defined_seeds"] == n_seeds and (
            v["min_events_either_half"] >= floor_min
        )
        assert v["gate_eligible"] is expected, key
        if v["gate_eligible"]:
            gate_eligible.add(key)
            # Every gate-eligible cell carries a noise-floor sd basis.
            assert key in inf["noise_floor_seeds_0_4"], key
    # A real, non-degenerate set of gate-eligible cells (report-only may be
    # empty here -- the transition cells are all well powered).
    assert gate_eligible
    assert len(stability) == len(art["reference_moments"])


# --------------------------------------------------------------------------
# Conversion validation vs 6.B5.1 (always runnable)
# --------------------------------------------------------------------------
def test_conversion_ratio_recomputes_and_is_reported_below_one():
    art = _artifact()
    cv = art["conversion_validation"]["by_sex"]
    for sex in ("male", "female"):
        b = cv[sex]
        admin_mean = b["admin_6b51_conversion_share_pct"]["mean_1998_2022"]
        assert b["ratio_psid_analog_to_admin_mean"] == pytest.approx(
            round(b["psid_conversion_analog_pct"] / admin_mean, 4)
        )
        # The self-reported analog sits well below the administrative
        # share -- reported honestly, never calibrated to 1.
        assert b["ratio_psid_analog_to_admin_mean"] < 1.0
    assert "concept_deltas" in art  # the deltas that explain the gap


def test_conversion_targets_match_committed_claim_reference():
    """The 6.B5.1 shares cited come from the committed claim-age ref."""
    art = _artifact()
    ref = json.loads(CLAIM_REF.read_text())["data"]
    cv = art["conversion_validation"]["by_sex"]
    for sex in ("male", "female"):
        by_year = cv[sex]["admin_6b51_conversion_share_pct"][
            "by_reference_year"
        ]
        for year, share in by_year.items():
            expected = ref[sex][year]["categories"]["disability_conversion"]
            assert share == pytest.approx(expected), (sex, year)


# --------------------------------------------------------------------------
# Concept deltas + wanted SSA tables (always runnable)
# --------------------------------------------------------------------------
def test_concept_deltas_present_and_named():
    art = _artifact()
    deltas = art["concept_deltas"]
    assert len(deltas) >= 6
    for d in deltas:
        assert d["name"] and d["delta"]
    names = " ".join(d["name"] for d in deltas).lower()
    # The headline deltas the module must never conflate.
    assert "self-report" in names or "adjudication" in names
    assert "transience" in names or "recovery" in names


def test_wanted_ssa_tables_named():
    art = _artifact()
    wanted = art["wanted_ssa_tables"]
    assert len(wanted) >= 5
    statuses = [w["status"] for w in wanted]
    # 6.B5.1 is in hand (validated); the incidence/termination/prevalence
    # series are the WANTED ones the orchestrator browser-fetches.
    assert any("IN HAND" in s for s in statuses)
    assert any(s == "WANTED" for s in statuses)
    for w in wanted:
        assert w["series"] and w["table"] and w["needed_for"]
    blob = json.dumps(wanted).lower()
    assert "incidence" in blob and "termination" in blob


# --------------------------------------------------------------------------
# Reproduction (needs PSID; NO populace-fit)
# --------------------------------------------------------------------------
@needs_real_ind
def test_moments_seed0_and_conversion_reproduce_without_populace_fit():
    assert "populace.fit" not in sys.modules
    builder = _import_builder()
    assert (
        "populace.fit" not in sys.modules
    ), "importing the disability builder pulled populace.fit"

    from populace_dynamics.data import deaths, disability

    status = disability.read_disability_status()
    panel = disability.build_disability_panel(
        status, deaths.read_death_records()
    )
    art = _artifact()

    # Reference moments reproduce to float precision.
    got = disability.reference_moments(panel)
    ref_moments = art["reference_moments"]
    assert set(got) == set(ref_moments)
    for key, ref_cell in ref_moments.items():
        assert got[key]["rate"] == pytest.approx(
            ref_cell["rate"], abs=1e-12
        ), key
        assert got[key]["n_events"] == ref_cell["n_events"], key

    # Seed-0 half-split cells reproduce.
    got0 = builder.measure_seed_halfsplit(0, panel)
    ref0 = next(
        s for s in art["internal_noise_floor"]["per_seed"] if s["seed"] == 0
    )
    assert got0["n_persons_side_a"] == ref0["n_persons_side_a"]
    for key, ref_cell in ref0["cells"].items():
        got_cell = got0["cells"][key]
        assert got_cell["rate_a"] == pytest.approx(
            ref_cell["rate_a"], abs=1e-12
        ), key
        if ref_cell["log_ratio_abs"] is None:
            assert got_cell["log_ratio_abs"] is None, key
        else:
            assert got_cell["log_ratio_abs"] == pytest.approx(
                ref_cell["log_ratio_abs"], abs=1e-12
            ), key

    # Conversion validation reproduces.
    from populace_dynamics import claiming

    got_cv = builder.conversion_validation(
        got, claiming.load_claim_age_reference()
    )
    ref_cv = art["conversion_validation"]["by_sex"]
    for sex in ("male", "female"):
        assert got_cv["by_sex"][sex][
            "psid_conversion_analog_pct"
        ] == pytest.approx(ref_cv[sex]["psid_conversion_analog_pct"])

    assert "populace.fit" not in sys.modules
