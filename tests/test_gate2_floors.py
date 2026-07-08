"""Tests for the gate-2 family-transition floors (runs/gate2_floors_v2.json).

The artifact is PRE-LOCK EVIDENCE (reads no gate, changes no gate): the
committed reference moments, the 100-seed person-disjoint half-split noise
floor the DRAFT gate-2 thresholds derive from, the coherent option-(a)
scoring protocol, the power-cap partition, the joint/sequence statistics,
and the concept-decomposed / period-matched NCHS anchors. It is pinned like
the other ``runs/`` floors. v2 applies the round-1 referee amendments (PR
#79 comment 4910467957); v1 is retained as the round-1 record.

Two tiers, mirroring ``tests/test_mortality_floors.py``:

* Always-runnable internal-consistency tests touching only committed
  files: the schema is a reported anchor whose thresholds are draft, not
  ratified; the NCHS references it cites are pinned by sha256; every
  pooled floor statistic (mean/sd/realised sigma) recomputes from the
  per-seed ``values``; every DRAFT threshold recomputes as
  round(floor mean + k*sd) and is capped at T_max; the gate-eligible /
  report-only partition derives from the events + power-cap rule with the
  pre-registered aggregate supersession; the training-copy disclosure and
  faithful-candidate OC recompute; and every external ratio / residual
  recomputes from its parts.
* A seed-0 + anchor reproduction pin (skipped when the PSID history files
  are absent) that rebuilds the panel and reruns the seed-0 half-split
  (including the added cells), the holdout-id commitment and the ASFR
  anchor -- with populace.fit never imported.
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
ARTIFACT = ROOT / "runs" / "gate2_floors_v2.json"
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

FLOOR_KEY = "noise_floor_seeds_0_99"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_gate2_floors as builder

    return builder


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# Schema and pre-lock framing (always runnable)
# --------------------------------------------------------------------------
def test_schema_and_reported_anchor():
    art = _artifact()
    assert art["schema_version"] == "gate2_floors.v2"
    assert art["run"] == "gate2_floors_v2"
    assert "gate2_floors_v1" in art["supersedes"]
    assert "4910467957" in art["referee_round"]
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
# 100-seed floor + realised sigma recompute from the values (always runnable)
# --------------------------------------------------------------------------
def test_floor_is_measured_on_100_seeds():
    art = _artifact()
    assert art["internal_noise_floor"]["floor_seeds"] == list(range(100))
    assert art["internal_noise_floor"]["gate_seeds"] == [0, 1, 2, 3, 4]
    for block in art[FLOOR_KEY].values():
        assert block["n_seeds"] == 100
        assert len(block["values"]) == 100


def test_pooled_floor_recomputes_from_values():
    """mean / sd / realised sigma recompute from the 100 stored draws."""
    art = _artifact()
    for key, block in art[FLOOR_KEY].items():
        values = block["values"]
        assert block["mean"] == pytest.approx(np.mean(values)), key
        assert block["sd"] == pytest.approx(np.std(values, ddof=1)), key
        rms = math.sqrt(sum(v * v for v in values) / len(values))
        assert block["realized_sigma"] == pytest.approx(rms), key
        pct = block["pct_diff_abs"]
        assert pct["mean"] == pytest.approx(np.mean(pct["values"]))


def test_stored_gate_seed_values_match_the_floor_head():
    """The five stored per-seed entries are the floor's first five draws."""
    art = _artifact()
    per_seed = art["noise_floor_per_seed"]
    assert [s["seed"] for s in per_seed] == art["internal_noise_floor"][
        "gate_seeds"
    ]
    for key, block in art[FLOOR_KEY].items():
        head = [s["cells"][key]["log_ratio_abs"] for s in per_seed]
        assert all(v is not None for v in head), key
        assert block["values"][: len(head)] == pytest.approx(head), key


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
# DRAFT thresholds recompute + power cap (always runnable)
# --------------------------------------------------------------------------
def test_draft_thresholds_recompute_from_floor():
    """Every DRAFT tolerance == round(floor mean + k*sd, rounding), and the
    realised-sigma diagnostic == tolerance / RMS of the floor."""
    art = _artifact()
    k = art["draft_thresholds"]["k"]
    rounding = art["draft_thresholds"]["rounding"]
    floor = art[FLOOR_KEY]
    for key, spec in art["draft_thresholds"]["cells"].items():
        mean = floor[key]["mean"]
        sd = floor[key]["sd"]
        assert spec["derivation"]["floor_mean"] == pytest.approx(mean)
        assert spec["derivation"]["floor_sd"] == pytest.approx(sd)
        derived = round(mean + k * sd, rounding)
        assert spec["log_ratio_abs_max"] == pytest.approx(derived), key
        sigma = floor[key]["realized_sigma"]
        assert spec["realized_sigma"] == pytest.approx(sigma)
        assert spec["tolerance_sigma_units"] == pytest.approx(
            round(derived / sigma, 3)
        )


def test_every_gated_tolerance_respects_the_power_cap():
    """Round-1 finding 3: no gated cell's tolerance exceeds T_max=ln(1.5)."""
    art = _artifact()
    t_max = art["internal_noise_floor"]["t_max"]
    assert t_max == pytest.approx(math.log(1.5))
    assert art["draft_thresholds"]["t_max"] == pytest.approx(math.log(1.5))
    for key, spec in art["draft_thresholds"]["cells"].items():
        assert spec["log_ratio_abs_max"] <= t_max, key


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
    assert drafts == set(art["gate_partition"]["gate_eligible"])
    assert report_only == set(art["gate_partition"]["report_only"])
    assert drafts.isdisjoint(report_only)
    assert set(art["reference_moments"]) == gate_eligible | report_only


# --------------------------------------------------------------------------
# Power-cap partition + aggregation supersession (always runnable)
# --------------------------------------------------------------------------
def test_partition_derives_from_events_and_power_cap():
    """gate_eligible == defined-on-all + >=20 events + tol<=T_max, unless a
    per-age cell is superseded by a gating aggregate (then report-only)."""
    art = _artifact()
    stability = art["cell_stability"]
    n_seeds = len(art["internal_noise_floor"]["floor_seeds"])
    min_events = art["internal_noise_floor"]["min_events_for_gate"]
    t_max = art["internal_noise_floor"]["t_max"]

    for key, v in stability.items():
        tol = v.get("tolerance")
        events_ok = (
            v["defined_seeds"] == n_seeds
            and v["min_events_either_half"] >= min_events
        )
        passes = events_ok and tol is not None and tol <= t_max
        superseded = str(v.get("report_reason", "")).startswith(
            "superseded_by:"
        )
        if v["gate_eligible"]:
            assert passes, key
            assert not superseded, key
        else:
            # not gated => either fails the rule, or is superseded.
            assert (not passes) or superseded, key


def test_aggregation_supersession_is_coherent():
    """A gating aggregate demotes every per-age member it spans."""
    art = _artifact()
    stability = art["cell_stability"]
    aggs = art["aggregations"]
    gate_eligible = set(art["gate_partition"]["gate_eligible"])
    for agg, spec in aggs.items():
        assert (agg in gate_eligible) == spec["gated"], agg
        if spec["gated"]:
            for member in spec["members"]:
                assert member not in gate_eligible, (agg, member)
                assert stability[member]["report_reason"] == (
                    f"superseded_by:{agg}"
                )


def test_report_only_reasons_are_recorded():
    """Every report-only cell carries a machine-readable demotion reason."""
    art = _artifact()
    stability = art["cell_stability"]
    valid_prefixes = (
        "below_20_events",
        "undefined_on_some_seed",
        "tolerance_above_t_max",
        "aggregate_",
        "superseded_by:",
        "no_floor",
    )
    for key in art["gate_partition"]["report_only"]:
        reason = stability[key]["report_reason"]
        assert reason and reason.startswith(valid_prefixes), (key, reason)


# --------------------------------------------------------------------------
# Holdout-id commitment (always runnable format; reproduced under real data)
# --------------------------------------------------------------------------
def test_holdout_ids_committed_for_every_gate_seed():
    art = _artifact()
    hold = art["holdout_ids"]
    assert hold["gate_seeds"] == [0, 1, 2, 3, 4]
    assert hold["fraction"] == 0.5
    assert "split_panel_by_person" in hold["numpy_generator"]
    seeds = [e["seed"] for e in hold["per_seed"]]
    assert seeds == [0, 1, 2, 3, 4]
    for entry in hold["per_seed"]:
        assert entry["n_holdout_persons"] > 0
        assert len(entry["holdout_person_id_sha256"]) == 64


# --------------------------------------------------------------------------
# Training-copy disclosure + faithful OC (always runnable)
# --------------------------------------------------------------------------
def test_training_copy_disclosure_is_honest():
    """A train-copy passes at the noise floor; disclosed as procedural."""
    art = _artifact()
    check = art["training_copy_check"]
    assert check["passes_4_of_5"] is True
    # It passes well within tolerance -- the score IS the floor value.
    assert check["max_score_over_tolerance"] < 1.0
    assert "procedural" in check["interpretation"]
    assert "no_self_rescue" in check["interpretation"]


def test_faithful_candidate_oc_recomputes():
    """The OC recomputes from the stabilised floor: per-cell 2*Phi(tol/sig)-1,
    seed = product over gated cells, gate = P(>=4/5)."""
    art = _artifact()
    oc = art["faithful_candidate_oc"]
    floor = art[FLOOR_KEY]
    tol = art["draft_thresholds"]["cells"]
    p_seed = 1.0
    for key, cell in oc["per_cell"].items():
        sigma = floor[key]["realized_sigma"]
        t = tol[key]["log_ratio_abs_max"]
        p = 2.0 * _normal_cdf(t / sigma) - 1.0
        assert cell["cell_pass_prob"] == pytest.approx(round(p, 6)), key
        p_seed *= p
    assert oc["n_gated_cells"] == len(art["gate_partition"]["gate_eligible"])
    assert oc["p_seed_pass"] == pytest.approx(round(p_seed, 4))
    p_gate = p_seed**5 + 5 * p_seed**4 * (1.0 - p_seed)
    assert oc["p_gate_pass_4_of_5"] == pytest.approx(round(p_gate, 4))
    # The stabilised floor delivers the intended ~4 sigma design.
    assert oc["p_gate_pass_4_of_5"] > 0.9


# --------------------------------------------------------------------------
# External anchors recompute (always runnable)
# --------------------------------------------------------------------------
def test_asfr_window_ratios_recompute_from_parts():
    art = _artifact()
    for window in art["external_anchor"]["asfr"]["windows"].values():
        assert window["vintage_confounded"] is True
        for band, cell in window["by_band"].items():
            if cell["ratio"] is None or cell["nchs_asfr_per_1000"] == 0:
                continue
            assert cell["ratio"] == pytest.approx(
                cell["psid_asfr_per_1000"] / cell["nchs_asfr_per_1000"]
            ), band


def test_asfr_period_matched_recomputes_and_tracks_reality():
    """Finding 6b: the period-matched ratio is observed / expected under
    each band's own-year national ASFR, and lands near 1."""
    art = _artifact()
    pm = art["external_anchor"]["asfr"]["period_matched"]
    # matched years are within the PSID panel range.
    assert pm["matched_years"]
    assert max(pm["matched_years"]) <= 2023
    for band, cell in pm["by_band"].items():
        r = cell["period_matched_ratio"]
        if r is None:
            continue
        # ratio = psid_asfr / exposure-weighted national asfr.
        assert r == pytest.approx(
            cell["psid_asfr_per_1000"]
            / cell["exposure_weighted_nchs_asfr_per_1000"]
        ), band
    assert 0.7 < pm["ratio_summary"]["median_ratio"] < 1.4


def test_marriage_divorce_concept_decomposition_recomputes():
    """Finding 6a: residual == raw ratio / (2 * 1/pop_15plus_share), and the
    recent-window residual is a near-bullseye ~1."""
    art = _artifact()
    md = art["external_anchor"]["marriage_divorce"]
    decomp = md["concept_decomposition"]
    factor = decomp["person_to_couple_factor"] * (
        1.0 / decomp["pop_15plus_share"]
    )
    assert decomp["concept_factor"] == pytest.approx(round(factor, 4))
    for window in md["windows"].values():
        m = window["psid_marriage_rate_per_1000_py15plus"]
        d = window["psid_divorce_rate_per_1000_py15plus"]
        assert window["marriage_ratio"] == pytest.approx(
            m / window["nchs_marriage_rate_per_1000_totalpop"]
        )
        assert window["marriage_residual_after_concept"] == pytest.approx(
            round(window["marriage_ratio"] / factor, 3)
        )
        assert window["divorce_residual_after_concept"] == pytest.approx(
            round(window["divorce_ratio"] / factor, 3)
        )
        assert window["marriage_ratio"] > 1.0
        assert m > d
    recent = md["windows"]["recent"]
    assert 0.9 < recent["marriage_residual_after_concept"] < 1.15
    assert 0.9 < recent["divorce_residual_after_concept"] < 1.15


# --------------------------------------------------------------------------
# Stability partition covers every cell (always runnable)
# --------------------------------------------------------------------------
def test_partition_is_a_real_cover_of_every_cell():
    art = _artifact()
    gate_eligible = set(art["gate_partition"]["gate_eligible"])
    report_only = set(art["gate_partition"]["report_only"])
    assert gate_eligible and report_only
    assert gate_eligible.isdisjoint(report_only)
    assert gate_eligible | report_only == set(art["reference_moments"])
    for key in gate_eligible:
        assert key in art[FLOOR_KEY], key


# --------------------------------------------------------------------------
# Seed-0 + holdout + anchor reproduction (needs PSID; NO populace-fit)
# --------------------------------------------------------------------------
@needs_real
def test_seed0_holdout_and_anchor_reproduce_without_populace_fit():
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
    # The added families reproduce alongside the originals.
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

    # Holdout-id commitment reproduces from the pinned generator.
    got_hold = builder.holdout_id_commitment(panel)
    for got, ref in zip(
        got_hold["per_seed"], art["holdout_ids"]["per_seed"], strict=True
    ):
        assert got["seed"] == ref["seed"]
        assert got["n_holdout_persons"] == ref["n_holdout_persons"]
        assert (
            got["holdout_person_id_sha256"] == ref["holdout_person_id_sha256"]
        ), got["seed"]

    got_asfr = builder.asfr_anchor(fert)
    ref_pm = art["external_anchor"]["asfr"]["period_matched"]["by_band"]
    got_pm = got_asfr["period_matched"]["by_band"]
    for band, ref_cell in ref_pm.items():
        if ref_cell["period_matched_ratio"] is None:
            continue
        assert got_pm[band]["period_matched_ratio"] == pytest.approx(
            ref_cell["period_matched_ratio"], abs=1e-9
        ), band

    assert "populace.fit" not in sys.modules
