"""Tests for the gate-2b household-composition floors (gate2b_floors_v1).

The artifact ``runs/gate2b_floors_v1.json`` is PRE-LOCK EVIDENCE -- the
first step of the gate_2b lock ceremony (``gates.yaml`` gate_2b,
``lock_ceremony.required_before_any_2b_pass``): the committed
household-composition reference moments, the 100-seed person-disjoint
half-split noise floor the DRAFT tolerances derive from, the coherent
option-(a) scoring protocol and the T_max power-cap partition. It reads no
gate, changes no gate, and -- unlike tranche 2a's build -- writes NO
``gates.yaml`` block (the lock-ceremony flip inserts the thresholds later).
It is pinned like the other ``runs/`` floors.

Two tiers, mirroring ``tests/test_gate2_floors.py``:

* Always-runnable internal-consistency tests touching only committed
  files: the schema is a reported anchor whose thresholds are draft, not
  ratified and not in gates.yaml; every pooled floor statistic recomputes
  from the per-seed ``values``; every DRAFT threshold recomputes as
  round(floor mean + k*sd) and is capped at T_max; the gate-eligible /
  report-only partition derives from the events + power-cap rule with the
  pre-registered aggregate supersession; the external-anchor GAP is
  honestly declared; and the training-copy disclosure + faithful-candidate
  OC recompute.
* A seed-0 + holdout reproduction pin (skipped when the PSID products are
  absent) that rebuilds the panel and reruns the seed-0 half-split and the
  holdout-id commitment -- with populace.fit never imported.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_floors_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real = pytest.mark.skipif(
    not (REAL_DATA / "MX23REL").is_dir()
    or not (REAL_DATA / "ind2023er").is_dir(),
    reason="PSID products (MX23REL / ind2023er) not staged",
)

FLOOR_KEY = "noise_floor_seeds_0_99"


def _artifact() -> dict:
    import json

    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_gate2b_floors as builder

    return builder


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# Schema and pre-lock framing (always runnable)
# --------------------------------------------------------------------------
def test_schema_and_reported_anchor():
    art = _artifact()
    assert art["schema_version"] == "gate2b_floors.v1"
    assert art["run"] == "gate2b_floors_v1"
    assert art["reported_anchor_not_gated"] is True
    assert "changes no gate" in art["purpose"]
    assert "household" in art["component"]
    assert art["holdout_basis"] == ["MX23REL"]
    note = art["draft_thresholds_note"]
    assert "NOT RATIFIED" in note
    assert "lock ceremony" in note


def test_ceremony_block_writes_no_gates_yaml():
    """gate-2b's floor is step 1; only the ratifying flip touches gates.yaml.
    The artifact must declare that and carry no ratified/locked flag."""
    art = _artifact()
    cer = art["ceremony"]
    assert cer["tranche"] == "2b_relationship_household"
    assert cer["gates_yaml_untouched"] is True
    assert "runs/gate2_floors_v2.json" in cer["mirrors_tranche_2a"]
    # It is not a gate run and did not write a threshold block anywhere.
    assert "reform" not in art
    assert "gate_result" not in art
    assert art["draft_thresholds"]["note"].find("NOT yet mirrored") >= 0


def test_gates_yaml_gate_2b_still_an_unlocked_stub():
    """Guard the 'gates.yaml UNTOUCHED' constraint: gate_2b must remain an
    unlocked stub with no thresholds block (the flip comes later)."""
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load((ROOT / "gates.yaml").read_text())
    gate_2b = spec["gates"]["gate_2"]["gate_2b"]
    assert gate_2b["locked"] is False
    assert gate_2b["status"] == "unlocked"
    assert gate_2b["holdout_basis"] == ["MX23REL"]
    assert "thresholds" not in gate_2b
    assert gate_2b["lock_ceremony"]["exists"] is False


def test_provenance_pins_present():
    art = _artifact()
    pins = art["revision_pins"]
    assert pins["artifact_schema_version"] == "gate2b_floors.v1"
    assert "populace_dynamics_sha" in pins


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


def test_reference_moments_rate_recomputes():
    art = _artifact()
    for key, cell in art["reference_moments"].items():
        if cell["den_wt"] > 0:
            assert cell["rate"] == pytest.approx(
                cell["num_wt"] / cell["den_wt"]
            ), key


# --------------------------------------------------------------------------
# DRAFT thresholds recompute + power cap (always runnable)
# --------------------------------------------------------------------------
def test_draft_thresholds_recompute_from_floor():
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
            assert (not passes) or superseded, key


def test_aggregation_supersession_is_coherent():
    """A gating aggregate demotes every per-age member it spans; a
    non-gating aggregate supersedes nothing (no self-rescue by pooling)."""
    art = _artifact()
    stability = art["cell_stability"]
    aggs = art["aggregations"]
    gate_eligible = set(art["gate_partition"]["gate_eligible"])
    for agg, spec in aggs.items():
        assert (agg in gate_eligible) == spec["gated"], agg
        for member in spec["members"]:
            if spec["gated"]:
                assert member not in gate_eligible, (agg, member)
                assert stability[member]["report_reason"] == (
                    f"superseded_by:{agg}"
                )
            else:
                assert stability[member]["report_reason"] != (
                    f"superseded_by:{agg}"
                )


def test_report_only_reasons_are_recorded():
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


def test_partition_is_a_real_cover_of_every_cell():
    art = _artifact()
    gate_eligible = set(art["gate_partition"]["gate_eligible"])
    report_only = set(art["gate_partition"]["report_only"])
    assert gate_eligible and report_only
    assert gate_eligible.isdisjoint(report_only)
    assert gate_eligible | report_only == set(art["reference_moments"])
    for key in gate_eligible:
        assert key in art[FLOOR_KEY], key


def test_coresident_spouse_family_is_fully_gated():
    """The dense, low-noise coresident-spouse share gates on every band x
    sex -- the family that anchors the gate's power."""
    art = _artifact()
    gated = set(art["gate_partition"]["gate_eligible"])
    for lo, hi in [
        (15, 24),
        (25, 34),
        (35, 44),
        (45, 54),
        (55, 64),
        (65, 74),
        (75, 120),
    ]:
        band = f"{lo}-{hi}" if hi < 120 else f"{lo}+"
        for sex in ("female", "male"):
            assert f"coresident_spouse.{band}|{sex}" in gated


# --------------------------------------------------------------------------
# External-anchor gap (honest declaration; always runnable)
# --------------------------------------------------------------------------
def test_external_anchor_gap_is_declared_not_fabricated():
    art = _artifact()
    anchor = art["external_anchor"]
    assert anchor["status"] == "none_bundled"
    assert anchor["reported_anchor_not_gated"] is True
    assert "KNOWN GAP" in anchor["note"]
    assert "family unit" in anchor["concept_delta"].lower()
    assert len(anchor["candidate_sources"]) >= 3
    # No fabricated external ratios anywhere in the artifact.
    assert "nchs_reference_sha256" not in str(anchor)


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
    art = _artifact()
    check = art["training_copy_check"]
    assert check["passes_4_of_5"] is True
    assert check["max_score_over_tolerance"] < 1.0
    assert "procedural" in check["interpretation"]
    assert "no_self_rescue" in check["interpretation"]


def test_faithful_candidate_oc_recomputes():
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
    assert oc["p_gate_pass_4_of_5"] > 0.9


# --------------------------------------------------------------------------
# Seed-0 + holdout reproduction (needs PSID; NO populace-fit)
# --------------------------------------------------------------------------
@needs_real
def test_seed0_and_holdout_reproduce_without_populace_fit():
    assert "populace.fit" not in sys.modules
    builder = _import_builder()
    assert (
        "populace.fit" not in sys.modules
    ), "importing the gate-2b builder pulled populace.fit"

    panel, _ = builder.load_panel()
    art = _artifact()

    got0 = builder.measure_seed_halfsplit(0, panel)
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

    got_hold = builder.holdout_id_commitment(panel)
    for got, ref in zip(
        got_hold["per_seed"], art["holdout_ids"]["per_seed"], strict=True
    ):
        assert got["seed"] == ref["seed"]
        assert got["n_holdout_persons"] == ref["n_holdout_persons"]
        assert (
            got["holdout_person_id_sha256"] == ref["holdout_person_id_sha256"]
        ), got["seed"]

    assert "populace.fit" not in sys.modules
