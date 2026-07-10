"""Tests for the gate-2c marriage x earnings floors (gate2c_floors_v1).

The artifact ``runs/gate2c_floors_v1.json`` is PRE-LOCK EVIDENCE -- the
first step of the gate_2c lock ceremony (``gates.yaml`` gate_2c,
``lock_ceremony.required_before_any_2c_pass``): the committed marriage x
earnings JOINT reference moments, the 100-seed person-disjoint half-split
noise floor the DRAFT tolerances derive from, the ratified 2a
mean-over-K=20-draws scoring protocol (adopted from the start), and the
T_max power-cap partition. It reads no gate, changes no gate, and -- like
tranche 2a's / gate 2b's build -- writes NO ``gates.yaml`` block (the
lock-ceremony flip inserts the thresholds later).

Two tiers, mirroring ``tests/test_gate2b_floors.py``:

* Always-runnable internal-consistency tests touching only committed files
  (schema, floor recompute, DRAFT-threshold and power-cap derivation, the
  partition + machine-recorded reasons, the estimand / selection
  disclosure, the K=20 protocol, the external-anchor GAP, the training-copy
  disclosure and the faithful OC).
* A seed-0 + holdout reproduction pin (skipped when the PSID products or the
  pe-us checkout are absent, since the AIME chain needs the certified NAWI
  series) that rebuilds the panel and reruns the seed-0 half-split and the
  holdout-id commitment -- with populace.fit never imported.

This module references POPULACE_DYNAMICS_PE_US_DIR (the AIME chain) and the
PSID data root, so it is the oracle_policyengine tier; its always-runnable
tests still execute under a plain ``pytest`` run (only the reproduction pin
skips when the data is absent).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2c_floors_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
PE_US_DIR = os.environ.get("POPULACE_DYNAMICS_PE_US_DIR")
needs_real = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir()
    or not (REAL_DATA / "family").is_dir()
    or not (REAL_DATA / "ind2023er").is_dir()
    or not PE_US_DIR
    or not Path(PE_US_DIR).expanduser().is_dir(),
    reason="PSID products (mh85_23 / family / ind2023er) or the pe-us "
    "checkout (POPULACE_DYNAMICS_PE_US_DIR, for the AIME chain) not staged",
)

FLOOR_KEY = "noise_floor_seeds_0_99"


def _artifact() -> dict:
    import json

    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_gate2c_floors as builder

    return builder


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# Schema and pre-lock framing (always runnable)
# --------------------------------------------------------------------------
def test_schema_and_reported_anchor():
    art = _artifact()
    assert art["schema_version"] == "gate2c_floors.v1"
    assert art["run"] == "gate2c_floors_v1"
    assert art["reported_anchor_not_gated"] is True
    assert "changes no gate" in art["purpose"]
    assert "marriage x earnings" in art["component"]
    assert art["holdout_basis"] == [
        "mh85_23",
        "family_earnings_panel_gate1_certified",
    ]
    note = art["draft_thresholds_note"]
    assert "NOT RATIFIED" in note
    assert "lock ceremony" in note


def test_ceremony_block_writes_no_gates_yaml():
    """gate-2c's floor is step 1; only the ratifying flip touches gates.yaml.
    The artifact must declare that and carry no ratified/locked flag."""
    art = _artifact()
    cer = art["ceremony"]
    assert cer["tranche"] == "2c_marriage_earnings_joint"
    assert cer["gates_yaml_untouched"] is True
    assert "gate2b_floors_v1.json" in cer["mirrors"]
    assert "reform" not in art
    assert "gate_result" not in art
    assert art["draft_thresholds"]["note"].find("NOT yet mirrored") >= 0


def test_gates_yaml_gate_2c_still_an_unlocked_stub():
    """Guard the 'gates.yaml UNTOUCHED' constraint: gate_2c must remain an
    unlocked stub with no thresholds block (the flip comes later), and the
    locked 2a block must be untouched."""
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load((ROOT / "gates.yaml").read_text())
    gate_2c = spec["gates"]["gate_2"]["gate_2c"]
    assert gate_2c["locked"] is False
    assert gate_2c["status"] == "unlocked"
    assert gate_2c["id"] == "2c_marriage_earnings_joint"
    assert "thresholds" not in gate_2c
    assert gate_2c["lock_ceremony"]["exists"] is False
    # holdout_basis is the gate_2c stub's descriptive string.
    assert "mh85_23" in gate_2c["holdout_basis"]
    assert "earnings" in gate_2c["holdout_basis"]
    # The locked tranche-2a block is untouched.
    assert spec["gates"]["gate_2"]["thresholds"]["locked"] is True


def test_provenance_pins_present():
    art = _artifact()
    pins = art["revision_pins"]
    assert pins["artifact_schema_version"] == "gate2c_floors.v1"
    # No parent-master populace_dynamics_sha pin (2b finding H).
    assert "populace_dynamics_sha" not in pins
    assert "base_sha" in pins
    # The AIME chain pins the pe-us revision (classifier_version_pin).
    assert pins["pe_us_revision"]
    note = pins["build_commit_note"]
    assert "gate2c-floors" in note
    assert "pe_us_revision" in note


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
# Power-cap partition + machine-recorded reasons (always runnable)
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


def test_no_aggregations_by_design():
    """gate-2c declares no coverage-recovery aggregates (finding-6b-aware);
    the artifact records the deliberate absence with its rationale."""
    art = _artifact()
    assert art["aggregations"] == {}
    assert "EMPTY by design" in art["aggregations_note"]
    # No report reason is a supersession one (no aggregate to supersede).
    for key in art["gate_partition"]["report_only"]:
        reason = art["cell_stability"][key]["report_reason"]
        assert not reason.startswith("superseded_by:"), (key, reason)


def test_partition_is_a_real_cover_of_every_cell():
    art = _artifact()
    gate_eligible = set(art["gate_partition"]["gate_eligible"])
    report_only = set(art["gate_partition"]["report_only"])
    assert gate_eligible and report_only
    assert gate_eligible.isdisjoint(report_only)
    assert gate_eligible | report_only == set(art["reference_moments"])
    for key in gate_eligible:
        assert key in art[FLOOR_KEY], key


def test_assortative_contingency_family_is_fully_gated():
    """The dense 3x3 own-tercile x spouse-tercile contingency -- the family
    that anchors the assortative-mating gate -- gates on every cell."""
    art = _artifact()
    gated = set(art["gate_partition"]["gate_eligible"])
    for o in (1, 2, 3):
        for u in (1, 2, 3):
            assert f"assort_mating.own{o}_spouse{u}" in gated


# --------------------------------------------------------------------------
# Statistic families present + power-priced (always runnable)
# --------------------------------------------------------------------------
def test_all_four_families_present_and_priced():
    """The four marriage x earnings families are present and each cell is
    either gated or report-only (a real cover), with the expected counts."""
    art = _artifact()
    ref = art["reference_moments"]
    gated = set(art["gate_partition"]["gate_eligible"])
    report = set(art["gate_partition"]["report_only"])
    families = {
        "assort_mating.": 9,
        "first_marriage_by_earnings.": 24,
        "remarriage_by_earnings.": 6,
        "earnings_around_marriage.": 3,
        "earnings_around_divorce.": 3,
        "shared_earnings_ratio.": 4,
    }
    for prefix, count in families.items():
        cells = [k for k in ref if k.startswith(prefix)]
        assert len(cells) == count, prefix
        for k in cells:
            assert (k in gated) ^ (k in report), k
    # earnings-conditional marriage hazards and the shape family clear the
    # cap on their dense cells; the shared-earnings ratios all gate.
    assert "first_marriage_by_earnings.t1.18-24|female" in gated
    assert "shared_earnings_ratio.q40_q20" in gated
    assert "earnings_around_marriage.all" in gated


def test_assortative_correlation_is_report_only():
    """The within-couple AIME Spearman rho is reported, never gated (a
    correlation is not a scale-free |ln ratio|)."""
    art = _artifact()
    corr = art["assortative_correlation_report_only"]
    assert corr["gated"] is False
    assert corr["report_only"] is True
    assert corr["overall"]["spearman_aime"] is not None
    assert "by_marriage_decade" in corr
    # It never leaks into the gated cell set.
    gated = set(art["gate_partition"]["gate_eligible"])
    assert not any("spearman" in k for k in gated)


# --------------------------------------------------------------------------
# Estimand / selection disclosure (2b finding 3; always runnable)
# --------------------------------------------------------------------------
def test_estimand_named_and_selection_disclosed():
    """Fix D analogue: the pooled estimand is named as the SELECTED couple
    universe (not all marriages), with the join-coverage numbers, the
    partial-career AIME proxy caveat, and the report-only marriage-decade
    slices disclosed."""
    art = _artifact()
    estimand = art["data"]["estimand"]
    assert "SELECTED" in estimand
    assert "NOT all PSID marriages" in estimand
    sel = art["selection_estimand"]
    assert sel["join_coverage_both_over_joinable"] is not None
    assert sel["join_coverage_both_over_ego_supply"] is not None
    assert sel["aime_is_partial_career_proxy"] is True
    slices = art["marriage_decade_slices"]
    assert "REPORT-ONLY" in slices["note"]
    # At least a few dense decades carry couples.
    dense = [
        d
        for d, v in slices.items()
        if d != "note" and v["n_directed_couples"] >= 100
    ]
    assert len(dense) >= 3


def test_couple_correlation_wart_disclosed():
    """The person-disjoint-by-ego split is not couple-disjoint; the wart is
    disclosed (a couple can straddle the split via its two directed
    records)."""
    art = _artifact()
    wart = art["panel_construction"]["couple_correlation_wart"]
    assert "NOT couple-disjoint" in wart
    assert "conservative" in wart
    caveats = art["panel_construction"]["coverage_caveats"]
    assert any("partial career" in c for c in caveats)


# --------------------------------------------------------------------------
# External-anchor gap (honest declaration; always runnable)
# --------------------------------------------------------------------------
def test_external_anchor_gap_is_declared_not_fabricated():
    art = _artifact()
    anchor = art["external_anchor"]
    assert anchor["status"] == "none_bundled"
    assert anchor["reported_anchor_not_gated"] is True
    assert "KNOWN GAP" in anchor["note"]
    assert "partial-career" in anchor["concept_delta"].lower()
    assert len(anchor["candidate_sources"]) >= 3
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
# The ratified K=20 estimator adopted from the start (2b finding 1; runnable)
# --------------------------------------------------------------------------
def test_protocol_is_the_ratified_k20_estimator():
    """Fix A pre-empted: the scoring protocol is tranche 2a's ratified
    mean-over-K=20 estimator (amendment 1) from the START, not the
    superseded single frozen draw the 2b round-1 referee flagged."""
    art = _artifact()
    proto = art["protocol"]
    assert proto["candidate_draws"] == 20
    assert "5200" in proto["candidate_draw_stream"]
    assert "K=20" in proto["estimator"]
    assert "NOT the mean of the per-draw" in proto["estimator"]
    basis = proto["basis_note"]
    assert "DRAW-NOISE-FREE" in basis
    assert "UNACHIEVABLE" in basis
    schema = proto["fresh_run_artifact_schema"]
    n_gated = art["gate_partition"]["n_gate_eligible"]
    assert schema["per_draw_per_cell_rates"]["shape"] == [20, n_gated, 5]
    assert schema["undefined_draw_rule"]["pre_specified"] is True
    assert schema["per_draw_dispersion_disclosure"]["report_only"] is True


# --------------------------------------------------------------------------
# Seed-0 + holdout reproduction (needs PSID + pe-us AIME; NO populace-fit)
# --------------------------------------------------------------------------
@needs_real
def test_seed0_and_holdout_reproduce_without_populace_fit():
    assert "populace.fit" not in sys.modules
    builder = _import_builder()
    assert (
        "populace.fit" not in sys.modules
    ), "importing the gate-2c builder pulled populace.fit"

    panel = builder.ce.build_couple_panel()
    art = _artifact()

    got0 = builder.measure_seed_halfsplit(0, panel)
    ref0 = next(s for s in art["noise_floor_per_seed"] if s["seed"] == 0)
    assert got0["n_persons_side_a"] == ref0["n_persons_side_a"]
    assert got0["n_persons_side_b"] == ref0["n_persons_side_b"]
    for key, ref_cell in ref0["cells"].items():
        got_cell = got0["cells"][key]
        assert got_cell["rate_a"] == pytest.approx(
            ref_cell["rate_a"], abs=1e-9
        ), key
        assert got_cell["n_events_a"] == ref_cell["n_events_a"], key
        if ref_cell["log_ratio_abs"] is None:
            assert got_cell["log_ratio_abs"] is None, key
        else:
            assert got_cell["log_ratio_abs"] == pytest.approx(
                ref_cell["log_ratio_abs"], abs=1e-9
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
