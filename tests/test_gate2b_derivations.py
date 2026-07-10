"""Perturbation-hardened bindings for the gate-2b floor derivations.

Companion to ``tests/test_gate2b_floors.py``. Where that file checks each
derivation *recomputes*, this file proves each is genuinely *bound* to the
committed floor -- perturbing an input changes the output, so a stored
value equal to its derivation is not a coincidence -- and reconstructs the
whole gate-eligible / report-only partition from the raw stability data
through the build script's own ``partition_cells`` and matches it to the
committed ``gate_partition``. This is the machine-binding the gate-2b lock
ceremony's verification round exercises; it replaces the gates.yaml <->
floor bindings that ``tests/test_gates_derivations.py`` holds for the
LOCKED tranches (gate-2b has no gates.yaml threshold block yet, so there is
nothing there to bind -- the flip adds it). Always runnable (committed
artifact only).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_floors_v1.json"
SCRIPTS = ROOT / "scripts"
FLOOR_KEY = "noise_floor_seeds_0_99"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_gate2b_floors as builder

    return builder


# --------------------------------------------------------------------------
# Tolerance binding: round(mean + k*sd, r), and perturbation moves it
# --------------------------------------------------------------------------
def test_tolerance_is_bound_to_the_committed_floor():
    art = _artifact()
    k = art["draft_thresholds"]["k"]
    r = art["draft_thresholds"]["rounding"]
    floor = art[FLOOR_KEY]
    for key, spec in art["draft_thresholds"]["cells"].items():
        mean = floor[key]["mean"]
        sd = floor[key]["sd"]
        assert spec["log_ratio_abs_max"] == round(mean + k * sd, r), key
        # Perturbing the floor mean by a decisive amount changes the
        # tolerance -- the stored value tracks THIS floor, not a constant.
        bumped = round(mean + 0.05 + k * sd, r)
        assert bumped != spec["log_ratio_abs_max"], key


def test_k_and_rounding_are_the_2a_draft_values():
    art = _artifact()
    assert art["draft_thresholds"]["k"] == 4
    assert art["draft_thresholds"]["rounding"] == 3
    assert art["internal_noise_floor"]["min_events_for_gate"] == 20
    assert art["internal_noise_floor"]["t_max"] == pytest.approx(math.log(1.5))


# --------------------------------------------------------------------------
# Full partition reconstructed through the build script's own logic
# --------------------------------------------------------------------------
def _reconstruct_inputs(art: dict):
    """Rebuild (stability, tolerances) from the committed artifact, as the
    build script held them just before partitioning."""
    k = art["draft_thresholds"]["k"]
    r = art["draft_thresholds"]["rounding"]
    floor = art[FLOOR_KEY]
    stability = {}
    for key, v in art["cell_stability"].items():
        stability[key] = {
            "defined_seeds": v["defined_seeds"],
            "n_seeds": v["n_seeds"],
            "min_events_either_half": v["min_events_either_half"],
        }
    tolerances = {
        key: round(block["mean"] + k * block["sd"], r)
        for key, block in floor.items()
    }
    return stability, tolerances


def test_partition_reconstructs_through_partition_cells():
    art = _artifact()
    builder = _builder()
    stability, tolerances = _reconstruct_inputs(art)
    gated, report, reasons = builder.partition_cells(stability, tolerances)
    assert gated == set(art["gate_partition"]["gate_eligible"])
    assert report == set(art["gate_partition"]["report_only"])
    for key, reason in reasons.items():
        assert art["cell_stability"][key]["report_reason"] == reason, key


def test_builder_aggregations_match_the_module_a_priori_map():
    """The supersession map is imported from the moment module (fixed a
    priori), not hand-written in the build script."""
    builder = _builder()
    from populace_dynamics.data import household_composition as hc

    assert builder.AGGREGATIONS == hc.aggregation_members()
    art = _artifact()
    assert set(art["aggregations"]) == set(builder.AGGREGATIONS)


def test_demoting_events_would_flip_a_gated_cell():
    """Binding check on the >=20-events rule: force a gated cell below the
    threshold and it must leave the gated set."""
    art = _artifact()
    builder = _builder()
    stability, tolerances = _reconstruct_inputs(art)
    gated0, _, _ = builder.partition_cells(stability, tolerances)
    victim = sorted(gated0)[0]
    stability[victim]["min_events_either_half"] = 5
    gated1, report1, _ = builder.partition_cells(stability, tolerances)
    assert victim not in gated1
    assert victim in report1


def test_loosening_a_tolerance_past_the_cap_demotes_it():
    """Binding check on the T_max power cap."""
    art = _artifact()
    builder = _builder()
    stability, tolerances = _reconstruct_inputs(art)
    gated0, _, _ = builder.partition_cells(stability, tolerances)
    # pick a gated non-aggregate cell and push its tolerance over the cap.
    victim = next(c for c in sorted(gated0) if c not in builder.AGGREGATIONS)
    tolerances[victim] = builder.T_MAX + 0.5
    gated1, _, reasons1 = builder.partition_cells(stability, tolerances)
    assert victim not in gated1
    assert reasons1[victim] in (
        "tolerance_above_t_max",
        "undefined_on_some_seed",
    )


# --------------------------------------------------------------------------
# Faithful OC binding
# --------------------------------------------------------------------------
def test_faithful_oc_is_bound_to_tolerances_and_sigmas():
    art = _artifact()
    builder = _builder()
    floor = art[FLOOR_KEY]
    tolerances = {
        c: s["log_ratio_abs_max"]
        for c, s in art["draft_thresholds"]["cells"].items()
    }
    gated = set(art["gate_partition"]["gate_eligible"])
    oc = builder.faithful_candidate_oc(floor, tolerances, gated)
    assert oc["p_seed_pass"] == art["faithful_candidate_oc"]["p_seed_pass"]
    assert (
        oc["p_gate_pass_4_of_5"]
        == art["faithful_candidate_oc"]["p_gate_pass_4_of_5"]
    )
    # Perturb one tolerance -> the seed pass probability must move.
    bumped = dict(tolerances)
    victim = sorted(gated)[0]
    bumped[victim] = tolerances[victim] * 0.5
    oc2 = builder.faithful_candidate_oc(floor, bumped, gated)
    assert oc2["p_seed_pass"] < oc["p_seed_pass"]


def test_no_gated_cell_is_a_pooled_aggregate_member():
    """A gating aggregate and its per-age members are never both gated."""
    art = _artifact()
    builder = _builder()
    gated = set(art["gate_partition"]["gate_eligible"])
    for agg, members in builder.AGGREGATIONS.items():
        if agg in gated:
            assert not (set(members) & gated), agg
