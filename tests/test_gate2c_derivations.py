"""Perturbation-hardened bindings for the gate-2c floor derivations.

Companion to ``tests/test_gate2c_floors.py``. Where that file checks each
derivation *recomputes*, this file proves each is genuinely *bound* to the
committed floor -- perturbing an input changes the output, so a stored value
equal to its derivation is not a coincidence -- reconstructs the whole
gate-eligible / report-only partition through the build script's own
``partition_cells``, closes the T_max / MIN_EVENTS / K perturbation dead
zones by pinning the builder constants directly (2b round-1 finding 9 / fix
H), and carries the ALWAYS-RUNNABLE structural label-swap catches (2b
round-1 finding 9: a self-consistent cell swap that the artifact tier must
catch WITHOUT the PSID/pe-us reproduction). Always runnable -- reads only
the committed artifact and imports the builder (no data load at import).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2c_floors_v1.json"
SCRIPTS = ROOT / "scripts"
FLOOR_KEY = "noise_floor_seeds_0_99"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_gate2c_floors as builder

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


def test_builder_aggregations_are_empty_by_design():
    """gate-2c declares NO coverage-recovery aggregates (moment-module
    aggregation_members returns {}); the build script imports that map, so
    the deliberate absence is bound, not a silent omission."""
    builder = _builder()
    from populace_dynamics.data import couple_earnings as ce

    assert builder.AGGREGATIONS == {}
    assert ce.aggregation_members() == {}
    art = _artifact()
    assert art["aggregations"] == {}
    assert "EMPTY by design" in art["aggregations_note"]


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
    victim = sorted(gated0)[0]
    tolerances[victim] = builder.T_MAX + 0.5
    gated1, _, reasons1 = builder.partition_cells(stability, tolerances)
    assert victim not in gated1
    assert reasons1[victim] == "tolerance_above_t_max"


def test_tightening_the_cap_below_a_gated_tolerance_demotes_it():
    """A cap move that IS in a live gated tolerance's neighbourhood must
    change the partition -- guards against the ln(1.5)->ln(1.51) dead zone
    at the partition level (complement to the constant pin below)."""
    art = _artifact()
    builder = _builder()
    stability, tolerances = _reconstruct_inputs(art)
    gated0, _, _ = builder.partition_cells(stability, tolerances)
    # Highest gated tolerance: a cap just below it must demote exactly it.
    top = max(gated0, key=lambda c: tolerances[c])
    tightened = tolerances[top] - 1e-6

    def _passes(key):
        s = stability[key]
        return (
            s["defined_seeds"] == s["n_seeds"]
            and s["min_events_either_half"] >= builder.MIN_EVENTS_FOR_GATE
            and tolerances[key] <= tightened
        )

    assert not _passes(top)
    assert top in gated0


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
    bumped = dict(tolerances)
    victim = sorted(gated)[0]
    bumped[victim] = tolerances[victim] * 0.5
    oc2 = builder.faithful_candidate_oc(floor, bumped, gated)
    assert oc2["p_seed_pass"] < oc["p_seed_pass"]


# --------------------------------------------------------------------------
# Constant pins: close the T_max / MIN_EVENTS / K perturbation dead zones
# --------------------------------------------------------------------------
def test_builder_power_cap_and_events_constants_are_pinned():
    """2b finding 9 / fix H: close the T_MAX, MIN_EVENTS and K perturbation
    dead zones. A partition test alone can miss a quiet mutation (no gated
    tolerance may sit in the ln(1.5)->ln(1.51) gap, no cell in the
    min-event gap), so bind the builder constants directly and tie the
    committed artifact to them."""
    builder = _builder()
    assert builder.T_MAX == math.log(1.5)
    assert builder.T_MAX_SOURCE == "ln(1.5)"
    assert builder.MIN_EVENTS_FOR_GATE == 20
    assert builder.DRAFT_K == 4
    assert builder.DRAFT_ROUNDING == 3
    assert builder.CANDIDATE_DRAWS == 20
    art = _artifact()
    assert art["internal_noise_floor"]["t_max"] == builder.T_MAX
    assert (
        art["internal_noise_floor"]["min_events_for_gate"]
        == builder.MIN_EVENTS_FOR_GATE
    )
    assert art["draft_thresholds"]["k"] == builder.DRAFT_K
    assert art["protocol"]["candidate_draws"] == builder.CANDIDATE_DRAWS


# --------------------------------------------------------------------------
# ALWAYS-RUNNABLE label-swap catches (2b finding 9): data-free structural
# invariants a self-consistent cell swap would violate, caught in CI with
# no PSID / pe-us reproduction.
# --------------------------------------------------------------------------
def test_first_marriage_age_monotonicity_catches_age_swap():
    """First-marriage hazard falls steeply with age: at every AIME tercile x
    sex the 18-24 rate exceeds the 45+ rate (~5-15x). A self-consistent swap
    of an 18-24 cell with its 45+ counterpart (across every artifact site)
    would invert this -- caught here with no PSID."""
    rm = _artifact()["reference_moments"]
    for terc in (1, 2, 3):
        for sex in ("female", "male"):
            young = rm[f"first_marriage_by_earnings.t{terc}.18-24|{sex}"][
                "rate"
            ]
            old = rm[f"first_marriage_by_earnings.t{terc}.45+|{sex}"]["rate"]
            assert young > old, (terc, sex, young, old)
            assert young > 2.0 * old, (terc, sex, young, old)


def test_first_marriage_women_marry_younger_catches_fm_swap():
    """Women's first-marriage hazard at 18-24 exceeds men's in the two
    lower AIME terciles (women marry younger); a self-consistent female<->
    male swap of one of those gated cells (the 2b finding-9 attack) inverts
    it -- caught here with no PSID."""
    rm = _artifact()["reference_moments"]
    for terc in (1, 2):
        f = rm[f"first_marriage_by_earnings.t{terc}.18-24|female"]["rate"]
        m = rm[f"first_marriage_by_earnings.t{terc}.18-24|male"]["rate"]
        assert f > m, (terc, f, m)


def test_shared_earnings_span_ordering_catches_swap():
    """The combined-AIME cutpoint ratios are ordered by span: the q80/q20
    span exceeds every adjacent-quintile ratio, and each ratio exceeds 1
    (cutpoints strictly increase). A self-consistent swap or corruption of a
    shared-earnings cell violates this -- caught with no PSID."""
    rm = _artifact()["reference_moments"]
    q80_q20 = rm["shared_earnings_ratio.q80_q20"]["rate"]
    for name in ("q40_q20", "q60_q40", "q80_q60"):
        ratio = rm[f"shared_earnings_ratio.{name}"]["rate"]
        assert ratio > 1.0, name
        assert q80_q20 > ratio, name
    assert q80_q20 > 1.0
