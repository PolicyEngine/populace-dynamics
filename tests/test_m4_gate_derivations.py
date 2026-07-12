"""Perturbation-hardened bindings for the M4 disability gate floors
(runs/m4_gate_floors_v1.json).

Companion to ``tests/test_m4_gate_floors.py`` (the PSID reproduction). This
file proves each derivation is genuinely *bound* to the committed floor --
perturbing an input changes the output -- reconstructs the internal
gate-eligible / report-only partition through the build script's own
``partition_cells``, closes the T_max / MIN_EVENTS / k dead zones by pinning
the builder constants, re-derives the sha-pinned SSA anchor values from the
committed in-repo table, and carries the ALWAYS-RUNNABLE structural
label-swap catches (a band swap on the prevalence shape, a retirement<->work
swap on the conversion exit) with NO PSID / pe-us reproduction. Reads only
the committed artifact + the committed anchor table + imports the builder
(no data load at import).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "m4_gate_floors_v1.json"
FOUNDATION = ROOT / "runs" / "m4_disability_v1.json"
ANCHOR_TABLES = ROOT / "data" / "external" / "di_asr_2023" / "tables.json"
GATES_YAML = ROOT / "gates.yaml"
SCRIPTS = ROOT / "scripts"
FLOOR_KEY = "noise_floor_seeds_0_99"
BRIDGED = ("20-29", "30-39", "40-49", "50-59")


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_m4_gate_floors as builder

    return builder


# --------------------------------------------------------------------------
# Reported-anchor framing + gates.yaml untouched
# --------------------------------------------------------------------------
def test_schema_and_reported_anchor():
    art = _artifact()
    assert art["schema_version"] == "m4_gate_floors.v1"
    assert art["run"] == "m4_gate_floors_v1"
    assert art["reported_anchor_not_gated"] is True
    assert "changes no gate" in art["purpose"]
    assert art["ceremony"]["gates_yaml_untouched"] is True
    note = art["draft_thresholds_note"]
    assert "NOT RATIFIED" in note
    assert "CEREMONY STEP 1" in note
    assert "referee" in note.lower()


def test_no_gate_mutated_and_gates_yaml_has_no_gate_m4():
    """The evidence changes no gate and gates.yaml carries NO gate_m4 block --
    the stub is only PROPOSED in the note (gates.yaml UNTOUCHED)."""
    art = _artifact()
    assert "gate_result" not in art
    assert "reform" not in art
    assert "gate_m4:" not in GATES_YAML.read_text()
    # the proposal text names the stub, so a reader can find it.
    assert "gate_m4" in art["draft_thresholds_note"]


def test_reference_moments_bound_to_the_m4_foundation():
    """The gate reuses the m4_disability_v1 reference moments verbatim -- a
    single source of truth for the hazards, bound here."""
    art = _artifact()
    foundation = json.loads(FOUNDATION.read_text())["reference_moments"]
    assert art["reference_moments"] == foundation
    assert len(art["reference_moments"]) == 5 * 2 * 3 + 2


# --------------------------------------------------------------------------
# Anchor tables: sha pin + value re-derivation from the committed table
# --------------------------------------------------------------------------
def test_anchor_table_sha_is_pinned():
    art = _artifact()
    digest = hashlib.sha256(ANCHOR_TABLES.read_bytes()).hexdigest()
    assert art["anchor_tables"]["sha256"] == digest
    assert art["revision_pins"]["anchor_tables_sha256"] == digest
    assert art["anchor_tables"]["n_bytes"] == ANCHOR_TABLES.stat().st_size


def test_anchor_values_rederive_from_committed_table():
    """Every extracted anchor value recomputes from the committed tables.json
    through the builder's own extractors -- so the pinned anchors are bound to
    the source, not free-floating."""
    art = _artifact()
    builder = _builder()
    tables = json.loads(ANCHOR_TABLES.read_text())
    at = art["anchor_tables"]
    assert (
        builder._t19_within_sex_shares(tables)
        == at["t19_prevalence_age_shape_2023"]
    )
    assert (
        builder._t50_worker_terminations(tables)
        == at["t50_worker_terminations_2023"]
    )
    assert (
        builder._t49_worker_rate_trend(tables)
        == at["t49_worker_termination_rate"]
    )


def test_t50_conversion_share_arithmetic_and_dominance():
    art = _artifact()
    t50 = art["anchor_tables"]["t50_worker_terminations_2023"]
    conv, cess = t50["fra_conversion"], t50["medical_cessation"]
    assert t50["fra_conversion_share_of_nondeath_exits"] == pytest.approx(
        conv / (conv + cess)
    )
    # FRA-conversion is the DOMINANT non-death worker termination reason.
    assert t50["fra_conversion_share_of_nondeath_exits"] > 0.5
    assert conv > cess


# --------------------------------------------------------------------------
# Internal (surface d) tolerance binding + constant pins
# --------------------------------------------------------------------------
def test_internal_tolerance_is_bound_to_the_committed_floor():
    art = _artifact()
    k = art["draft_thresholds"]["k"]
    r = art["draft_thresholds"]["rounding"]
    floor = art["internal_noise_floor"][FLOOR_KEY]
    cells = art["draft_thresholds"]["cells"]
    assert cells, "at least one internal cell must gate"
    for key, spec in cells.items():
        mean = floor[key]["mean"]
        sd = floor[key]["sd"]
        assert spec["log_ratio_abs_max"] == round(mean + k * sd, r), key
        bumped = round(mean + 0.05 + k * sd, r)
        assert bumped != spec["log_ratio_abs_max"], key


def test_builder_constants_are_pinned_against_the_artifact():
    """Close the T_MAX / MIN_EVENTS / k / K dead zones: pin the builder
    constants directly and tie the artifact to them."""
    builder = _builder()
    art = _artifact()
    assert builder.T_MAX == math.log(1.5)
    assert builder.T_MAX_SOURCE == "ln(1.5)"
    assert builder.MIN_EVENTS_FOR_GATE == 20
    assert builder.DRAFT_K == 3
    assert builder.MARGIN_K == 3
    assert builder.DRAFT_ROUNDING == 3
    assert builder.CANDIDATE_DRAWS == 20
    assert len(builder.SEEDS) == 100
    inf = art["internal_noise_floor"]
    assert inf["t_max"] == builder.T_MAX
    assert inf["min_events_for_gate"] == builder.MIN_EVENTS_FOR_GATE
    assert art["draft_thresholds"]["k"] == builder.DRAFT_K
    assert len(inf["floor_seeds"]) == 100


# --------------------------------------------------------------------------
# Internal partition reconstructed through the build script's own logic
# --------------------------------------------------------------------------
def _reconstruct(art: dict):
    k = art["draft_thresholds"]["k"]
    r = art["draft_thresholds"]["rounding"]
    floor = art["internal_noise_floor"][FLOOR_KEY]
    stability = {
        key: {
            "family": v["family"],
            "defined_seeds": v["defined_seeds"],
            "n_seeds": v["n_seeds"],
            "min_events_either_half": v["min_events_either_half"],
        }
        for key, v in art["internal_noise_floor"]["cell_stability"].items()
    }
    tolerances = {
        key: round(block["mean"] + k * block["sd"], r)
        for key, block in floor.items()
    }
    return stability, tolerances


def test_internal_partition_reconstructs_through_partition_cells():
    art = _artifact()
    builder = _builder()
    stability, tolerances = _reconstruct(art)
    gated, report, reasons = builder.partition_cells(stability, tolerances)
    assert gated == set(art["gate_partition"]["internal_gate_eligible"])
    for key, reason in reasons.items():
        assert (
            art["internal_noise_floor"]["cell_stability"][key]["report_reason"]
            == reason
        ), key


def test_prevalence_and_conversion_never_internally_gated():
    """Family rule: only incidence/recovery are internal-gate candidates;
    prevalence/conversion levels are bridged by the anchor shape cells."""
    art = _artifact()
    stability = art["internal_noise_floor"]["cell_stability"]
    for key, v in stability.items():
        if key.split(".")[0] in ("prevalence", "conversion"):
            assert v["gate_eligible"] is False, key
            assert v["report_reason"] == "level_bridged_via_anchor_shape", key


def test_demoting_events_flips_a_gated_internal_cell():
    art = _artifact()
    builder = _builder()
    stability, tolerances = _reconstruct(art)
    gated0, _, _ = builder.partition_cells(stability, tolerances)
    victim = sorted(gated0)[0]
    stability[victim]["min_events_either_half"] = 5
    gated1, report1, reasons1 = builder.partition_cells(stability, tolerances)
    assert victim not in gated1
    assert reasons1[victim] == "below_20_events"


def test_tightening_the_cap_below_a_gated_tolerance_demotes_it():
    """No cap dead zone: a cap just below the highest gated tolerance must
    demote exactly it (guards the ln(1.5)->ln(1.5)+eps gap)."""
    art = _artifact()
    builder = _builder()
    stability, tolerances = _reconstruct(art)
    gated0, _, _ = builder.partition_cells(stability, tolerances)
    top = max(gated0, key=lambda c: tolerances[c])
    orig = builder.T_MAX
    try:
        builder.T_MAX = tolerances[top] - 1e-6
        gated1, _, reasons1 = builder.partition_cells(stability, tolerances)
        assert top not in gated1
        assert reasons1[top] == "tolerance_above_t_max"
    finally:
        builder.T_MAX = orig


def test_k4_empties_internal_surface_k3_gates_the_disclosed_set():
    """The k-sensitivity disclosure: k=4 (the gate-2c inheritance) gates 0
    cells; this DRAFT k=3 gates the committed set. Bound to the wart."""
    art = _artifact()
    ksens = art["internal_noise_floor"]["k_sensitivity"]
    assert ksens["4"]["n_gate_eligible"] == 0
    assert set(ksens["3"]["cells"]) == set(
        art["gate_partition"]["internal_gate_eligible"]
    )
    assert ksens["2"]["n_gate_eligible"] >= ksens["3"]["n_gate_eligible"]
    warts = " ".join(w["wart"] + w["detail"] for w in art["warts"])
    assert "k=4" in warts


# --------------------------------------------------------------------------
# Bounded anchor cells: ordinal (unit-honest), NOT |ln ratio| floors
# --------------------------------------------------------------------------
def test_anchor_cells_are_ordinal_not_logratio_gates():
    """Unit-correlation-honest lesson: the prevalence share and the
    conversion-exit share are bounded, so they are gated as ordinal
    invariants -- never as |ln ratio| tolerances."""
    art = _artifact()
    for key, v in art["anchor_checks"].items():
        assert "log_ratio_abs_max" not in v, key
        assert "gate_rule" in v and "margin_sigma_units" in v, key
    # draft_thresholds.cells holds ONLY internal |ln ratio| cells.
    for key in art["draft_thresholds"]["cells"]:
        assert key.split(".")[0] in ("incidence", "recovery"), key


def test_prevalence_ageshape_gated_and_anchor_corroborated():
    art = _artifact()
    for sex in ("female", "male"):
        c = art["anchor_checks"][f"prevalence_ageshape.comonotone|{sex}"]
        assert c["gated"] is True
        assert c["rank_agreement_over_bridged_bands"] is True
        assert c["psid_working_age_peak"] == "50-59"
        assert c["half_split_floor"]["holds_on_all_halves"] is True
        assert c["margin_sigma_units"] >= art["draft_thresholds"]["margin_k"]
        # PSID shares strictly rise over the bridged bands ...
        ps = [c["psid_share_by_band"][b] for b in BRIDGED]
        assert all(ps[i] < ps[i + 1] for i in range(3)), sex
        # ... and so does the Table 19 anchor (the corroboration).
        t19 = [c["t19_share_by_band"][b] for b in BRIDGED]
        assert all(t19[i] < t19[i + 1] for i in range(3)), sex
        assert c["t19_strictly_rises_bridged"] is True


def test_prevalence_label_swap_band_is_caught():
    """A self-consistent 20-29<->50-59 band swap inverts the monotone rise --
    the co-monotone gate catches it with NO PSID reproduction."""
    art = _artifact()
    for sex in ("female", "male"):
        c = art["anchor_checks"][f"prevalence_ageshape.comonotone|{sex}"]
        ps = [c["psid_share_by_band"][b] for b in BRIDGED]
        swapped = ps.copy()
        swapped[0], swapped[3] = swapped[3], swapped[0]  # 20-29 <-> 50-59
        gaps = [swapped[i + 1] - swapped[i] for i in range(3)]
        assert min(gaps) < 0, sex  # no longer strictly rising -> rejected


def test_conversion_exit_gated_and_brackets_t50():
    art = _artifact()
    for sex in ("female", "male"):
        c = art["anchor_checks"][f"conversion_exit.retirement_dominant|{sex}"]
        assert c["gated"] is True
        assert c["psid_retirement_exit_share"] > 0.5
        assert c["t50_is_dominant"] is True
        assert c["half_split_floor"]["holds_on_all_halves"] is True
        assert c["margin_sigma_units"] >= art["draft_thresholds"]["margin_k"]
        assert (
            c["psid_n_exits"]
            >= art["internal_noise_floor"]["min_events_for_gate"]
        )


def test_conversion_exit_label_swap_is_caught():
    """A retirement<->return-to-work swap sends the share below 0.5 -- the
    dominance gate catches it with NO PSID reproduction."""
    art = _artifact()
    for sex in ("female", "male"):
        c = art["anchor_checks"][f"conversion_exit.retirement_dominant|{sex}"]
        swapped = 1.0 - c["psid_retirement_exit_share"]
        assert swapped < 0.5, sex  # work-return is not dominant -> rejected


# --------------------------------------------------------------------------
# Report-only surfaces (delta named, never calibrated)
# --------------------------------------------------------------------------
def test_conversion_level_and_sexorder_report_only():
    art = _artifact()
    cr = art["conversion_reports"]
    assert cr["gated"] is False
    for sex in ("female", "male"):
        assert cr["level_ratio"]["by_sex"][sex]["ratio_psid_to_admin"] < 1.0
    so = cr["sex_ordering"]
    # The sex-ordering matches on the full sample but is half-split fragile
    # (flips on some halves) -> report-only, not a gated co-movement.
    assert so["psid_male_gt_female_full_sample"] is True
    assert so["psid_male_gt_female_on_n_halves"] < so["n_halves"]
    assert "fragile" in so["note"]


def test_termination_trend_report_only_with_pooling_delta():
    art = _artifact()
    tr = art["termination_trend_report"]
    assert tr["gated"] is False
    assert tr["report_reason"] == "no_pooled_year_series"
    t49 = tr["table_49_worker_rate"]
    assert t49["recent_trend_direction"] == "rising"
    assert t49["worker_rate_2023"] > t49["worker_rate_2015"]


# --------------------------------------------------------------------------
# Partition totals + operating characteristic
# --------------------------------------------------------------------------
def test_gate_partition_is_a_clean_disjoint_cover():
    art = _artifact()
    gp = art["gate_partition"]
    gated, report = set(gp["gate_eligible"]), set(gp["report_only"])
    assert gated & report == set()
    assert gp["n_gate_eligible"] == len(gated)
    assert gp["n_report_only"] == len(report)
    # 6 internal + 4 anchor gated; 26 report-only reference-moment cells.
    assert set(gp["internal_gate_eligible"]) <= gated
    assert set(gp["anchor_gate_eligible"]) <= gated
    assert len(gp["anchor_gate_eligible"]) == 4
    # every reference-moment cell + anchor cell is classified exactly once.
    universe = set(art["reference_moments"]) | set(art["anchor_checks"])
    assert gated | report == universe


def test_faithful_oc_and_training_copy_bound():
    art = _artifact()
    builder = _builder()
    floor = art["internal_noise_floor"][FLOOR_KEY]
    tolerances = {
        c: s["log_ratio_abs_max"]
        for c, s in art["draft_thresholds"]["cells"].items()
    }
    gated = set(art["gate_partition"]["internal_gate_eligible"])
    # A verbatim train-copy passes at ~the floor (max score below 1x tol).
    assert art["training_copy_check"]["passes_4_of_5"] is True
    assert art["training_copy_check"]["max_score_over_tolerance"] < 1.0
    # OC recomputes and is bound to the tolerances.
    oc = builder.faithful_candidate_oc(floor, tolerances, gated)
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
# Concept deltas, wanted tables, warts
# --------------------------------------------------------------------------
def test_concept_deltas_and_wanted_tables():
    art = _artifact()
    deltas = art["concept_deltas"]
    assert len(deltas) == 7
    names = " ".join(d["name"] for d in deltas).lower()
    assert "self-report" in names or "adjudication" in names
    assert "transience" in names or "recovery" in names
    wanted = art["wanted_ssa_tables"]
    blob = json.dumps(wanted)
    assert "Table 19" in blob and "Table 50" in blob and "Table 49" in blob
    assert any("IN HAND" in w["status"] for w in wanted)
    assert any(w["status"] == "WANTED" for w in wanted)


def test_warts_are_present_and_named():
    art = _artifact()
    warts = art["warts"]
    assert len(warts) >= 4
    blob = " ".join(w["wart"] for w in warts).lower()
    assert "k=4" in blob
    assert "anchor" in blob
    assert "censor" in blob or "death" in blob


def test_protocol_is_k20_from_the_start():
    """The K=20 mean-over-draws estimator is first-class (adopted from the
    START), bound to the builder constant and the gated-cell counts."""
    art = _artifact()
    builder = _builder()
    proto = art["protocol"]
    assert proto["candidate_draws"] == builder.CANDIDATE_DRAWS == 20
    assert proto["draw_stream_base"] == builder.DRAW_STREAM_BASE
    assert "mean-over-K=20" in proto["candidate_estimator"]
    assert proto["n_gated_internal_cells"] == len(
        art["gate_partition"]["internal_gate_eligible"]
    )
    assert proto["n_gated_anchor_cells"] == len(
        art["gate_partition"]["anchor_gate_eligible"]
    )
    assert "4 of 5" in proto["conjunction"]
