"""Tests for the W1 transport-gate floors (gate_w1_floors_v1).

The artifact ``runs/gate_w1_floors_v1.json`` is PRE-LOCK EVIDENCE -- step 1
of the gate-W1 lock ceremony (#151 design; #113 M5; #100 W1), the same role
PR #79 played for tranche 2a, #118 for gate 2b, #124 for gate 2c. It reads no
gate, changes no gate, and writes NO ``gates.yaml`` block (the ratifying flip
inserts the proposed gate_w1). Three target families:

* A -- CPS-observable joints, priced by a 100-seed household-disjoint
  half-split floor on the certified frame's own weighted moments;
* B -- SSA administrative anchors, priced by named machine-derivable
  vintage/measurement tolerances (no sampling floor);
* C -- the two ordinal compression fingerprints, a binary check vs the
  committed before/after orderings (no floor).

Almost every test touches only the committed artifact (always runnable). One
reproduction pin rebuilds the family-A seed-0 half-split from the certified
frame and is skipped unless ``huggingface_hub`` + ``tables`` are importable
AND the pinned h5 is already in the local HF cache (no network in CI).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate_w1_floors_v1.json"
SCRIPTS = ROOT / "scripts"
FLOOR_KEY = "noise_floor_seeds_0_99"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_gate_w1_floors as builder

    return builder


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# Schema + pre-lock framing (always runnable)
# --------------------------------------------------------------------------
def test_schema_and_reported_anchor():
    art = _artifact()
    assert art["schema_version"] == "gate_w1_floors.v1"
    assert art["run"] == "gate_w1_floors_v1"
    assert art["reported_anchor_not_gated"] is True
    assert "changes no gate" in art["purpose"]
    assert "TRANSPORT" in art["purpose"] or "transport" in art["component"]
    assert art["design_issue"] == "#151"
    assert art["roadmap_issue"] == "#113"
    assert art["seam_issue"] == "#100"
    note = art["draft_thresholds_note"]
    assert "NOT RATIFIED" in note
    assert "lock ceremony" in note


def test_ceremony_writes_no_gates_yaml():
    art = _artifact()
    cer = art["ceremony"]
    assert cer["gates_yaml_untouched"] is True
    assert "gate2c_floors_v1.json" in cer["mirrors"]
    assert "reform" not in art
    assert "gate_result" not in art
    assert "NOT yet mirrored" in art["draft_thresholds"]["note"]


def test_gates_yaml_gate_w1_block_is_locked_by_the_flip():
    """Flip-time update: the W1 lock flip (2026-07-12) INSERTS the gate_w1
    block the #151 stub proposed, so gates.yaml now carries a LOCKED gate_w1
    as a top-level gate (the floors PR left it absent; this assertion inverts
    at the flip, exactly as the gate-2c flip inverted its sibling assertion).
    The frozen floor's own ceremony.gates_yaml_untouched stays True -- that is
    a statement about the FLOORS step, not the live contract. The locked gates
    1/2a/2b/2c stay locked and untouched."""
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load((ROOT / "gates.yaml").read_text())
    gates = spec["gates"]
    gate_w1 = gates["gate_w1"]
    assert gate_w1["locked"] is True
    assert gate_w1["status"] == "locked"
    assert gate_w1["id"] == "w1_representative_frame_transport"
    assert gate_w1["thresholds"]["floor_run"] == "runs/gate_w1_floors_v1.json"
    # the FROZEN floor still records that the FLOORS step wrote no gates.yaml.
    assert _artifact()["ceremony"]["gates_yaml_untouched"] is True
    # locked siblings untouched by the flip.
    assert gates["gate_2"]["thresholds"]["locked"] is True
    assert gates["gate_2"]["gate_2c"]["locked"] is True


def test_provenance_pins_present():
    art = _artifact()
    pins = art["revision_pins"]
    assert pins["artifact_schema_version"] == "gate_w1_floors.v1"
    assert "populace_dynamics_sha" not in pins
    assert "base_sha" in pins
    assert (
        pins["certified_artifact_sha256"]
        == "c2065b642ab00da74746afdfd9f06890e5f32f9b10bd6610ff236452d40f39c5"
    )
    assert "release_bundle" in pins["build_env"]


def test_estimand_pins_the_certified_frame():
    art = _artifact()
    est = art["estimand"]
    frame = est["deployment_frame"]
    assert frame["bundle_id"] == "us-4.18.8"
    assert frame["dataset"] == "populace_us_2024"
    assert "sparse-l0-refit-57k" in frame["revision"]
    assert est["n_persons"] == 166302
    assert est["n_households"] == 57240
    assert 3.3e8 < est["total_weight"] < 3.5e8
    # the sparse-file populated-column disclosure is present and passes floors.
    frac = est["populated_source_fractions"]
    assert frac["employment_income_before_lsr"] > 0.30
    assert frac["A_MARITL"] >= 0.99
    assert "EXACTLY" in est["claim"]


# --------------------------------------------------------------------------
# Family A: 100-seed household-disjoint floor (always runnable)
# --------------------------------------------------------------------------
def test_floor_is_measured_on_100_household_disjoint_seeds():
    art = _artifact()
    nf = art["internal_noise_floor"]
    assert nf["floor_seeds"] == list(range(100))
    assert nf["gate_seeds"] == [0, 1, 2, 3, 4]
    assert nf["split_unit"] == "household"
    assert nf["method"].startswith("half_vs_half")
    for block in art[FLOOR_KEY].values():
        assert block["n_seeds"] == 100
        assert len(block["values"]) == 100


def test_pooled_floor_recomputes_from_values():
    import numpy as np

    art = _artifact()
    for key, block in art[FLOOR_KEY].items():
        values = block["values"]
        assert block["mean"] == pytest.approx(np.mean(values)), key
        assert block["sd"] == pytest.approx(np.std(values, ddof=1)), key
        assert block["max"] == pytest.approx(max(values)), key
        rms = math.sqrt(sum(v * v for v in values) / len(values))
        assert block["realized_sigma"] == pytest.approx(rms), key


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


def test_draft_thresholds_recompute_from_floor():
    art = _artifact()
    k = art["draft_thresholds"]["k"]
    rounding = art["draft_thresholds"]["rounding"]
    floor = art[FLOOR_KEY]
    for key, spec in art["draft_thresholds"]["cells"].items():
        mean = floor[key]["mean"]
        sd = floor[key]["sd"]
        derived = round(mean + k * sd, rounding)
        assert spec["log_ratio_abs_max"] == pytest.approx(derived), key
        sigma = floor[key]["realized_sigma"]
        assert spec["tolerance_sigma_units"] == pytest.approx(
            round(derived / sigma, 3)
        )


def test_every_gated_tolerance_respects_power_cap_and_heavy_tail_guard():
    art = _artifact()
    t_max = art["internal_noise_floor"]["t_max"]
    assert t_max == pytest.approx(math.log(1.5))
    floor = art[FLOOR_KEY]
    for key, spec in art["draft_thresholds"]["cells"].items():
        tol = spec["log_ratio_abs_max"]
        assert tol <= t_max, key
        # the W1 heavy-tail guard: the tolerance covers the floor's own max.
        assert floor[key]["max"] <= tol, key


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
    assert set(art["reference_moments"]) == gate_eligible | report_only


def test_partition_is_a_real_cover():
    art = _artifact()
    gate_eligible = set(art["gate_partition"]["gate_eligible"])
    report_only = set(art["gate_partition"]["report_only"])
    assert gate_eligible and report_only
    assert gate_eligible.isdisjoint(report_only)
    assert gate_eligible | report_only == set(art["reference_moments"])
    for key in gate_eligible:
        assert key in art[FLOOR_KEY], key


def test_report_only_reasons_are_recorded():
    art = _artifact()
    stability = art["cell_stability"]
    valid = (
        "below_20_events",
        "undefined_on_some_seed",
        "tolerance_above_t_max",
        "floor_max_exceeds_tolerance",
        "no_floor",
    )
    for key in art["gate_partition"]["report_only"]:
        reason = stability[key]["report_reason"]
        assert reason and reason.startswith(valid), (key, reason)


def test_heavy_tail_guard_demotes_are_real_and_disclosed():
    """The W1-specific guard: at least one cell demotes because its
    heavy-tailed floor max exceeds the mean+4sd tolerance, and the training
    copy consequently passes at <=1x (unlike the un-guarded build)."""
    art = _artifact()
    stab = art["cell_stability"]
    demotes = [
        k
        for k, v in stab.items()
        if v["report_reason"] == "floor_max_exceeds_tolerance"
    ]
    assert demotes, "expected heavy-tail-guard demotes on the sparse frame"
    # every such demote genuinely has floor_max > tolerance.
    floor = art[FLOOR_KEY]
    for k in demotes:
        tol = round(
            floor[k]["mean"] + art["draft_thresholds"]["k"] * floor[k]["sd"],
            art["draft_thresholds"]["rounding"],
        )
        assert floor[k]["max"] > tol, k
    assert art["training_copy_check"]["max_score_over_tolerance"] <= 1.0


def test_all_family_a_families_present_and_priced():
    art = _artifact()
    ref = art["reference_moments"]
    gated = set(art["gate_partition"]["gate_eligible"])
    report = set(art["gate_partition"]["report_only"])
    prefixes = (
        "earnings_participation",
        "earnings_profile",
        "earnings_p90p50",
        "earnings_p50p10",
        "marital_share",
        "hh_size_share",
        "coresident_spouse",
    )
    for prefix in prefixes:
        cells = [k for k in ref if k.startswith(prefix + ".")]
        assert cells, prefix
        for k in cells:
            assert (k in gated) ^ (k in report), k
    # the dense prime-age cells gate.
    assert "earnings_participation.35-44|male" in gated
    assert "hh_size_share.1" in gated
    assert "coresident_spouse.45-54|female" in gated


# --------------------------------------------------------------------------
# Family B: SSA anchors + machine-derivable vintage tolerances (runnable)
# --------------------------------------------------------------------------
def test_family_b_anchor_pricing_and_partition():
    art = _artifact()
    fb = art["family_b"]
    assert fb["pricing"].startswith("anchor")
    knobs = fb["knobs"]
    assert knobs["vintage_window_years"] == 10
    assert knobs["k_vintage"] == 2.0
    assert knobs["t_max_pp"] == 3.0
    assert knobs["anchor_frame_year"] == 2024
    assert knobs["claim_age_delta_years"] == 2
    assert knobs["di_delta_years"] == 1
    # claim-age: 8 categories x 2 sexes present.
    assert len([k for k in fb["claim_age"]]) == 16
    # DI prevalence: 8 age bands.
    assert len([k for k in fb["di_prevalence"]]) == 8
    part = fb["gate_partition"]
    # POST-FIX-A partition: 10 gated (2 disability_conversion + 8 DI bands),
    # 15 report-only (14 circular claim-age + 1 benefit level).
    assert part["n_gate_eligible"] == 10
    assert part["n_report_only"] == 15
    assert part["n_circular_report_only"] == 14
    gated = set(part["gate_eligible"])
    assert gated == {
        "claim_age.disability_conversion|male",
        "claim_age.disability_conversion|female",
        *(
            f"di_prevalence.{b}"
            for b in (
                "under30",
                "30-34",
                "35-39",
                "40-44",
                "45-49",
                "50-54",
                "55-59",
                "60-fra",
            )
        ),
    }
    assert "benefit_level.report_only" in part["report_only"]


def test_family_b_claim_age_circularity_is_named_and_gated_content_is_conversion():
    """fix A / finding 1: the 14 non-conversion claim-age cells are report-only
    (sampled from their own anchor by the v1 claiming module); the 2
    disability-conversion cells (the M4-simulated margin) stay gated."""
    art = _artifact()
    fb = art["family_b"]
    for key, cell in fb["claim_age"].items():
        is_conv = key.endswith(
            ("disability_conversion|male", "disability_conversion|female")
        )
        assert cell["is_conversion_margin"] == is_conv, key
        assert cell["circular_under_v1_candidate"] == (not is_conv), key
        if not is_conv:
            assert cell["gate_eligible"] is False, key
            assert cell["report_reason"] == (
                "circular_under_v1_claiming_candidate"
            ), key
        else:
            # a conversion cell gates iff its reference-period tolerance clears.
            assert cell["gate_eligible"] == (
                cell["tolerance_gate_eligible"]
            ), key
    # the circularity is named in a wart and the design note.
    assert "family_b_claim_age_circularity" in {w["id"] for w in art["warts"]}
    assert "circular" in fb["circularity_rule"]


def test_family_b_candidate_protocol_is_specified():
    """fix A / finding 2: family B now has a full candidate protocol -- object,
    population, estimator, reference-period rule, no-frame-DI-column rule,
    pass rule."""
    art = _artifact()
    proto = art["family_b"]["candidate_protocol"]
    assert "FULL certified deployment frame" in proto["population"]
    assert proto["candidate_draws"] == 20
    assert proto["family_b_draw_stream_base"] == 9200
    assert "claim_age.disability_conversion" in proto["simulated_object"]
    assert "December-2023" in proto["simulated_object"]["di_prevalence"]
    assert "M4 dynamics" in proto["no_frame_di_column_rule"]
    assert "social_security_disability" in proto["no_frame_di_column_rule"]
    assert "CONJUNCTION" in proto["pass_rule"]
    assert str(art["family_b"]["knobs"]["anchor_frame_year"]) in (
        proto["reference_period_rule"]
    )


def test_family_b_reference_period_prices_the_trend():
    """fix A / finding 2: the tolerance carries a |trend| * Delta term, so the
    di_prevalence.60-fra cell that would fail a faithful candidate at 1.90x the
    detrended tolerance is now PRICED (tolerance covers the trend drift)."""
    art = _artifact()
    fb = art["family_b"]
    cell = fb["di_prevalence"]["di_prevalence.60-fra"]
    # the trend component is the priced drift over Delta=1 year.
    assert cell["reference_period_delta_years"] == 1
    assert cell["trend_component_pp"] == pytest.approx(
        abs(cell["trend_pp_per_year"]) * 1, abs=1e-6
    )
    # tolerance = detrended + trend component (both disclosed).
    assert cell["tolerance_pp"] == pytest.approx(
        round(cell["detrended_tolerance_pp"] + cell["trend_component_pp"], 2),
        abs=0.011,
    )
    # the priced tolerance now covers the drift the v1 detrended rule excluded.
    assert cell["trend_component_pp"] <= cell["tolerance_pp"]
    assert cell["gate_eligible"] is True


def test_family_b_vintage_tolerance_recomputes():
    """The reference-period tolerance is round(k*detrended_residual_sd +
    |trend|*Delta + meas, 2), a machine rule reconstructible from the recorded
    residual sd, trend, and Delta (fix A / finding 2). tolerance_gate_eligible
    is the tolerance's own T_max_pp check; the FINAL gate_eligible also removes
    the circular claim-age cells."""
    art = _artifact()
    fb = art["family_b"]
    k = fb["knobs"]["k_vintage"]
    meas = fb["knobs"]["measurement_pp"]
    tmax = fb["knobs"]["t_max_pp"]
    for group in ("claim_age", "di_prevalence"):
        for key, v in fb[group].items():
            # trend_component_pp is |trend| * Delta at FULL trend precision;
            # recompute from the recorded (3dp) trend is exact to the rounding
            # (<= Delta * 5e-4).
            trend_comp = (
                abs(v["trend_pp_per_year"]) * v["reference_period_delta_years"]
            )
            expect = round(
                k * v["detrended_residual_sd_pp"] + trend_comp + meas, 2
            )
            assert v["tolerance_pp"] == pytest.approx(expect, abs=0.02), key
            assert v["trend_component_pp"] == pytest.approx(
                trend_comp, abs=1.5e-3
            ), key
            assert v["tolerance_gate_eligible"] == (
                v["tolerance_pp"] <= tmax
            ), key
            # final gate_eligible = tolerance-eligible AND not circular.
            assert v["gate_eligible"] == (
                v["tolerance_gate_eligible"]
                and not v["circular_under_v1_candidate"]
            ), key
            assert v["anchor_pp"] >= 0


def test_family_b_benefit_levels_are_report_only():
    art = _artifact()
    bene = art["family_b"]["benefit_level"]
    assert bene["status"] == "report_only"
    assert "transported_aime" in bene["reason"]
    assert bene["anchors"]["avg_pia_all"] == 1984.09
    assert bene["anchors"]["avg_monthly_benefit_all"] == 1908.86


def test_family_b_anchor_values_match_staged_sources():
    """The recorded anchors match the in-repo staged SSA files byte-for-byte
    (no fabricated numbers): 6.B5.1 2022 collapsed categories + DI T19 2023."""
    art = _artifact()
    fb = art["family_b"]
    claim_src = json.loads(
        (
            ROOT / "data" / "external" / "ssa_claim_ages_2023supplement.json"
        ).read_text()
    )
    m2022 = claim_src["data"]["male"]["2022"]["categories"]
    assert (
        fb["claim_age"]["claim_age.age62|male"]["anchor_pp"] == m2022["age62"]
    )
    assert (
        fb["claim_age"]["claim_age.disability_conversion|male"]["anchor_pp"]
        == m2022["disability_conversion"]
    )
    di_src = json.loads(
        (
            ROOT / "data" / "external" / "di_asr_2023" / "tables.json"
        ).read_text()
    )
    # 2023 All-disabled-workers 60-FRA share is 45.4 (documented headline).
    assert fb["di_prevalence"]["di_prevalence.60-fra"]["anchor_pp"] == 45.4
    assert "Table 19" in di_src["Table 19"]["caption"]


# --------------------------------------------------------------------------
# Family C: the two ordinal compression fingerprints (binary; runnable)
# --------------------------------------------------------------------------
def test_family_c_fingerprints_are_the_committed_before_after_orderings():
    art = _artifact()
    fc = art["family_c"]
    assert fc["pricing"].startswith("binary ordinal")
    c1 = fc["fingerprints"]["c1"]
    c2 = fc["fingerprints"]["c2"]
    # C1 PPI<->NRA (the #115 T2 / #117 F4 swap).
    assert c1["psid_frame_order"] == [
        "price_indexing",
        "nra_raised_to_70",
        "progressive_price_indexing",
        "reduced_cola",
    ]
    assert c1["required_representative_order"] == [
        "price_indexing",
        "progressive_price_indexing",
        "nra_raised_to_70",
        "reduced_cola",
    ]
    assert set(c1["swap_pair"]) == {
        "nra_raised_to_70",
        "progressive_price_indexing",
    }
    assert c1["kendall_tau_before"] == pytest.approx(2.0 / 3.0)
    assert c1["kendall_tau_after_required"] == 1.0
    # C2 elimination<->+2pp (the #117 F2 swap).
    assert c2["psid_frame_order"] == [
        "payroll_plus_2pp",
        "elimination",
        "payroll_plus_1pp",
        "cap_150k",
    ]
    assert c2["required_representative_order"] == [
        "elimination",
        "payroll_plus_2pp",
        "payroll_plus_1pp",
        "cap_150k",
    ]
    assert set(c2["swap_pair"]) == {"payroll_plus_2pp", "elimination"}


def test_family_c_reversal_is_a_single_adjacent_swap_each():
    """Each fingerprint's before->after difference is EXACTLY the swap_pair
    adjacent transposition, nothing else (the '#113 and nothing else in the
    orderings' commitment)."""
    art = _artifact()
    for fp in art["family_c"]["fingerprints"].values():
        before = fp["psid_frame_order"]
        after = fp["required_representative_order"]
        assert sorted(before) == sorted(after)
        differ = [i for i in range(len(before)) if before[i] != after[i]]
        # exactly two adjacent positions differ (one transposition).
        assert len(differ) == 2, fp["id"]
        assert differ[1] - differ[0] == 1, fp["id"]
        assert {before[differ[0]], before[differ[1]]} == set(fp["swap_pair"])
        # after restores the anchor: the two swapped are transposed.
        assert before[differ[0]] == after[differ[1]]
        assert before[differ[1]] == after[differ[0]]


def test_family_c_committed_orderings_match_the_source_artifacts():
    """The before-orderings are the committed #115/#117 results, not
    fabricated: C1 == m2 F4 outlay order, C2 == m2 F2 exhaustion order."""
    art = _artifact()
    m2 = json.loads(
        (ROOT / "runs" / "m2_pseudo_projection_v1.json").read_text()
    )
    f4 = m2["results_vs_forecasts"]["F4"]["result_order"]
    f2 = m2["results_vs_forecasts"]["F2"]["result_order"]
    assert art["family_c"]["fingerprints"]["c1"]["psid_frame_order"] == f4
    assert art["family_c"]["fingerprints"]["c2"]["psid_frame_order"] == f2


# --------------------------------------------------------------------------
# Estimand / K=20 protocol / OC / training copy / WARTS (always runnable)
# --------------------------------------------------------------------------
def test_protocol_is_the_ratified_k20_estimator():
    art = _artifact()
    proto = art["protocol"]
    assert proto["candidate_draws"] == 20
    assert "9100" in proto["candidate_draw_stream"]
    assert "K=20" in proto["estimator"]
    assert "NOT the mean of the per-draw" in proto["estimator"]
    schema = proto["fresh_run_artifact_schema"]
    n_gated = art["gate_partition"]["n_gate_eligible"]
    assert schema["per_draw_per_cell_rates"]["shape"] == [20, n_gated, 5]
    assert schema["undefined_draw_rule"]["pre_specified"] is True
    assert schema["per_draw_dispersion_disclosure"]["report_only"] is True


def test_training_copy_disclosure_is_honest():
    art = _artifact()
    check = art["training_copy_check"]
    assert check["passes_4_of_5"] is True
    assert check["max_score_over_tolerance"] <= 1.0
    assert "procedural" in check["interpretation"]


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


def test_holdout_ids_committed_for_every_gate_seed():
    art = _artifact()
    hold = art["holdout_ids"]
    assert hold["gate_seeds"] == [0, 1, 2, 3, 4]
    assert hold["split_unit"] == "household"
    assert "split_panel_by_person" in hold["numpy_generator"]
    for entry in hold["per_seed"]:
        assert entry["n_holdout_households"] > 0
        assert len(entry["holdout_household_id_sha256"]) == 64


def test_warts_preempt_the_referee_classes():
    art = _artifact()
    ids = {w["id"] for w in art["warts"]}
    for required in (
        "sparse_frame_zeroes_untargeted_inputs",
        "weight_concentration",
        "family_a_is_internal_consistency_not_census_fidelity",
        "family_b_non_stationary_anchors",
        "family_b_claim_age_circularity",
        "age_top_code_85",
        "family_b_benefit_levels_report_only",
        "family_c_records_not_runs_the_reversal",
        "di_prevalence_rate_denominator_absent",
        "earnings_concept_before_lsr",
    ):
        assert required in ids, required


# --------------------------------------------------------------------------
# Fixes-round additions: regenerated surface, degenerate identity candidate,
# A' published-value block, boundary bootstrap, records (always runnable)
# --------------------------------------------------------------------------
def test_regenerated_surface_is_pinned_in_the_protocol():
    """fix B / finding 3: the protocol states, per cell family, which columns
    the candidate must RE-GENERATE; copying a scored column is non-conformant.
    """
    art = _artifact()
    schema = art["protocol"]["fresh_run_artifact_schema"]
    surf = schema["regenerated_surface"]
    assert surf["identity_candidate_is_non_conformant"] is True
    assert "NON-CONFORMANT" in surf["rule"]
    assert set(surf["per_family"]) == {
        "earnings_participation|profile|p90p50|p50p10",
        "marital_share|coresident_spouse",
        "hh_size_share",
    }
    # the gate-2 "fit complement" language is grounded, not inherited verbatim.
    assert "fit complement" in art["protocol"]["varies_per_seed"]
    assert "nothing REFITS per W1 seed" in art["protocol"]["varies_per_seed"]


def test_degenerate_identity_candidate_is_disclosed():
    """fix B / finding 3: the identity candidate (copies the scored columns,
    scores 0) is named in the degenerate table and caught by the regenerated-
    surface rule + the zero across-draw dispersion."""
    art = _artifact()
    deg = art["degenerate_candidates"]
    ident = deg["identity_candidate"]
    assert ident["conformance"] == "NON-CONFORMANT"
    assert ident["across_draw_sd"] == 0.0
    assert "max_per_draw_abs_ln_per_cell" in ident["caught_by"]
    assert "nothing about the generators" in ident["certifies"]
    # the train-copy is still carried (its disclosure is unchanged).
    assert deg["train_copy"]["passes_4_of_5"] is True


def test_per_draw_dispersion_discloses_both_fields():
    """fix E / finding 9: the locked gate-2 per_draw_dispersion disclosure has
    BOTH the across-draw sd and max_per_draw_abs_ln_per_cell (the v1 build
    dropped the max, which is what exposes an identity candidate)."""
    art = _artifact()
    disc = art["protocol"]["fresh_run_artifact_schema"][
        "per_draw_dispersion_disclosure"
    ]
    assert disc["fields"] == [
        "per_cell_across_draw_sd",
        "max_per_draw_abs_ln_per_cell",
    ]
    assert "max_per_draw_abs_ln_per_cell == 0 EXPOSES" in disc["note"]


def test_family_a_prime_published_value_block_is_sourced_and_report_only():
    """fix D / finding 6: a report-only A' comparison of the frame's covered
    joints vs published CPS/ACS values, sourced from committed census files
    (sha256-pinned), NOT gated."""
    art = _artifact()
    ap = art["family_a_prime"]
    assert ap["status"] == "report_only"
    # household size: all five person-level shares compared to HH-4.
    hh = ap["household_size_person_level"]
    assert set(hh) == {
        f"hh_size_share.{c}" for c in ("1", "2", "3", "4", "5plus")
    }
    for row in hh.values():
        assert "frame_rate" in row and "published_share" in row
        assert row["abs_diff_pp"] >= 0
    # coresident spouse: the one aligned AD-3 band (25-34).
    sp = ap["coresident_spouse_aligned_band"]
    assert set(sp) == {
        "coresident_spouse.25-34|male",
        "coresident_spouse.25-34|female",
    }
    # sources carry the committed file + a sha256 for the household file.
    assert "census_household_size_2023.json" in (
        ap["sources"]["household_size"]["file"]
    )
    assert len(ap["sources"]["household_size"]["file_sha256"]) == 64
    assert "report_only" in ap["status"]
    assert "REPORT-ONLY" in ap["caveats"]


def test_family_a_prime_household_values_match_census_source():
    """The A' household-size published shares are the committed HH-4 person-
    level shares verbatim (no fabricated numbers)."""
    art = _artifact()
    src = json.loads(
        (
            ROOT / "data" / "external" / "census_household_size_2023.json"
        ).read_text()
    )
    pls = src["derived"]["person_level_share"]
    rows = art["family_a_prime"]["household_size_person_level"]
    assert rows["hh_size_share.1"]["published_share"] == pls["1"]
    assert rows["hh_size_share.5plus"]["published_share"] == pls["5+"]


def test_calibration_coverage_statement_present():
    """fix D / finding 6: the artifact states which family-A families ride on
    populace calibration and which ride uncertified."""
    art = _artifact()
    cov = art["family_a"]["calibration_coverage"]
    assert cov["covered_by_populace_calibration"]
    assert cov["not_calibration_covered_ride_uncertified"]
    assert any(
        "marital" in x for x in cov["not_calibration_covered_ride_uncertified"]
    )
    # the thin-coverage facts sentence is present and derived (13/50, 1/8).
    facts = art["family_a"]["coverage_facts"]
    assert "marital family gates 13 of 50" in facts
    assert "lower-tail dispersion gates 1 of 8" in facts


def test_heavy_tail_boundary_bootstrap_is_disclosed():
    """fix G / finding 7: the boundary-fragility bootstrap and the seed-count-
    dependence note are carried (2b-7c / 2c-8ii)."""
    art = _artifact()
    htb = art["heavy_tail_boundary_bootstrap"]
    assert htb["n_bootstrap"] == 5000
    # the 5 heavy-tail demotes each get a re-entry probability, ranked.
    reentry = htb["demote_reentry_prob"]
    assert len(reentry) == 5
    probs = list(reentry.values())
    assert probs == sorted(probs, reverse=True)
    assert 0.0 <= min(probs) and max(probs) <= 1.0
    assert "seed_count_dependence" in htb
    assert "in-sample max" in htb["seed_count_dependence"].lower()


def test_uprating_knob_is_stripped():
    """fix H / finding 10iv: the underived 6.0% uprating knob is removed (no
    machine rule, no series)."""
    art = _artifact()
    bene = art["family_b"]["benefit_level"]
    assert "uprating_context_tolerance_pct" not in bene
    assert "STRIPPED" in bene["note"]


def test_age_top_code_disclosed_in_estimand():
    """fix H / finding 5: the frame's age top-code is disclosed."""
    art = _artifact()
    est = art["estimand"]
    assert est["age_top_code"] == 85
    assert "TOP-CODES age" in est["age_top_code_note"]


def test_build_commit_note_documents_parent_sha_convention():
    """fix H / finding 10i: the base_sha parent-commit pin convention is
    documented (the 2b-8(iii) chicken-and-egg)."""
    art = _artifact()
    note = art["revision_pins"]["build_commit_note"]
    assert "base_sha = HEAD at BUILD time" in note
    assert "PARENT" in note
    assert "chicken-and-egg" in note
    # the .venv-gate skip of the only data-bound test is noted.
    assert "venv-gate" in art["revision_pins"]["certified_repro_env_note"]


# --------------------------------------------------------------------------
# Certified-frame reproduction pin (skips without hf/tables or a cached h5)
# --------------------------------------------------------------------------
def _cached_certified_h5():
    """Resolve the pinned h5 from the LOCAL HF cache only (no network)."""
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("tables")
    from huggingface_hub import hf_hub_download

    from populace_dynamics.data import deployment_frame as dfm

    pin = dfm.CERTIFIED_PIN
    try:
        return hf_hub_download(
            repo_id=pin["hf_repo_id"],
            filename=pin["hf_filename"],
            repo_type=pin["hf_repo_type"],
            revision=pin["revision"],
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001
        pytest.skip("certified populace h5 not in local HF cache")


def test_seed0_reproduces_from_the_certified_frame():
    path = _cached_certified_h5()
    builder = _import_builder()
    from populace_dynamics.data import deployment_frame as dfm

    persons, _ = dfm.load_certified_persons(path)
    got0 = builder.measure_seed_halfsplit(0, persons)
    art = _artifact()
    ref0 = next(s for s in art["noise_floor_per_seed"] if s["seed"] == 0)
    assert got0["n_households_side_a"] == ref0["n_households_side_a"]
    assert got0["n_persons_side_a"] == ref0["n_persons_side_a"]
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


def test_holdout_universe_and_sha256s_bind_to_the_certified_frame():
    """fix H / finding 10ii: the committed household-id universe IS the frame's
    own sorted household ids, and every gate seed's holdout sha256 recomputes
    from the frame (the v1 repro checked cells only, so a corrupted holdout
    sha was invisible even here -- mutation A6)."""
    path = _cached_certified_h5()
    builder = _import_builder()
    from populace_dynamics.data import deployment_frame as dfm

    persons, _ = dfm.load_certified_persons(path)
    art = _artifact()
    hold = art["holdout_ids"]
    universe = sorted(int(x) for x in persons["household_id"].unique())
    # the committed universe (CSV) is the frame's own household-id set.
    assert ",".join(str(i) for i in universe) == (
        hold["household_id_universe_csv"]
    )
    # every gate seed's holdout sha256 recomputes from the frame.
    for entry in hold["per_seed"]:
        ids = builder._holdout_household_ids(persons, entry["seed"])
        assert len(ids) == entry["n_holdout_households"], entry["seed"]
        assert (
            builder._sha256_ids(ids) == entry["holdout_household_id_sha256"]
        ), entry["seed"]
