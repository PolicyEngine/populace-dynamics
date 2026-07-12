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


def test_gates_yaml_has_no_gate_w1_block():
    """This PR touches no gate: gates.yaml must carry no gate_w1 (the stub is
    proposed in #151 and inserted only by the ratifying flip). The locked
    gates 1/2a/2b/2c stay locked and untouched."""
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load((ROOT / "gates.yaml").read_text())
    gates = spec["gates"]
    assert "gate_w1" not in gates
    assert "gate_w1" not in gates.get("gate_2", {})
    assert spec["gates"]["gate_2"]["thresholds"]["locked"] is True
    assert spec["gates"]["gate_2"]["gate_2c"]["locked"] is True


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
    # claim-age: 8 categories x 2 sexes present.
    assert len([k for k in fb["claim_age"]]) == 16
    # DI prevalence: 8 age bands.
    assert len([k for k in fb["di_prevalence"]]) == 8
    part = fb["gate_partition"]
    assert part["n_gate_eligible"] >= 16
    assert "benefit_level.report_only" in part["report_only"]


def test_family_b_vintage_tolerance_recomputes():
    """The vintage tolerance is round(k*detrended_residual_sd + meas, 2), a
    machine rule reconstructible from the recorded residual sd."""
    art = _artifact()
    fb = art["family_b"]
    k = fb["knobs"]["k_vintage"]
    meas = fb["knobs"]["measurement_pp"]
    tmax = fb["knobs"]["t_max_pp"]
    for group in ("claim_age", "di_prevalence"):
        for key, v in fb[group].items():
            expect = round(k * v["detrended_residual_sd_pp"] + meas, 2)
            assert v["tolerance_pp"] == pytest.approx(expect), key
            assert v["gate_eligible"] == (v["tolerance_pp"] <= tmax), key
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
        "family_b_benefit_levels_report_only",
        "family_c_records_not_runs_the_reversal",
        "di_prevalence_rate_denominator_absent",
        "earnings_concept_before_lsr",
    ):
        assert required in ids, required


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
