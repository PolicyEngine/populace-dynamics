"""Perturbation-hardened bindings for the gate-W1 floor derivations.

Companion to ``tests/test_gate_w1_floors.py``. Where that file checks each
derivation *recomputes*, this file proves each is genuinely *bound* -- a
perturbation changes the output, so a stored value equal to its derivation is
not a coincidence. It reconstructs the family-A partition through the build
script's own ``partition_cells`` (INCLUDING the W1 heavy-tail guard), pins the
builder constants directly (no T_max / MIN_EVENTS / K / heavy-tail-guard
perturbation dead zones), binds the family-B vintage tolerance rule and the
family-C committed orderings through the builder's own functions, and carries
ALWAYS-RUNNABLE data-free structural invariants a self-consistent cell swap
would violate. Reads only the committed artifact + imports the builder (no
data load at import).
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


def _builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_gate_w1_floors as builder

    return builder


# --------------------------------------------------------------------------
# Tolerance binding
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


def test_k_events_cap_and_stream_are_the_precedent_values():
    art = _artifact()
    assert art["draft_thresholds"]["k"] == 4
    assert art["draft_thresholds"]["rounding"] == 3
    assert art["internal_noise_floor"]["min_events_for_gate"] == 20
    assert art["internal_noise_floor"]["t_max"] == pytest.approx(math.log(1.5))
    assert art["protocol"]["candidate_draws"] == 20


# --------------------------------------------------------------------------
# Full family-A partition reconstructed through partition_cells
# --------------------------------------------------------------------------
def _reconstruct_inputs(art: dict):
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
    noise_floor = {key: {"max": block["max"]} for key, block in floor.items()}
    return stability, tolerances, noise_floor


def test_partition_reconstructs_through_partition_cells():
    art = _artifact()
    builder = _builder()
    stability, tolerances, noise_floor = _reconstruct_inputs(art)
    gated, report, reasons = builder.partition_cells(
        stability, tolerances, noise_floor
    )
    assert gated == set(art["gate_partition"]["gate_eligible"])
    assert report == set(art["gate_partition"]["report_only"])
    for key, reason in reasons.items():
        assert art["cell_stability"][key]["report_reason"] == reason, key


def test_demoting_events_flips_a_gated_cell():
    art = _artifact()
    builder = _builder()
    stability, tolerances, noise_floor = _reconstruct_inputs(art)
    gated0, _, _ = builder.partition_cells(stability, tolerances, noise_floor)
    victim = sorted(gated0)[0]
    stability[victim]["min_events_either_half"] = 5
    gated1, report1, reasons1 = builder.partition_cells(
        stability, tolerances, noise_floor
    )
    assert victim not in gated1
    assert reasons1[victim] == "below_20_events"


def test_loosening_a_tolerance_past_the_cap_demotes_it():
    art = _artifact()
    builder = _builder()
    stability, tolerances, noise_floor = _reconstruct_inputs(art)
    gated0, _, _ = builder.partition_cells(stability, tolerances, noise_floor)
    victim = sorted(gated0)[0]
    tolerances[victim] = builder.T_MAX + 0.5
    # keep the heavy-tail guard satisfiable so the cap reason is exercised.
    noise_floor[victim]["max"] = tolerances[victim] - 0.01
    gated1, _, reasons1 = builder.partition_cells(
        stability, tolerances, noise_floor
    )
    assert victim not in gated1
    assert reasons1[victim] == "tolerance_above_t_max"


def test_heavy_tail_guard_binds_the_partition():
    """The W1-specific criterion: inflate a gated cell's floor max above its
    tolerance and it must demote with reason floor_max_exceeds_tolerance."""
    art = _artifact()
    builder = _builder()
    stability, tolerances, noise_floor = _reconstruct_inputs(art)
    gated0, _, _ = builder.partition_cells(stability, tolerances, noise_floor)
    victim = sorted(gated0)[0]
    noise_floor[victim]["max"] = tolerances[victim] + 0.5
    gated1, _, reasons1 = builder.partition_cells(
        stability, tolerances, noise_floor
    )
    assert victim not in gated1
    assert reasons1[victim] == "floor_max_exceeds_tolerance"


def test_builder_constants_are_pinned():
    """Close the T_MAX / MIN_EVENTS / K / split-unit perturbation dead zones."""
    builder = _builder()
    assert builder.T_MAX == math.log(1.5)
    assert builder.T_MAX_SOURCE == "ln(1.5)"
    assert builder.MIN_EVENTS_FOR_GATE == 20
    assert builder.DRAFT_K == 4
    assert builder.DRAFT_ROUNDING == 3
    assert builder.CANDIDATE_DRAWS == 20
    assert builder.DRAW_STREAM_BASE == 9100
    assert builder.SPLIT_COLUMN == "household_id"
    assert builder.SPLIT_FRACTION == 0.5  # closes mutation M7
    assert builder.AGGREGATIONS == {}
    # family-B reference-period + circularity constants (fix A).
    assert builder.ANCHOR_FRAME_YEAR == 2024
    assert builder.FAMILY_B_DRAW_STREAM_BASE == 9200
    assert builder.CONVERSION_CATEGORIES == frozenset(
        {"disability_conversion"}
    )
    art = _artifact()
    assert art["internal_noise_floor"]["t_max"] == builder.T_MAX
    assert (
        art["internal_noise_floor"]["min_events_for_gate"]
        == builder.MIN_EVENTS_FOR_GATE
    )
    assert art["internal_noise_floor"]["split_unit"] == "household"
    assert (
        art["internal_noise_floor"]["split_fraction"] == builder.SPLIT_FRACTION
    )
    assert str(builder.DRAW_STREAM_BASE) in builder.CANDIDATE_DRAW_STREAM


def test_load_bearing_module_knobs_are_pinned_against_the_artifact():
    """Mutating a band edge, the profile reference band, or the
    spouse-present codes changes the moment set -- caught WITHOUT the
    certified-frame reproduction."""
    builder = _builder()
    from populace_dynamics.data import deployment_frame as dfm

    knobs = _artifact()["knobs"]
    assert [list(b) for b in dfm.EARN_BANDS] == knobs["earn_bands"]
    assert [list(b) for b in dfm.DISPERSION_BANDS] == knobs["dispersion_bands"]
    assert [list(b) for b in dfm.ADULT_BANDS] == knobs["adult_bands"]
    assert list(dfm.PROFILE_REF_BAND) == knobs["profile_ref_band"]
    assert list(dfm.SPOUSE_PRESENT_CODES) == knobs["spouse_present_codes"]
    assert knobs["draw_stream_base"] == builder.DRAW_STREAM_BASE


def test_required_source_columns_and_floors_are_pinned():
    """fix C / finding 4 (closes M14): the guard's column SET and its support
    FLOORS are pinned against the module, so weakening a floor against a
    future zeroed frame is caught always-runnable (it was invisible
    everywhere). Both earnings source columns are present."""
    from populace_dynamics.data import deployment_frame as dfm

    pinned = _artifact()["knobs"]["required_source_columns"]
    assert pinned == dict(dfm.REQUIRED_SOURCE_COLUMNS)
    assert "self_employment_income_before_lsr" in pinned
    assert "employment_income_before_lsr" in pinned
    # a weakened floor (M14: 0.30 -> 0.01) would break this pin.
    assert pinned["employment_income_before_lsr"] == 0.30
    assert pinned["self_employment_income_before_lsr"] == 0.03


def test_heavy_tail_boundary_bootstrap_recomputes():
    """fix G / finding 7: the boundary-fragility bootstrap re-derives from the
    committed floor values + the pinned seed (deterministic), so the disclosed
    P(flip) numbers are bound, not a pasted table."""
    builder = _builder()
    art = _artifact()
    floor = art[FLOOR_KEY]
    tolerances = {
        key: round(
            block["mean"] + builder.DRAFT_K * block["sd"],
            builder.DRAFT_ROUNDING,
        )
        for key, block in floor.items()
    }
    gated = set(art["gate_partition"]["gate_eligible"])
    reasons = {k: v["report_reason"] for k, v in art["cell_stability"].items()}
    rebuilt = builder.heavy_tail_boundary_bootstrap(
        floor, tolerances, gated, reasons
    )
    committed = art["heavy_tail_boundary_bootstrap"]
    assert rebuilt["demote_reentry_prob"] == committed["demote_reentry_prob"]
    assert rebuilt["gated_flipout_prob"] == committed["gated_flipout_prob"]
    # exactly the 5 heavy-tail demotes get a re-entry probability.
    demotes = {
        k
        for k, v in art["cell_stability"].items()
        if v["report_reason"] == "floor_max_exceeds_tolerance"
    }
    assert set(committed["demote_reentry_prob"]) == demotes
    assert len(demotes) == 5


def test_holdout_sha256_recomputes_always_runnable_from_committed_universe():
    """fix H / finding 10ii (closes A6): the committed household-id universe +
    the split rule recompute each gate seed's holdout sha256 with NO h5, so a
    corrupted committed sha (previously invisible everywhere incl. the
    certified repro) is caught at an always-runnable tier."""
    import hashlib

    import pandas as pd

    from populace_dynamics.harness import panel as hpanel

    art = _artifact()
    hold = art["holdout_ids"]
    universe_csv = hold["household_id_universe_csv"]
    universe = [int(x) for x in universe_csv.split(",")]
    assert len(universe) == hold["n_households_universe"]
    assert universe == sorted(universe)
    # the committed universe sha256 is self-consistent (hashes the CSV string).
    uni_sha = hashlib.sha256(universe_csv.encode()).hexdigest()
    assert uni_sha == hold["household_id_universe_sha256"]
    # reconstruct each gate seed's holdout the way the builder does and
    # recompute the sha256 -- no certified frame needed.
    frame = pd.DataFrame({"household_id": universe})
    for entry in hold["per_seed"]:
        side_a, _ = hpanel.split_panel_by_person(
            frame,
            "household_id",
            fraction=hold["fraction"],
            seed=entry["seed"],
        )
        ids = sorted(int(x) for x in side_a["household_id"].unique())
        assert len(ids) == entry["n_holdout_households"], entry["seed"]
        sha = hashlib.sha256(
            ",".join(str(i) for i in ids).encode()
        ).hexdigest()
        assert sha == entry["holdout_household_id_sha256"], entry["seed"]


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
# Family B: vintage-tolerance rule bound + recomputed through the builder
# --------------------------------------------------------------------------
def test_family_b_vintage_rule_is_bound_and_perturbs():
    builder = _builder()
    years = list(range(2013, 2023))
    # a pure trend, priced at Delta=0, has ~0 detrended sd -> tol ~ measurement.
    trend = [10.0 + 0.5 * (y - 2013) for y in years]
    tol0 = builder._vintage_tolerance(years, trend, 0)
    assert tol0["detrended_residual_sd_pp"] == pytest.approx(0.0, abs=1e-9)
    assert tol0["tolerance_pp"] == pytest.approx(builder.MEASUREMENT_PP)
    assert tol0["trend_pp_per_year"] == pytest.approx(0.5)
    assert tol0["trend_component_pp"] == pytest.approx(0.0)
    # the REFERENCE-PERIOD term prices the trend drift over Delta years (fix A):
    # a Delta=2 gap adds |trend|*2 = 1.0 pp to the tolerance.
    tol2 = builder._vintage_tolerance(years, trend, 2)
    assert tol2["reference_period_delta_years"] == 2
    assert tol2["trend_component_pp"] == pytest.approx(1.0)
    assert tol2["tolerance_pp"] == pytest.approx(
        round(builder.MEASUREMENT_PP + 1.0, 2)
    )
    assert tol2["tolerance_pp"] > tol0["tolerance_pp"]
    # adding noise raises the residual sd and the tolerance.
    noisy = [v + (1.0 if i % 2 else -1.0) for i, v in enumerate(trend)]
    tol_noisy = builder._vintage_tolerance(years, noisy, 0)
    assert tol_noisy["tolerance_pp"] > tol0["tolerance_pp"]


def test_family_b_delta_years_pinned_to_anchor_vintages():
    """fix A / finding 2: Delta = ANCHOR_FRAME_YEAR - vintage. Claim-age
    (2022 award flow) -> 2; DI (December-2023 stock) -> 1 -- pinned via the
    recorded per-cell deltas so a vintage/frame-year change is caught."""
    art = _artifact()
    fb = art["family_b"]
    for v in fb["claim_age"].values():
        assert v["reference_period_delta_years"] == 2
        assert v["anchor_year"] == 2022
    for v in fb["di_prevalence"].values():
        assert v["reference_period_delta_years"] == 1
        assert v["anchor_year"] == 2023
    assert fb["knobs"]["anchor_frame_year"] == 2024


def test_family_b_claim_age_recomputes_from_staged_file():
    builder = _builder()
    art = _artifact()
    rebuilt = builder.claim_age_anchor()
    for key, v in art["family_b"]["claim_age"].items():
        assert rebuilt[key]["anchor_pp"] == pytest.approx(v["anchor_pp"]), key
        assert rebuilt[key]["tolerance_pp"] == pytest.approx(
            v["tolerance_pp"]
        ), key
        assert rebuilt[key]["gate_eligible"] == v["gate_eligible"], key


def test_family_b_di_prevalence_recomputes_from_staged_file():
    builder = _builder()
    art = _artifact()
    rebuilt = builder.di_prevalence_anchor()
    for key, v in art["family_b"]["di_prevalence"].items():
        assert rebuilt[key]["anchor_pp"] == pytest.approx(v["anchor_pp"]), key
        assert rebuilt[key]["tolerance_pp"] == pytest.approx(
            v["tolerance_pp"]
        ), key


# --------------------------------------------------------------------------
# Family C: committed orderings bound through the builder
# --------------------------------------------------------------------------
def test_family_c_block_recomputes_all_four_order_fields():
    """fix F / finding 8a: ALL FOUR order fields (c1/c2 before + after)
    recompute through the builder, bound to the committed anchor artifacts."""
    builder = _builder()
    art = _artifact()
    rebuilt = builder.family_c_block()
    for cid in ("c1", "c2"):
        for field in (
            "psid_frame_order",
            "required_representative_order",
            "swap_pair",
            "anchor_values",
        ):
            assert rebuilt["fingerprints"][cid][field] == (
                art["family_c"]["fingerprints"][cid][field]
            ), (cid, field)


def test_family_c_required_orders_derive_from_committed_anchors():
    """fix F / finding 8a: the required after-orders are DERIVED by ranking the
    committed Mermin payroll-pct / Smith solvency deltas descending -- the
    anchor itself is machine-checked (closes M10: a hand-swapped builder order
    no longer exists to swap)."""
    builder = _builder()
    art = _artifact()
    c1 = art["family_c"]["fingerprints"]["c1"]
    c2 = art["family_c"]["fingerprints"]["c2"]
    # C1: ranking Mermin payroll-pct desc reproduces the required order.
    assert (
        builder._rank_desc(c1["anchor_values"], list(c1["anchor_values"]))
        == c1["required_representative_order"]
    )
    assert c1["required_representative_order"] == [
        "price_indexing",
        "progressive_price_indexing",
        "nra_raised_to_70",
        "reduced_cola",
    ]
    # C2: ranking Smith deltas desc reproduces the required order.
    assert (
        builder._rank_desc(c2["anchor_values"], list(c2["anchor_values"]))
        == c2["required_representative_order"]
    )
    assert c2["required_representative_order"] == [
        "elimination",
        "payroll_plus_2pp",
        "payroll_plus_1pp",
        "cap_150k",
    ]


def test_family_c_order_derivation_perturbs():
    """Perturbing an anchor value changes the derived order (the ranking is
    genuinely bound to the anchor, not coincidental)."""
    builder = _builder()
    anchor = {"a": 0.68, "b": -0.14, "c": -0.5, "d": -1.12}
    provs = list(anchor)
    assert builder._rank_desc(anchor, provs) == ["a", "b", "c", "d"]
    bumped = dict(anchor, b=-0.9)  # b now ranks below c.
    assert builder._rank_desc(bumped, provs) == ["a", "c", "b", "d"]


def test_family_c_candidate_procedure_is_pinned():
    """fix E / finding 8b: the reversal-time C-candidate procedure (per-
    fingerprint statistic, committed encodings, engine pins, pass rule)."""
    art = _artifact()
    proc = art["family_c"]["candidate_procedure"]
    assert "outlay-side" in proc["c1_statistic"]
    assert "exhaustion-delay" in proc["c2_statistic"]
    assert "#115" in proc["c1_statistic"] or "F4" in proc["c1_statistic"]
    assert "#117" in proc["c2_statistic"] or "F2" in proc["c2_statistic"]
    assert "engine_pins" in proc
    assert "Kendall tau 1.0" in proc["pass_rule"]


# --------------------------------------------------------------------------
# ALWAYS-RUNNABLE data-free structural invariants (label-swap catches)
# --------------------------------------------------------------------------
def test_participation_falls_from_prime_age_to_retirement():
    """Labor participation is a hump: prime-age (35-44) participation strictly
    exceeds retirement-age (62-69) for each sex. A self-consistent swap of a
    prime cell with its 62-69 counterpart inverts it -- caught with no h5."""
    rm = _artifact()["reference_moments"]
    for sex in ("male", "female"):
        prime = rm[f"earnings_participation.35-44|{sex}"]["rate"]
        retire = rm[f"earnings_participation.62-69|{sex}"]["rate"]
        assert prime > retire, (sex, prime, retire)
        assert prime > 1.5 * retire, (sex, prime, retire)


def test_prime_age_participation_gender_gap():
    """Prime-age male participation exceeds female (a robust CPS fact); a
    female<->male swap of a gated participation cell inverts it."""
    rm = _artifact()["reference_moments"]
    for band in ("25-34", "35-44", "45-54"):
        m = rm[f"earnings_participation.{band}|male"]["rate"]
        f = rm[f"earnings_participation.{band}|female"]["rate"]
        assert m > f, (band, m, f)


def test_never_married_share_falls_with_age():
    """The never-married share falls from young-adult to mid-life for each
    sex; a self-consistent age swap of a marital cell inverts it."""
    rm = _artifact()["reference_moments"]
    for sex in ("male", "female"):
        young = rm[f"marital_share.never_married.25-34|{sex}"]["rate"]
        mid = rm[f"marital_share.never_married.45-54|{sex}"]["rate"]
        assert young > mid, (sex, young, mid)


def test_coresident_spouse_rises_into_midlife():
    """Living-with-a-spouse rises from 25-34 into 45-54 for each sex; a
    self-consistent age swap inverts it."""
    rm = _artifact()["reference_moments"]
    for sex in ("male", "female"):
        young = rm[f"coresident_spouse.25-34|{sex}"]["rate"]
        mid = rm[f"coresident_spouse.45-54|{sex}"]["rate"]
        assert mid > young, (sex, young, mid)


def test_hh_size_shares_are_a_person_level_distribution():
    """The five person-level household-size shares are positive and sum to 1
    (a real cover of the population); a corruption breaks the sum."""
    rm = _artifact()["reference_moments"]
    shares = [
        rm[f"hh_size_share.{c}"]["rate"] for c in ("1", "2", "3", "4", "5plus")
    ]
    assert all(s > 0 for s in shares)
    assert sum(shares) == pytest.approx(1.0, abs=1e-6)


def test_earnings_dispersion_ratios_exceed_one():
    """p90/p50 and p50/p10 earnings ratios strictly exceed 1 (quantiles
    increase); a corrupted dispersion cell violates it."""
    rm = _artifact()["reference_moments"]
    for key, cell in rm.items():
        if key.startswith("earnings_p90p50.") or key.startswith(
            "earnings_p50p10."
        ):
            assert cell["rate"] > 1.0, key


def test_family_c_reversal_directions_are_consistent():
    """The committed reversal direction is internally consistent: in each
    fingerprint's PSID order the 'wrong' member leads, and the required
    representative order puts the anchor-correct member first."""
    fc = _artifact()["family_c"]["fingerprints"]
    # C1: NRA leads PPI before; PPI leads NRA after.
    c1b = fc["c1"]["psid_frame_order"]
    c1a = fc["c1"]["required_representative_order"]
    assert c1b.index("nra_raised_to_70") < c1b.index(
        "progressive_price_indexing"
    )
    assert c1a.index("progressive_price_indexing") < c1a.index(
        "nra_raised_to_70"
    )
    # C2: +2pp leads elimination before; elimination leads +2pp after.
    c2b = fc["c2"]["psid_frame_order"]
    c2a = fc["c2"]["required_representative_order"]
    assert c2b.index("payroll_plus_2pp") < c2b.index("elimination")
    assert c2a.index("elimination") < c2a.index("payroll_plus_2pp")
