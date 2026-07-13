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


# ==========================================================================
# gate_w1 (the W1 representative-frame transport gate, M5) LOCKED-HOT
# bindings. gate_w1 is ADDED FRESH as a locked top-level gate by the W1 lock
# flip (like gate_m4; unlike gate_2b/2c it had no pre-existing unlocked stub).
# The flip READS the ratified, FROZEN floor runs/gate_w1_floors_v1.json and
# writes NO number of its own: family-A tolerances == round(floor mean + 4*sd,
# 3) capped at ln(1.5); family-B tolerances == the frozen anchor tolerance_pp
# (reference-period rule, Delta pinned {2, 1}); family-C required orderings ==
# rank(committed anchor values) descending. These bindings are LOCKED-HOT (the
# 2a lesson: they run unconditionally, committed files only), touch NO locked
# sibling block, and a D1-style SUBSET master-compare proves the flip leaves
# gate_1 / gate_2 / gate_3 (and gate_m4 if its sibling flip has landed)
# byte-identical to origin/master, gate_w1 the sole new key.
# ==========================================================================
import subprocess  # noqa: E402

import yaml  # noqa: E402

GATES = ROOT / "gates.yaml"


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _gate_w1_block() -> dict:
    return yaml.safe_load(GATES.read_text())["gates"]["gate_w1"]


def _gate_w1() -> dict:
    """gate_w1.thresholds. Post-lock the bindings run unconditionally."""
    return _gate_w1_block()["thresholds"]


def _w1_family_a_tolerances() -> dict:
    tol: dict = {}
    for view in _gate_w1()["family_a"]["views"].values():
        tol.update(view["tolerances"])
    return tol


def _w1_derive(cell: str, k: float, rounding: int) -> float:
    stats = _artifact()[FLOOR_KEY][cell]
    return round(stats["mean"] + k * stats["sd"], rounding)


def _master_gates_mapping():
    """origin/master gates mapping, self-fetching the ref if needed."""
    for attempt in range(2):
        try:
            text = subprocess.run(
                ["git", "show", "origin/master:gates.yaml"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            return yaml.safe_load(text)["gates"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            if attempt == 0:
                subprocess.run(
                    ["git", "fetch", "origin", "master"],
                    cwd=ROOT,
                    capture_output=True,
                )
                continue
            return None
    return None


def test_gate_w1_locked_added_as_transport_top_level_gate():
    """gate_w1 is locked, kind transport, and a NEW top-level gate (it did not
    pre-exist as an unlocked stub the way gate_2b/2c did)."""
    b = _gate_w1_block()
    assert b["locked"] is True
    assert b["status"] == "locked"
    assert b["kind"] == "transport"
    assert b["id"] == "w1_representative_frame_transport"
    assert b["holdout_basis"] == _artifact()["holdout_basis"]
    assert b["holdout_basis"] == [
        "populace_us_2024_certified",
        "ssa_supplement_6b",
        "di_asr_2023",
        "mermin_smith_committed_orderings",
    ]
    g = _gate_w1()
    assert g["locked"] is True
    assert g["status"] == "locked"
    assert g["tranche_id"] == "w1_representative_frame_transport"
    assert g["kind"] == "transport"


def test_gate_w1_locked_with_ceremony_record():
    """The full lock ceremony is recorded in-contract: the FOUR ceremony
    comment ids, the ratifying squash merge d9d0a68 (PR #154), the #158
    tier-manifest repair, the delegated-authority note, and no_self_rescue."""
    record = json.dumps(_gate_w1()["ceremony_record"])
    for token in (
        "4950451145",  # round-1 adversarial referee AMEND BEFORE LOCK
        "4950611796",  # fixes A-H
        "4950670492",  # verification AMEND text-only
        "4950673980",  # body amendment executed / ratifying
        "PR #154",  # ratifying PR
        "d9d0a68",  # ratifying squash merge commit
        "#158",  # the maintainer-side tier-manifest repair
        "199",  # the CI-authoritative unit count #158 restored
        "no_self_rescue",
    ):
        assert token in record, token
    assert "delegated" in record.lower()
    assert "text-only" in record.lower()
    # the four verification flip-time notes are enumerated as implemented.
    notes = _gate_w1()["ceremony_record"]["flip_notes_implemented"]
    assert set(notes) == {
        "note_1_gate_w1_stub_from_151",
        "note_2_anchor_bundle_conventions",
        "note_3_manifest_transition",
        "note_4_binding_scope_record",
    }
    for key, text in notes.items():
        assert text.strip(), key


def test_gate_w1_floor_run_is_the_ratified_frozen_anchor():
    """The floor the locked thresholds cite is the ratified, frozen artifact
    that reads no gate."""
    g = _gate_w1()
    assert g["floor_run"] == "runs/gate_w1_floors_v1.json"
    art = _artifact()
    assert art["schema_version"] == "gate_w1_floors.v1"
    assert art["ceremony"]["gates_yaml_untouched"] is True


def test_gate_w1_family_a_tolerances_bind_to_floor():
    """Every locked family-A tolerance == round(100-seed floor mean + k*sd,
    rounding) on the frozen floor, keyed on the floor cell itself."""
    g = _gate_w1()
    for view_name, view in g["family_a"]["views"].items():
        rules = view["derivations"]["rules"]
        tolerances = view["tolerances"]
        assert set(rules) == set(tolerances), view_name
        assert (
            view["derivations"]["floor_run"] == "runs/gate_w1_floors_v1.json"
        )
        assert view["derivations"]["floor_key"] == FLOOR_KEY
        for cell, rule in rules.items():
            assert rule["key"] == cell, f"{view_name}.{cell}"
            assert rule["k"] == 4
            assert rule["rounding"] == 3
            derived = _w1_derive(cell, rule["k"], rule["rounding"])
            assert derived == pytest.approx(tolerances[cell]), (
                f"{view_name}.{cell}: derived {derived} != tolerance "
                f"{tolerances[cell]}"
            )


AMENDMENT2_DEMOTED_A = {
    "earnings_participation.18-24|female",
    "earnings_participation.18-24|male",
    "marital_share.married.65+|female",
    "marital_share.married.65+|male",
    "coresident_spouse.65+|female",
    "coresident_spouse.65+|male",
}


def test_gate_w1_family_a_power_cap_and_partition_47_58():
    """Every gated family-A tolerance <= T_max = ln(1.5); post-amendment-2 the
    contract's gated set == the FROZEN floor's derived partition (53 / 52)
    minus the 6 amendment-2 demotes (47 / 58); the floor's 52 keep the floor's
    own machine reasons, the 6 demotes carry the amendment's per-cell reasons,
    and their tolerances are RETAINED verbatim (zero threshold movement)."""
    g = _gate_w1()
    fa = g["family_a"]
    assert fa["power_cap"]["t_max"] == "ln(1.5)"
    assert fa["power_cap"]["aggregations"] == {}
    art = _artifact()
    t_max = art["internal_noise_floor"]["t_max"]
    assert t_max == pytest.approx(math.log(1.5))
    tolerances = _w1_family_a_tolerances()
    for cell, tol in tolerances.items():
        assert tol <= t_max, f"{cell} tolerance {tol} exceeds T_max"
    gated = set(tolerances)
    report_only = set(fa["report_only"])
    partition = art["gate_partition"]
    # amendment 2: the frozen floor still lists all 53 as gate_eligible; the
    # CONTRACT surface is the floor partition minus the 6 demoted cells.
    assert gated == set(partition["gate_eligible"]) - AMENDMENT2_DEMOTED_A
    assert report_only == set(partition["report_only"]) | AMENDMENT2_DEMOTED_A
    assert len(gated) == 47
    assert len(report_only) == 58
    assert gated.isdisjoint(report_only)
    assert gated | report_only == set(art["reference_moments"])
    # the floor's 52 report-only cells keep the floor's own machine reasons.
    from collections import Counter

    reasons = Counter(
        art["cell_stability"][c]["report_reason"]
        for c in report_only - AMENDMENT2_DEMOTED_A
    )
    assert reasons == {
        "tolerance_above_t_max": 44,
        "floor_max_exceeds_tolerance": 5,
        "below_20_events": 2,
        "undefined_on_some_seed": 1,
    }
    # the 6 demotes carry the amendment's per-cell machine reasons.
    assert fa["report_reasons"] == {
        "earnings_participation.18-24|female": (
            "population_concept_delta_head_spouse_universe"
        ),
        "earnings_participation.18-24|male": (
            "population_concept_delta_head_spouse_universe"
        ),
        "marital_share.married.65+|female": (
            "cohort_vintage_hazard_frame_mismatch"
        ),
        "marital_share.married.65+|male": (
            "cohort_vintage_hazard_frame_mismatch"
        ),
        "coresident_spouse.65+|male": "cohort_vintage_hazard_frame_mismatch",
        "coresident_spouse.65+|female": (
            "scored_duplicate_of_demoted_married_quantity"
        ),
    }
    # zero threshold movement: the 6 tolerances are retained verbatim.
    retained = fa["retained_tolerances"]
    assert set(retained) == AMENDMENT2_DEMOTED_A
    assert {c: r["tolerance"] for c, r in retained.items()} == {
        "earnings_participation.18-24|female": 0.211,
        "earnings_participation.18-24|male": 0.221,
        "marital_share.married.65+|female": 0.163,
        "marital_share.married.65+|male": 0.084,
        "coresident_spouse.65+|female": 0.168,
        "coresident_spouse.65+|male": 0.094,
    }


def test_gate_w1_family_a_oc_recomputes_from_tolerances_and_sigmas():
    """Post-amendment-2 the faithful-candidate OC (p_seed 0.9344 / p_gate
    0.9623) recomputes from the 47 gated family-A tolerances and the frozen
    floor sigmas on the draw-noise-free half-normal basis, independent of the
    stored value. The frozen artifact keeps the 53-cell pre-amendment OC
    (0.922 / 0.9481), which also still recomputes from the full floor
    partition (the artifact is untouched by the flip)."""
    art = _artifact()
    per = art["faithful_candidate_oc"]["per_cell"]
    tolerances = _w1_family_a_tolerances()
    assert len(tolerances) == 47
    p_seed = 1.0
    for cell, tol in tolerances.items():
        sigma = per[cell]["realized_sigma"]
        p_seed *= 2.0 * _normal_cdf(tol / sigma) - 1.0
    p_gate = p_seed**5 + 5 * p_seed**4 * (1.0 - p_seed)
    assert round(p_seed, 4) == 0.9344
    assert round(p_gate, 4) == 0.9623
    oc = _gate_w1()["family_a"]["faithful_candidate_oc"]
    assert oc["p_seed_pass"] == 0.9344
    assert oc["p_gate_pass_4_of_5"] == 0.9623
    assert oc["n_gated_cells"] == 47
    # the FROZEN artifact still carries the 53-cell pre-amendment OC, and it
    # still recomputes from the floor's own full partition.
    p_seed_53 = 1.0
    retained = _gate_w1()["family_a"]["retained_tolerances"]
    for cell, tol in tolerances.items():
        sigma = per[cell]["realized_sigma"]
        p_seed_53 *= 2.0 * _normal_cdf(tol / sigma) - 1.0
    for cell, r in retained.items():
        sigma = per[cell]["realized_sigma"]
        p_seed_53 *= 2.0 * _normal_cdf(r["tolerance"] / sigma) - 1.0
    p_gate_53 = p_seed_53**5 + 5 * p_seed_53**4 * (1.0 - p_seed_53)
    assert round(p_seed_53, 4) == 0.922
    assert round(p_gate_53, 4) == 0.9481
    assert art["faithful_candidate_oc"]["p_seed_pass"] == 0.922
    assert art["faithful_candidate_oc"]["p_gate_pass_4_of_5"] == 0.9481


def test_gate_w1_protocol_is_the_ratified_k20_estimator_stream_9100():
    """The family-A protocol is the ratified mean-over-K=20 estimator on stream
    9100; the fresh-run schema is [20, 47, 5] post-amendment-2 with BOTH
    dispersion fields; the regenerated-surface rule + identity-candidate
    prohibition are pinned."""
    proto = _gate_w1()["protocol"]
    assert proto["candidate_draws"] == 20
    assert proto["draw_stream_base"] == 9100
    assert "9100" in proto["candidate_draw_stream"]
    assert "K=20" in proto["estimator"]
    assert "NOT the mean of the per-draw" in proto["estimator"]
    schema = proto["fresh_run_artifact_schema"]
    assert schema["per_draw_per_cell_rates"]["shape"] == [20, 47, 5]
    assert schema["undefined_draw_rule"]["pre_specified"] is True
    disp = schema["per_draw_dispersion_disclosure"]
    assert disp["report_only"] is True
    assert disp["fields"] == [
        "per_cell_across_draw_sd",
        "max_per_draw_abs_ln_per_cell",
    ]
    reg = schema["regenerated_surface"]
    assert reg["identity_candidate_is_non_conformant"] is True
    assert "NON-CONFORMANT" in reg["rule"]
    assert set(reg["per_family"]) == {
        "earnings_participation|profile|p90p50|p50p10",
        "marital_share|coresident_spouse",
        "hh_size_share",
    }


def test_gate_w1_identity_candidate_prohibited():
    """The identity candidate (copying scored columns) scores 0 with zero
    across-draw dispersion and is NON-CONFORMANT, caught by the
    regenerated-surface rule + the max_per_draw_abs_ln_per_cell == 0
    disclosure."""
    idc = _gate_w1()["family_a"]["degenerate_candidates"]["identity_candidate"]
    art_idc = _artifact()["degenerate_candidates"]["identity_candidate"]
    assert idc["conformance"] == "NON-CONFORMANT" == art_idc["conformance"]
    assert idc["across_draw_sd"] == 0.0
    assert "regenerated_surface" in idc["caught_by"]
    assert "max_per_draw_abs_ln_per_cell == 0" in idc["caught_by"]
    # the train-copy (a real, procedurally-barred attack) is distinguished.
    tc = _gate_w1()["family_a"]["degenerate_candidates"]["train_copy"]
    assert tc["max_score_over_tolerance"] == pytest.approx(
        art["training_copy_check"]["max_score_over_tolerance"]
        if (art := _artifact())
        else 0.8089
    )


def test_gate_w1_family_a_heavy_tail_bootstrap_carried():
    """fix G / finding 7: the boundary-fragility bootstrap (B=5000, seed 91000)
    rides in the lock with the 5 demote re-entry probabilities and the
    seed-count-dependence note, matching the frozen floor exactly."""
    htb = _gate_w1()["family_a"]["heavy_tail_boundary_bootstrap"]
    art = _artifact()["heavy_tail_boundary_bootstrap"]
    assert htb["n_bootstrap"] == 5000
    assert htb["seed"] == 91000
    assert htb["demote_reentry_prob"] == art["demote_reentry_prob"]
    assert htb["gated_flipout_prob"] == art["gated_flipout_prob"]
    assert len(htb["demote_reentry_prob"]) == 5
    assert htb["seed_count_dependence"].strip()


def test_gate_w1_family_a_prime_is_report_only():
    """fix D / finding 6: the A' published-value family rides report-only with
    the Census HH-4 + AD-3 rows and the sha-pinned source; NOT gated."""
    fap = _gate_w1()["family_a"]["family_a_prime"]
    assert fap["status"] == "report_only"
    art = _artifact()["family_a_prime"]
    assert set(fap["household_size_person_level"]) == set(
        art["household_size_person_level"]
    )
    for cell, row in fap["household_size_person_level"].items():
        assert row["frame_rate"] == pytest.approx(
            art["household_size_person_level"][cell]["frame_rate"]
        )
    assert (
        fap["sources"]["household_size"]["file_sha256"]
        == art["sources"]["household_size"]["file_sha256"]
    )


def test_gate_w1_family_b_partition_0_25_after_amendment1():
    """Amendment 1 (2026-07-12-w1-family-b-di-bands) demoted ALL 10 family-B
    cells to report-only: family B gates NOTHING (0 gated), report-only grew
    15 -> 25, and every anchor value + tolerance is RETAINED verbatim under
    retained_anchors (zero threshold movement -- byte-equal to the FROZEN floor
    the lock cited, which still lists the 10 as gate_eligible). Machine reasons
    are per-cell: 8 DI bands concept_bridge_undefined_di_stock; 2 conversion
    cells that plus conversion_level_match_never_certified. This is the
    locked-hot re-binding of the 10/15 surface to the amended 0/25 partition,
    exactly as the gate-2 amendment flips updated theirs."""
    fb = _gate_w1()["family_b"]
    art_fb = _artifact()["family_b"]
    # family B gates NOTHING after the flip.
    assert fb["gated_cells"] == {}
    assert len(fb["gated_cells"]) == 0
    # report-only grew from 15 to 25 (the 15 originals + the 10 demoted).
    report_only = fb["report_only"]
    assert len(report_only) == 25
    # the frozen floor STILL lists the 10 as gate_eligible -- the amendment
    # demotes them in the CONTRACT, not the floor (frozen, untouched).
    frozen_gated = set(art_fb["gate_partition"]["gate_eligible"])
    assert len(frozen_gated) == 10
    # the demoted set is exactly those 10, keyed in report_reasons.
    reasons = fb["report_reasons"]
    assert set(reasons) == frozen_gated
    conv = [c for c in reasons if "disability_conversion" in c]
    di = [c for c in reasons if c.startswith("di_prevalence.")]
    assert len(conv) == 2
    assert len(di) == 8
    # all 10 demoted cells are now IN report_only (partition set-equality).
    assert frozen_gated <= set(report_only)
    # per-cell machine reasons (fix A): 8 bands one reason; 2 conversion two.
    for cell in di:
        assert reasons[cell] == ["concept_bridge_undefined_di_stock"], cell
    for cell in conv:
        assert reasons[cell] == [
            "concept_bridge_undefined_di_stock",
            "conversion_level_match_never_certified",
        ], cell

    # anchors + tolerances RETAINED byte-equal to the frozen floor (zero
    # threshold movement -- nothing deleted, published report-only).
    retained = fb["retained_anchors"]
    assert set(retained) == frozen_gated

    def frozen(cell):
        if cell.startswith("claim_age."):
            return art_fb["claim_age"][cell]
        return art_fb["di_prevalence"][cell]

    for cell, row in retained.items():
        src = frozen(cell)
        assert row["tolerance_pp"] == src["tolerance_pp"], cell
        assert row["anchor_pp"] == src["anchor_pp"], cell
        expected_delta = 2 if cell.startswith("claim_age.") else 1
        assert row["reference_period_delta_years"] == expected_delta, cell
    # knobs unchanged (anchors stay in-contract, report-only).
    assert fb["knobs"]["anchor_frame_year"] == 2024
    assert fb["knobs"]["claim_age_delta_years"] == 2
    assert fb["knobs"]["di_delta_years"] == 1


def test_gate_w1_family_b_candidate_protocol_stream_9200_and_no_di_column():
    """The family-B candidate protocol pins stream base 9200 (distinct from
    family A's 9100), the M4-simulated conversion object, the no-frame-DI-column
    rule (forbidding social_security_disability), and the SS-proxy laundering /
    conditioning-column rule."""
    proto = _gate_w1()["family_b"]["candidate_protocol"]
    assert proto["family_b_draw_stream_base"] == 9200
    assert proto["candidate_draws"] == 20
    assert "M4" in proto["simulated_object"]["claim_age.disability_conversion"]
    no_di = proto["no_frame_di_column_rule"]
    assert "social_security_disability" in no_di
    assert "M4" in no_di
    launder = proto["ss_proxy_laundering_rule"]
    assert "SS_VAL" in launder
    assert "SS_YN" in launder
    assert "enumerat" in proto["candidate_conditioning_columns_rule"].lower()


def test_gate_w1_family_c_orderings_derive_from_committed_anchors():
    """Post-amendment-2 family C GATES the C2 fingerprint only; C1 is
    report-only (fingerprint_reversal_not_realized) with its ordering fields
    RETAINED verbatim. Each required representative order == the committed
    anchor values ranked descending, and equals the frozen floor's committed
    order (no hand-written order to swap)."""
    fc = _gate_w1()["family_c"]
    assert fc["gate_partition"]["n_gate_eligible"] == 1
    assert fc["gate_partition"]["n_report_only"] == 1
    assert fc["gate_partition"]["gate_eligible"] == [
        "fingerprint.elimination_plus2pp"
    ]
    assert fc["report_only"] == ["fingerprint.ppi_nra"]
    assert fc["report_reasons"] == {
        "fingerprint.ppi_nra": "fingerprint_reversal_not_realized"
    }
    assert "amendment 2" in fc["demoted_disclosure_note"]
    assert "RETAINED" in fc["demoted_disclosure_note"]
    art_fc = _artifact()["family_c"]["fingerprints"]

    def rank_desc(values):
        order = list(values)
        return sorted(order, key=lambda k: (-values[k], order.index(k)))

    for cid in ("c1", "c2"):
        fp = fc["fingerprints"][cid]
        af = art_fc[cid]
        for field in (
            "psid_frame_order",
            "required_representative_order",
            "swap_pair",
            "anchor_values",
        ):
            assert fp[field] == af[field], (cid, field)
        assert (
            rank_desc(fp["anchor_values"])
            == fp["required_representative_order"]
        ), cid
    # C1 the Mermin payroll-pct order; C2 the Smith solvency-delta order.
    assert fc["fingerprints"]["c1"]["required_representative_order"] == [
        "price_indexing",
        "progressive_price_indexing",
        "nra_raised_to_70",
        "reduced_cola",
    ]
    assert fc["fingerprints"]["c2"]["required_representative_order"] == [
        "elimination",
        "payroll_plus_2pp",
        "payroll_plus_1pp",
        "cap_150k",
    ]
    proc = fc["candidate_procedure"]
    assert "engine_pins" in proc
    assert "Kendall tau 1.0" in proc["pass_rule"]
    # amendment 2: the operative conjunction is C2 alone; C1 publishes.
    assert "GATES the C2 fingerprint only" in proc["pass_rule"]
    assert "the C2 fingerprint reverses" in fc["check"]
    assert "BOTH fingerprints reverse" not in fc["check"]


def test_gate_w1_frame_pin_matches_the_certified_bundle_sha():
    """The deployment-frame pin (bundle sha256 c2065b64...) matches the frozen
    floor's certified pin; the binding-scope record rides in frame_pin."""
    b = _gate_w1_block()
    g = _gate_w1()
    art = _artifact()
    frame_sha = art["estimand"]["deployment_frame"]["artifact_sha256"]
    assert b["deployment_frame"]["artifact_sha256"] == frame_sha
    assert b["deployment_frame"]["bundle"] == "us-4.18.8"
    assert b["deployment_frame"]["pe_us_version"] == "1.752.2"
    assert (
        g["frame_pin"]["certified_artifact_sha256"]
        == art["revision_pins"]["certified_artifact_sha256"]
        == frame_sha
    )
    assert "binding_scope" in g["frame_pin"]
    assert "artifact-side" in g["frame_pin"]["binding_scope"]


def test_gate_w1_covers_and_certification_scope():
    """Post-amendment-2: the covers text names the 48 gated surface
    (description-claims-exactly) — family A 47 gated / 58 report-only, family
    C 1 gated / 1 report-only — with family B's inner figures still CONTRACT
    text (flip-note N1: 0 gated / 25 report-only), and certification_scope
    certifies 48 cells / C2-only reversal / no SSA family-B margin, still not
    the dynamics (which stay gate-1/2a/2b/2c/M4-certified)."""
    b = _gate_w1_block()
    covers = b["covers"]
    assert "TRANSPORT" in covers
    assert "47 gated / 58 report-only" in covers
    # the roll-up flipped 55 -> 48; the pre-amendment roll-ups are gone.
    assert "48 gated / 84 report-only" in covers
    assert "55 gated" not in covers
    assert "65 gated" not in covers
    assert "53 gated" not in covers
    assert "1 gated / 1 report-only" in covers
    assert "C1 demoted by amendment 2" in covers
    # flip-note N1: section 7.5's inner family-B figures are now contract text.
    assert "0 gated (family B gates nothing)" in covers
    assert "25 report-only" in covers
    assert "Does NOT re-certify the dynamics" in covers
    csc = _gate_w1()["certification_scope"]
    assert csc["tranche"] == "w1_representative_frame_transport"
    assert "48 gated cells" in csc["certifies"]
    assert "55 gated cells" not in csc["certifies"]
    assert "65 gated cells" not in csc["certifies"]
    assert "reverses the C2 fingerprint" in csc["certifies"]
    assert "report-only pending" in csc["certifies"]
    assert "family B certifies NO" in csc["certifies"]
    dns = " ".join(csc["does_not_support"])
    assert "DYNAMICS" in dns
    assert "transport, not re-estimation" in dns


def test_gate_w1_governance_inherits_no_self_rescue():
    """The lock carries the inherited no_self_rescue + the promoted standing
    description-claims-exactly rule + a weight_definition."""
    gov = _gate_w1()["governance"]
    amend = gov["amendment_rules"]
    assert amend["inherits"] == "gate_1"
    assert "committed run verdict changes" in amend["no_self_rescue"]
    assert "runs registered after its ratification" in amend["no_self_rescue"]
    assert "EXACTLY" in amend["description_claims_exactly_the_scored_surface"]
    assert "issue #42" in gov["registration"]
    assert "unweighted" in gov["weight_definition"].lower()


def test_gate_w1_history_entry_records_the_lock():
    """gate_w1.history carries the 2026-07-12-w1-gate-lock entry with the four
    ceremony comment ids, the ratifying merge (PR #154 / d9d0a68),
    no_self_rescue, and ZERO threshold movement."""
    hist = _gate_w1_block()["history"]
    entry = next(e for e in hist if e["id"] == "2026-07-12-w1-gate-lock")
    assert entry["flipped_live"] == "this pull request"
    rr = entry["referee_round"]
    assert "4950451145" in rr["review"]
    assert "4950611796" in rr["fixes"]
    assert "4950670492" in rr["verification"]
    assert "4950673980" in rr["verification"]
    assert "d9d0a68" in entry["ratified"]
    assert "PR #154" in entry["ratified"]
    assert "no_self_rescue" in entry["ratified"]
    assert "ZERO threshold movement" in entry["content"]


def test_gate_w1_amendment1_flip_history_and_ceremony_record():
    """gate_w1.history records amendment 1 (2026-07-12-w1-family-b-di-bands)
    with the full ceremony (round-1 AMEND 4951701300 / fixes 9bc4e32 + comment
    4951835634 / verification RATIFY AS-IS 4951903239 / ratifying merge PR #164
    bb2e0de), the ZERO-threshold-movement all-10-demoted content, and the four
    flip-time notes N1-N4 as implemented -- the amendment ceremony record."""
    hist = _gate_w1_block()["history"]
    ids = [e["id"] for e in hist]
    assert "2026-07-12-w1-gate-lock" in ids  # the lock entry survives
    entry = next(
        e for e in hist if e["id"] == "2026-07-12-w1-family-b-di-bands"
    )
    assert entry["flipped_live"] == "this pull request"
    rr = entry["referee_round"]
    assert "4951701300" in rr["review"]  # round-1 AMEND THE AMENDMENT
    assert "9bc4e32" in rr["fixes"] and "4951835634" in rr["fixes"]
    assert "4951903239" in rr["verification"]  # verification RATIFY AS-IS
    assert "PR #164" in entry["ratified"]
    assert "bb2e0de" in entry["ratified"]  # ratifying merge commit
    content = entry["content"]
    assert "ZERO THRESHOLD MOVEMENT" in content
    assert "ALL 10 family-B gated cells" in content
    assert "concept_bridge_undefined_di_stock" in content
    assert "conversion_level_match_never_certified" in content
    assert "65 -> 55 gated, 67 -> 77 report-only" in content
    assert "0.9481" in content
    # the four flip-time notes N1-N4 are recorded as implemented.
    notes = entry["flip_notes_implemented"]
    assert set(notes) == {
        "note_1_section7_5_inner_figures_bound",
        "note_2_historical_descriptor_fixed",
        "note_3_env_only_test_restated",
        "note_4_5p23_5p78_attribution",
    }
    for key, text in notes.items():
        assert text.strip(), key


def test_gate_w1_flip_leaves_locked_siblings_byte_identical():
    """D1 SUBSET master-compare (the form that held at #122 / #130 / M4, no
    red-master window): the flip ADDS gate_w1 and changes nothing else, so
    gate_1 / gate_2 (2a + gate_2b + gate_2c) / gate_3 (and gate_m4 if the
    sibling flip has landed) stay deep-equal to origin/master and gate_w1 is
    the SOLE new key. These asserts hold in every state (before and after the
    flip merges), so they are unconditional."""
    master = _master_gates_mapping()
    if master is None:
        pytest.skip("origin/master gates.yaml unreachable")
    current = yaml.safe_load(GATES.read_text())["gates"]
    added = set(current) - set(master)
    removed = set(master) - set(current)
    # gate_w1 may already be on master once the flip merges; then the set diff
    # is empty. Before merge it is exactly {gate_w1}. A concurrent gate_m4
    # sibling flip is preserved (present in BOTH once rebased), never dropped.
    assert added in (set(), {"gate_w1"}), added
    assert removed == set(), removed
    for key in master:
        if key == "gate_w1":
            continue
        assert current[key] == master[key], f"{key} changed vs master!"
    # gate_2's locked tranche-2a thresholds + gate_2b + gate_2c untouched.
    assert current["gate_2"]["thresholds"]["locked"] is True
    assert current["gate_2"]["gate_2b"]["locked"] is True
    assert current["gate_2"]["gate_2c"]["locked"] is True
