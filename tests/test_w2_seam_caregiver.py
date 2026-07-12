"""Tests for the W2 interim seam diagnostic (runs/w2_seam_caregiver_v1.json).

The artifact is the ADR-0001 W2 interim product end to end: caregiver-credit
(Biden) benefit deltas from the committed anchor replication, transported by an
interim cell-mean mapping onto the certified populace default file, fed to the
managed policyengine runner as a ``social_security`` perturbation, yielding full
current-year tax-and-benefit incidence. It is REPORTED, NOT GATED: it reads no
gate and changes no gate, and certifies the SEAM, not the levels.

Always-runnable tiers only (this file touches the committed artifacts and the
pure helpers in :mod:`scripts.w2_seam_caregiver`; it references no PSID root and
no policyengine-us checkout, so the collector marks it ``artifact``):

* the schema is sane and marked reported-not-gated; the registration pointer
  (#42 comment 4950247511) and the six named interim-transport deltas ride the
  artifact; the Biden encoding matches the committed anchor replication and its
  own registration; the observed frame's person count binds to the committed
  anchor artifact;
* the fiscal block recomputes (net = gross - income-tax clawback - means-tested
  savings; net/gross; the engine gross agrees with the transport gross to
  floating precision -- the seam self-consistency check); the program offsets
  are enumerated; SPM poverty and decile incidence recompute; the delta-cell
  5-seed floor recomputes from its per-seed gaps; and every pre-registered
  expectation verdict recomputes from the stored quantities;
* pure helpers reproduce closed-form answers (age band, weighted deciles,
  the robust ratio-of-means, the fallback chain, the net-cost arithmetic).

The reproduction pin that rebuilds the observed frame from PSID + the SSA oracle
lives in ``test_w2_seam_caregiver_build.py`` (skipped off-machine).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "w2_seam_caregiver_v1.json"
ANCHOR_ARTIFACT = ROOT / "runs" / "replication_caregiver_v1.json"
SCRIPTS = ROOT / "scripts"

OFFSET_PROGRAMS = ("snap", "ssi", "medicaid", "aca_ptc", "tanf")
SEEDS = [0, 1, 2, 3, 4]
REL = dict(rel=1e-9, abs=1e-6)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _anchor() -> dict:
    return json.loads(ANCHOR_ARTIFACT.read_text())


def _builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import w2_seam_caregiver as builder

    return builder


# ----------------------------------------------------------------------
# Schema, framing, registration pointer, named interim deltas
# ----------------------------------------------------------------------
def test_schema_and_reported_not_gated():
    art = _artifact()
    assert art["schema_version"] == "w2_seam_caregiver.v1"
    assert art["run"] == "w2_seam_caregiver_v1"
    assert art["reported_not_gated"] is True
    assert "changes no gate" in art["purpose"]
    assert "SEAM, not the levels" in art["purpose"]
    assert "SEAM" in art["what_is_certified"]
    assert art["registration_pointer"] == "#42 comment 4950247511"
    assert art["registration"].endswith("4950247511")
    assert art["adr"].endswith("0001-populace-axiom-seam-ownership.md")


def test_named_interim_transport_deltas_present():
    """Every scored claim carries the named interim-transport deltas."""
    art = _artifact()
    named = art["interim_transport_delta_named_on_every_claim"]
    assert isinstance(named, list) and len(named) >= 6
    blob = " ".join(named).lower()
    # The load-bearing named limitations must each appear.
    assert "not the w1-certified" in blob
    assert "social_security rank" in blob
    assert "1943-1957" in blob or "single" in blob
    assert "recipients only" in blob
    assert "sparse-57k" in blob  # the certified-default caveat
    assert "no behavioural response" in blob or "static incidence" in blob


def test_reform_encoding_matches_anchor_replication():
    """The Biden encoding is the committed anchor replication's PLANS[0]."""
    art = _artifact()
    enc = art["reform_encoding"]
    assert enc["credit_fraction"] == 0.5
    assert enc["child_age_limit_exclusive"] == 12
    assert enc["year_cap"] == 5
    assert enc["anchor_replication_registration"] == "#42 comment 4911453454"
    # The anchor artifact really carries a Biden row with these parameters.
    anchor = _anchor()
    biden = next(r for r in anchor["four_plan_table"] if r["plan"] == "Biden")
    assert biden["credit_fraction"] == enc["credit_fraction"]
    assert biden["child_age_limit_max"] == enc["child_age_limit_exclusive"] - 1
    assert biden["year_cap"] == enc["year_cap"]


def test_observed_frame_binds_to_committed_anchor():
    """Step 1's career-frame count equals the committed anchor artifact's."""
    art = _artifact()
    anchor = _anchor()
    step1 = art["step1_observed_frame"]
    assert (
        step1["n_career_frame"] == anchor["study_population"]["n_career_frame"]
    )
    assert 0 < step1["n_gainers"] <= step1["n_career_frame"]
    shares = step1["weighted_sex_shares"]
    assert set(shares) == {"male", "female"}
    assert abs(shares["male"] + shares["female"] - 1.0) < 1e-9
    assert "ER32000" in step1["sex_source"] or "SEX" in step1["sex_source"]


# ----------------------------------------------------------------------
# Step 2 transport structure
# ----------------------------------------------------------------------
def test_step2_transport_structure():
    art = _artifact()
    s2 = art["step2_transport"]
    assert s2["cell_definition"] == "age band x sex x AIME-proxy decile"
    assert s2["age_bands"] == ["62-74", "75-79", "80+"]
    assert s2["n_deciles"] == 10
    assert s2["n_cells_full_grid"] == 3 * 2 * 10
    assert "sum(w*gain)/sum(w*base_PIA)" in s2["cell_statistic"]
    assert "social_security" in s2["aime_proxy_populace"]
    assert (
        "cell -> (sex, decile) -> (decile) -> overall" in s2["fallback_chain"]
    )
    assert s2["winsorization_applied"] is False
    # Fallback accounting covers every recipient across the source levels.
    src = s2["fallback_source_counts_recipients"]
    assert set(src) <= {"cell", "sex_decile", "decile", "overall"}
    assert (
        sum(src.values())
        == art["step3_policyengine"]["counts"]["n_recipients"]
    )
    # The overall proportion is a small positive benefit increase.
    assert 0.0 < s2["overall_proportion"] < 0.2


# ----------------------------------------------------------------------
# Step 3: dataset + counts
# ----------------------------------------------------------------------
def test_step3_certified_default_and_counts():
    art = _artifact()
    s3 = art["step3_policyengine"]
    assert s3["policy_year"] == 2026
    assert "managed_microsimulation" in s3["runner"]
    assert "set_input" in s3["seam_mechanism"]
    c = s3["counts"]
    assert c["n_persons"] > c["n_households"] > 0
    assert 0 < c["n_recipients"] < c["n_persons"]
    assert c["pop_65plus"] < c["pop_total"]


# ----------------------------------------------------------------------
# Fiscal: net = gross - clawback - means-tested savings; seam self-consistency
# ----------------------------------------------------------------------
def test_fiscal_recompute_and_signs():
    art = _artifact()
    f = art["fiscal"]
    offsets = art["program_offsets"]["program_deltas"]
    # Gross benefit cost is positive (the reform raises benefits).
    assert f["gross_benefit_cost"] > 0.0
    # means-tested savings = -(sum of program deltas).
    assert f["means_tested_savings"] == pytest.approx(
        -sum(offsets.values()), **REL
    )
    # net = gross - income-tax clawback - means-tested savings.
    assert f["net_fiscal_cost"] == pytest.approx(
        f["gross_benefit_cost"]
        - f["income_tax_clawback"]
        - f["means_tested_savings"],
        **REL,
    )
    assert f["net_over_gross"] == pytest.approx(
        f["net_fiscal_cost"] / f["gross_benefit_cost"], **REL
    )
    # Net cost is a fraction of gross (some clawback), and positive.
    assert 0.0 < f["net_over_gross"] <= 1.0


def test_seam_self_consistency_engine_vs_transport_gross():
    """The engine's Δsocial_security equals the transport's weighted Σ(w·δ).

    set_input adds exactly the per-person delta, so the engine aggregate change
    must match the transport gross to floating precision -- the seam moved the
    dollars it was handed and nothing else.
    """
    art = _artifact()
    f = art["fiscal"]
    assert f["gross_from_transport_sum"] > 0.0
    rel_gap = f["gross_engine_vs_transport_abs_gap"] / f["gross_benefit_cost"]
    assert rel_gap < 1e-3, rel_gap


def test_program_offsets_enumerated():
    art = _artifact()
    off = art["program_offsets"]
    assert set(off["program_deltas"]) == set(OFFSET_PROGRAMS)
    assert off["snap_falls"] == (off["program_deltas"]["snap"] < 0.0)
    # Every "moved" program really exceeds the threshold.
    thr = off["moved_threshold_dollars"]
    for p in off["programs_that_moved"]:
        assert abs(off["program_deltas"][p]) >= thr
    for p in OFFSET_PROGRAMS:
        if abs(off["program_deltas"][p]) >= thr:
            assert p in off["programs_that_moved"]


# ----------------------------------------------------------------------
# Poverty + decile incidence recompute
# ----------------------------------------------------------------------
def test_poverty_block_recompute_and_direction():
    art = _artifact()
    p = art["poverty"]
    for side in ("baseline", "reform"):
        assert 0.0 < p[side]["spm_poverty_rate_overall"] < 1.0
        assert 0.0 < p[side]["spm_poverty_rate_65plus"] < 1.0
    assert p["reduction_overall_pp"] == pytest.approx(
        100.0
        * (
            p["baseline"]["spm_poverty_rate_overall"]
            - p["reform"]["spm_poverty_rate_overall"]
        ),
        **REL,
    )
    assert p["reduction_65plus_pp"] == pytest.approx(
        100.0
        * (
            p["baseline"]["spm_poverty_rate_65plus"]
            - p["reform"]["spm_poverty_rate_65plus"]
        ),
        **REL,
    )
    # A benefit increase weakly reduces SPM poverty on both cuts.
    assert (
        p["reform"]["spm_poverty_rate_overall"]
        <= p["baseline"]["spm_poverty_rate_overall"] + 1e-9
    )
    assert (
        p["reform"]["spm_poverty_rate_65plus"]
        <= p["baseline"]["spm_poverty_rate_65plus"] + 1e-9
    )


def test_decile_incidence_recompute():
    art = _artifact()
    d = art["decile_incidence"]
    rows = d["by_decile"]
    assert [r["decile"] for r in rows] == list(range(1, 11))
    assert d["bottom_half_share_of_gain_pct"] == pytest.approx(
        sum(r["share_of_aggregate_gain_pct"] for r in rows[:5]), **REL
    )
    assert d["bottom_quintile_share_of_gain_pct"] == pytest.approx(
        sum(r["share_of_aggregate_gain_pct"] for r in rows[:2]), **REL
    )
    # The 10 deciles plus the out-of-range (negative-income) block partition
    # the total gain (~100%).
    total = sum(r["share_of_aggregate_gain_pct"] for r in rows)
    total += d["out_of_range"]["share_of_aggregate_gain_pct"]
    assert total == pytest.approx(100.0, abs=1e-6)
    assert d["out_of_range"]["n_households"] >= 0


# ----------------------------------------------------------------------
# Delta-cell 5-seed floor recompute
# ----------------------------------------------------------------------
def test_delta_cell_floor_recompute():
    art = _artifact()
    floors = art["delta_cell_floors_5seed"]
    assert floors["seeds"] == SEEDS
    gross = floors["gross_cost_floor"]
    per_seed = gross["per_seed"]
    assert [s["seed"] for s in per_seed] == SEEDS
    gaps = [s["gross_abs_gap"] for s in per_seed]
    for s in per_seed:
        assert s["gross_abs_gap"] == pytest.approx(
            abs(s["gross_side_a"] - s["gross_side_b"]), **REL
        )
    assert gross["mean"] == pytest.approx(float(np.mean(gaps)), **REL)
    assert gross["max"] == pytest.approx(float(np.max(gaps)), **REL)
    # The floor is small relative to the gross cost (a stable transport).
    assert gross["as_share_of_gross"] == pytest.approx(
        gross["mean"] / art["fiscal"]["gross_benefit_cost"], **REL
    )
    cell = art["delta_cell_floors_5seed"]["cell_proportion_floor"]
    assert cell["n_cells"] >= 1
    assert cell["max_abs_gap_over_cells"] >= cell["mean_abs_gap_over_cells"]


# ----------------------------------------------------------------------
# Pre-registered expectation verdicts recompute from stored quantities
# ----------------------------------------------------------------------
def test_registered_expectations_recompute():
    art = _artifact()
    e = art["registered_expectations"]
    f = art["fiscal"]
    off = art["program_offsets"]
    p = art["poverty"]
    d = art["decile_incidence"]

    net = e["net_cost_75_92pct_of_gross"]
    assert net["net_over_gross"] == pytest.approx(f["net_over_gross"], **REL)
    assert net["held"] == (0.75 <= f["net_over_gross"] <= 0.92)

    snap = e["snap_spending_falls"]
    assert snap["held"] == (off["program_deltas"]["snap"] < 0.0)

    pov = e["poverty_reduction_concentrates_65plus"]
    assert pov["held"] == (
        p["reduction_65plus_pp"] >= p["reduction_overall_pp"]
        and p["reduction_65plus_pp"] > 0
    )

    dec = e["decile_gains_bottom_half"]
    assert dec["held"] == (d["bottom_half_share_of_gain_pct"] > 50.0)

    # The aggregate verdict is exactly "no failure modes".
    assert e["all_expectations_held"] == (len(e["seam_failure_modes"]) == 0)
    blocks = [
        e["net_cost_75_92pct_of_gross"]["held"],
        e["snap_spending_falls"]["held"],
        e["poverty_reduction_concentrates_65plus"]["held"],
        e["decile_gains_bottom_half"]["held"],
    ]
    assert e["all_expectations_held"] == all(blocks)


def test_revision_pins():
    art = _artifact()
    pins = art["revision_pins"]
    assert pins["artifact_schema_version"] == "w2_seam_caregiver.v1"
    assert pins["pe_us_version"]
    assert art["step3_policyengine"]["pe_us_version"] == pins["pe_us_version"]
    # Step 1 pins the SSA-oracle (policyengine-us) revision it was scored on.
    assert pins["ss_oracle_pe_us_revision"]


# ----------------------------------------------------------------------
# Pure-helper unit tests (import the builder; neutral top-level imports)
# ----------------------------------------------------------------------
def test_age_band_boundaries():
    b = _builder()
    assert b.age_band(62) == "62-74"
    assert b.age_band(74.999) == "62-74"
    assert b.age_band(75) == "75-79"
    assert b.age_band(79.999) == "75-79"
    assert b.age_band(80) == "80+"
    assert b.age_band(101) == "80+"


def test_weighted_decile_index_partition():
    b = _builder()
    g = b.weighted_decile_index(np.arange(100.0), np.ones(100))
    assert set(g.tolist()) == set(range(10))
    assert np.bincount(g, minlength=10).tolist() == [10] * 10
    # A 30% mass at zero fills the bottom three deciles.
    vals = np.concatenate([np.zeros(30), np.arange(1, 71.0)])
    g2 = b.weighted_decile_index(vals, np.ones(100))
    assert set(g2[:30].tolist()) <= {0, 1, 2}


def test_ratio_of_means_robust_to_small_base():
    """Per-person gain/base would blow up on a tiny base; ratio-of-means is
    dominated by the weighted totals, not the outlier."""
    b = _builder()
    gain = np.array([10.0, 10.0, 10.0, 10.0])
    base = np.array([100.0, 100.0, 100.0, 0.001])
    prop, den = b.ratio_of_means(gain, base, np.ones(4))
    assert prop == pytest.approx(40.0 / 300.001, abs=1e-9)
    assert prop < 0.15  # not the ~10000 a per-person ratio would give
    assert den == pytest.approx(300.001, abs=1e-9)
    # Degenerate (zero base) -> zero proportion, not a division error.
    assert b.ratio_of_means(np.array([1.0]), np.array([0.0]), np.ones(1)) == (
        0.0,
        0.0,
    )


def test_lookup_cell_prop_fallback_chain():
    b = _builder()
    tables = {
        "full": {
            ("62-74", "female", 0): {"prop": 0.5, "n": 10, "wbase": 1.0},
            ("62-74", "male", 1): {"prop": 0.2, "n": 2, "wbase": 1.0},
        },
        "sex_decile": {("male", 1): {"prop": 0.25, "n": 8, "wbase": 1.0}},
        "decile": {5: {"prop": 0.05, "n": 3, "wbase": 1.0}},
        "overall": {"prop": 0.03, "n": 100, "wbase": 1.0},
    }
    assert b.lookup_cell_prop(tables, "62-74", "female", 0) == (0.5, "cell")
    # Thin cell (n<MIN_CELL_N) falls back to the (sex, decile) marginal.
    assert b.lookup_cell_prop(tables, "62-74", "male", 1) == (
        0.25,
        "sex_decile",
    )
    # No cell, no marginal -> decile -> overall.
    assert b.lookup_cell_prop(tables, "80+", "female", 5) == (0.05, "decile")
    assert b.lookup_cell_prop(tables, "80+", "female", 9) == (0.03, "overall")


def test_net_fiscal_cost_closed_form():
    b = _builder()
    nf = b.net_fiscal_cost(
        65.79,
        3.90,
        {
            "snap": -1.0,
            "ssi": -1.61,
            "medicaid": 0.02,
            "aca_ptc": -0.25,
            "tanf": 0.0,
        },
    )
    assert nf["means_tested_savings"] == pytest.approx(2.84, abs=1e-9)
    assert nf["net_fiscal_cost"] == pytest.approx(59.05, abs=1e-9)
    assert nf["net_over_gross"] == pytest.approx(59.05 / 65.79, abs=1e-12)
    assert b.net_fiscal_cost(0.0, 0.0, {"snap": 0.0})["net_over_gross"] is None


def test_transport_gross_and_cell_tables_closed_form():
    """build_cell_tables + transport_gross on a hand-built observed frame."""
    import pandas as pd

    b = _builder()
    observed = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "gain": [10.0, 0.0, 5.0, 0.0],
            "base": [100.0, 100.0, 50.0, 50.0],
            "band": ["62-74", "62-74", "80+", "80+"],
            "sex": ["female", "female", "male", "male"],
            "dec": [0, 0, 9, 9],
        }
    )
    tables = b.build_cell_tables(observed)
    # Cell (62-74, female, 0): gain 10 over base 200 -> 0.05.
    assert tables["full"][("62-74", "female", 0)]["prop"] == pytest.approx(
        0.05
    )
    assert tables["overall"]["prop"] == pytest.approx(15.0 / 300.0)
    # Transport a single recipient in the (80+, male, 9) cell (prop 5/100=0.05).
    delta, gross = b.transport_gross(
        tables,
        np.array(["80+"]),
        np.array(["male"]),
        np.array([9]),
        np.array([20000.0]),
        np.array([2.0]),
    )
    assert delta[0] == pytest.approx(0.05 * 20000.0)
    assert gross == pytest.approx(2.0 * 0.05 * 20000.0)
