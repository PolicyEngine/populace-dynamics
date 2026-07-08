"""Spouse/survivor auxiliary benefits (#74, Phase C): 42 USC 402(b)/(c)/(e)/(f).

Three tiers, mirroring the repo's oracle discipline:

* **Pure units** — reduction floors, dual-entitlement offset, RIB-LIM
  branches, the remarriage predicate, on a hand-built parameter bundle
  (the auxiliary rates are statutory constants, defaulted on
  ``SSAParameters``), so no checkout is needed.
* **SSA worked examples** — every published auxiliary figure (spousal
  50%/32.5%/35%, the 71.5% survivor floor, the 81%-at-62 FRA-66 cell,
  RIB-LIM 82.5%, DRC pass-through), cited in the committed artifact.
* **policyengine-us PIA foundation** — the own-benefit layer the
  auxiliary amounts are fractions of, cross-checked against a live
  policyengine-us Simulation run in a separate interpreter (located by
  ``POPULACE_DYNAMICS_PE_US_PYTHON``), skipped if unavailable. pe-us has
  NO spousal/survivor computation, so this validates the foundation, not
  the auxiliary amounts (which the SSA examples carry).

None of these is a gate; the artifact is REPORTED, not gated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from populace_dynamics.ss import benefits
from populace_dynamics.ss.params import SSAParameters

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "runs" / "aux_benefit_examples_v1.json"
SCRIPTS = ROOT / "scripts"

PE_US = Path("~/PolicyEngine/policyengine-us").expanduser()
needs_pe_us = pytest.mark.skipif(
    not PE_US.is_dir()
    and "POPULACE_DYNAMICS_PE_US_DIR" not in __import__("os").environ,
    reason="policyengine-us not checked out and "
    "POPULACE_DYNAMICS_PE_US_DIR unset",
)


def _pure_params(**overrides) -> SSAParameters:
    """FRA 65/66/67 tiers + the 8%/yr credit; auxiliary rates default to
    their statutory values. Overridable (e.g. the survivor span)."""
    base = dict(
        nawi={},
        wage_base={},
        pia_factors=(0.90, 0.32, 0.15),
        fra_months_by_birth_year=[(1900, 780), (1943, 792), (1960, 804)],
        early_monthly_rates=(0.00555556, 0.00416667),
        early_first_bracket_months=36,
        pe_us_revision="pure-test-bundle",
        delayed_credit_by_birth_year=[(1900, 0.03), (1943, 0.08)],
        max_delayed_months=48,
    )
    base.update(overrides)
    return SSAParameters(**base)


# ==========================================================================
# Tier 1: statutory constants default onto the bundle
# ==========================================================================
def test_auxiliary_constants_default_to_statute():
    p = _pure_params()
    assert p.spousal_pia_share == 0.5
    assert p.spousal_early_monthly_rates == (25 / 3600, 5 / 1200)
    assert p.spousal_early_first_bracket_months == 36
    assert p.survivor_pia_share == 1.0
    assert p.survivor_reduction_floor == 0.715
    assert p.survivor_reduction_period_months == 84  # survivor FRA 67
    assert p.rib_lim_pia_share == 0.825
    assert p.remarriage_protected_age == 60
    assert p.remarriage_protected_age_disabled == 50


# ==========================================================================
# Tier 1: spouse's benefit — 402(b)/(c), 402(q)
# ==========================================================================
def test_spousal_early_reduction_brackets():
    p = _pure_params()
    assert benefits.spousal_early_reduction(0, p) == 0.0
    # First 36 months at 25/36 of 1% -> exactly 25%.
    assert benefits.spousal_early_reduction(36, p) == pytest.approx(0.25)
    # 48 months (62 vs FRA 66): 25% + 12*5/12% = 30%.
    assert benefits.spousal_early_reduction(48, p) == pytest.approx(0.30)
    # 60 months (62 vs FRA 67): 25% + 24*5/12% = 35%.
    assert benefits.spousal_early_reduction(60, p) == pytest.approx(0.35)
    # Steeper first bracket than the worker's own 5/9%.
    assert benefits.spousal_early_reduction(36, p) > benefits.early_reduction(
        36, p
    )


def test_spousal_benefit_headline_shares():
    p = _pure_params()
    # 50% of the worker's PIA at FRA.
    assert benefits.spousal_benefit(0.0, 2000.0, 0, p) == pytest.approx(1000.0)
    # 32.5% at 62 against FRA 67.
    assert benefits.spousal_benefit(0.0, 2000.0, 60, p) == pytest.approx(650.0)
    # 35% at 62 against FRA 66.
    assert benefits.spousal_benefit(0.0, 2000.0, 48, p) == pytest.approx(700.0)


def test_spousal_dual_entitlement_offset():
    p = _pure_params()
    # Excess of half the worker's PIA over the claimant's own PIA.
    assert benefits.spousal_benefit(600.0, 2000.0, 0, p) == pytest.approx(
        400.0
    )
    # Own PIA at least half the worker's -> no excess spousal benefit.
    assert benefits.spousal_benefit(1000.0, 2000.0, 0, p) == 0.0
    assert benefits.spousal_benefit(1500.0, 2000.0, 0, p) == 0.0
    # The offset is applied before the age reduction, so the reduced
    # excess is (0.5*worker - own) * (1 - reduction).
    assert benefits.spousal_benefit(400.0, 2000.0, 60, p) == pytest.approx(
        (1000.0 - 400.0) * 0.65
    )


# ==========================================================================
# Tier 1: widow(er)'s benefit — 402(e)/(f), 402(q), RIB-LIM
# ==========================================================================
def test_survivor_reduction_ramp_and_floor():
    p = _pure_params()
    assert benefits.survivor_reduction(0, p) == 0.0
    # Maximum reduction 28.5% at the full 84-month span (age 60).
    assert benefits.survivor_reduction(84, p) == pytest.approx(0.285)
    # Linear midpoint.
    assert benefits.survivor_reduction(42, p) == pytest.approx(0.1425)
    # 60 months early (age 62, FRA 67): 0.285 * 60/84.
    assert benefits.survivor_reduction(60, p) == pytest.approx(0.285 * 60 / 84)
    # Clamped at the floor beyond the span (disabled-widow deep early claim).
    assert benefits.survivor_reduction(120, p) == pytest.approx(0.285)


def test_widow_reduction_71_5_floor_at_60():
    p = _pure_params()
    # Widow at 60 (84 months early, FRA 67) gets exactly 71.5% of PIA.
    assert benefits.widow_benefit(0.0, 1000.0, 84, 1.0, p) == pytest.approx(
        715.0
    )
    # 81% at 62 for a survivor FRA of 66 (72-month span).
    p66 = _pure_params(survivor_reduction_period_months=72)
    assert benefits.widow_benefit(0.0, 1000.0, 48, 1.0, p66) == pytest.approx(
        810.0
    )


def test_widow_full_pia_and_dual_entitlement():
    p = _pure_params()
    # 100% of the deceased's PIA at survivor FRA (deceased claimed at FRA).
    assert benefits.widow_benefit(0.0, 1000.0, 0, 1.0, p) == pytest.approx(
        1000.0
    )
    # Dual entitlement: the survivor is paid the larger of own or widow.
    assert benefits.widow_benefit(900.0, 1000.0, 84, 1.0, p) == pytest.approx(
        900.0
    )
    assert benefits.widow_benefit(600.0, 1000.0, 84, 1.0, p) == pytest.approx(
        715.0
    )


def test_rib_lim_cases():
    p = _pure_params()
    pia = 1000.0
    # Deceased claimed a reduced RIB of 0.75*PIA (< 0.825) -> widow at
    # survivor FRA is capped at 82.5% of PIA.
    assert benefits.widow_benefit(0.0, pia, 0, 0.75, p) == pytest.approx(825.0)
    # Deceased's actual reduced benefit exceeds 82.5% -> widow gets it.
    assert benefits.widow_benefit(0.0, pia, 0, 0.90, p) == pytest.approx(900.0)
    # Right at the 82.5% hinge.
    assert benefits.widow_benefit(0.0, pia, 0, 0.825, p) == pytest.approx(
        825.0
    )
    # Delayed-retirement credits pass through (no RIB-LIM when factor>=1).
    assert benefits.widow_benefit(0.0, pia, 0, 1.32, p) == pytest.approx(
        1320.0
    )
    # RIB-LIM ceiling still binds when the survivor also claims early:
    # deceased 0.75 -> ceiling 825; survivor 84 months early would give
    # 1000*0.715=715 (< 825), so the survivor's own reduction binds.
    assert benefits.widow_benefit(0.0, pia, 84, 0.75, p) == pytest.approx(
        715.0
    )
    # But a milder survivor reduction is capped at the 825 ceiling:
    # 42 months early -> 1000*0.8575=857.5, capped to 825.
    assert benefits.widow_benefit(0.0, pia, 42, 0.75, p) == pytest.approx(
        825.0
    )


def test_widow_inherits_credits_but_capped_at_deceased_amount():
    p = _pure_params()
    # Deceased delayed to earn +32%: widow base is the credit-enhanced
    # 1320, reduced for the survivor's own early claiming.
    at_fra = benefits.widow_benefit(0.0, 1000.0, 0, 1.32, p)
    early = benefits.widow_benefit(0.0, 1000.0, 84, 1.32, p)
    assert at_fra == pytest.approx(1320.0)
    assert early == pytest.approx(1320.0 * 0.715)


# ==========================================================================
# Tier 1: remarriage predicate — 402(e)(3)/(f)(4)
# ==========================================================================
def test_remarriage_predicate():
    p = _pure_params()
    assert benefits.widow_benefit_survives_remarriage(59, p) is False
    assert benefits.widow_benefit_survives_remarriage(60, p) is True
    assert benefits.widow_benefit_survives_remarriage(61, p) is True
    # Disabled survivor: protected from age 50.
    assert benefits.widow_benefit_survives_remarriage(55, p, disabled=True)
    assert not benefits.widow_benefit_survives_remarriage(49, p, disabled=True)


# ==========================================================================
# Tier 2: the committed worked-example artifact
# ==========================================================================
@pytest.fixture(scope="module")
def artifact() -> dict:
    with ARTIFACT.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_artifact_schema_and_reported_not_gated(artifact: dict):
    assert artifact["schema_version"] == "aux_benefit_examples.v1"
    assert artifact["run"] == "aux_benefit_examples_v1"
    assert artifact["reported_not_gated"] is True
    assert artifact["pe_us_revision"] not in ("", "unknown")


def test_artifact_ssa_examples_match_published(artifact: dict):
    block = artifact["ssa_worked_examples"]
    # Every SSA example agrees with its published figure to the cent.
    for group in ("spousal", "survivor"):
        for ex in block[group]:
            assert abs(ex["our_output"] - ex["expected"]) < 0.01, ex["name"]
            assert ex["citation"]
    assert block["max_abs_deviation"] < 0.01
    # The rock-solid independent numeric anchors are present.
    names = {e["name"] for e in block["spousal"] + block["survivor"]}
    assert {
        "spouse_at_fra_half_pia",
        "spouse_at_62_fra67",
        "widow_at_60_floor",
        "widow_at_62_fra66",
        "rib_lim_82_5_floor",
    } <= names


def test_artifact_foundation_agrees_with_pe_us(artifact: dict):
    f = artifact["pe_us_pia_foundation"]
    if not f["oracle_available"]:
        pytest.skip("pe-us oracle was unavailable when the artifact was built")
    # PIA agrees to float32 storage precision (< 1e-4 dollars); the
    # adjustment factor to ~1e-7.
    assert f["max_pia_abs_deviation"] < 1e-3
    assert f["max_factor_abs_deviation"] < 1e-5
    assert f["n_cases"] == 40


def test_artifact_reports_panel_coverage(artifact: dict):
    cov = artifact["panel_coverage"]
    if not (cov and cov.get("available")):
        pytest.skip("Marriage History File not staged when built")
    # Both-spouse coverage is the reported ceiling; widowhood the survivor
    # subset. Sanity bounds (values pinned in test_household).
    assert 0.5 < cov["both_computable_share"] <= cov["joinable_spouse_share"]
    assert 0.0 < cov["widowhood_share"] < 0.2


@needs_pe_us
def test_artifact_rebuild_reproduces(artifact: dict):
    # build() loads parameters from policyengine-us, so this reproduction
    # check needs the checkout; the committed-artifact reads above do not.
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_aux_benefit_examples as builder

    rebuilt = builder.build()
    # The SSA and couple/survivor blocks are deterministic and pe-us-free;
    # they must reproduce exactly (the foundation/coverage blocks depend on
    # external checkouts, so are compared only when present and equal-keyed).
    for key in ("ssa_worked_examples", "couple_grid", "survivor_grid"):
        assert rebuilt[key] == artifact[key], key


# ==========================================================================
# Tier 3: live policyengine-us Simulation oracle (subprocess; may skip)
# ==========================================================================
def _load_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_aux_benefit_examples as builder

    return builder


@needs_pe_us
def test_pe_us_simulation_matches_oracle_foundation_live():
    """Recompute a subset of the PIA foundation through a LIVE
    policyengine-us Simulation and assert our pia()/benefit_factor() agree
    to float32 precision — a fresh proof, not the committed file. Skips
    unless POPULACE_DYNAMICS_PE_US_PYTHON points at a pe-us interpreter."""
    from populace_dynamics import claiming
    from populace_dynamics.ss.params import load_ssa_parameters

    builder = _load_builder()
    python = builder._pe_us_python()
    if not builder._pe_us_available(python):
        pytest.skip(
            "POPULACE_DYNAMICS_PE_US_PYTHON unset or cannot import "
            "policyengine_us; set it to a pe-us venv interpreter"
        )
    params = load_ssa_parameters()
    cases = builder._foundation_cases()[:12]
    oracle = {
        (r["aime"], r["age"], r["claim_age"]): r
        for r in builder._pe_us_oracle(python, cases)
    }
    checked = 0
    for c in cases:
        aime, age, claim_age = c["aime"], c["age"], c["claim_age"]
        birth_year = builder.FOUNDATION_YEAR - age
        our_pia = benefits.pia(aime, builder.FOUNDATION_YEAR, params)
        our_factor = claiming.benefit_factor(
            claim_age * 12, birth_year, params
        )
        o = oracle[(aime, age, claim_age)]
        assert abs(our_pia - o["pia"]) < 1e-3, (c, our_pia, o["pia"])
        assert abs(our_factor - o["factor"]) < 1e-5, c
        checked += 1
    assert checked == 12
