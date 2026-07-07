"""Tests for Title II formulas against SSA's own published figures."""

from __future__ import annotations

from pathlib import Path

import pytest

from populace_dynamics.ss import benefits, params

PE_US = Path("~/PolicyEngine/policyengine-us").expanduser()
needs_pe_us = pytest.mark.skipif(
    not PE_US.is_dir(), reason="policyengine-us not checked out"
)

if PE_US.is_dir():
    PARAMS = params.load_ssa_parameters()


@needs_pe_us
def test_loader_cross_checks_pass_and_pin_revision():
    assert PARAMS.pe_us_revision != ""
    assert PARAMS.pia_factors == (0.90, 0.32, 0.15)


def _ssa_official_2026_params():
    """Minimal parameter bundle with SSA's OWN published inputs.

    SSA's 2026 determination uses realized NAWI(2024) = 69,846.57
    (policyengine-us carries a Trustees projection there), so the
    external worked-example anchor builds its own bundle.
    """
    from populace_dynamics.ss.params import SSAParameters

    return SSAParameters(
        nawi={1977: 9_779.44, 2024: 69_846.57},
        wage_base={},
        pia_factors=(0.90, 0.32, 0.15),
        fra_months_by_birth_year=[(1900, 780), (1943, 792), (1960, 804)],
        early_monthly_rates=(0.00555556, 0.00416667),
        early_first_bracket_months=36,
        pe_us_revision="ssa-official-anchor",
        delayed_credit_by_birth_year=[(1900, 0.03), (1943, 0.08)],
        max_delayed_months=48,
    )


def test_2026_bend_points_match_ssa_determination():
    params_2026 = _ssa_official_2026_params()
    assert params_2026.bend_points(2026) == (1286.0, 7749.0)


def test_pia_worked_example_2026():
    # 0.9*1286 + 0.32*(7749-1286) + 0.15*(8000-7749)
    #   = 1157.40 + 2068.16 + 37.65 = 3263.21 -> 3263.20 per 415(g).
    params_2026 = _ssa_official_2026_params()
    assert benefits.pia(8000, 2026, params_2026) == pytest.approx(3263.20)


def test_pia_below_first_bend_point_is_90_percent():
    params_2026 = _ssa_official_2026_params()
    assert benefits.pia(1000, 2026, params_2026) == pytest.approx(900.0)


@needs_pe_us
def test_derived_2015_bend_points_match_ssa_published():
    # SSA's 2015 determination: 826 / 4,980 (realized NAWI era).
    assert PARAMS.bend_points(2015) == (826.0, 4980.0)


@needs_pe_us
def test_aime_caps_at_wage_base_and_floors_to_dollar():
    # A worker at exactly the wage base every year from age 22 to 61.
    birth_year = 1955
    history = {}
    for age in range(22, 62):
        year = birth_year + age
        history[year] = 10_000_000.0  # far above every cap
    value = benefits.aime(history, birth_year, PARAMS)
    capped = benefits.creditable_history(history, PARAMS)
    indexed = benefits.indexed_history(capped, birth_year, PARAMS)
    top = sorted(indexed.values(), reverse=True)[:35]
    assert value == int(sum(top) / 420)


@needs_pe_us
def test_indexing_is_neutral_for_constant_relative_earner():
    # Earning exactly NAWI every year indexes to the same amount.
    birth_year = 1950
    indexing_year = birth_year + 60
    history = {
        year: PARAMS.nawi[year]
        for year in range(birth_year + 22, birth_year + 60)
    }
    indexed = benefits.indexed_history(history, birth_year, PARAMS)
    for value in indexed.values():
        assert value == pytest.approx(PARAMS.nawi[indexing_year])


@needs_pe_us
def test_early_reduction_at_fra_67_claiming_62():
    # 60 months early: 36 * 5/9% + 24 * 5/12% = 20% + 10% = 30%.
    assert benefits.early_reduction(60, PARAMS) == pytest.approx(0.30)
    # Born 1960+ has FRA 67 -> age-62 benefit is 70% of PIA.
    assert benefits.age62_monthly_benefit(
        1000.0, 1962, PARAMS
    ) == pytest.approx(700.0)


@needs_pe_us
def test_fra_schedule_matches_statute():
    assert PARAMS.fra_months(1937) == 65 * 12
    assert PARAMS.fra_months(1943) == 66 * 12
    assert PARAMS.fra_months(1960) == 67 * 12
    assert PARAMS.fra_months(1990) == 67 * 12


def test_delayed_credit_pure_bundle():
    # FRA 66 (born 1943): claiming at 70 is 48 months late = 4 years of
    # the 8%/yr credit = 32%.
    p = _ssa_official_2026_params()
    assert benefits.delayed_credit(48, 1943, p) == pytest.approx(0.32)
    # No credit at or before FRA.
    assert benefits.delayed_credit(0, 1943, p) == 0.0
    assert benefits.delayed_credit(-5, 1943, p) == 0.0
    # FRA 67 (born 1960): credits stop at age 70, so the window is only
    # 36 months even though max_delayed_months is 48 -> 3 * 8% = 24%.
    assert benefits.delayed_credit(48, 1960, p) == pytest.approx(0.24)
    assert benefits.delayed_credit(24, 1960, p) == pytest.approx(0.16)


@needs_pe_us
def test_delayed_credit_rate_loads_from_pe_us():
    # 42 USC 402(w): 8% per year for 1943 and later; 3% for the earliest
    # cohorts.
    assert PARAMS.delayed_credit_annual_rate(1943) == pytest.approx(0.08)
    assert PARAMS.delayed_credit_annual_rate(1990) == pytest.approx(0.08)
    assert PARAMS.delayed_credit_annual_rate(1917) == pytest.approx(0.03)
    assert PARAMS.max_delayed_months == 48
    # Born 1943 (FRA 66) claiming at 70 -> +32% of PIA.
    assert benefits.delayed_credit(48, 1943, PARAMS) == pytest.approx(0.32)
