"""Tests for the claiming-age module and its committed references (#74).

Two tiers, mirroring the repo's run/floor tests:

* **Always runnable** -- touch only the committed artifacts
  (``data/external/ssa_claim_ages_2023supplement.json`` and
  ``runs/claiming_reference_v1.json``), the build scripts, and a
  hand-built parameter bundle. These pin the four-part validation
  standard stated in ``populace_dynamics.claiming``: exact reference
  reproduction, share-sum integrity, sampler/factor mechanics, and the
  held-out numbers.
* **policyengine-us-dependent** (skipped when the checkout is absent) --
  bind Table 6.B5.1 footnote a's FRA schedule to the pinned 416(l)
  oracle and check the factor against SSA's own worked ratios.

None of these tests is a locked gate; the held-out artifact is REPORTED,
not gated (see :mod:`scripts.build_claiming_reference`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from populace_dynamics import claiming
from populace_dynamics.ss.params import SSAParameters

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "external" / "ssa_claim_ages_2023supplement.json"
HELDOUT = ROOT / "runs" / "claiming_reference_v1.json"
SCRIPTS = ROOT / "scripts"

PE_US = Path("~/PolicyEngine/policyengine-us").expanduser()
needs_pe_us = pytest.mark.skipif(
    not PE_US.is_dir(), reason="policyengine-us not checked out"
)


def _reference() -> dict:
    return json.loads(REFERENCE.read_text())


def _heldout() -> dict:
    return json.loads(HELDOUT.read_text())


def _import_build_reference():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_ssa_claim_ages as builder

    return builder


def _import_build_heldout():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_claiming_reference as builder

    return builder


def _pure_params() -> SSAParameters:
    """Minimal parameter bundle: FRA 65/66/67 tiers + the 8%/yr credit.

    Lets the sampler and factor mechanics be tested without a
    policyengine-us checkout (the FRA and 402(q)/402(w) rates it needs
    are the statutory constants SSA publishes).
    """
    return SSAParameters(
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


#: Published Table 6.B5.1 cells (raw columns), transcribed independently
#: of the build script for the reproduction spot-check.
PUBLISHED_SPOT_CHECK = {
    ("male", "2022"): {
        "number_thousands": 1580,
        "average_age": 65.1,
        "raw": {
            "age62": 23.8,
            "age63": 6.6,
            "age64": 6.8,
            "age65_before_fra": 11.7,
            "age65_at_fra": None,
            "age65_after_fra": None,
            "age66_before_fra": 2.1,
            "age66_at_fra": 14.5,
            "age66_after_fra": 4.9,
            "disability_conversion": 14.5,
            "age67_69": 7.0,
            "age70plus": 8.1,
        },
    },
    ("female", "2022"): {
        "number_thousands": 1563,
        "average_age": 65.0,
        "raw": {
            "age62": 25.6,
            "age63": 6.8,
            "age64": 7.4,
            "age65_before_fra": 12.0,
            "age65_at_fra": None,
            "age65_after_fra": None,
            "age66_before_fra": 2.0,
            "age66_at_fra": 13.0,
            "age66_after_fra": 4.2,
            "disability_conversion": 14.1,
            "age67_69": 6.2,
            "age70plus": 8.8,
        },
    },
    ("male", "1998"): {
        "number_thousands": 902,
        "average_age": 63.4,
        "raw": {
            "age62": 50.8,
            "age63": 6.7,
            "age64": 10.6,
            "age65_before_fra": None,
            "age65_at_fra": 12.1,
            "age65_after_fra": 2.5,
            "age66_before_fra": None,
            "age66_at_fra": None,
            "age66_after_fra": 1.4,
            "disability_conversion": 12.7,
            "age67_69": 2.1,
            "age70plus": 1.1,
        },
    },
}


# ==========================================================================
# Tier 1: the committed reference JSON
# ==========================================================================
def test_reference_schema_and_provenance():
    doc = _reference()
    assert doc["schema_version"] == "ssa_claim_ages.v1"
    assert doc["table"] == "6.B5.1"
    assert doc["supplement_year"] == 2023

    prov = doc["provenance"]
    assert prov["source_url"].startswith(
        "https://www.ssa.gov/policy/docs/statcomps/supplement/2023/6b"
    )
    # Browser-fetch provenance (ssa.gov 403s programmatic fetches).
    assert "authenticated browser session 2026-07-07" in prov["fetch_method"]
    assert "403" in prov["fetch_method"]
    assert "dynasim-refs" in prov["provenance_file"]
    assert "Master Beneficiary Record" in prov["data_basis"]
    assert "100 percent data" in prov["mbr_100_percent_note"]
    # FRA schedule (footnote a) and the disability-conversion definition
    # (footnote b) are both preserved verbatim.
    assert "attaining age 65 before 2003" in prov["footnote_a_fra_schedule"]
    assert "automatically converts" in prov["footnote_b_disability_conversion"]

    schema = doc["column_schema"]
    assert len(schema["raw_columns"]) == 12
    assert len(schema["collapsed_categories"]) == 8
    assert "SUBSET" in schema["fra_at_note"]
    assert "double-count" in schema["fra_at_note"]
    assert (
        "not a claiming choice" in schema["disability_conversion_note"].lower()
        or "NOT a claiming choice" in schema["disability_conversion_note"]
    )
    # Era map documents all four regimes; nothing silently merged.
    assert len(schema["era_map"]) == 4


def test_reference_published_spot_checks_exact():
    """Published cells reproduce to the digit (reproduction standard)."""
    doc = _reference()
    for (sex, year), expected in PUBLISHED_SPOT_CHECK.items():
        row = doc["data"][sex][year]
        assert row["number_thousands"] == expected["number_thousands"]
        assert row["average_age"] == expected["average_age"]
        assert row["raw"] == expected["raw"]


def test_reference_build_reproduces_committed_exactly():
    """Rerunning the build reproduces the committed JSON (bar timestamp)."""
    builder = _import_build_reference()
    rebuilt = builder.build()
    committed = _reference()
    for key in committed:
        if key == "build":
            continue
        assert rebuilt[key] == committed[key], key


def test_share_sum_integrity():
    doc = _reference()
    tol = doc["validation"]["sum_tolerance"]
    worst = 0.0
    for sex in ("male", "female"):
        for year, row in doc["data"][sex].items():
            cats = row["categories"]
            recomputed = round(sum(cats.values()), 1)
            assert recomputed == row["component_sum"]
            assert row["residual"] == round(recomputed - 100.0, 1)
            assert abs(row["residual"]) <= tol, (sex, year)
            worst = max(worst, abs(row["residual"]))
    assert doc["validation"]["max_abs_residual"] == pytest.approx(worst)
    # The residuals are real (SSA rounds components), not identically 0.
    assert worst > 0.0


def _sum_raw(raw: dict, keys: tuple[str, ...]) -> float:
    return round(sum(raw[k] for k in keys if raw[k] is not None), 1)


def test_collapsed_partition_recomputes_from_raw():
    doc = _reference()
    for sex in ("male", "female"):
        for row in doc["data"][sex].values():
            raw = row["raw"]
            cats = row["categories"]
            assert cats["age65"] == _sum_raw(
                raw, ("age65_before_fra", "age65_at_fra", "age65_after_fra")
            )
            assert cats["age66"] == _sum_raw(
                raw, ("age66_before_fra", "age66_at_fra", "age66_after_fra")
            )
            assert cats["disability_conversion"] == (
                raw["disability_conversion"] or 0.0
            )


def test_fra_at_is_the_single_populated_at_fra_column():
    doc = _reference()
    for sex in ("male", "female"):
        for year, row in doc["data"][sex].items():
            raw = row["raw"]
            populated = [
                k
                for k in ("age65_at_fra", "age66_at_fra")
                if raw[k] is not None
            ]
            assert len(populated) == 1, (sex, year)
            at_col = populated[0]
            assert row["fra_at"]["share"] == raw[at_col]
            assert row["fra_at"]["at_age"] == (
                65 if at_col == "age65_at_fra" else 66
            )
            # fra_at is a subset of the age65+age66 mass, never additive.
            assert (
                row["fra_at"]["share"]
                <= row["categories"]["age65"]
                + row["categories"]["age66"]
                + 1e-9
            )


def test_era_transition_columns_are_na_where_expected():
    """The FRA-transition N/A pattern is honoured, not back-filled."""
    doc = _reference()
    # 1998 (FRA=65): no 65-before-FRA, no 66 at/before FRA.
    m1998 = doc["data"]["male"]["1998"]["raw"]
    assert m1998["age65_before_fra"] is None
    assert m1998["age66_before_fra"] is None
    assert m1998["age66_at_fra"] is None
    # 2009 (FRA=66): 65 entirely before FRA; 66 at/after only.
    m2009 = doc["data"]["male"]["2009"]["raw"]
    assert m2009["age65_at_fra"] is None
    assert m2009["age65_after_fra"] is None
    assert m2009["age66_before_fra"] is None
    assert m2009["age66_at_fra"] is not None
    # 2022 (FRA>66): 66 splits three ways.
    m2022 = doc["data"]["male"]["2022"]["raw"]
    assert m2022["age66_before_fra"] is not None
    assert m2022["age66_at_fra"] is not None
    assert m2022["age66_after_fra"] is not None


# ==========================================================================
# Tier 1: module behaviour
# ==========================================================================
def test_claim_age_distribution_and_nearest_year_rule():
    exact = claiming.claim_age_distribution("men", 2010)
    assert exact.entitlement_year == 2010
    assert exact.nearest_year_used is False

    below = claiming.claim_age_distribution("female", 1950)
    assert below.entitlement_year == claiming.MIN_YEAR
    assert below.nearest_year_used is True

    above = claiming.claim_age_distribution("male", 2035)
    assert above.entitlement_year == claiming.MAX_YEAR
    assert above.nearest_year_used is True

    # Sex aliases resolve to the same row.
    assert (
        claiming.claim_age_distribution("m", 2022).categories
        == claiming.claim_age_distribution("male", 2022).categories
    )
    with pytest.raises(ValueError):
        claiming.claim_age_distribution("other", 2022)


def test_claim_age_pmf_excludes_conversions_and_is_normalised():
    pmf = claiming.claim_age_pmf("male", 2022)
    assert set(pmf) == set(claiming.CLAIM_AGES)
    assert sum(pmf.values()) == pytest.approx(1.0)
    # 67-69 split uniformly.
    assert pmf[67] == pytest.approx(pmf[68]) == pytest.approx(pmf[69])

    # Excluding conversions renormalises the non-conversion mass.
    row = claiming.claim_age_distribution("male", 2022)
    non_conv = sum(
        v for k, v in row.categories.items() if k != "disability_conversion"
    )
    assert pmf[62] == pytest.approx(row.categories["age62"] / non_conv)

    # include=False conserves the conversion mass at the year's at-FRA age.
    pmf_incl = claiming.claim_age_pmf("male", 2022, exclude_conversions=False)
    at_age = row.fra_at["at_age"]
    assert pmf_incl[at_age] > pmf[at_age]
    assert sum(pmf_incl.values()) == pytest.approx(1.0)


def test_conversion_share_matches_reference():
    assert claiming.conversion_share("male", 2022) == pytest.approx(14.5)
    assert claiming.conversion_share("female", 1998) == pytest.approx(9.7)


def test_draw_claim_ages_matches_pmf():
    rng = np.random.default_rng(0)
    n = 200_000
    draws = claiming.draw_claim_ages(rng, "female", 2015, n)
    assert draws.shape == (n,)
    assert set(int(a) for a in np.unique(draws)) <= set(claiming.CLAIM_AGES)

    pmf = claiming.claim_age_pmf("female", 2015)
    for age in claiming.CLAIM_AGES:
        empirical = float(np.mean(draws == age))
        assert empirical == pytest.approx(pmf[age], abs=0.005), age

    # Deterministic for a fixed seed; empty draw is allowed.
    a = claiming.draw_claim_ages(np.random.default_rng(7), "male", 2000, 500)
    b = claiming.draw_claim_ages(np.random.default_rng(7), "male", 2000, 500)
    assert np.array_equal(a, b)
    assert claiming.draw_claim_ages(rng, "male", 2000, 0).shape == (0,)


def test_months_early_late_and_benefit_factor_pure():
    p = _pure_params()
    # Born 1960 -> FRA 67 (804 months).
    assert claiming.months_early(62 * 12, 1960, p) == 60
    assert claiming.months_early(67 * 12, 1960, p) == 0
    assert claiming.months_late(70 * 12, 1960, p) == 36  # capped at 70
    assert claiming.months_late(65 * 12, 1960, p) == 0

    assert claiming.benefit_factor(62 * 12, 1960, p) == pytest.approx(0.70)
    assert claiming.benefit_factor(67 * 12, 1960, p) == pytest.approx(1.0)
    assert claiming.benefit_factor(70 * 12, 1960, p) == pytest.approx(1.24)

    # Monotone non-decreasing in claim age (reduction shrinks, credit
    # grows).
    factors = [
        claiming.benefit_factor(age * 12, 1960, p)
        for age in claiming.CLAIM_AGES
    ]
    assert factors == sorted(factors)


def test_expected_reduction_factor_bounds_and_direction():
    p = _pure_params()
    pmf = claiming.claim_age_pmf("male", 2022)
    lo = min(claiming.benefit_factor(a * 12, 1960, p) for a in pmf)
    hi = max(claiming.benefit_factor(a * 12, 1960, p) for a in pmf)

    erf_1960 = claiming.expected_reduction_factor("male", 2022, 1960, p)
    assert lo <= erf_1960 <= hi
    # 2022 awards are still early-claiming heavy against an FRA of 67.
    assert 0.80 < erf_1960 < 0.92

    # Lower FRA (born 1943 -> 66) means less reduction on the same age
    # mix, so a higher expected factor.
    erf_1943 = claiming.expected_reduction_factor("male", 2022, 1943, p)
    assert erf_1943 > erf_1960


# ==========================================================================
# Tier 1: the held-out reproduction artifact
# ==========================================================================
def test_heldout_schema_and_reported_not_gated():
    art = _heldout()
    assert art["schema_version"] == "claiming_reference.v1"
    assert art["run"] == "claiming_reference_v1"
    assert art["reported_not_gated"] is True
    assert "Changes no gate" in art["purpose"]
    proto = art["protocol"]
    assert proto["fit_years"] == [1998, 2019]
    assert proto["holdout_years"] == [2020, 2021, 2022]
    assert proto["default_rule"] == "nearest_year"
    assert set(art["results"]) == {"nearest_year", "linear_trend"}


def test_heldout_max_recomputes_from_per_cell():
    art = _heldout()
    doc = _reference()
    for _rule, block in art["results"].items():
        cells = block["per_cell"]
        assert block["n_cells"] == len(cells) == 2 * 3 * 8
        abs_devs = [abs(c["deviation"]) for c in cells]
        assert block["max_abs_deviation"] == pytest.approx(max(abs_devs))
        assert block["argmax"]["deviation"] == pytest.approx(
            max(cells, key=lambda c: abs(c["deviation"]))["deviation"]
        )
        # Each cell's actual matches the reference and the deviation is
        # predicted - actual.
        for c in cells:
            ref_val = doc["data"][c["sex"]][str(c["year"])]["categories"][
                c["category"]
            ]
            assert c["actual"] == pytest.approx(ref_val)
            assert c["deviation"] == pytest.approx(
                round(c["predicted"] - c["actual"], 4)
            )


def test_heldout_default_rule_is_nearest_year_and_more_robust():
    art = _heldout()
    head = art["headline"]
    assert head["default_rule"] == "nearest_year"
    # Nearest-year max deviation pins at 3.2 pp; the trend rule is worse
    # (it overshoots the age-66 spike that the FRA-past-66 transition
    # splits in 2021-2022).
    assert head["nearest_year_max_abs_deviation"] == pytest.approx(3.2)
    assert (
        head["linear_trend_max_abs_deviation"]
        > head["nearest_year_max_abs_deviation"]
    )
    assert head["better_rule"] == "nearest_year"


def test_heldout_build_reproduces_committed():
    builder = _import_build_heldout()
    rebuilt = builder.build()
    committed = _heldout()
    for key in committed:
        if key == "build":
            continue
        assert rebuilt[key] == committed[key], key


# ==========================================================================
# Tier 2: policyengine-us oracle consistency
# ==========================================================================
@needs_pe_us
def test_fra_footnote_schedule_matches_oracle():
    """Footnote a's FRA schedule (by attain-65 year) reproduces the
    416(l) oracle (by birth year) under birth = attain65 - 65."""
    from populace_dynamics.ss.params import load_ssa_parameters

    params = load_ssa_parameters()
    doc = _reference()
    schedule = doc["fra_schedule"]["schedule"]
    # Every published bracket boundary agrees with the oracle, over the
    # oracle's covered birth-year domain (its lowest bracket is birth
    # 1900). The lowest schedule entry is a "before 2003" sentinel whose
    # implied birth year predates the oracle; the true FRA-65 boundary
    # (attain-65 year 2002 -> birth 1937) is checked in the sweep below.
    for entry in schedule:
        attain_65 = entry["attain_65_year_from"]
        birth_year = attain_65 - 65
        if birth_year < 1900:
            continue
        assert params.fra_months(birth_year) == entry["fra_months"], (
            attain_65,
            birth_year,
        )
    # And a sweep of individual attain-65 years across the transition.
    footnote_expected = {
        2002: 780,  # before 2003 -> 65
        2003: 782,
        2007: 790,
        2008: 792,  # 66
        2019: 792,
        2020: 794,
        2024: 802,
        2025: 804,  # 67
        2030: 804,
    }
    for attain_65, months in footnote_expected.items():
        assert params.fra_months(attain_65 - 65) == months, attain_65


@needs_pe_us
def test_benefit_factor_matches_ssa_worked_ratios_with_oracle():
    from populace_dynamics.ss import benefits
    from populace_dynamics.ss.params import load_ssa_parameters

    params = load_ssa_parameters()
    # Born 1960 (FRA 67): claim at 62 -> 70% of PIA (matches
    # benefits.age62_monthly_benefit), claim at 70 -> 124%.
    assert claiming.benefit_factor(62 * 12, 1960, params) == pytest.approx(
        0.70
    )
    assert benefits.age62_monthly_benefit(
        1000.0, 1960, params
    ) == pytest.approx(700.0)
    assert claiming.benefit_factor(70 * 12, 1960, params) == pytest.approx(
        1.24
    )
    # Born 1943 (FRA 66): claim at 70 -> 132%.
    assert claiming.benefit_factor(70 * 12, 1943, params) == pytest.approx(
        1.32
    )


@needs_pe_us
def test_expected_reduction_factor_with_oracle_params():
    from populace_dynamics.ss.params import load_ssa_parameters

    params = load_ssa_parameters()
    erf = claiming.expected_reduction_factor("male", 2022, 1960, params)
    assert 0.80 < erf < 0.92
