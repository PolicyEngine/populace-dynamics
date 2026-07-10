"""Tests for the DI->retirement conversion wiring (roadmap #113, M4).

Pure unit tier: the statutory conversion factor and FRA age, the
administrative 6.B5.1 conversion-share re-export, and the partition that
composes with the claiming sampler's conversion exclusion -- all on a
hand-built parameter bundle plus the committed claim-age reference, no
staged microdata and no policyengine-us checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from populace_dynamics import claiming
from populace_dynamics import disability_conversion as dc
from populace_dynamics.ss.params import SSAParameters


def _pure_params() -> SSAParameters:
    """FRA 65/66/67 tiers + the 8%/yr credit -- the claiming test bundle."""
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


# --------------------------------------------------------------------------
# The statutory conversion identity
# --------------------------------------------------------------------------
def test_conversion_factor_is_one():
    assert dc.CONVERSION_BENEFIT_FACTOR == 1.0
    assert dc.conversion_benefit_factor() == 1.0


def test_conversion_age_is_fra():
    p = _pure_params()
    # Born 1960 -> FRA 67 (804 months); born 1943 -> 66 (792).
    assert dc.conversion_claim_age_months(1960, p) == 804
    assert dc.conversion_claim_age_months(1943, p) == 792
    assert dc.conversion_claim_age_months(1930, p) == 780


# --------------------------------------------------------------------------
# Administrative 6.B5.1 conversion share (re-export of claiming)
# --------------------------------------------------------------------------
def test_administrative_share_matches_claiming_reference():
    assert dc.administrative_conversion_share("male", 2022) == pytest.approx(
        14.5
    )
    assert dc.administrative_conversion_share("female", 1998) == pytest.approx(
        9.7
    )
    # Identical to the claiming module's own reading (a thin re-export).
    for sex in ("male", "female"):
        for year in (1998, 2010, 2022):
            assert dc.administrative_conversion_share(
                sex, year
            ) == claiming.conversion_share(sex, year)


# --------------------------------------------------------------------------
# The claim/conversion partition
# --------------------------------------------------------------------------
def _people() -> list[dict]:
    return [
        {"person_id": 1, "sex": "male", "birth_year": 1960, "on_di": True},
        {"person_id": 2, "sex": "male", "birth_year": 1960, "on_di": False},
        {"person_id": 3, "sex": "female", "birth_year": 1955, "on_di": True},
        {"person_id": 4, "sex": "female", "birth_year": 1955, "on_di": False},
    ]


def test_partition_conversions_get_fra_and_unit_factor():
    import pandas as pd

    p = _pure_params()
    out = dc.assign_claim_or_conversion(
        np.random.default_rng(0), pd.DataFrame(_people()), 2015, p
    )
    conv = out[out["claim_kind"] == "conversion"]
    # DI rows convert: benefit factor 1.0 at each cohort's FRA.
    assert set(conv["person_id"]) == {1, 3}
    assert (conv["benefit_factor"] == 1.0).all()
    assert out.loc[out.person_id == 1, "claim_age_months"].item() == 804
    assert out.loc[out.person_id == 3, "claim_age_months"].item() == 792


def test_partition_claimers_draw_valid_ages_and_matching_factor():
    import pandas as pd

    p = _pure_params()
    out = dc.assign_claim_or_conversion(
        np.random.default_rng(1), pd.DataFrame(_people()), 2015, p
    )
    claim = out[out["claim_kind"] == "claim"]
    assert set(claim["person_id"]) == {2, 4}
    for _, row in claim.iterrows():
        months = int(row["claim_age_months"])
        assert months % 12 == 0
        assert 62 * 12 <= months <= 70 * 12
        # The benefit factor is exactly claiming's for that age/cohort.
        assert row["benefit_factor"] == pytest.approx(
            claiming.benefit_factor(months, int(row["birth_year"]), p)
        )


def test_partition_is_deterministic_for_a_seed():
    import pandas as pd

    p = _pure_params()
    frame = pd.DataFrame(_people())
    a = dc.assign_claim_or_conversion(np.random.default_rng(7), frame, 2015, p)
    b = dc.assign_claim_or_conversion(np.random.default_rng(7), frame, 2015, p)
    assert a["claim_age_months"].tolist() == b["claim_age_months"].tolist()
    assert a["benefit_factor"].tolist() == b["benefit_factor"].tolist()


def test_partition_composes_with_claiming_exclusion():
    """Conversions are exactly the on_di rows; claimers draw from the
    conversion-excluded distribution, so the two halves do not overlap and
    together cover everyone (the 6.B5.1 partition)."""
    import pandas as pd

    p = _pure_params()
    rng = np.random.default_rng(3)
    n = 4000
    frame = pd.DataFrame(
        {
            "person_id": np.arange(n),
            "sex": np.where(np.arange(n) % 2 == 0, "male", "female"),
            "birth_year": 1960,
            "on_di": np.arange(n) % 5 == 0,  # 20% on DI
        }
    )
    out = dc.assign_claim_or_conversion(rng, frame, 2020, p)
    is_conv = out["claim_kind"] == "conversion"
    # Partition: conversion iff on_di; disjoint and exhaustive.
    assert (is_conv == out["on_di"]).all()
    assert (out["claim_kind"].isin({"conversion", "claim"})).all()
    # Every claimer age is a claiming.CLAIM_AGES value (conversions excluded
    # from that sampler, supplied here instead).
    claimer_ages = (out.loc[~is_conv, "claim_age_months"] // 12).unique()
    assert set(int(a) for a in claimer_ages) <= set(claiming.CLAIM_AGES)


def test_partition_requires_expected_columns():
    import pandas as pd

    p = _pure_params()
    with pytest.raises(KeyError, match="on_di"):
        dc.assign_claim_or_conversion(
            np.random.default_rng(0),
            pd.DataFrame({"sex": ["male"], "birth_year": [1960]}),
            2015,
            p,
        )
