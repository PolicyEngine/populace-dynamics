"""Couple/survivor panel scoring helpers (#74, Phase C).

Pure tests over synthetic marriage frames and supplied PIA/claim/birth
maps — :mod:`populace_dynamics.household` never loads data or a model, so
these need no checkout. The scalar arithmetic is validated in
``tests/ss/test_aux_benefits.py``; here we test the panel *wiring*:
coverage accounting, the unordered-pair dedupe, the widowhood restriction,
and the survivor-FRA timing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from populace_dynamics import household
from populace_dynamics.ss import benefits
from populace_dynamics.ss.params import SSAParameters


def _pure_params(**overrides) -> SSAParameters:
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


def _episodes() -> pd.DataFrame:
    """Four episodes: an intact couple (1,2) recorded from both sides, a
    widowhood (3 survives 4), and a marriage to a non-joinable spouse."""
    return pd.DataFrame(
        [
            dict(person_id=1, spouse_person_id=2, how_ended="intact"),
            dict(person_id=2, spouse_person_id=1, how_ended="intact"),
            dict(person_id=3, spouse_person_id=4, how_ended="widowhood"),
            dict(person_id=5, spouse_person_id=pd.NA, how_ended="divorce"),
        ]
    )


# ==========================================================================
# Survivor-FRA timing
# ==========================================================================
def test_survivor_fra_and_months_early():
    p = _pure_params()
    # Default span 84 -> survivor FRA 67 (804 months).
    assert household.survivor_fra_months(p) == 804
    assert household.survivor_months_early(60 * 12, p) == 84  # claim at 60
    assert household.survivor_months_early(67 * 12, p) == 0  # claim at FRA
    assert household.survivor_months_early(70 * 12, p) == 0  # at/after FRA
    # FRA-66 survivor bundle: span 72 -> FRA 66 (792 months).
    p66 = _pure_params(survivor_reduction_period_months=72)
    assert household.survivor_fra_months(p66) == 792
    assert household.survivor_months_early(60 * 12, p66) == 72


# ==========================================================================
# Scalar couple / survivor helpers
# ==========================================================================
def test_couple_benefit_decomposition():
    p = _pure_params()
    # Worker PIA 2000 claiming at FRA 67; spouse PIA 800 claiming at 62.
    cb = household.couple_benefit(
        2000.0, 800.0, 67 * 12, 62 * 12, 1960, 1960, p
    )
    assert cb.own_a == pytest.approx(2000.0)  # at FRA
    assert cb.own_b == pytest.approx(800.0 * 0.70)  # 62, FRA 67 -> 70%
    # Higher earner gets no excess; lower earner's excess = (1000-800)*0.65.
    assert cb.excess_spousal_a == pytest.approx(0.0)
    assert cb.excess_spousal_b == pytest.approx((1000.0 - 800.0) * 0.65)
    assert cb.total == pytest.approx(2000.0 + 560.0 + 0.0 + (200.0 * 0.65))


def test_survivor_benefit_at_death_wires_widow_benefit():
    p = _pure_params()
    # Deceased (born 1943, FRA 66) claimed at 62 -> 0.75 factor; survivor
    # (born 1960) claims widow at survivor FRA 67; own PIA 500.
    sb = household.survivor_benefit_at_death(
        500.0, 2000.0, 67 * 12, 62 * 12, 1960, 1943, p
    )
    assert sb.deceased_own_factor == pytest.approx(0.75)
    assert sb.survivor_months_early == 0
    # RIB-LIM: deceased 0.75*2000=1500 vs 0.825*2000=1650 -> 1650; survivor
    # at FRA, own 500 < 1650 -> widow benefit 1650.
    assert sb.widow_benefit == pytest.approx(1650.0)
    # Consistency with the low-level function.
    assert sb.widow_benefit == pytest.approx(
        benefits.widow_benefit(sb.survivor_own_benefit, 2000.0, 0, 0.75, p)
    )


# ==========================================================================
# Coverage accounting
# ==========================================================================
def test_both_spouse_coverage():
    ep = _episodes()
    pia = {1: 2000.0, 2: 800.0, 3: 1500.0, 4: 2200.0, 5: 1000.0}
    cov = household.both_spouse_coverage(ep, pia)
    assert cov.n_episodes == 4
    assert cov.n_joinable_spouse == 3  # episode 5 has NA spouse
    assert cov.n_both_pia == 3  # (1,2),(2,1),(3,4) all have both PIAs
    assert cov.joinable_spouse_share == pytest.approx(0.75)
    assert cov.both_pia_share == pytest.approx(0.75)
    # Drop person 4's PIA: the widowhood pair is no longer coverable.
    partial = {1: 2000.0, 2: 800.0, 3: 1500.0, 5: 1000.0}
    cov2 = household.both_spouse_coverage(ep, partial)
    assert cov2.n_both_pia == 2  # only (1,2) and (2,1)
    # Empty frame is safe.
    empty = household.both_spouse_coverage(ep.iloc[:0], pia)
    assert empty.n_episodes == 0
    assert empty.both_pia_share == 0.0


# ==========================================================================
# Frame helpers
# ==========================================================================
def test_couple_benefits_frame_dedupes_and_filters():
    p = _pure_params()
    ep = _episodes()
    pia = {1: 2000.0, 2: 800.0, 3: 1500.0, 4: 2200.0, 5: 1000.0}
    claim = {1: 67 * 12, 2: 62 * 12, 3: 66 * 12, 4: 62 * 12, 5: 65 * 12}
    birth = {1: 1960, 2: 1960, 3: 1958, 4: 1958, 5: 1962}
    frame = household.couple_benefits_frame(ep, pia, claim, birth, p)
    # Intact couple (1,2) appears once (deduped), plus (3,4): two pairs.
    assert len(frame) == 2
    assert set(
        map(
            frozenset,
            zip(frame.person_id, frame.spouse_person_id, strict=True),
        )
    ) == {
        frozenset((1, 2)),
        frozenset((3, 4)),
    }
    row12 = frame[frame.person_id == 1].iloc[0]
    assert row12.couple_total == pytest.approx(2000.0 + 560.0 + (200.0 * 0.65))
    # unique_pairs=False keeps both recordings of the intact marriage.
    both = household.couple_benefits_frame(
        ep, pia, claim, birth, p, unique_pairs=False
    )
    assert len(both) == 3

    # A missing birth year drops that pair.
    birth_missing = dict(birth)
    del birth_missing[4]
    frame2 = household.couple_benefits_frame(ep, pia, claim, birth_missing, p)
    assert len(frame2) == 1


def test_survivor_benefits_frame_restricts_to_widowhood():
    p = _pure_params()
    ep = _episodes()
    pia = {1: 2000.0, 2: 800.0, 3: 1500.0, 4: 2200.0, 5: 1000.0}
    claim = {1: 67 * 12, 2: 62 * 12, 3: 66 * 12, 4: 62 * 12, 5: 65 * 12}
    birth = {1: 1960, 2: 1960, 3: 1958, 4: 1958, 5: 1962}
    frame = household.survivor_benefits_frame(ep, pia, claim, birth, p)
    # Only the widowhood episode (3 survives 4).
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.person_id == 3 and row.spouse_person_id == 4
    # Deceased 4 (born 1958, FRA 66y8m) claimed at 62.
    assert 0.7 < row.deceased_own_factor < 0.8
    # Widow benefit matches the scalar helper.
    sb = household.survivor_benefit_at_death(
        1500.0, 2200.0, 66 * 12, 62 * 12, 1958, 1958, p
    )
    assert row.widow_benefit == pytest.approx(sb.widow_benefit)
    # A distinct widow-claim-age map is honoured.
    frame2 = household.survivor_benefits_frame(
        ep, pia, claim, birth, p, survivor_claim_age_months={3: 60 * 12}
    )
    assert frame2.iloc[0].survivor_months_early == 84  # claimed widow at 60
