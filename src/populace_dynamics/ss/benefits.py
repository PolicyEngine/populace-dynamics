"""AIME, PIA, and claiming-age adjustment (42 USC 415, 402(q)).

Pure formula machinery over :class:`~populace_dynamics.ss.params.
SSAParameters`. Statute references:

* 415(b): AIME — creditable earnings capped at the contribution and
  benefit base, indexed by NAWI to the year the worker attains age
  60 (unindexed thereafter), highest-35 selection for retirement,
  divided by 420 months, truncated to the dollar.
* 415(a): PIA — 90/32/15 percent brackets over the eligibility
  year's bend points; 415(g) rounds the total down to the next
  lower multiple of ten cents.
* 402(q): early-claiming reduction — 5/9 of one percent per month
  for the first 36 months before full retirement age, 5/12 of one
  percent per month beyond.
* 402(w): delayed-retirement credit — an increase per month of delay
  past full retirement age at the birth cohort's annual rate (8 percent
  per year for 1943 and later), stopping at age 70.
"""

from __future__ import annotations

import math

from populace_dynamics.ss.params import SSAParameters

__all__ = [
    "creditable_history",
    "indexed_history",
    "aime",
    "pia",
    "early_reduction",
    "delayed_credit",
    "age62_monthly_benefit",
]

_COMPUTATION_YEARS = 35
_MONTHS = 12
_INDEXING_AGE = 60
_EARLIEST_CLAIM_MONTHS = 62 * 12
_AGE_70_MONTHS = 70 * 12


def creditable_history(
    history: dict[int, float], params: SSAParameters
) -> dict[int, float]:
    """Cap each year's earnings at that year's wage base."""
    return {
        year: min(float(earnings), params.wage_base_for(year))
        for year, earnings in history.items()
    }


def indexed_history(
    history: dict[int, float],
    birth_year: int,
    params: SSAParameters,
) -> dict[int, float]:
    """Index creditable earnings by NAWI to the age-60 year.

    Years at or after the indexing year enter at nominal value.
    """
    indexing_year = birth_year + _INDEXING_AGE
    if indexing_year not in params.nawi:
        raise KeyError(f"NAWI for indexing year {indexing_year} unavailable.")
    base = params.nawi[indexing_year]
    out = {}
    for year, earnings in history.items():
        if year >= indexing_year:
            out[year] = earnings
        else:
            out[year] = earnings * base / params.nawi[year]
    return out


def aime(
    history: dict[int, float],
    birth_year: int,
    params: SSAParameters,
) -> float:
    """Average indexed monthly earnings from a nominal history.

    ``history`` maps calendar year to that year's labor earnings.
    Years absent from the mapping contribute zero if selected —
    callers are responsible for only passing histories whose
    coverage supports that reading (415(b) counts every computation
    year, earnings or not).
    """
    creditable = creditable_history(history, params)
    indexed = indexed_history(creditable, birth_year, params)
    top = sorted(indexed.values(), reverse=True)[:_COMPUTATION_YEARS]
    top += [0.0] * (_COMPUTATION_YEARS - len(top))
    return math.floor(sum(top) / (_COMPUTATION_YEARS * _MONTHS))


def pia(
    aime_value: float, eligibility_year: int, params: SSAParameters
) -> float:
    """Primary insurance amount at eligibility (415(a), 415(g))."""
    first, second = params.bend_points(eligibility_year)
    f1, f2, f3 = params.pia_factors
    amount = (
        f1 * min(aime_value, first)
        + f2 * max(0.0, min(aime_value, second) - first)
        + f3 * max(0.0, aime_value - second)
    )
    return math.floor(amount * 10.0 + 1e-9) / 10.0


def early_reduction(months_early: int, params: SSAParameters) -> float:
    """Fractional reduction for claiming ``months_early`` months
    before full retirement age (402(q))."""
    if months_early <= 0:
        return 0.0
    first_rate, later_rate = params.early_monthly_rates
    cap = params.early_first_bracket_months
    first = min(months_early, cap)
    later = max(0, months_early - cap)
    return first * first_rate + later * later_rate


def delayed_credit(
    months_late: int, birth_year: int, params: SSAParameters
) -> float:
    """Fractional delayed-retirement credit for claiming ``months_late``
    months after full retirement age (42 USC 402(w)).

    Credits accrue at the birth cohort's annual rate (a twelfth per
    month) and stop at age 70; the accrual window is additionally capped
    at the statutory maximum (``max_delayed_years``). Returns 0 for a
    claim at or before full retirement age.
    """
    if months_late <= 0:
        return 0.0
    fra = params.fra_months(birth_year)
    window = min(params.max_delayed_months, _AGE_70_MONTHS - fra)
    credited = min(months_late, window)
    if credited <= 0:
        return 0.0
    return (credited / _MONTHS) * params.delayed_credit_annual_rate(birth_year)


def age62_monthly_benefit(
    pia_value: float, birth_year: int, params: SSAParameters
) -> float:
    """Monthly benefit for a worker claiming at exactly age 62.

    Uses the full-retirement-age schedule (416(l)) and the 402(q)
    reduction. The one-month eligibility subtlety for persons born
    on the first two days of a month is ignored — a documented
    simplification of at most one month's reduction factor.
    """
    months_early = params.fra_months(birth_year) - _EARLIEST_CLAIM_MONTHS
    reduction = early_reduction(months_early, params)
    return pia_value * (1.0 - reduction)
