"""AIME, PIA, claiming-age adjustment, and the spouse/survivor
auxiliary benefits (42 USC 415, 402(b)/(c)/(e)/(f)/(q)/(w)/(k)).

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
  percent per month beyond (the worker's own reduction). The spousal
  first-bracket rate is 25/36 of one percent (:func:`spousal_benefit`);
  the widow(er) reduction is a floor-anchored ramp to 71.5% at age 60
  (:func:`survivor_reduction`).
* 402(w): delayed-retirement credit — an increase per month of delay
  past full retirement age at the birth cohort's annual rate (8 percent
  per year for 1943 and later), stopping at age 70.
* 402(b)/(c): spouse's insurance benefit — one-half of the worker's
  PIA, offset by the spouse's own PIA (dual entitlement), reduced for
  the spouse's early claiming (:func:`spousal_benefit`).
* 402(e)/(f): widow(er)'s insurance benefit — 100% of the deceased's
  PIA at survivor FRA, reduced to a 71.5% floor at age 60, capped by
  the RIB-LIM when the deceased claimed early, and paid as the larger
  of the survivor's own or widow(er)'s benefit (:func:`widow_benefit`).

Scope note. This module remains the frozen Python oracle (see
``ss/__init__.py``): these auxiliary functions encode the statute as a
validated reference, cross-checked against SSA's published worked
examples and — for the PIA foundation they build on — against a live
policyengine-us Simulation. They are NOT a gate component; the
pre-registered bar (issue #74) requires validation before any scored
use, and family/household transitions clear the gate-2 ceremony first.
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
    "spousal_early_reduction",
    "spousal_benefit",
    "survivor_reduction",
    "widow_benefit",
    "widow_benefit_survives_remarriage",
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


# ---------------------------------------------------------------------------
# Spouse's insurance benefit — 42 USC 402(b) (wife's) / 402(c) (husband's)
# ---------------------------------------------------------------------------
def spousal_early_reduction(months_early: int, params: SSAParameters) -> float:
    """Fractional reduction of a *spouse's* benefit for claiming
    ``months_early`` months before full retirement age (42 USC 402(q)(1)).

    Structurally identical to :func:`early_reduction` but on the spousal
    rate schedule: 25/36 of one percent per month for the first 36
    months (a steeper first bracket than the worker's own 5/9 of one
    percent) and 5/12 of one percent per month beyond. At the maximum 60
    months early (claim at 62 against an FRA of 67) this is
    ``36·25/36% + 24·5/12% = 25% + 10% = 35%``.
    """
    if months_early <= 0:
        return 0.0
    first_rate, later_rate = params.spousal_early_monthly_rates
    cap = params.spousal_early_first_bracket_months
    first = min(months_early, cap)
    later = max(0, months_early - cap)
    return first * first_rate + later * later_rate


def spousal_benefit(
    own_pia: float,
    spouse_pia: float,
    months_early: int,
    params: SSAParameters,
) -> float:
    """Excess spouse's insurance benefit (42 USC 402(b)/(c), 402(k)(3)).

    Returns the auxiliary amount payable *on top of* the claimant's own
    retirement benefit under dual entitlement — i.e. the excess of
    one-half of the worker's PIA over the claimant's own PIA, reduced for
    the claimant's early claiming:

    * ``own_pia`` — the claiming spouse's own primary insurance amount
      (0 for a spouse with no covered work of their own).
    * ``spouse_pia`` — the *worker* spouse's PIA (the benefit is one-half
      of this, ``params.spousal_pia_share``, per 402(b)(2)/(c)(2)).
    * ``months_early`` — whole months the claiming spouse is claiming
      before their own full retirement age (0 at or after FRA).

    The full spouse-side amount a household receives is the claimant's
    own (reduced) retirement benefit *plus* this excess; the dual-
    entitlement offset (402(k)(3)(A)) is applied before the age
    reduction, so a spouse with own PIA at least half the worker's PIA
    receives no excess. When ``own_pia`` is 0 this reduces to the
    headline figures — 50% of the worker's PIA at FRA, 32.5% at age 62
    against an FRA of 67.
    """
    base = params.spousal_pia_share * spouse_pia
    excess = max(0.0, base - own_pia)
    reduction = spousal_early_reduction(months_early, params)
    return excess * (1.0 - reduction)


# ---------------------------------------------------------------------------
# Widow(er)'s insurance benefit — 42 USC 402(e)/(f), 402(q), RIB-LIM 402(k)
# ---------------------------------------------------------------------------
def survivor_reduction(
    survivor_months_early: int, params: SSAParameters
) -> float:
    """Fractional reduction of a widow(er)'s benefit for claiming
    ``survivor_months_early`` months before survivor full retirement age
    (42 USC 402(q); SSA POMS RS 00615.302).

    Unlike the worker's fixed per-month rates, the widow(er) reduction is
    floor-anchored: the maximum reduction (``1 −
    params.survivor_reduction_floor`` = 28.5%, giving the 71.5% floor at
    age 60) is spread linearly across the
    ``params.survivor_reduction_period_months`` months from age 60 to
    survivor FRA. So the per-month rate depends on the span — 0.285/84
    for a survivor FRA of 67, 0.285/72 for 66 — which is why it is
    encoded as a ramp rather than a constant. Clamped at the floor.
    """
    if survivor_months_early <= 0:
        return 0.0
    max_reduction = 1.0 - params.survivor_reduction_floor
    period = params.survivor_reduction_period_months
    capped = min(survivor_months_early, period)
    return max_reduction * capped / period


def widow_benefit(
    own_pia: float,
    deceased_pia: float,
    survivor_months_early: int,
    deceased_claimed_early_factor: float,
    params: SSAParameters,
) -> float:
    """Widow(er)'s insurance benefit (42 USC 402(e)/(f), 402(q), 402(k)).

    Arguments:
        own_pia: the survivor's own benefit amount for the dual-
            entitlement comparison — their own primary insurance amount,
            or their own *reduced* retirement benefit if they have
            already claimed it (the caller decides; see
            :mod:`populace_dynamics.household`).
        deceased_pia: the deceased worker's primary insurance amount.
        survivor_months_early: whole months the survivor claims before
            *survivor* full retirement age (0 at or after it). Drives the
            71.5%-floor ramp (:func:`survivor_reduction`).
        deceased_claimed_early_factor: the 402(q)/(w) benefit-to-PIA
            factor the deceased locked in by their own claiming age
            (< 1 if they claimed a reduced retirement benefit before
            FRA, > 1 if they earned delayed-retirement credits). Drives
            the RIB-LIM and the credit pass-through.

    The computation, per SSA POMS RS 00615.300–.320:

    1. **Base.** 100% of the deceased's PIA (``survivor_pia_share``) is
       the maximum widow(er)'s benefit at survivor FRA. If the deceased
       had *delayed*-retirement credits (factor > 1), the widow(er)
       inherits them, so the base is the deceased's actual
       (credit-enhanced) benefit.
    2. **RIB-LIM (402(e)(2)(D)/(k)(3)(A)).** If the deceased took a
       *reduced* retirement benefit (factor < 1), the widow(er)'s benefit
       is capped at the larger of the deceased's actual benefit or 82.5%
       of the deceased's PIA (``rib_lim_pia_share``).
    3. **Survivor's own age reduction.** The base is reduced by the
       survivor's own early-claiming ramp to the 71.5% floor at age 60.
       The RIB-LIM ceiling then still binds.
    4. **Dual entitlement (402(k)).** The survivor is paid the larger of
       their own benefit or the widow(er)'s benefit.
    """
    deceased_benefit = deceased_pia * deceased_claimed_early_factor
    took_reduced_rib = deceased_claimed_early_factor < 1.0

    if took_reduced_rib:
        # RIB-LIM ceiling: larger of the deceased's actual benefit or
        # 82.5% of the deceased's PIA.
        rib_lim_ceiling = max(
            deceased_benefit, params.rib_lim_pia_share * deceased_pia
        )
        widow_base = params.survivor_pia_share * deceased_pia
    else:
        # No RIB-LIM; any delayed-retirement credits pass through.
        rib_lim_ceiling = None
        widow_base = params.survivor_pia_share * deceased_benefit

    survivor_factor = 1.0 - survivor_reduction(survivor_months_early, params)
    widow_reduced = widow_base * survivor_factor
    if rib_lim_ceiling is not None:
        widow_reduced = min(widow_reduced, rib_lim_ceiling)

    return max(own_pia, widow_reduced)


def widow_benefit_survives_remarriage(
    remarriage_age: float,
    params: SSAParameters,
    *,
    disabled: bool = False,
) -> bool:
    """Whether a widow(er)'s benefit survives a remarriage (402(e)(3)/(f)(4)).

    Remarriage at or after age 60 (age 50 if the surviving spouse is
    disabled) does not terminate entitlement to a widow(er)'s benefit;
    remarriage before that age does (until that later marriage itself
    ends). Returns True if the benefit is preserved.
    """
    threshold = (
        params.remarriage_protected_age_disabled
        if disabled
        else params.remarriage_protected_age
    )
    return remarriage_age >= threshold
