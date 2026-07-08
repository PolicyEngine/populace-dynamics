"""Couple and survivor benefit scoring helpers over the panel (#74).

Phase-C plumbing for the spouse/survivor auxiliary benefits: pure
functions that join a marriage frame (:func:`populace_dynamics.data.
marriage.marriage_episodes`) to *supplied* per-person primary insurance
amounts, claim ages and birth years and produce, for each married pair,

* the **couple benefit** — each spouse's own reduced retirement benefit
  plus the excess spouse's benefit each way (42 USC 402(b)/(c)); and
* the **survivor benefit** — the widow(er)'s benefit for the surviving
  spouse when a marriage ends in the other spouse's death (42 USC
  402(e)/(f), with the RIB-LIM and the survivor's own age reduction).

Design constraints (issue #74, pre-registered bar comment 4907496891):

* **No model dependency.** Every benefit is a pure function of *supplied*
  PIA / claim-age / birth-year inputs — real (from a certified earnings
  history) or generated — plugged in by the caller. This module never
  loads earnings or calls the earnings-history model; it only wires the
  statutory arithmetic in :mod:`populace_dynamics.ss.benefits` (and the
  pinned 402(q)/(w) reduction machinery in
  :mod:`populace_dynamics.claiming`) onto the marriage frame.
* **Not a gate component.** This is statutory encoding + panel wiring,
  validated against SSA worked examples before any use; it is *not*
  scored. Family/household transitions clear the gate-2 lock ceremony
  before any survivor/spousal figure is published (bar, step 3).

Coverage. A pair is *coverable* only when both spouses are joinable PSID
persons (``spouse_person_id`` non-null; see
:mod:`populace_dynamics.data.marriage`) **and** both have a supplied PIA.
:func:`both_spouse_coverage` reports that share; on the staged Release 2
Marriage History File it is 72.0% of the 42,666 actual marriage episodes
(the share whose spouse also has a file record), of which 99.0% also
carry both birth years — an operational both-spouse coverage of 71.3% of
episodes (computed in ``scripts/build_aux_benefit_examples.py``).

Survivor timing simplification. The survivor age reduction uses the
modern survivor full retirement age of 67 (an 84-month reduction span;
:data:`SURVIVOR_FRA_MONTHS`), exact for every survivor cohort born 1962
or later — the model's whole scoring window. See
:attr:`populace_dynamics.ss.params.SSAParameters.
survivor_reduction_period_months`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from populace_dynamics import claiming
from populace_dynamics.ss import benefits
from populace_dynamics.ss.params import SSAParameters

__all__ = [
    "SURVIVOR_EARLIEST_CLAIM_MONTHS",
    "CoupleBenefit",
    "SurvivorBenefit",
    "CoverageReport",
    "survivor_fra_months",
    "survivor_months_early",
    "own_retirement_benefit",
    "excess_spousal_benefit",
    "couple_benefit",
    "survivor_benefit_at_death",
    "both_spouse_coverage",
    "couple_benefits_frame",
    "survivor_benefits_frame",
]

_MONTHS = 12
#: Earliest aged-widow(er) claim age, in months (42 USC 402(e)(1)(B)).
SURVIVOR_EARLIEST_CLAIM_MONTHS = 60 * _MONTHS


# ---------------------------------------------------------------------------
# Scalar helpers — the statutory arithmetic, one couple at a time
# ---------------------------------------------------------------------------
def survivor_fra_months(params: SSAParameters) -> int:
    """Survivor full retirement age in months implied by the reduction
    span — age 60 plus ``survivor_reduction_period_months`` (804 = age
    67 by default; see the module docstring's timing simplification)."""
    return (
        SURVIVOR_EARLIEST_CLAIM_MONTHS
        + params.survivor_reduction_period_months
    )


def survivor_months_early(
    survivor_claim_age_months: int, params: SSAParameters
) -> int:
    """Whole months a survivor claims a widow(er)'s benefit before
    survivor FRA (0 at or after it), the driver of
    :func:`populace_dynamics.ss.benefits.survivor_reduction`."""
    return max(0, survivor_fra_months(params) - int(survivor_claim_age_months))


def own_retirement_benefit(
    own_pia: float,
    claim_age_months: int,
    birth_year: int,
    params: SSAParameters,
) -> float:
    """A worker's own reduced/credited retirement benefit — PIA times the
    pinned 402(q)/(w) benefit factor for their claim age and cohort
    (:func:`populace_dynamics.claiming.benefit_factor`)."""
    return own_pia * claiming.benefit_factor(
        claim_age_months, birth_year, params
    )


def excess_spousal_benefit(
    own_pia: float,
    spouse_pia: float,
    claim_age_months: int,
    birth_year: int,
    params: SSAParameters,
) -> float:
    """The excess spouse's benefit a claimant receives on top of their
    own, given their own claim age and cohort (42 USC 402(b)/(c)).

    Converts the claim age to whole months before the claimant's *own*
    full retirement age (the spouse's benefit reduction runs to the
    claimant's FRA) via :func:`populace_dynamics.claiming.months_early`,
    then applies :func:`populace_dynamics.ss.benefits.spousal_benefit`.
    """
    months_early = claiming.months_early(claim_age_months, birth_year, params)
    return benefits.spousal_benefit(own_pia, spouse_pia, months_early, params)


@dataclass(frozen=True)
class CoupleBenefit:
    """One married pair's benefit decomposition (monthly dollars)."""

    own_a: float
    own_b: float
    excess_spousal_a: float
    excess_spousal_b: float

    @property
    def total(self) -> float:
        """Household total: both own benefits plus both excess spousal."""
        return (
            self.own_a
            + self.own_b
            + self.excess_spousal_a
            + self.excess_spousal_b
        )


def couple_benefit(
    pia_a: float,
    pia_b: float,
    claim_age_months_a: int,
    claim_age_months_b: int,
    birth_year_a: int,
    birth_year_b: int,
    params: SSAParameters,
) -> CoupleBenefit:
    """Benefit for a married couple: own retirement each way plus the
    excess spouse's benefit each way (42 USC 402(a)/(b)/(c), 402(k)).

    Each spouse receives their own reduced/credited retirement benefit
    and, under dual entitlement, the excess of half the *other* spouse's
    PIA over their own PIA (reduced for their own early claiming). The
    excess is typically positive for at most one spouse (the lower
    earner).
    """
    own_a = own_retirement_benefit(
        pia_a, claim_age_months_a, birth_year_a, params
    )
    own_b = own_retirement_benefit(
        pia_b, claim_age_months_b, birth_year_b, params
    )
    exc_a = excess_spousal_benefit(
        pia_a, pia_b, claim_age_months_a, birth_year_a, params
    )
    exc_b = excess_spousal_benefit(
        pia_b, pia_a, claim_age_months_b, birth_year_b, params
    )
    return CoupleBenefit(own_a, own_b, exc_a, exc_b)


@dataclass(frozen=True)
class SurvivorBenefit:
    """A surviving spouse's benefit at the other spouse's death."""

    survivor_own_benefit: float
    deceased_own_factor: float
    survivor_months_early: int
    widow_benefit: float


def survivor_benefit_at_death(
    surviving_own_pia: float,
    deceased_pia: float,
    surviving_claim_age_months: int,
    deceased_claim_age_months: int,
    surviving_birth_year: int,
    deceased_birth_year: int,
    params: SSAParameters,
) -> SurvivorBenefit:
    """Widow(er)'s benefit for a surviving spouse (42 USC 402(e)/(f)).

    Wires the panel inputs into
    :func:`populace_dynamics.ss.benefits.widow_benefit`:

    * the **deceased's own claiming factor** (for the RIB-LIM / credit
      pass-through) comes from
      :func:`populace_dynamics.claiming.benefit_factor` on the deceased's
      claim age and cohort;
    * the **survivor's own benefit** (the dual-entitlement floor) is the
      survivor's own reduced retirement benefit at their claim age; and
    * the **survivor's months early** are measured against the survivor
      FRA (:func:`survivor_months_early`).
    """
    deceased_factor = claiming.benefit_factor(
        deceased_claim_age_months, deceased_birth_year, params
    )
    survivor_own = own_retirement_benefit(
        surviving_own_pia,
        surviving_claim_age_months,
        surviving_birth_year,
        params,
    )
    months_early = survivor_months_early(surviving_claim_age_months, params)
    widow = benefits.widow_benefit(
        survivor_own, deceased_pia, months_early, deceased_factor, params
    )
    return SurvivorBenefit(
        survivor_own_benefit=survivor_own,
        deceased_own_factor=deceased_factor,
        survivor_months_early=months_early,
        widow_benefit=widow,
    )


# ---------------------------------------------------------------------------
# Frame helpers — mapping the scalar arithmetic over the marriage panel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CoverageReport:
    """Both-spouse coverage of a marriage frame under a PIA supply."""

    n_episodes: int
    n_joinable_spouse: int
    n_both_pia: int

    @property
    def joinable_spouse_share(self) -> float:
        """Share of episodes with a joinable spouse person id."""
        return (
            self.n_joinable_spouse / self.n_episodes
            if self.n_episodes
            else 0.0
        )

    @property
    def both_pia_share(self) -> float:
        """Share of episodes where both spouses have a supplied PIA — the
        coverable-pair share this module can actually score."""
        return self.n_both_pia / self.n_episodes if self.n_episodes else 0.0


def _has_pia(pid, pia: Mapping) -> bool:
    if pid is None or pd.isna(pid):
        return False
    value = pia.get(int(pid))
    return value is not None and not pd.isna(value)


def both_spouse_coverage(
    episodes: pd.DataFrame, pia: Mapping
) -> CoverageReport:
    """Report the both-spouse-in-panel coverage of a marriage frame.

    ``episodes`` is a :func:`populace_dynamics.data.marriage.
    marriage_episodes`-shaped frame (needs ``person_id`` and
    ``spouse_person_id``); ``pia`` maps ``person_id`` to a primary
    insurance amount. A pair is coverable when the spouse is joinable
    (``spouse_person_id`` non-null) *and* both persons have a supplied
    PIA. Pure — pass any subset of episodes and any PIA supply.
    """
    n = len(episodes)
    joinable = episodes["spouse_person_id"].notna()
    both = [
        _has_pia(row.person_id, pia) and _has_pia(row.spouse_person_id, pia)
        for row in episodes.itertuples(index=False)
    ]
    return CoverageReport(
        n_episodes=int(n),
        n_joinable_spouse=int(joinable.sum()),
        n_both_pia=int(pd.Series(both).sum()) if n else 0,
    )


def _lookup(pid, table: Mapping):
    return table.get(int(pid))


def couple_benefits_frame(
    episodes: pd.DataFrame,
    pia: Mapping,
    claim_age_months: Mapping,
    birth_year: Mapping,
    params: SSAParameters,
    *,
    unique_pairs: bool = True,
) -> pd.DataFrame:
    """Couple benefits for every coverable married pair in ``episodes``.

    For each episode whose spouse is joinable and where both spouses have
    a supplied PIA, claim age and birth year, computes
    :func:`couple_benefit`. With ``unique_pairs`` (default) each unordered
    couple is emitted once (a marriage is recorded once per spouse in the
    file). Returns a frame with one row per coverable pair and columns
    ``person_id``, ``spouse_person_id``, ``own_a``, ``own_b``,
    ``excess_spousal_a``, ``excess_spousal_b``, ``couple_total``.

    Pure over the supplied maps — no model or data-loading dependency.
    """
    rows = []
    seen: set[frozenset] = set()
    for ep in episodes.itertuples(index=False):
        a, b = ep.person_id, ep.spouse_person_id
        if not (_has_pia(a, pia) and _has_pia(b, pia)):
            continue
        a, b = int(a), int(b)
        if not (
            a in claim_age_months
            and b in claim_age_months
            and a in birth_year
            and b in birth_year
        ):
            continue
        if unique_pairs:
            key = frozenset((a, b))
            if key in seen:
                continue
            seen.add(key)
        cb = couple_benefit(
            _lookup(a, pia),
            _lookup(b, pia),
            _lookup(a, claim_age_months),
            _lookup(b, claim_age_months),
            _lookup(a, birth_year),
            _lookup(b, birth_year),
            params,
        )
        rows.append(
            {
                "person_id": a,
                "spouse_person_id": b,
                "own_a": cb.own_a,
                "own_b": cb.own_b,
                "excess_spousal_a": cb.excess_spousal_a,
                "excess_spousal_b": cb.excess_spousal_b,
                "couple_total": cb.total,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "person_id",
            "spouse_person_id",
            "own_a",
            "own_b",
            "excess_spousal_a",
            "excess_spousal_b",
            "couple_total",
        ],
    )


def survivor_benefits_frame(
    episodes: pd.DataFrame,
    pia: Mapping,
    claim_age_months: Mapping,
    birth_year: Mapping,
    params: SSAParameters,
    *,
    survivor_claim_age_months: Mapping | None = None,
) -> pd.DataFrame:
    """Survivor benefits for every widowhood-ending coverable pair.

    Restricts ``episodes`` to those with ``how_ended == "widowhood"`` —
    where ``person_id`` is the survivor and ``spouse_person_id`` the
    deceased — and, for each coverable pair, computes
    :func:`survivor_benefit_at_death`. The survivor's widow-claim age
    defaults to their supplied ``claim_age_months`` (a documented proxy;
    pass ``survivor_claim_age_months`` for a distinct widow-claim age).

    Returns one row per coverable survivor with columns ``person_id``
    (survivor), ``spouse_person_id`` (deceased), ``survivor_own_benefit``,
    ``deceased_own_factor``, ``survivor_months_early`` and
    ``widow_benefit``. Pure over the supplied maps.
    """
    widow_claim = survivor_claim_age_months or claim_age_months
    ended = episodes[episodes["how_ended"] == "widowhood"]
    rows = []
    for ep in ended.itertuples(index=False):
        survivor, deceased = ep.person_id, ep.spouse_person_id
        if not (_has_pia(survivor, pia) and _has_pia(deceased, pia)):
            continue
        survivor, deceased = int(survivor), int(deceased)
        if not (
            survivor in widow_claim
            and deceased in claim_age_months
            and survivor in birth_year
            and deceased in birth_year
        ):
            continue
        sb = survivor_benefit_at_death(
            _lookup(survivor, pia),
            _lookup(deceased, pia),
            _lookup(survivor, widow_claim),
            _lookup(deceased, claim_age_months),
            _lookup(survivor, birth_year),
            _lookup(deceased, birth_year),
            params,
        )
        rows.append(
            {
                "person_id": survivor,
                "spouse_person_id": deceased,
                "survivor_own_benefit": sb.survivor_own_benefit,
                "deceased_own_factor": sb.deceased_own_factor,
                "survivor_months_early": sb.survivor_months_early,
                "widow_benefit": sb.widow_benefit,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "person_id",
            "spouse_person_id",
            "survivor_own_benefit",
            "deceased_own_factor",
            "survivor_months_early",
            "widow_benefit",
        ],
    )
