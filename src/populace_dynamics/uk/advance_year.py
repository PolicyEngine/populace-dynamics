"""One-year composer for the UK panel pipeline.

Ported from archived
`policyengine-uk-data#346 <https://github.com/PolicyEngine/policyengine-uk-data/pull/346>`_
per populace#148 / ADR 0002, onto :class:`~populace_dynamics.uk.dataset.UKPanelDataset`.

Ordering rationale (the sequence is load-bearing, unchanged from the
source):

1. **Migration** first so immigrants joining this year are exposed to
   every subsequent transition, and emigrants leave before they
   accidentally marry / retire / age.
2. **Separations** before marriages so a person who separates this
   year cannot marry in the same step (matches how ONS registers
   marriage / divorce by year).
3. **Children leaving home** next: adult children who move out this
   year are then exposed to marriage as their own benunit.
4. **Marriages** draw on the (now-expanded) single pool.
5. **Employment transitions** redraw labour-market state for anyone
   who survived the composition changes above.
6. **Income-decile transitions** reposition each worker in the income
   distribution.
7. **Demographic ageing** (mortality -> fertility -> age increment)
   last so women who just married can give birth in the same step and
   newborns enter at age 0 and stay 0 at year end.
8. An optional caller-supplied **uprating** hook closes the step.

Differences from the source: this repo has no OBR uprating machinery,
so ``uprate`` is an optional callable ``(dataset, target_year) ->
dataset`` instead of a built-in step; and every person-removal path
inside the ported transitions prunes orphaned benunits/households
(the #346 review fix).

Every transition draws from one seeded generator, so the whole year
is deterministic; every function returns a fresh dataset, so the
caller never sees mutation.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from populace_dynamics.uk.dataset import UKPanelDataset
from populace_dynamics.uk.demographic_ageing import age_dataset
from populace_dynamics.uk.household_transitions import (
    apply_children_leaving_home,
    apply_employment_transitions,
    apply_income_decile_transitions,
    apply_marriages,
    apply_migration,
    apply_separations,
)

__all__ = ["advance_year"]


def advance_year(
    dataset: UKPanelDataset,
    *,
    target_year: int | None = None,
    seed: int = 0,
    mortality_rates: Any = None,
    fertility_rates: Any = None,
    marriage_rates: Any = None,
    separation_rates: Any = None,
    leaving_home_rates: Any = None,
    net_migration_rates: Any = None,
    ukhls_employment_rates: Mapping | None = None,
    ukhls_decile_rates: Mapping | None = None,
    state_pension_age: int = 66,
    job_loss_rate: float = 0.03,
    job_gain_rate: float = 0.05,
    wage_drift: float = 0.04,
    uprate: Callable[[UKPanelDataset, int], UKPanelDataset] | None = None,
) -> UKPanelDataset:
    """Run one full year of the panel pipeline against ``dataset``.

    Args:
        dataset: the starting-year dataset. Not mutated.
        target_year: the year the output represents. ``None`` takes
            ``dataset.fiscal_year + 1``.
        seed: reproducibility seed; every transition draws from the
            same seeded generator sequence.
        mortality_rates, fertility_rates, marriage_rates,
        separation_rates, leaving_home_rates, net_migration_rates:
            passed through to the corresponding ``apply_*`` / ageing
            function. ``None`` uses that function's default table;
            ``{}`` disables the transition.
        ukhls_employment_rates: if provided, replaces the rule-based
            job loss/gain path with age_band x sex x state draws
            (output of
            :func:`populace_dynamics.data.ukhls.load_employment_transitions`).
        ukhls_decile_rates: if provided, adds a decile-transition step
            (output of
            :func:`populace_dynamics.data.ukhls.load_income_decile_transitions`).
        state_pension_age, job_loss_rate, job_gain_rate, wage_drift:
            employment-transition knobs.
        uprate: optional ``(dataset, target_year) -> dataset`` hook
            run at the end of the step (this repo carries no OBR
            uprating tables of its own).

    Returns:
        A new :class:`UKPanelDataset` one year forward, with
        ``fiscal_year`` set to ``target_year``.
    """
    if target_year is None:
        target_year = int(dataset.fiscal_year) + 1

    rng = np.random.default_rng(seed)

    ds = apply_migration(
        dataset,
        net_migration_rates=net_migration_rates,
        rng=rng,
    )
    ds = apply_separations(
        ds,
        separation_rates=separation_rates,
        rng=rng,
    )
    ds = apply_children_leaving_home(
        ds,
        leaving_home_rates=leaving_home_rates,
        rng=rng,
    )
    ds = apply_marriages(
        ds,
        marriage_rates=marriage_rates,
        rng=rng,
    )
    ds = apply_employment_transitions(
        ds,
        state_pension_age=state_pension_age,
        job_loss_rate=job_loss_rate,
        job_gain_rate=job_gain_rate,
        wage_drift=wage_drift,
        ukhls_rates=ukhls_employment_rates,
        rng=rng,
    )
    if ukhls_decile_rates:
        ds = apply_income_decile_transitions(
            ds,
            decile_rates=ukhls_decile_rates,
            rng=rng,
        )
    ds = age_dataset(
        ds,
        years=1,
        seed=int(rng.integers(0, 2**31 - 1)),
        mortality_rates=mortality_rates,
        fertility_rates=fertility_rates,
    )
    if uprate is not None:
        ds = uprate(ds, target_year)
    ds.fiscal_year = int(target_year)
    return ds
