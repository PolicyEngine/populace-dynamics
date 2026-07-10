"""DI -> retirement conversion at the full retirement age, and its wiring
to the claiming sampler (roadmap #113, M4; task counterpart to B2).

The statutory fact
------------------
A disabled worker's benefit equals the primary insurance amount with no
early-retirement reduction (42 USC 423(a)(2), 423(e)). At the full
retirement age (FRA) the disability benefit **automatically converts** to
an old-age (retirement) benefit of the same amount -- the worker is
deemed to have attained retirement age and the disability benefit ends
(42 USC 402(a), 416(i)(2)(D); SSA POMS DI 13005.010, "the conversion is
automatic"). Because there is no age reduction on the converted benefit
and no delayed-retirement credit is earned while on DI, the
benefit-to-PIA factor of a conversion is exactly ``1.0``.

This is precisely why *Annual Statistical Supplement* Table 6.B5.1
carries disability conversions in their **own column** (footnote b: they
"automatically convert" and are "not a claiming choice"), and why
:mod:`populace_dynamics.claiming` **excludes** them from its claim-age
sampler (:func:`claiming.claim_age_pmf` with ``exclude_conversions=True``,
the default). Those two facts are the two halves of one partition:

    every retired-worker benefit entrant is EITHER a choice-claimer
    (age drawn from the 6.B5.1 age-at-entitlement distribution, with
    conversions removed) OR a DI -> retirement conversion at FRA.

The claiming module owns the first half; this module owns the second and
:func:`assign_claim_or_conversion` composes them so their union recovers
the whole entrant population. The DI-conversion population itself is the
one supplied by the M4 disability hazards
(:mod:`populace_dynamics.data.disability`): the disabled-worker stock
that reaches FRA.

Statute-cited, like the auxiliary constants
-------------------------------------------
policyengine-us has no DI-conversion formula (SSDI is an uprated survey
input there; see :mod:`populace_dynamics.ss.params`), so the conversion
factor is carried here as a constant of the statute -- the same
STATUTE-CITED precedent as the 402(b)/(c)/(e)/(f) auxiliary rates and the
claiming module's at-FRA factor. The FRA itself is NOT re-implemented: it
is read from the pinned 416(l) schedule
(:meth:`populace_dynamics.ss.params.SSAParameters.fra_months`).

v1 is mechanical: whether a worker is on DI at FRA is an input (from the
disability stock), not a modeled labor-supply response. Behavioral DI
entry plugs in later behind its own gate, stated as a domain-of-validity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from populace_dynamics import claiming
from populace_dynamics.ss.params import SSAParameters

__all__ = [
    "CONVERSION_BENEFIT_FACTOR",
    "conversion_claim_age_months",
    "conversion_benefit_factor",
    "administrative_conversion_share",
    "assign_claim_or_conversion",
]

#: Benefit-to-PIA factor of a DI -> retirement conversion: exactly 1.0
#: (the disability benefit is PIA with no age reduction, and it converts
#: at FRA to a retirement benefit of the same amount -- no 402(q)
#: reduction, no 402(w) credit). 42 USC 423(a)(2) / 402(a) / 416(i)(2)(D).
CONVERSION_BENEFIT_FACTOR = 1.0


def conversion_claim_age_months(birth_year: int, params: SSAParameters) -> int:
    """Age (in months) at which a DI benefit converts: the FRA (416(l)).

    Read from the pinned oracle schedule, never duplicated here.
    """
    return params.fra_months(birth_year)


def conversion_benefit_factor() -> float:
    """The benefit-to-PIA factor of a conversion: ``1.0`` (see module).

    A function (not just the constant) so callers read it the same way
    they read :func:`populace_dynamics.claiming.benefit_factor` for a
    choice-claimer.
    """
    return CONVERSION_BENEFIT_FACTOR


def administrative_conversion_share(
    sex: str,
    entitlement_year: int,
    *,
    reference: claiming.ClaimAgeReference | None = None,
) -> float:
    """SSA 6.B5.1 disability-conversion share (percent) for a (sex, year).

    A thin, documented re-export of
    :func:`populace_dynamics.claiming.conversion_share` so the
    administrative auto-conversion share -- conversions as a percentage
    of that year's retired-worker awards -- sits beside the conversion
    wiring it validates. This is the ADMINISTRATIVE concept (an award
    flow among insured workers); the PSID analog
    (:func:`populace_dynamics.data.disability.conversion_cells`) is a
    self-reported-labor-force transition and the two are compared with
    their concept delta named, never equated.
    """
    return claiming.conversion_share(
        sex, entitlement_year, reference=reference
    )


def assign_claim_or_conversion(
    rng: np.random.Generator,
    people: pd.DataFrame,
    entitlement_year: int,
    params: SSAParameters,
    *,
    reference: claiming.ClaimAgeReference | None = None,
) -> pd.DataFrame:
    """Partition retirement-benefit entrants into conversions and claims.

    ``people`` needs columns ``sex``, ``birth_year`` and ``on_di`` (bool:
    on a disability benefit as FRA is reached). Returns a copy with three
    added columns:

    * ``claim_kind`` -- ``"conversion"`` for ``on_di`` rows, else
      ``"claim"``;
    * ``claim_age_months`` -- FRA (:func:`conversion_claim_age_months`)
      for a conversion; for a claim, an integer claim age (in months)
      drawn from :func:`populace_dynamics.claiming.draw_claim_ages` for
      the row's ``(sex, entitlement_year)`` with conversions **excluded**
      (so the two halves do not double-count);
    * ``benefit_factor`` -- :data:`CONVERSION_BENEFIT_FACTOR` (1.0) for a
      conversion, else :func:`populace_dynamics.claiming.benefit_factor`
      at the drawn age and the row's birth cohort.

    The conversion population is an INPUT (the disabled stock reaching
    FRA), not a modeled response. Drawing the choice-claimers from the
    conversion-excluded distribution and appending the conversions here
    reconstitutes the full 6.B5.1 entrant mix; the resulting conversion
    share is what the artifact compares to
    :func:`administrative_conversion_share`.
    """
    for col in ("sex", "birth_year", "on_di"):
        if col not in people.columns:
            raise KeyError(f"people is missing required column {col!r}.")

    out = people.copy().reset_index(drop=True)
    n = len(out)
    claim_kind = np.where(
        out["on_di"].to_numpy(dtype=bool), "conversion", "claim"
    )
    birth_year = out["birth_year"].astype("int64").to_numpy()
    fra_months = np.array(
        [params.fra_months(int(by)) for by in birth_year], dtype="int64"
    )

    claim_age_months = np.zeros(n, dtype="int64")
    benefit_factor = np.zeros(n, dtype="float64")

    is_conv = out["on_di"].to_numpy(dtype=bool)
    claim_age_months[is_conv] = fra_months[is_conv]
    benefit_factor[is_conv] = CONVERSION_BENEFIT_FACTOR

    # Choice-claimers: draw ages per sex group (the distribution is keyed
    # by sex x year), then map each to its 402(q)/(w) benefit factor.
    claimers = ~is_conv
    for sex in ("female", "male"):
        mask = claimers & (out["sex"].to_numpy() == sex)
        k = int(mask.sum())
        if k == 0:
            continue
        ages = claiming.draw_claim_ages(
            rng,
            sex,
            entitlement_year,
            k,
            exclude_conversions=True,
            reference=reference,
        )
        claim_age_months[mask] = ages.astype("int64") * 12
        idx = np.flatnonzero(mask)
        for j, months in zip(idx, claim_age_months[mask], strict=True):
            benefit_factor[j] = claiming.benefit_factor(
                int(months), int(birth_year[j]), params
            )

    out["claim_kind"] = claim_kind
    out["claim_age_months"] = claim_age_months
    out["benefit_factor"] = benefit_factor
    return out
