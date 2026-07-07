"""Claiming age: the SSA Statistical Supplement reference and the
benefit-adjustment factor it implies (task B2 of the replication
program, #74).

This module reads the committed reference built by
``scripts/build_ssa_claim_ages.py`` from *Annual Statistical
Supplement, 2023*, Table 6.B5.1 -- the percentage distribution of
retired-worker awardees by age at month of entitlement, by sex and
entitlement year, 1998-2022 -- and turns it into three things the
reform-scoring stack needs:

* the category shares for a (sex, entitlement year), with a documented
  nearest-year rule for out-of-range requests
  (:func:`claim_age_distribution`);
* a sampler over integer claim ages (:func:`draw_claim_ages`), which
  **excludes disability conversions** -- an auto-conversion at full
  retirement age (FRA), not a claiming choice (footnote b) -- and
  exposes their share separately (:func:`conversion_share`);
* the months a claim falls before or after FRA
  (:func:`months_early`/:func:`months_late`) and the resulting expected
  benefit-to-PIA factor (:func:`expected_reduction_factor`).

The FRA schedule and the 402(q)/402(w) actuarial adjustments are **not**
re-implemented here: :func:`months_early`/:func:`months_late` read FRA
from :meth:`populace_dynamics.ss.params.SSAParameters.fra_months`
(416(l), pinned to policyengine-us), and the factor reuses
:func:`populace_dynamics.ss.benefits.early_reduction` (402(q)) and
:func:`populace_dynamics.ss.benefits.delayed_credit` (402(w)).

Validation standard (stated here before first scored use; NOT a locked
gate -- see ``runs/claiming_reference_v1.json``):

1. **Reference reproduction (exact).** The committed JSON is reproduced
   bit-for-bit by the build script (modulo its build timestamp), and
   published spot-check cells match the Supplement to the digit.
2. **Share-sum integrity.** Every row's twelve published components sum
   to 100 within SSA's own rounding tolerance ("Totals do not
   necessarily equal the sum of rounded components"); the residual is
   recorded per row.
3. **FRA-schedule consistency with the oracle.** Table 6.B5.1 footnote
   a's FRA schedule (keyed by attain-65 year) reproduces the
   policyengine-us 416(l) schedule (keyed by birth year) under
   ``birth_year = attain_65_year - 65``.
4. **Out-of-sample note.** The standard for any *future* gate is
   held-out Supplement-year reproduction: fit to 1998-2019 and predict
   2020-2022. That is computed now as a REPORTED (not gated) number in
   ``runs/claiming_reference_v1.json`` under both a nearest-year rule
   (the module's documented default fallback) and a linear-trend rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from populace_dynamics.ss import benefits
from populace_dynamics.ss.params import SSAParameters

__all__ = [
    "ClaimAgeReference",
    "ClaimAgeRow",
    "load_claim_age_reference",
    "claim_age_distribution",
    "claim_age_pmf",
    "draw_claim_ages",
    "conversion_share",
    "months_early",
    "months_late",
    "benefit_factor",
    "expected_reduction_factor",
    "CLAIM_AGES",
    "MIN_YEAR",
    "MAX_YEAR",
]

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REFERENCE = (
    _ROOT / "data" / "external" / "ssa_claim_ages_2023supplement.json"
)

#: Entitlement-year span covered by Table 6.B5.1.
MIN_YEAR = 1998
MAX_YEAR = 2022

#: The integer claim ages the sampler can emit.
CLAIM_AGES: tuple[int, ...] = (62, 63, 64, 65, 66, 67, 68, 69, 70)

#: Within-band allocation for the published 67-69 aggregate: uniform
#: over the three single years. The table publishes only the 3-year
#: total, so a uniform split is the maximum-entropy allocation
#: consistent with that single number -- it adds no within-band
#: behavioural assumption the source cannot support. Finer resolution
#: would require MBR single-year-of-age tabulations, which 6.B5.1 does
#: not provide.
_BAND_67_69 = (67, 68, 69)

_AGE_70_MONTHS = 70 * 12

_SEX_ALIASES = {
    "male": "male",
    "m": "male",
    "men": "male",
    "female": "female",
    "f": "female",
    "women": "female",
}


def _normalize_sex(sex: str) -> str:
    key = str(sex).strip().lower()
    if key not in _SEX_ALIASES:
        raise ValueError(
            f"Unknown sex {sex!r}; expected one of "
            f"{sorted(set(_SEX_ALIASES))}."
        )
    return _SEX_ALIASES[key]


# --------------------------------------------------------------------------
# Reference loading
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ClaimAgeReference:
    """The committed Table 6.B5.1 reference, parsed once."""

    schema_version: str
    table: str
    supplement_year: int
    raw_columns: tuple[str, ...]
    collapsed_categories: tuple[str, ...]
    provenance: dict
    validation: dict
    fra_schedule: dict
    _data: dict

    def years(self) -> list[int]:
        return sorted(int(y) for y in self._data["male"])

    def row(self, sex: str, year: int) -> dict:
        return self._data[_normalize_sex(sex)][str(year)]


def _load(path: Path) -> ClaimAgeReference:
    doc = json.loads(path.read_text())
    if doc.get("schema_version") != "ssa_claim_ages.v1":
        raise ValueError(
            f"{path} has schema {doc.get('schema_version')!r}, expected "
            "ssa_claim_ages.v1."
        )
    return ClaimAgeReference(
        schema_version=doc["schema_version"],
        table=doc["table"],
        supplement_year=doc["supplement_year"],
        raw_columns=tuple(doc["column_schema"]["raw_columns"]),
        collapsed_categories=tuple(
            doc["column_schema"]["collapsed_categories"]
        ),
        provenance=doc["provenance"],
        validation=doc["validation"],
        fra_schedule=doc["fra_schedule"],
        _data=doc["data"],
    )


@cache
def _load_cached(path_str: str) -> ClaimAgeReference:
    return _load(Path(path_str))


def load_claim_age_reference(
    path: Path | str | None = None,
) -> ClaimAgeReference:
    """Load (and cache) the committed claim-age reference JSON."""
    resolved = Path(path) if path is not None else _DEFAULT_REFERENCE
    return _load_cached(str(resolved.resolve()))


# --------------------------------------------------------------------------
# (a) category shares, with a documented nearest-year rule
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ClaimAgeRow:
    """Age-at-entitlement shares for one (sex, entitlement year)."""

    sex: str
    entitlement_year: int  # the year actually used
    requested_year: int
    nearest_year_used: bool
    number_thousands: int
    average_age: float
    raw: dict  # 12 published columns; None for "not applicable"
    categories: dict  # collapsed 8-way partition, percentages
    fra_at: dict  # {"share": float, "at_age": 65 | 66}


def _resolve_year(requested: int) -> tuple[int, bool]:
    """Snap an out-of-range year to the nearest covered one.

    Every year in [MIN_YEAR, MAX_YEAR] is present, so an in-range
    request is always exact. Requests below MIN_YEAR use MIN_YEAR;
    above MAX_YEAR use MAX_YEAR. The nearest-year rule (as opposed to a
    trend extrapolation) is the module's documented default; see the
    held-out artifact for the measured cost of that choice.
    """
    if requested < MIN_YEAR:
        return MIN_YEAR, True
    if requested > MAX_YEAR:
        return MAX_YEAR, True
    return requested, False


def claim_age_distribution(
    sex: str,
    entitlement_year: int,
    *,
    reference: ClaimAgeReference | None = None,
) -> ClaimAgeRow:
    """Category shares for a (sex, entitlement year).

    Out-of-range years snap to the nearest covered year (documented in
    the returned :class:`ClaimAgeRow`).
    """
    ref = reference or load_claim_age_reference()
    norm_sex = _normalize_sex(sex)
    year, snapped = _resolve_year(int(entitlement_year))
    row = ref.row(norm_sex, year)
    return ClaimAgeRow(
        sex=norm_sex,
        entitlement_year=year,
        requested_year=int(entitlement_year),
        nearest_year_used=snapped,
        number_thousands=row["number_thousands"],
        average_age=row["average_age"],
        raw=dict(row["raw"]),
        categories=dict(row["categories"]),
        fra_at=dict(row["fra_at"]),
    )


def conversion_share(
    sex: str,
    entitlement_year: int,
    *,
    reference: ClaimAgeReference | None = None,
) -> float:
    """Disability-conversion share (percent) for a (sex, year).

    Exposed separately because conversions are auto-conversions at FRA,
    not claiming choices; the sampler excludes them by default.
    """
    row = claim_age_distribution(sex, entitlement_year, reference=reference)
    return row.categories["disability_conversion"]


# --------------------------------------------------------------------------
# (b) sampler over integer claim ages
# --------------------------------------------------------------------------
def claim_age_pmf(
    sex: str,
    entitlement_year: int,
    *,
    exclude_conversions: bool = True,
    reference: ClaimAgeReference | None = None,
) -> dict[int, float]:
    """Probability mass over integer claim ages implied by (sex, year).

    Category -> age: ``age62``/``age63``/``age64``/``age65``/``age66``
    map to that age; ``age67_69`` splits uniformly over {67, 68, 69}
    (:data:`_BAND_67_69`); ``age70plus`` -> 70.

    ``disability_conversion`` is not a claim age. With
    ``exclude_conversions=True`` (default) it is dropped and the
    remaining mass renormalised to 1. With ``exclude_conversions=False``
    its mass is placed at the year's at-FRA integer age (65 or 66) only
    so the total is conserved -- this is an accounting convenience, NOT
    a behavioural claim, and callers should prefer the default.
    """
    row = claim_age_distribution(sex, entitlement_year, reference=reference)
    cats = row.categories
    band = cats["age67_69"] / len(_BAND_67_69)
    mass: dict[int, float] = {
        62: cats["age62"],
        63: cats["age63"],
        64: cats["age64"],
        65: cats["age65"],
        66: cats["age66"],
        67: band,
        68: band,
        69: band,
        70: cats["age70plus"],
    }
    if not exclude_conversions:
        mass[row.fra_at["at_age"]] += cats["disability_conversion"]
    total = sum(mass.values())
    if total <= 0.0:
        raise ValueError(
            f"Degenerate claim-age mass for {row.sex} {row.entitlement_year}."
        )
    return {age: value / total for age, value in mass.items()}


def draw_claim_ages(
    rng: np.random.Generator,
    sex: str,
    entitlement_year: int,
    n: int,
    *,
    exclude_conversions: bool = True,
    reference: ClaimAgeReference | None = None,
) -> np.ndarray:
    """Draw ``n`` integer claim ages from the (sex, year) distribution.

    Disability conversions are excluded by default (see
    :func:`claim_age_pmf`). ``rng`` is a NumPy generator
    (``np.random.default_rng(seed)``).
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")
    pmf = claim_age_pmf(
        sex,
        entitlement_year,
        exclude_conversions=exclude_conversions,
        reference=reference,
    )
    ages = np.array(sorted(pmf), dtype=int)
    probs = np.array([pmf[int(a)] for a in ages], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(ages, size=n, p=probs)


# --------------------------------------------------------------------------
# (c) months early / late relative to the oracle's FRA schedule
# --------------------------------------------------------------------------
def months_early(
    claim_age_months: int, birth_year: int, params: SSAParameters
) -> int:
    """Whole months a claim precedes full retirement age (0 if at/after).

    FRA is read from the oracle's 416(l) schedule
    (:meth:`SSAParameters.fra_months`); it is not duplicated here.
    """
    return max(0, params.fra_months(birth_year) - int(claim_age_months))


def months_late(
    claim_age_months: int, birth_year: int, params: SSAParameters
) -> int:
    """Whole months a claim follows full retirement age (0 if at/before).

    Capped at age 70, past which no delayed credit accrues.
    """
    capped = min(int(claim_age_months), _AGE_70_MONTHS)
    return max(0, capped - params.fra_months(birth_year))


def benefit_factor(
    claim_age_months: int, birth_year: int, params: SSAParameters
) -> float:
    """Benefit-to-PIA multiplier for a claim age and birth cohort.

    ``1 - early_reduction`` before FRA (402(q)), ``1.0`` at FRA,
    ``1 + delayed_credit`` after FRA (402(w), capped at 70) -- each term
    from :mod:`populace_dynamics.ss.benefits`, so the pinned actuarial
    math is the single source of truth.
    """
    early = months_early(claim_age_months, birth_year, params)
    if early > 0:
        return 1.0 - benefits.early_reduction(early, params)
    late = months_late(claim_age_months, birth_year, params)
    if late > 0:
        return 1.0 + benefits.delayed_credit(late, birth_year, params)
    return 1.0


# --------------------------------------------------------------------------
# (d) expected benefit-to-PIA factor over the claim-age distribution
# --------------------------------------------------------------------------
def expected_reduction_factor(
    sex: str,
    entitlement_year: int,
    birth_year: int,
    params: SSAParameters,
    *,
    exclude_conversions: bool = True,
    reference: ClaimAgeReference | None = None,
) -> float:
    """Population-weighted expected benefit-to-PIA factor.

    The empirical claim-age mix from Table 6.B5.1 for
    ``(sex, entitlement_year)`` supplies the claiming behaviour;
    ``birth_year`` fixes the full retirement age. This is a
    single-cohort reading -- apply one birth cohort's 416(l) FRA and its
    402(q)/402(w) adjustments to the observed age mix -- documented as
    such because within a real entitlement year the birth cohort varies
    by claim age. Each integer age's factor comes from
    :func:`benefit_factor` (i.e. from ``ss.benefits``); conversions are
    excluded by default.

    A value below 1 means the average award is reduced relative to PIA
    (early claiming dominates); above 1 means delayed credits dominate.
    """
    pmf = claim_age_pmf(
        sex,
        entitlement_year,
        exclude_conversions=exclude_conversions,
        reference=reference,
    )
    return sum(
        prob * benefit_factor(age * 12, birth_year, params)
        for age, prob in pmf.items()
    )
