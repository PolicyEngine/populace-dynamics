"""SSA benefit-formula parameters, sourced from policyengine-us.

Every numeric series loads from the policyengine-us parameter tree on
disk (NAWI, the taxable wage base, PIA formula factors, the full
retirement age schedule, early-retirement reduction rates) — nothing
is hand-typed except the two statutory 1978-base bend-point dollar
amounts of 42 USC 415(a)(1)(B), which are constants of the statute
itself and are cross-checked at load time against both
policyengine-us's stored bracket thresholds and SSA's published 2026
determination (1,286 / 7,749).

The loader records the policyengine-us git revision so every
artifact pins its parameter provenance. The most recent NAWI
entries in policyengine-us are Trustees projections, not realized
SSA determinations, so scoring restricted to historical eligibility
years stays on realized values.

Auxiliary (spouse/survivor) rate constants — a documented exception
========================================================================
The spousal and survivor auxiliary benefits (42 USC 402(b)/(c)/(e)/(f))
have **no** parameter node anywhere in the policyengine-us tree:
policyengine-us carries ``social_security_retirement``,
``social_security_survivors`` and ``social_security_dependents`` as
uprated survey **inputs** (no formula), and only computes the worker's
*own* PIA and 402(q)/(w) retirement-age adjustment. So the auxiliary
rates cannot be sourced from pe-us the way the own-benefit series are.
They are therefore carried here as statutory constants, each cited to
its 42 USC section on the field below — the same "constant of the
statute itself" exception already made for the 1978-base bend points,
and the STATUTE-CITED precedent the pia-proxy floor and the claiming
module follow. They are *not* cross-checked against pe-us (there is
nothing to check against); they are validated against SSA's published
worked examples in ``tests/ss/test_aux_benefits.py`` and the committed
``runs/aux_benefit_examples_v1.json`` grid. The one shape simplification
— the survivor reduction period defaults to the modern 84-month span
(age 60 to a survivor full retirement age of 67, exact for every
survivor cohort born 1962 or later, i.e. the model's whole scoring
window) — is documented on ``survivor_reduction_period_months`` below.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = ["SSAParameters", "load_ssa_parameters"]

_PE_US_ENV = "POPULACE_DYNAMICS_PE_US_DIR"
_PE_US_DEFAULT = Path("~/PolicyEngine/policyengine-us").expanduser()
_SSA = Path("policyengine_us/parameters/gov/ssa")

#: 42 USC 415(a)(1)(B)(ii)-(iii): bend points for a given year equal
#: the 1979 amounts ($180 and $1,085 monthly) scaled by the ratio of
#: the national average wage index for the second year before the
#: year in question to that for 1977, rounded to the nearest dollar.
_BASE_FIRST_BEND = 180.0
_BASE_SECOND_BEND = 1_085.0
_BASE_NAWI_YEAR = 1977


@dataclass(frozen=True)
class SSAParameters:
    """Parameter bundle for Title II benefit computation."""

    nawi: dict[int, float]
    wage_base: dict[int, float]
    pia_factors: tuple[float, float, float]
    fra_months_by_birth_year: list[tuple[int, int]]
    early_monthly_rates: tuple[float, float]
    early_first_bracket_months: int
    pe_us_revision: str
    #: 42 USC 402(w): delayed-retirement credit, annual rate by birth
    #: year (a step function; 8 percent per year for 1943 and later).
    #: Defaulted so the historical (early-side) constructors in the test
    #: suite need not supply it.
    delayed_credit_by_birth_year: list[tuple[int, float]] = field(
        default_factory=list
    )
    #: Statutory cap on the credit-accrual window (max_delayed_years x
    #: 12); credits also stop at age 70 regardless.
    max_delayed_months: int = 48

    # ---- Auxiliary (spouse/survivor) statutory constants -------------
    # Not in the policyengine-us tree (see the module docstring); each is
    # a constant of the statute, cited to its 42 USC section. Defaulted to
    # the statutory value so historical/pure test bundles pick them up
    # automatically, exactly as they do delayed_credit_by_birth_year.

    #: 42 USC 402(b)(2)/(c)(2): the spouse's insurance benefit equals
    #: one-half of the worker's primary insurance amount.
    spousal_pia_share: float = 0.5
    #: 42 USC 402(q)(1): a spouse's benefit is reduced 25/36 of 1% per
    #: month for the first 36 months before full retirement age and
    #: 5/12 of 1% per month beyond — the SAME later rate as the worker's
    #: own reduction (benefits.early_reduction) but a STEEPER first
    #: bracket (25/36% vs the worker's 5/9%).
    spousal_early_monthly_rates: tuple[float, float] = (25 / 3600, 5 / 1200)
    #: 42 USC 402(q)(1): the 36-month boundary between the two spousal
    #: reduction rates.
    spousal_early_first_bracket_months: int = 36
    #: 42 USC 402(e)(2)(A)/(f)(3)(A): a widow(er)'s insurance benefit at
    #: survivor full retirement age equals 100% of the deceased worker's
    #: primary insurance amount (before RIB-LIM and inherited credits).
    survivor_pia_share: float = 1.0
    #: 42 USC 402(q): the widow(er)'s benefit floor at age 60 is 71.5% of
    #: the deceased's PIA (a maximum age reduction of 28.5%).
    survivor_reduction_floor: float = 0.715
    #: Months over which the 28.5% survivor age reduction is spread —
    #: age 60 to survivor full retirement age. STATUTE-SHAPED default of
    #: 84 = a survivor FRA of 67 (age 804 months − age 720 months), which
    #: is exact for every survivor cohort born 1962 or later (42 USC
    #: 416(l) as applied to survivors). Earlier survivor cohorts had a
    #: 60-to-65 (60-month) or 60-to-66 (72-month) span; using the modern
    #: 84-month span for them is the module's one documented survivor
    #: simplification and is immaterial to the model's post-1962 scoring
    #: window. Build a bundle with a different value to reproduce SSA's
    #: FRA-65/66 survivor reduction cells (see the aux-benefit tests).
    survivor_reduction_period_months: int = 84
    #: 42 USC 402(e)(1)(B)/(f)(1)(B): earliest aged-widow(er) claim age.
    survivor_earliest_claim_age: int = 60
    #: 42 USC 402(e)(2)(D)/(k)(3)(A) — the "RIB-LIM": when the deceased
    #: took a reduced retirement benefit, the widow(er)'s benefit is the
    #: larger of the deceased's actual benefit or 82.5% of the deceased's
    #: PIA.
    rib_lim_pia_share: float = 0.825
    #: 42 USC 402(e)(3)/(f)(4): remarriage at or after this age does not
    #: terminate widow(er)'s benefits (age 50 if the survivor is
    #: disabled).
    remarriage_protected_age: int = 60
    #: 42 USC 402(e)(3)(A)(ii): the protected-remarriage age for a
    #: disabled survivor.
    remarriage_protected_age_disabled: int = 50

    def bend_points(self, year: int) -> tuple[float, float]:
        """Statutory bend points for an eligibility year (415(a))."""
        index_year = year - 2
        if index_year not in self.nawi:
            raise KeyError(
                f"NAWI for {index_year} (needed for {year} bend "
                "points) is not in the policyengine-us series."
            )
        ratio = self.nawi[index_year] / self.nawi[_BASE_NAWI_YEAR]
        first = float(round(_BASE_FIRST_BEND * ratio))
        second = float(round(_BASE_SECOND_BEND * ratio))
        return first, second

    def wage_base_for(self, year: int) -> float:
        """Contribution and benefit base in effect for ``year``.

        The policyengine-us series lists only change years; the base
        persists until the next entry (a step function).
        """
        applicable = [y for y in self.wage_base if y <= year]
        if not applicable:
            raise KeyError(f"No wage base on or before {year}.")
        return self.wage_base[max(applicable)]

    def fra_months(self, birth_year: int) -> int:
        """Full retirement age in months for a birth year (416(l))."""
        months = None
        for threshold, amount in self.fra_months_by_birth_year:
            if birth_year >= threshold:
                months = amount
        if months is None:
            raise KeyError(f"No FRA bracket covers {birth_year}.")
        return months

    def delayed_credit_annual_rate(self, birth_year: int) -> float:
        """Annual delayed-retirement-credit rate for a birth year
        (42 USC 402(w)); a step function like the FRA schedule."""
        rate = None
        for threshold, amount in self.delayed_credit_by_birth_year:
            if birth_year >= threshold:
                rate = amount
        if rate is None:
            raise KeyError(
                f"No delayed-retirement-credit bracket covers "
                f"{birth_year}; parameters may not have been loaded."
            )
        return rate


def _resolve_pe_us(pe_us_dir: Path | None) -> Path:
    if pe_us_dir is not None:
        return Path(pe_us_dir).expanduser()
    env = os.environ.get(_PE_US_ENV)
    if env:
        return Path(env).expanduser()
    return _PE_US_DEFAULT


def _year_values(path: Path) -> dict[int, float]:
    """Read a pe-us ``values:`` parameter into ``{year: value}``."""
    data = yaml.safe_load(path.read_text())
    out: dict[int, float] = {}
    for key, value in data["values"].items():
        year = int(str(key)[:4])
        out[year] = float(value)
    return out


def load_ssa_parameters(
    pe_us_dir: Path | None = None,
) -> SSAParameters:
    """Load the Title II parameter bundle from policyengine-us.

    Raises:
        FileNotFoundError: If the policyengine-us checkout is absent.
        ValueError: If the derived bend points disagree with
            policyengine-us's stored bracket thresholds or SSA's
            published 2026 determination — the cross-check that
            guards the two statutory base constants.
    """
    root = _resolve_pe_us(pe_us_dir)
    ssa = root / _SSA
    if not ssa.is_dir():
        raise FileNotFoundError(
            f"policyengine-us SSA parameters not found at {ssa}; "
            f"set {_PE_US_ENV} or clone policyengine-us."
        )

    nawi = _year_values(ssa / "nawi.yaml")
    wage_base = _year_values(ssa / "social_security" / "wage_base.yaml")

    factors_doc = yaml.safe_load(
        (ssa / "social_security" / "pia" / "formula_factors.yaml").read_text()
    )
    rates = []
    stored_thresholds: list[dict[int, float]] = []
    for bracket in factors_doc["brackets"]:
        rate_values = bracket["rate"]
        rates.append(float(next(iter(rate_values.values()))))
        threshold = bracket["threshold"]
        values = (
            threshold.get("values", threshold)
            if isinstance(threshold, dict)
            else {}
        )
        stored_thresholds.append(
            {
                int(str(k)[:4]): float(v)
                for k, v in values.items()
                if str(v).replace(".", "").isdigit()
                or isinstance(v, (int, float))
            }
        )
    if len(rates) != 3:
        raise ValueError(f"Expected three PIA formula factors, found {rates}.")
    pia_factors = (rates[0], rates[1], rates[2])

    fra_doc = yaml.safe_load(
        (
            ssa / "social_security" / "full_retirement_age_by_birth_year.yaml"
        ).read_text()
    )
    fra_schedule = []
    for bracket in fra_doc["brackets"]:
        threshold = int(next(iter(bracket["threshold"].values())))
        amount = int(next(iter(bracket["amount"].values())))
        fra_schedule.append((threshold, amount))
    fra_schedule.sort()

    early_doc = yaml.safe_load(
        (
            ssa
            / "social_security"
            / "retirement_age_adjustment"
            / "early_retirement"
            / "reduction_rates.yaml"
        ).read_text()
    )
    early_brackets = early_doc["brackets"]
    first_rate = float(next(iter(early_brackets[0]["rate"].values())))
    second_rate = float(next(iter(early_brackets[1]["rate"].values())))
    first_bracket_months = int(
        next(iter(early_brackets[1]["threshold"].values()))
    )

    credit_doc = yaml.safe_load(
        (
            ssa
            / "social_security"
            / "retirement_age_adjustment"
            / "delayed_retirement"
            / "credit_rates.yaml"
        ).read_text()
    )
    delayed_credit_schedule = []
    for bracket in credit_doc["brackets"]:
        threshold = int(next(iter(bracket["threshold"].values())))
        amount = float(next(iter(bracket["amount"].values())))
        delayed_credit_schedule.append((threshold, amount))
    delayed_credit_schedule.sort()

    max_delayed_doc = yaml.safe_load(
        (
            ssa
            / "social_security"
            / "retirement_age_adjustment"
            / "max_delayed_years.yaml"
        ).read_text()
    )
    max_delayed_years = int(next(iter(max_delayed_doc["values"].values())))

    try:
        revision = subprocess.run(
            ["git", "log", "-1", "--format=%h"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "unknown"

    params = SSAParameters(
        nawi=nawi,
        wage_base=wage_base,
        pia_factors=pia_factors,
        fra_months_by_birth_year=fra_schedule,
        early_monthly_rates=(first_rate, second_rate),
        early_first_bracket_months=first_bracket_months,
        pe_us_revision=revision,
        delayed_credit_by_birth_year=delayed_credit_schedule,
        max_delayed_months=max_delayed_years * 12,
    )

    # Cross-check 1: the statutory base-year index. SSA's bend-point
    # determinations all divide by NAWI(1977) = 9,779.44.
    if abs(nawi.get(_BASE_NAWI_YEAR, 0.0) - 9_779.44) > 0.005:
        raise ValueError(
            f"NAWI({_BASE_NAWI_YEAR}) is "
            f"{nawi.get(_BASE_NAWI_YEAR)!r}, expected 9779.44; the "
            "policyengine-us series changed shape."
        )
    # Cross-check 2: derived bend points agree with policyengine-us's
    # stored bracket thresholds for every stored year, both brackets.
    # (The most recent NAWI values in policyengine-us are Trustees
    # projections rather than realized determinations, so external
    # anchors belong in tests with SSA's own figures; this check
    # guards internal consistency across the whole stored range.)
    for bracket_index in (1, 2):
        for year, stored in stored_thresholds[bracket_index].items():
            derived = params.bend_points(year)[bracket_index - 1]
            if abs(derived - stored) > 1.0:
                raise ValueError(
                    f"Derived bend point {bracket_index} for {year} "
                    f"({derived}) disagrees with policyengine-us "
                    f"({stored})."
                )
    return params
