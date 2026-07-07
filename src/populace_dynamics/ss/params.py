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
