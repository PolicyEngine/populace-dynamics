"""Mermin claiming/COLA-row replication (reported, not gated): the
retirement-age (NRA->70) and reduced-COLA rows on real careers vs Mermin
(2005).

REPORTED, NOT GATED. This artifact reads no gate and changes no gate. It
is the remaining-rows external-anchor replication of PolicyEngine/
populace-dynamics (issue #74, anchor 1): does our stack -- real PSID
Phase-A careers + the committed claim-age distribution (B2,
:mod:`populace_dynamics.claiming`) + the statutory actuarial machinery
(42 USC 402(q) early reduction / 402(w) delayed credit,
:mod:`populace_dynamics.ss.benefits`) + the committed NCHS x PSID-band
survival (mortality module) -- reproduce the incidence structure of the
two remaining Mermin (2005) provisions: raising the Normal Retirement Age
to 70 and reducing the annual COLA by 0.4 percentage points?

Frozen spec: issue #42 comment 4911609804. Where this module and the
registration disagree, the registration wins.

=====================================================================
THE ANCHOR (Urban 411260, Tables 1/2/4)
=====================================================================
Mermin, G. B. T. (2005), "The Effect of Benefit Reductions on the
Distribution of Social Security Benefits" (Urban Institute report 411260,
DYNASIM3 Runid 432; 2005 Trustees intermediate assumptions). Benefits as
a percent of scheduled for retired workers, by policy scenario:

* NRA->70 (raise the Normal Retirement Age to 70 for those turning 62
  after 2028; printed p.2 "Policy Scenarios"). Table 2 (retired workers
  ages 62-67 in 2050, N=5351): 79.7% of scheduled overall, roughly
  uniform 79.4/79.5/79.6/79.8/79.9 across shared-lifetime-income
  quintiles (the reduction factor does not depend on lifetime earnings,
  so the row is arithmetically flat). Disability beneficiaries are
  unaffected (printed p.7): "unlike retired worker benefits, disability
  benefits are unaffected by age of take-up".
* COLA -0.4pp (reduce the annual COLA from 2.8% to 2.4% beginning in
  2012; printed p.2 and fn.6). Little effect at claim, compounding with
  age: Table 2 (62-67) 98.9% of scheduled (Table 1's 2050 all-ages row is
  98.3%); Table 4 (retired workers ages 80-85 in 2050, N=3088) 92.4%
  (Table 5, all beneficiary types, 92.0%). "COLAs ... continue to reduce
  benefits relative to scheduled amounts after initial entitlement"
  (printed p.2).

=====================================================================
FROZEN MECHANIC (per the registration)
=====================================================================
Population: the Phase-A career-selection frame under the common 2050
transport, exactly as scripts/replication_caregiver.py / replication_r7_
sharing.py build it (coverage >= 0.8 of ages 22-61, age-62 eligibility in
2005-2019, born 1943-1957; r7's :func:`_person_history` and frame
constants reused byte-for-byte; the full statutory 42 USC 415(b) top-35
transported AIME reused verbatim from replication_caregiver).

*NRA->70.* Each person's benefit = PIA x the expected benefit-to-PIA
factor over their sex-specific, evaluation-era claim-age distribution
(the B2 module's :func:`claiming.claim_age_pmf`, excluding disability
conversions), with the SAME claim distribution for baseline and reform
("same draw"). Only the Normal Retirement Age changes: FRA = 67 baseline
vs 70 reform. Per claim age the factor is 1 - 402(q) early reduction
below the NRA (:func:`benefits.early_reduction`, rates unchanged, months
early recomputed against the imposed NRA) and 1 + 402(w) delayed credit
above it (:func:`benefits.delayed_credit`, only reachable in the FRA-67
baseline since the drawn claim ages top out at 70). Because the reduction
factor is PIA-independent, percent-of-scheduled = reform-factor /
baseline-factor cancels the PIA; it is reported by own-distribution AIME
quintile (the Table 2 comparison) plus the all-quintile weighted mean vs
the anchor's flat 79.7.

*COLA -0.4pp.* The benefit at an evaluation age a is the claim-age
benefit x (1 + COLA) ** (a - claim age); reform / scheduled cancels the
claim-age benefit and equals (1.024 / 1.028) ** (a - claim age), with the
claim age drawn from the same B2 distribution and the years-since-claim
from it. Percent-of-scheduled is reported per age group -- 62-67 and
80-85 -- as the population-weighted mean of that ratio over the joint
(person, evaluation age, claim age) mass, restricted at each evaluation
age to claim ages that have already begun (a retired worker at age a has
claimed by a). The 80-85 row weights each (person, evaluation age) by the
committed NCHS x PSID-band probability of surviving from 62 to that age
(the mortality module); 62-67 is not survival-weighted (per the
registration).

Floors: a 5-seed person-disjoint half-split (:func:`hpanel.split_panel_
by_person`, fraction=0.5, seeds 0-4) on each reported ratio -- the
across-seed |side_a - side_b| is the sampling-noise scale.

=====================================================================
NAMED POPULATION DELTAS vs the DYNASIM 2050 projection (documented, not
hidden -- the rows match in STRUCTURE, not level)
=====================================================================
* COHORT: observed PSID retirees eligible 2005-2019 (born 1943-1957),
  each benefit path evaluated at the anchor's 62-67 / 80-85 ages, vs
  DYNASIM's projected 2050 cross-section.
* FRA vs CLAIM ERA: the claim-age BEHAVIOUR is the observed 2005-2019
  distribution (Statistical Supplement Table 6.B5.1, the committed B2
  reference), while the Normal Retirement Age is the 2050-policy value
  (67 current law, 70 reform). Observed-era behaviour + 2050-policy FRA,
  as the transport requires.
* QUINTILE VARIABLE: individual own-record career-average earnings
  (transported AIME) vs the paper's SHARED lifetime-income quintiles
  (half of both spouses' earnings in married years; printed p.3, fn.9) --
  no marriage join here.
* COLA START: we compound COLAs from the CLAIM age (registration:
  "years-since-claim from the claim-age distribution"), whereas the paper
  credits COLAs from the age-62 first-eligibility year even when claiming
  is later (printed p.2, fn.7). For claims after 62 this makes our
  compounding window shorter, so our COLA percent-of-scheduled runs
  slightly ABOVE the paper's -- a named convention delta, not a model
  failure.
* WEIGHTING: person-weighted (anchor weight x survival) mean of the
  PIA-independent factor ratio, matching the sibling replications, vs any
  dollar-weighted aggregate. The ratios are PIA-independent, so the
  choice only reweights composition.
* NO EARNINGS TEST, no behavioural response, no disabled/auxiliary
  beneficiaries (Tables 2/4 are retired workers only; the disabled are
  unaffected by the NRA, printed p.7).

Run (from the repository root, PSID family + marriage files staged)::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv/bin/python scripts/replication_mermin_rows.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Weighted-mean helper and the seed-stable across-seed summary, imported
# byte-for-byte from the merged builders (single source of truth), exactly
# as the sibling replications do.
from build_downstream_relevance import _weighted_mean  # noqa: E402
from reform_delta_diagnostic import _summary  # noqa: E402
from replication_caregiver import (  # noqa: E402
    assign_quintiles,
    transported_aime,
)

# The 2050 transport + own-distribution quintile assignment, reused
# verbatim from the PPI/Mermin replication; the full statutory 415(b)
# top-35 transported AIME reused verbatim from the caregiver replication
# (the Phase-A frame carries full annual careers).
from replication_ppi_mermin import (  # noqa: E402
    N_QUINTILES,
    build_transport,
)

# The Phase-A career-selection frame: r7's per-person history builder and
# the frame constants, reused byte-for-byte (the pia_observed rule).
from replication_r7_sharing import (  # noqa: E402
    COVERAGE_FLOOR,
    ELIGIBILITY_AGE,
    ELIGIBILITY_HI,
    ELIGIBILITY_LO,
    N_CAREER_AGES,
    SEEDS,
    _person_history,
)

from populace_dynamics import claiming  # noqa: E402
from populace_dynamics.data.family import (  # noqa: E402
    family_earnings_panel,
)
from populace_dynamics.data.marriage import marriage_history  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.ss import benefits  # noqa: E402
from populace_dynamics.ss.params import load_ssa_parameters  # noqa: E402

ARTIFACT_PATH = ROOT / "runs" / "replication_mermin_rows_v1.json"
ARTIFACT_SCHEMA_VERSION = "replication_mermin_rows.v1"
RUN_NAME = "replication_mermin_rows_v1"

#: This replication's frozen-spec registration (issue #42 comment).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911609804"
)
SPEC_REGISTRATION_POINTER = "#42 comment 4911609804"
PROGRAM_DESIGN_ISSUE = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/74"
)

SEXES = ("male", "female")

# ---- NRA->70 scenario (Mermin, printed p.2) -----------------------------
#: Current-law Normal Retirement Age (fully phased to 67) vs the reform's
#: age 70; the 402(q) reduction rates are unchanged -- only the number of
#: months early changes with the NRA.
BASELINE_FRA_MONTHS = 67 * 12
REFORM_FRA_MONTHS = 70 * 12
_AGE70_MONTHS = 70 * 12
#: A 2050 current-law eligibility cohort (age 62 in 2050): statutory FRA
#: 67 and an 8% delayed-retirement credit. It fixes the 402(w) accrual
#: window (70 - 67 = 3 years) and rate for the reuse of
#: :func:`benefits.delayed_credit` under the imposed FRA-67 baseline, and
#: makes the baseline factor equal :func:`claiming.benefit_factor` at that
#: cohort exactly (see the tests).
FRA67_COHORT_BIRTH_YEAR = 1988

# ---- COLA -0.4pp scenario (Mermin, printed p.2, fn.6) -------------------
#: The paper's modelled annual COLAs: 2.8% current law, 2.4% reform.
BASELINE_COLA = 0.028
REFORM_COLA = 0.024
#: Per year since claim, the reform benefit is this multiple of the
#: scheduled benefit; the PIA and claim-age reduction cancel.
COLA_RATIO = (1.0 + REFORM_COLA) / (1.0 + BASELINE_COLA)
#: The two evaluation-age groups Mermin reports (Tables 2 and 4).
AGE_GROUP_62_67 = tuple(range(62, 68))
AGE_GROUP_80_85 = tuple(range(80, 86))
#: Condition survival on reaching first eligibility (age 62).
SURVIVAL_START_AGE = 62

# ---- Committed mortality inputs (NCHS x PSID-band survival) --------------
MORTALITY_WINDOW = "all"
NCHS_LIFE_TABLE_PATH = (
    ROOT / "data" / "external" / "nchs_life_tables_2023.json"
)
MORTALITY_FLOORS_PATH = ROOT / "runs" / "mortality_floors_v1.json"

# =====================================================================
# Anchor: transcribed Mermin (2005) target rows + citations. Verified
# against ~/PolicyEngine/dynasim-refs/411260-benefit-reductions.{pdf,txt}
# on 2026-07-08. Printed page numbers cite the report's own pagination.
# =====================================================================
#: Table 2 (retired workers ages 62-67 in 2050, N=5351), "age raised to
#: 70" column, by shared-lifetime-income quintile lowest -> highest.
ANCHOR_NRA_62_67_BY_QUINTILE = (79.4, 79.5, 79.6, 79.8, 79.9)
ANCHOR_NRA_62_67_ALL = 79.7
#: Table 4 (retired workers ages 80-85 in 2050, N=3088), NRA-70 "All".
ANCHOR_NRA_80_85_ALL = 80.9
#: Table 2 "reduced cost of living adjustment" column, "All" (62-67).
ANCHOR_COLA_62_67_ALL = 98.9
#: Table 1 (2050 row), COLA percent of scheduled at ages 62-67.
ANCHOR_COLA_62_67_TABLE1_2050 = 98.3
#: Table 4 COLA "All" (retired workers 80-85); Table 5 (all beneficiary
#: types 80-85) reads 92.0 -- the 92.0-92.4 band the registration cites.
ANCHOR_COLA_80_85_ALL = 92.4
ANCHOR_COLA_80_85_TABLE5_ALL = 92.0
#: Table 1 (2050 row) 75-year OASDI effect (percent of taxable payroll).
ANCHOR_TABLE1_PAYROLL_PCT = {
    "scheduled_deficit": -1.69,
    "reduced_cola": -1.12,
    "nra_raised_to_70": -0.5,
}


# =====================================================================
# Actuarial factor at an imposed Normal Retirement Age
# =====================================================================
def benefit_factor_at_fra(
    claim_age_months: int, fra_months: int, params: Any
) -> float:
    """Benefit-to-PIA factor for a claim age against an EXPLICIT NRA.

    1 - 402(q) early reduction below the NRA
    (:func:`benefits.early_reduction`, which takes the months-early
    directly, so the rates are NRA-agnostic and reused unchanged); 1 +
    402(w) delayed credit above it (:func:`benefits.delayed_credit`, with
    :data:`FRA67_COHORT_BIRTH_YEAR` fixing the 8% rate and the 3-year
    accrual window under the imposed NRA); 1.0 at the NRA. Reuses
    :mod:`populace_dynamics.ss.benefits` so the pinned actuarial math is
    the single source of truth. At ``fra_months = BASELINE_FRA_MONTHS``
    this equals :func:`claiming.benefit_factor` at the FRA-67 cohort.
    """
    early = max(0, fra_months - claim_age_months)
    if early > 0:
        return 1.0 - benefits.early_reduction(early, params)
    late = max(0, min(claim_age_months, _AGE70_MONTHS) - fra_months)
    if late > 0:
        return 1.0 + benefits.delayed_credit(
            late, FRA67_COHORT_BIRTH_YEAR, params
        )
    return 1.0


def expected_nra_factors(
    pmf: dict[int, float], params: Any
) -> tuple[float, float]:
    """Expected baseline (FRA 67) and reform (FRA 70) benefit-to-PIA
    factors over one claim-age distribution ("same draw" both sides).

    The baseline reuses :func:`claiming.benefit_factor` at the FRA-67
    cohort (its documented 402(q)/(w) path); the reform uses the same
    per-age machinery at the imposed FRA of 70. The identical ``pmf``
    drives both, so only the Normal Retirement Age differs.
    """
    baseline = sum(
        prob
        * claiming.benefit_factor(age * 12, FRA67_COHORT_BIRTH_YEAR, params)
        for age, prob in pmf.items()
    )
    reform = sum(
        prob * benefit_factor_at_fra(age * 12, REFORM_FRA_MONTHS, params)
        for age, prob in pmf.items()
    )
    return baseline, reform


# =====================================================================
# NCHS 2023 x PSID-band survival (the two committed artifacts)
# =====================================================================
class Survival:
    """Sex-specific survival from age 62, from the committed artifacts.

    ``survival(sex, 62, age)`` is the product of single-year (1 - q'_x)
    for x in [62, age), where q'_x is the NCHS 2023 sex-specific
    probability of death (``data/external/nchs_life_tables_2023.json``)
    scaled by the committed PSID/NCHS central-death-rate band ratio
    (``runs/mortality_floors_v1.json``, window ``all``). A ratio below 1
    (PSID mortality undercount, a known literature fact) raises survival;
    it is used as committed, never calibrated away.
    """

    #: SSA/NCHS-style bands the mortality artifact reports on.
    _BANDS = (
        (55, 64, "55-64"),
        (65, 74, "65-74"),
        (75, 84, "75-84"),
        (85, 200, "85+"),
    )

    def __init__(self, window: str = MORTALITY_WINDOW) -> None:
        nchs = json.loads(NCHS_LIFE_TABLE_PATH.read_text())
        self.nchs_vintage = int(nchs["vintage_year"])
        self.qx = {
            sex: {int(r["age"]): float(r["qx"]) for r in nchs["tables"][sex]}
            for sex in SEXES
        }
        mort = json.loads(MORTALITY_FLOORS_PATH.read_text())
        band_sex = mort["external_anchor"]["windows"][window]["by_band_sex"]
        self.ratios = {
            key: float(cell["ratio"]) for key, cell in band_sex.items()
        }
        self.window = window
        self._cache: dict[tuple[str, int, int], float] = {}

    def _band(self, age: int) -> str:
        for lo, hi, name in self._BANDS:
            if lo <= age <= hi:
                return name
        raise ValueError(f"No mortality band covers age {age}.")

    def adjusted_qx(self, sex: str, age: int) -> float:
        """NCHS q_x scaled by the PSID/NCHS band ratio, clipped to 1."""
        ratio = self.ratios[f"{self._band(age)}|{sex}"]
        return min(1.0, self.qx[sex][age] * ratio)

    def survival(self, sex: str, start_age: int, age: int) -> float:
        """Probability of surviving from ``start_age`` to ``age``."""
        key = (sex, start_age, age)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        surviving = 1.0
        for x in range(start_age, age):
            surviving *= 1.0 - self.adjusted_qx(sex, x)
        self._cache[key] = surviving
        return surviving


# =====================================================================
# Study data: the Phase-A career frame + sex + transported AIME
# =====================================================================
class MerminStudy:
    """Phase-A careers, sex, transported baseline AIME, and the per-person
    NRA baseline/reform factors + claim distribution.

    The career frame is the Phase-A ``pia_observed`` rule built exactly as
    scripts/replication_r7_sharing.py / replication_caregiver.py build it
    (same :func:`_person_history`, same constants). Sex is joined from the
    Marriage History File (mh85_23) on the shared PSID person id, as r7
    does. The transported AIME is the full statutory 415(b) top-35 amount
    (:func:`transported_aime`, reused verbatim from the caregiver
    replication).
    """

    def __init__(self, params: Any, transport: dict[str, Any]) -> None:
        self.params = params
        self.transport = transport

        panel = family_earnings_panel()
        panel = panel[(panel["age"] >= 14) & (panel["age"] <= 90)].copy()
        panel["implied_birth_year"] = panel["period"] - panel["age"]
        birth_year = (
            panel.groupby("person_id")["implied_birth_year"]
            .median()
            .round()
            .astype(int)
        )
        self.birth_year = birth_year.to_dict()
        # Anchor weight = the person's chronologically last observed weight.
        self.weight = (
            panel.sort_values("period")
            .groupby("person_id")["weight"]
            .last()
            .to_dict()
        )

        # Every person's ages-22-61 annual earnings history once.
        self.history: dict[int, dict[int, float]] = {}
        for pid, sub in panel.groupby("person_id"):
            hist = _person_history(sub, self.birth_year[int(pid)])
            if hist:
                self.history[int(pid)] = hist

        # Career frame: coverage >= 0.8 of ages 22-61, eligibility window.
        self.careers: set[int] = set()
        for pid, hist in self.history.items():
            elig = self.birth_year[pid] + ELIGIBILITY_AGE
            if not (ELIGIBILITY_LO <= elig <= ELIGIBILITY_HI):
                continue
            if len(hist) / N_CAREER_AGES >= COVERAGE_FLOOR:
                self.careers.add(pid)

        # Sex from the Marriage History File (individual-level, one row
        # per person id), joined as r7 does.
        mh = marriage_history()
        self.sex = (
            mh.drop_duplicates("person_id").set_index("person_id")["sex"]
        ).to_dict()


def score_population(
    study: MerminStudy, params: Any, transport: dict[str, Any]
) -> pd.DataFrame:
    """Per career-frame person with a resolvable sex: sex, weight,
    eligibility year, transported AIME, and the NRA baseline/reform
    factors (the expectation over their own claim-age distribution)."""
    rows = []
    for pid in sorted(study.careers):
        sex = study.sex.get(pid)
        if sex not in SEXES:
            continue
        b = study.birth_year[pid]
        elig = b + ELIGIBILITY_AGE
        pmf = claiming.claim_age_pmf(sex, elig)
        baseline, reform = expected_nra_factors(pmf, params)
        aime = transported_aime(study.history[pid], params, transport)
        rows.append(
            {
                "person_id": pid,
                "sex": sex,
                "weight": float(study.weight.get(pid, 1.0)),
                "elig_year": int(elig),
                "base_aime": float(aime),
                "nra_baseline_factor": float(baseline),
                "nra_reform_factor": float(reform),
            }
        )
    return pd.DataFrame(rows)


# =====================================================================
# NRA->70: percent-of-scheduled by own-distribution AIME quintile + mean
# =====================================================================
def nra_quintile_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Weighted-mean reform/baseline (percent of scheduled) by own AIME
    quintile and overall.

    percent-of-scheduled_i = 100 * reform_factor_i / baseline_factor_i
    (the PIA cancels). Quintiles are the own-distribution weighted 20%
    cutpoints of the transported AIME (the caregiver convention).
    """
    aime = df["base_aime"].to_numpy(dtype=np.float64)
    weight = df["weight"].to_numpy(dtype=np.float64)
    reform = df["nra_reform_factor"].to_numpy(dtype=np.float64)
    baseline = df["nra_baseline_factor"].to_numpy(dtype=np.float64)
    ratio = 100.0 * reform / baseline
    quintile = assign_quintiles(aime, weight)

    by_quintile = []
    for k in range(N_QUINTILES):
        mask = quintile == k
        by_quintile.append(
            {
                "quintile": k + 1,
                "n_persons": int(np.sum(mask)),
                "pct_of_scheduled": (
                    _weighted_mean(ratio[mask], weight[mask])
                    if np.any(mask)
                    else None
                ),
                "mean_aime": (
                    _weighted_mean(aime[mask], weight[mask])
                    if np.any(mask)
                    else None
                ),
            }
        )
    return {
        "by_quintile": by_quintile,
        "overall_pct_of_scheduled": _weighted_mean(ratio, weight),
        "cross_quintile_spread_pp": _quintile_spread(by_quintile),
    }


def _quintile_spread(by_quintile: list[dict[str, Any]]) -> float | None:
    """Max - min of the defined per-quintile percent-of-scheduled."""
    vals = [
        q["pct_of_scheduled"]
        for q in by_quintile
        if q["pct_of_scheduled"] is not None
    ]
    return (max(vals) - min(vals)) if vals else None


# =====================================================================
# COLA -0.4pp: percent-of-scheduled per evaluation-age group
# =====================================================================
def cola_group_coefficients(
    pmf: dict[int, float],
    sex: str,
    age_group: tuple[int, ...],
    survival: Survival,
    *,
    survival_weighted: bool,
) -> tuple[float, float]:
    """The (numerator, denominator) coefficients one person of this (sex,
    claim distribution) contributes per unit weight to an age group.

    For each evaluation age ``a`` in the group and each claim age
    ``c <= a`` (a retired worker at age ``a`` has already claimed), the
    mass is the claim probability times the survival weight (survival from
    62 to ``a`` for the 80-85 row; 1 for 62-67). The numerator carries the
    reform/scheduled ratio ``COLA_RATIO ** (a - c)``; the denominator is
    the mass. The population percent-of-scheduled is then
    ``100 * sum(w * A) / sum(w * B)`` over persons.
    """
    numer = 0.0
    denom = 0.0
    for a in age_group:
        surv = (
            survival.survival(sex, SURVIVAL_START_AGE, a)
            if survival_weighted
            else 1.0
        )
        if surv <= 0.0:
            continue
        for claim_age, prob in pmf.items():
            if claim_age > a:
                continue
            mass = surv * prob
            numer += mass * COLA_RATIO ** (a - claim_age)
            denom += mass
    return numer, denom


def _coefficient_table(
    df: pd.DataFrame,
    survival: Survival,
) -> dict[tuple[str, int], dict[str, tuple[float, float]]]:
    """Per distinct (sex, eligibility year), the COLA coefficients for
    both age groups (cached; the claim distribution depends only on sex
    and eligibility year)."""
    table: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}
    for sex, elig in {
        (r.sex, int(r.elig_year)) for r in df.itertuples(index=False)
    }:
        pmf = claiming.claim_age_pmf(sex, elig)
        table[(sex, elig)] = {
            "62_67": cola_group_coefficients(
                pmf, sex, AGE_GROUP_62_67, survival, survival_weighted=False
            ),
            "80_85": cola_group_coefficients(
                pmf, sex, AGE_GROUP_80_85, survival, survival_weighted=True
            ),
        }
    return table


def cola_pct(
    df: pd.DataFrame,
    coeffs: dict[tuple[str, int], dict[str, tuple[float, float]]],
    group_key: str,
) -> float | None:
    """Population percent-of-scheduled for an age group from the cached
    coefficients: ``100 * sum(w * A) / sum(w * B)``."""
    weight = df["weight"].to_numpy(dtype=np.float64)
    ab = np.array(
        [
            coeffs[(row.sex, int(row.elig_year))][group_key]
            for row in df.itertuples(index=False)
        ],
        dtype=np.float64,
    )
    numer = float(np.sum(weight * ab[:, 0]))
    denom = float(np.sum(weight * ab[:, 1]))
    return 100.0 * numer / denom if denom > 0.0 else None


# =====================================================================
# Floors: 5-seed person-disjoint half-split on each reported ratio
# =====================================================================
def seed_half_metrics(
    df: pd.DataFrame,
    coeffs: dict[tuple[str, int], dict[str, tuple[float, float]]],
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Each person-disjoint half's reported ratios for one seed.

    Own-distribution AIME quintiles are recomputed within each half (the
    NRA row), and the COLA age-group percents use the cached coefficients.
    The A-vs-B gap is the sampling-noise floor at half sample.
    """
    side_a, side_b = hpanel.split_panel_by_person(
        df, "person_id", fraction=0.5, seed=seed
    )
    out: dict[str, dict[str, Any]] = {}
    for label, side in (("side_a", side_a), ("side_b", side_b)):
        nra = nra_quintile_metrics(side)
        out[label] = {
            "nra_by_quintile_pct": [
                q["pct_of_scheduled"] for q in nra["by_quintile"]
            ],
            "nra_overall_pct": nra["overall_pct_of_scheduled"],
            "cola_62_67_pct": cola_pct(side, coeffs, "62_67"),
            "cola_80_85_pct": cola_pct(side, coeffs, "80_85"),
            "n": int(len(side)),
        }
    return out


def _abs_gaps(per_seed: list[dict[str, Any]], getter) -> list[float]:
    """Per-seed |side_a - side_b| for the value ``getter(side)`` (drops
    seeds where either side is undefined)."""
    gaps = []
    for row in per_seed:
        hm = row["half_metrics"]
        a = getter(hm["side_a"])
        b = getter(hm["side_b"])
        if a is not None and b is not None:
            gaps.append(abs(float(a) - float(b)))
    return gaps


def build_floors(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Per reported ratio, the floor = summary of the across-seed
    |side_a - side_b| (recomputable from the stored per-seed halves)."""
    floors: dict[str, Any] = {"nra_by_quintile": [], "nra_overall": None}
    for k in range(N_QUINTILES):
        floors["nra_by_quintile"].append(
            _summary(
                _abs_gaps(
                    per_seed,
                    lambda s, k=k: s["nra_by_quintile_pct"][k],
                )
            )
        )
    floors["nra_overall"] = _summary(
        _abs_gaps(per_seed, lambda s: s["nra_overall_pct"])
    )
    floors["cola_62_67"] = _summary(
        _abs_gaps(per_seed, lambda s: s["cola_62_67_pct"])
    )
    floors["cola_80_85"] = _summary(
        _abs_gaps(per_seed, lambda s: s["cola_80_85_pct"])
    )
    return floors


# =====================================================================
# Provision tables (ours / floor / anchor) + the registered verdict
# =====================================================================
def build_nra_table(
    nra: dict[str, Any], floors: dict[str, Any]
) -> dict[str, Any]:
    """The NRA->70 percent-of-scheduled table: per quintile and overall,
    ours vs its floor vs the anchor (Table 2)."""
    by_quintile = []
    for k in range(N_QUINTILES):
        row = nra["by_quintile"][k]
        floor = floors["nra_by_quintile"][k]
        anchor = ANCHOR_NRA_62_67_BY_QUINTILE[k]
        by_quintile.append(
            {
                "quintile": k + 1,
                "n_persons": row["n_persons"],
                "our_pct_of_scheduled": row["pct_of_scheduled"],
                "floor_mean": floor["mean"],
                "floor_sd": floor["sd"],
                "anchor_pct": anchor,
                "abs_gap_vs_anchor": (
                    round(abs(row["pct_of_scheduled"] - anchor), 2)
                    if row["pct_of_scheduled"] is not None
                    else None
                ),
                "mean_aime": row["mean_aime"],
            }
        )
    overall = nra["overall_pct_of_scheduled"]
    return {
        "by_quintile": by_quintile,
        "overall": {
            "our_pct_of_scheduled": overall,
            "floor_mean": floors["nra_overall"]["mean"],
            "floor_sd": floors["nra_overall"]["sd"],
            "anchor_pct": ANCHOR_NRA_62_67_ALL,
            "abs_gap_vs_anchor": round(abs(overall - ANCHOR_NRA_62_67_ALL), 2),
            "cross_quintile_spread_pp": nra["cross_quintile_spread_pp"],
        },
    }


def build_cola_table(
    cola_62_67: float,
    cola_80_85: float,
    floors: dict[str, Any],
) -> list[dict[str, Any]]:
    """The COLA -0.4pp percent-of-scheduled table: per age group, ours vs
    its floor vs the anchor (Tables 2 and 4)."""
    return [
        {
            "age_group": "62-67",
            "our_pct_of_scheduled": cola_62_67,
            "floor_mean": floors["cola_62_67"]["mean"],
            "floor_sd": floors["cola_62_67"]["sd"],
            "anchor_pct": ANCHOR_COLA_62_67_ALL,
            "anchor_secondary_pct": ANCHOR_COLA_62_67_TABLE1_2050,
            "abs_gap_vs_anchor": round(
                abs(cola_62_67 - ANCHOR_COLA_62_67_ALL), 2
            ),
            "survival_weighted": False,
        },
        {
            "age_group": "80-85",
            "our_pct_of_scheduled": cola_80_85,
            "floor_mean": floors["cola_80_85"]["mean"],
            "floor_sd": floors["cola_80_85"]["sd"],
            "anchor_pct": ANCHOR_COLA_80_85_ALL,
            "anchor_secondary_pct": ANCHOR_COLA_80_85_TABLE5_ALL,
            "abs_gap_vs_anchor": round(
                abs(cola_80_85 - ANCHOR_COLA_80_85_ALL), 2
            ),
            "survival_weighted": True,
        },
    ]


def registered_expectation(
    nra: dict[str, Any],
    cola_62_67: float,
    cola_80_85: float,
) -> dict[str, Any]:
    """The registration's pre-registered expectations, recomputed.

    (1) NRA->70 mean lands 77-83% with a cross-quintile spread < 3pp (the
        anchor's uniformity is arithmetic -- the reduction factor does not
        depend on AIME).
    (2) COLA 62-67 >= 97.5% (little effect at claim) and 80-85 in 90-94%
        (compounding with age).
    """
    nra_mean = nra["overall_pct_of_scheduled"]
    spread = nra["cross_quintile_spread_pp"]
    nra_ok = bool(
        77.0 <= nra_mean <= 83.0 and spread is not None and spread < 3.0
    )
    cola_62_ok = bool(cola_62_67 >= 97.5)
    cola_80_ok = bool(90.0 <= cola_80_85 <= 94.0)
    return {
        "nra_mean_77_83_spread_lt_3pp": {
            "statement": (
                "NRA->70 all-quintile mean in 77-83% and the cross-quintile "
                "spread below 3pp (the reduction factor is PIA-independent, "
                "so the row is near-flat as in the anchor)"
            ),
            "our_mean_pct": round(nra_mean, 2),
            "cross_quintile_spread_pp": (
                round(spread, 3) if spread is not None else None
            ),
            "held": nra_ok,
        },
        "cola_62_67_ge_97_5": {
            "statement": "COLA 62-67 percent-of-scheduled >= 97.5%",
            "our_pct": round(cola_62_67, 2),
            "held": cola_62_ok,
        },
        "cola_80_85_in_90_94": {
            "statement": (
                "COLA 80-85 percent-of-scheduled in 90-94% (compounding "
                "with age; the survival-weighted row is the risk)"
            ),
            "our_pct": round(cola_80_85, 2),
            "held": cola_80_ok,
        },
        "all_held": bool(nra_ok and cola_62_ok and cola_80_ok),
    }


# =====================================================================
# Anchor provenance
# =====================================================================
def anchor_provenance() -> dict[str, Any]:
    """Mermin (2005) transcribed target rows + page/section citations.

    Every figure verified against the archived PDF and its pdftotext
    (~/PolicyEngine/dynasim-refs/411260-benefit-reductions.{pdf,txt}) on
    2026-07-08. Printed page numbers cite the report's own pagination.
    """
    return {
        "paper": (
            "Mermin, G. B. T. (2005). The Effect of Benefit Reductions on "
            "the Distribution of Social Security Benefits. Urban Institute "
            "report 411260. DYNASIM3, Runid 432. 2005 Trustees intermediate "
            "assumptions; CBO (2005) solvency scoring."
        ),
        "source_files": [
            "~/PolicyEngine/dynasim-refs/411260-benefit-reductions.pdf",
            "~/PolicyEngine/dynasim-refs/411260-benefit-reductions.txt",
        ],
        "nra_mechanics": {
            "citation": (
                "printed p.2 (Policy Scenarios: 'Raising the Normal "
                "Retirement Age') and p.2 fn (the months-early mechanism)"
            ),
            "quote_scenario": (
                "Accelerates the currently scheduled increase to age 67 and "
                "continues increasing the NRA until it reaches age 70 for "
                "those turning 62 after 2028."
            ),
            "quote_mechanism": (
                "raising the NRA will reduce initial benefits because it "
                "increases the number of months early for those retiring "
                "before the NRA."
            ),
            "quote_no_behaviour": (
                "The analysis assumes that raising the NRA does not affect "
                "when workers apply for benefits (printed p.3-4)."
            ),
            "disability_unaffected": (
                "This option has no impact on recipients of disability "
                "benefits ... disability benefits are unaffected by age of "
                "take-up (printed p.7)."
            ),
            "encoding": (
                "current-law FRA 67 vs reform FRA 70; per-claim-age factor is "
                "1 - 402(q) early reduction (rates unchanged, months-early "
                "recomputed against the imposed NRA) or 1 + 402(w) delayed "
                "credit above the NRA; expected over the sex-specific "
                "evaluation-era B2 claim-age distribution; same draw both "
                "sides; percent-of-scheduled = reform-factor / baseline-"
                "factor (PIA cancels)"
            ),
        },
        "cola_mechanics": {
            "citation": "printed p.2 (Policy Scenarios) and fn.6, fn.7",
            "quote_scenario": (
                "Beginning in 2012, the annual COLA is reduced by 0.4 "
                "percentage points for all beneficiaries."
            ),
            "quote_rates": (
                "The analysis assumes annual COLAs of 2.8 percent under "
                "current law and 2.4 percent under the reduced scenario "
                "(fn.6)."
            ),
            "quote_after_entitlement": (
                "Reducing the COLA ... continues to reduce benefits relative "
                "to scheduled amounts after initial entitlement."
            ),
            "paper_cola_start_note": (
                "The paper credits COLAs from the year of first eligibility "
                "(age 62), not first receipt (fn.7). This module compounds "
                "from the claim age per the registration ('years-since-claim "
                "from the claim-age distribution') -- a named convention "
                "delta that shortens the window for later claimers, so our "
                "percent-of-scheduled runs slightly above the paper's."
            ),
            "encoding": (
                "reform/scheduled at evaluation age a = (1.024/1.028) ** "
                "(a - claim age), claim age from the B2 distribution; per "
                "age group 62-67 and 80-85; 80-85 survival-weighted by the "
                "committed NCHS x PSID-band survival"
            ),
        },
        "table1_ages_62_67_by_year": {
            "citation": "Table 1 (printed p., DYNASIM3 Runid 432); 2050 row",
            "cola_pct_of_scheduled_2050": ANCHOR_COLA_62_67_TABLE1_2050,
            "nra_pct_of_scheduled_2050": 85.2,
            "note": (
                "raising the NRA holds steady at about 85% of scheduled once "
                "fully phased; the 79.7 in Table 2 is lower because retired "
                "workers ages 62-67 claim early"
            ),
            "seventy_five_year_payroll_pct": ANCHOR_TABLE1_PAYROLL_PCT,
        },
        "table2_retired_workers_62_67_in_2050": {
            "citation": (
                "Table 2 (Distribution of Annual Benefits at Ages 62 to 67 "
                "in 2050, Retired Workers Only, N=5351); Percent of Scheduled "
                "Benefits columns"
            ),
            "quintile_variable_paper": (
                "shared lifetime income quintile: a worker's entire earnings "
                "in single years plus half of both spouses' earnings in "
                "married years (printed p.3, fn.9)"
            ),
            "nra_raised_to_70_pct": {
                "by_quintile": list(ANCHOR_NRA_62_67_BY_QUINTILE),
                "all": ANCHOR_NRA_62_67_ALL,
            },
            "reduced_cola_pct": {
                "all": ANCHOR_COLA_62_67_ALL,
                "note": "~flat 98.8-99.0 across quintiles",
            },
        },
        "table4_retired_workers_80_85_in_2050": {
            "citation": (
                "Table 4 (Distribution of Annual Benefits at Ages 80 to 85 "
                "in 2050, Retired Workers Only, N=3088); Percent of scheduled "
                "benefits columns"
            ),
            "reduced_cola_pct": {
                "all": ANCHOR_COLA_80_85_ALL,
                "table5_all_beneficiary_types": ANCHOR_COLA_80_85_TABLE5_ALL,
                "note": (
                    "92.4 (retired workers) / 92.0 (all types) -- the "
                    "92.0-92.4 band the registration cites; COLA compounds "
                    "with age so the 80-85 cut is far larger than 62-67"
                ),
            },
            "nra_raised_to_70_pct": {"all": ANCHOR_NRA_80_85_ALL},
        },
        "named_population_deltas": [
            "observed PSID retirees eligible 2005-2019 (born 1943-1957), "
            "benefit paths evaluated at the anchor's 62-67 / 80-85 ages, vs "
            "DYNASIM's projected 2050 cross-section",
            "observed-era (2005-2019) claim-age BEHAVIOUR (B2 Table 6.B5.1) "
            "with the 2050-policy Normal Retirement Age (67 current law, 70 "
            "reform)",
            "individual own-record career-average earnings (transported "
            "AIME) quintiles vs the paper's spouse-shared lifetime-income "
            "quintiles (no marriage join)",
            "COLAs compounded from the claim age (registration) vs the "
            "paper's age-62 first-eligibility start (fn.7) -- our COLA "
            "percent-of-scheduled runs slightly above the paper's",
            "person-weighted (anchor weight x survival) mean of the "
            "PIA-independent factor ratio vs a dollar-weighted aggregate",
            "no earnings test, no behavioural response, retired workers "
            "only (the disabled are unaffected by the NRA, printed p.7)",
        ],
    }


# =====================================================================
# Provenance
# =====================================================================
def _sha(cwd: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _revision_pins(params: Any, survival: Survival) -> dict[str, Any]:
    return {
        "populace_dynamics_sha": _sha(ROOT),
        "pe_us_revision": getattr(params, "pe_us_revision", None),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "nchs_life_table_vintage": survival.nchs_vintage,
        "mortality_window": survival.window,
    }


# =====================================================================
# Driver
# =====================================================================
def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full Mermin NRA->70 + COLA row replication (reported)."""
    started = time.time()
    params = load_ssa_parameters()
    transport = build_transport(params)
    survival = Survival()
    if verbose:
        print(
            f"pe_us_revision={params.pe_us_revision} "
            f"nawi[{transport['index_year']}]="
            f"{transport['index_nawi']:.0f}; loading PSID ..."
        )

    study = MerminStudy(params, transport)
    df = score_population(study, params, transport)
    coeffs = _coefficient_table(df, survival)
    if verbose:
        n_by_sex = df["sex"].value_counts().to_dict()
        print(
            f"career frame {len(study.careers)} persons; scored {len(df)} "
            f"with sex {n_by_sex} ({time.time() - started:.0f}s)"
        )

    nra = nra_quintile_metrics(df)
    cola_62_67 = cola_pct(df, coeffs, "62_67")
    cola_80_85 = cola_pct(df, coeffs, "80_85")

    per_seed = []
    for seed in SEEDS:
        per_seed.append(
            {"seed": seed, "half_metrics": seed_half_metrics(df, coeffs, seed)}
        )
    floors = build_floors(per_seed)

    nra_table = build_nra_table(nra, floors)
    cola_table = build_cola_table(cola_62_67, cola_80_85, floors)
    expectation = registered_expectation(nra, cola_62_67, cola_80_85)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "reported_not_gated": True,
        "registration": SPEC_REGISTRATION,
        "registration_pointer": SPEC_REGISTRATION_POINTER,
        "program_design_issue": PROGRAM_DESIGN_ISSUE,
        "purpose": (
            "Mermin remaining-rows external-anchor replication: does our "
            "stack (real PSID Phase-A careers + the committed B2 claim-age "
            "distribution + the statutory 402(q)/(w) actuarial machinery + "
            "the committed NCHS x PSID-band survival) reproduce the "
            "incidence structure of raising the Normal Retirement Age to 70 "
            "(percent-of-scheduled by AIME quintile) and reducing the COLA "
            "by 0.4pp (percent-of-scheduled at ages 62-67 and 80-85) from "
            "Mermin (2005)? Reads no gate, changes no gate; publishes "
            "regardless of outcome."
        ),
        "what_is_tested": (
            "the incidence STRUCTURE (the near-flat NRA reduction across "
            "AIME quintiles, and the small-at-claim / compounding-with-age "
            "COLA pattern), not the population LEVELS -- every level "
            "difference vs the DYNASIM 2050 projection is a named delta"
        ),
        "real_data_only": True,
        "anchor_provenance": anchor_provenance(),
        "study_population": {
            "career_frame_rule": (
                "Phase-A pia_observed selection (reused from "
                "replication_r7_sharing / replication_caregiver): coverage "
                ">= 0.8 of ages 22-61, age-62 eligibility year in 2005-2019 "
                "(born 1943-1957), single-year gaps interpolated"
            ),
            "n_career_frame": int(len(study.careers)),
            "n_scored": int(len(df)),
            "n_by_sex": {sex: int((df["sex"] == sex).sum()) for sex in SEXES},
            "sex_join": (
                "Marriage History File mh85_23, one row per person id, "
                "joined on the shared PSID person id (as replication_r7_"
                "sharing does); career-frame persons without a resolvable "
                "sex are dropped"
            ),
            "n_career_frame_without_sex": int(
                sum(
                    1
                    for pid in study.careers
                    if study.sex.get(pid) not in SEXES
                )
            ),
            "weighting": (
                "anchor weight = the person's last observed PSID "
                "cross-sectional weight (times NCHS x PSID-band survival "
                "from 62 for the 80-85 COLA row)"
            ),
            "aime_quintile": (
                "own-distribution weighted 20% quintiles of the full "
                "statutory 415(b) top-35 transported AIME (Q1 = lowest)"
            ),
        },
        "conventions": {
            "transport": (
                "the 2050-eligibility transport from replication_ppi_mermin "
                "(build_transport); the transported AIME is the full 415(b) "
                "top-35 amount (replication_caregiver.transported_aime), the "
                "AIME-quintile ranking variable only"
            ),
            "claim_age_distribution": (
                "the committed B2 reference (claiming.claim_age_pmf, "
                "Statistical Supplement Table 6.B5.1, sex x entitlement "
                "year 1998-2022, disability conversions excluded), read at "
                "the person's own age-62 eligibility year (nearest-year "
                "rule inside range)"
            ),
            "nra_factor": (
                "baseline = expectation of claiming.benefit_factor at the "
                "FRA-67 cohort (born 1988); reform = expectation of the "
                "same 402(q)/(w) machinery at the imposed FRA of 70; same "
                "claim distribution both sides; percent-of-scheduled = "
                "100 * reform / baseline"
            ),
            "cola_ratio": (
                "reform/scheduled at evaluation age a = (1.024/1.028) ** "
                "(a - claim age); baseline COLA 2.8%, reform 2.4% (fn.6); "
                "per age group as the weighted mean over (person, "
                "evaluation age, claim age <= a) mass"
            ),
            "survival_weighting_80_85": (
                "each (person, evaluation age a) in 80-85 is weighted by the "
                "committed NCHS 2023 x PSID-band probability of surviving "
                "from 62 to a (q'_x = NCHS q_x * PSID/NCHS band ratio, "
                "window 'all'); 62-67 is not survival-weighted"
            ),
            "floor": (
                "5-seed person-disjoint half-split (split_panel_by_person, "
                "fraction=0.5); floor per ratio = summary of |side_a - "
                "side_b| across seeds, own-distribution AIME quintiles "
                "recomputed within each half"
            ),
        },
        "nra_raise_to_70": {
            "table": nra_table,
            "full_sample": {
                "overall_pct_of_scheduled": nra["overall_pct_of_scheduled"],
                "cross_quintile_spread_pp": nra["cross_quintile_spread_pp"],
            },
        },
        "cola_minus_0_4pp": {"table": cola_table},
        "registered_expectation": expectation,
        "per_seed": per_seed,
        "revision_pins": _revision_pins(params, survival),
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if verbose:
        _print_summary(nra_table, cola_table, expectation)
    return artifact


def _print_summary(
    nra_table: dict[str, Any],
    cola_table: list[dict[str, Any]],
    expectation: dict[str, Any],
) -> None:
    print(
        "\n=== NRA->70: percent of scheduled by AIME quintile "
        "(ours / floor / anchor) ==="
    )
    print(f"{'Q':>2}{'ours':>9}{'floor':>8}{'anchor':>8}{'gap':>7}")
    for row in nra_table["by_quintile"]:
        print(
            f"{row['quintile']:>2}"
            f"{row['our_pct_of_scheduled']:>9.2f}"
            f"{row['floor_mean']:>8.2f}"
            f"{row['anchor_pct']:>8.1f}"
            f"{row['abs_gap_vs_anchor']:>7.2f}"
        )
    ov = nra_table["overall"]
    print(
        f"{'all':>2}{ov['our_pct_of_scheduled']:>9.2f}"
        f"{ov['floor_mean']:>8.2f}{ov['anchor_pct']:>8.1f}"
        f"{ov['abs_gap_vs_anchor']:>7.2f}  "
        f"(spread {ov['cross_quintile_spread_pp']:.3f}pp)"
    )
    print(
        "\n=== COLA -0.4pp: percent of scheduled by age group "
        "(ours / floor / anchor) ==="
    )
    print(f"{'group':>7}{'ours':>9}{'floor':>8}{'anchor':>8}{'gap':>7}")
    for row in cola_table:
        print(
            f"{row['age_group']:>7}"
            f"{row['our_pct_of_scheduled']:>9.2f}"
            f"{row['floor_mean']:>8.2f}"
            f"{row['anchor_pct']:>8.1f}"
            f"{row['abs_gap_vs_anchor']:>7.2f}"
        )
    print("\nregistered expectations:")
    for name, check in expectation.items():
        if name == "all_held":
            continue
        print(f"  {name}: {check['held']}")
    print(f"  ALL HELD: {expectation['all_held']}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON-serializable: {type(obj)!r}")


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, indent=2, default=_json_default) + "\n"
    )
    print(f"\nwrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
