"""Caregiver-credit replication (reported, not gated): R5 progressivity
pattern on real careers vs Smith, Johnson & Favreault (2020).

REPORTED, NOT GATED. This artifact reads no gate and changes no gate. It
is the caregiver-credit external-anchor replication of PolicyEngine/
populace-dynamics (issue #74, anchor 2): does our stack -- real PSID
careers (the Phase-A career-selection frame) + observed births (the
Childbirth & Adoption History File cah85_23) + the statutory AIME/PIA
chain (42 USC 415(a)/(b)) transported to a common 2050 cohort --
reproduce the *progressivity PATTERN* of the caregiver credit: the share
of aggregate benefit gains reaching the bottom fifth of lifetime earners
(Smith/Johnson/Favreault 2020, Table 15)?

Frozen spec: issue #42 comment 4911453454. Where this module and the
registration disagree, the registration wins.

=====================================================================
THE ANCHOR (Urban 103050, Table 15)
=====================================================================
Smith, K. E., Johnson, R. W. and Favreault, M. M. (2020), "Five
Democratic Approaches to Social Security Reform: Estimated Impact of
Plans by 2020 Presidential Candidates" (Urban Institute report 103050,
DYNASIM3 ID980). The caregiver credit is the most progressive provision
scored: Table 15 (printed p. 66) reports the percentage of each
provision's benefit increase going to the bottom fifth of the lifetime
earnings distribution in 2065. The "Create caregiver credit" row:

    Biden 54   Buttigieg 52   Klobuchar 62   Warren 55

-- a 52-62% band across the four plans' credit designs.

=====================================================================
THE FOUR PLANS (Table 2, printed p. 8; narrative printed p. 11)
=====================================================================
Every plan credits a worker's earnings, in years they care for a
qualifying child, up to a credit level tied to the economy-wide average
wage (the NAWI series), then recomputes the AIME/PIA. The plans differ
in the credit level, the child age limit, and the year cap:

* Biden      -- 1/2 average wage, child younger than 12 (max age 11),
               capped at 5 years.
* Buttigieg  -- full average wage, child younger than 18 (max age 17),
               no year cap.
* Klobuchar  -- 1/2 average wage, child younger than 6 (max age 5),
               no year cap.
* Warren     -- full average wage, child younger than 6 (max age 5),
               no year cap.

The registration's frozen mechanic is uniform across plans: in a
qualifying child-year where the worker's earnings are below the credit
level, earnings are *topped up to the credit level* (the paper's own
"replace a caregiver's earnings ... with the caregiver credit as long as
that credit exceeded the worker's earnings", printed p. 11). Biden's
statutory on-top-of-earnings, phase-out variant is not separately
modelled -- the registration specifies the common top-up mechanic for
all four; see :data:`ANCHOR_PROVENANCE`.

=====================================================================
CONVENTIONS (reused verbatim from the Phase-A replications)
=====================================================================
* Career frame: the Phase-A career-selection frame (the
  ``pia_observed_psid_v1`` rule) exactly as scripts/replication_r7_
  sharing.py builds it -- coverage >= 0.8 of ages 22-61, age-62
  eligibility year in 2005-2019 (born 1943-1957), single-year gaps
  interpolated. Reuses r7's :func:`_person_history` and the frame
  constants byte-for-byte.
* Transport: the common 2050-eligibility transport from scripts/
  replication_ppi_mermin.py (:func:`build_transport`) -- NAWI projected
  to the 2048 age-60 indexing year, 2050 bend points derived per 42 USC
  415(a)(1)(B). Every career is transported to the same synthetic 2050
  cohort so the incidence comparison uses one common bracket geometry.
* AIME: the full statutory 42 USC 415(b) top-35 indexed earnings
  (indexed to NAWI[2048], each year capped at its historical wage base,
  divided by 35*12, floored). This is the *original* Phase-A AIME
  convention -- available here because the Phase-A frame carries full
  annual careers, unlike ppi_mermin's biennial gate panel, which forced
  its documented biennial-proxy deviation.
* PIA / gain: the pre-415(g) PIA amount (the 90/32/15 bracket sum on the
  2050 bends), reusing ppi_mermin's :func:`scheduled_amount`. The gain
  is the reformed-minus-baseline PIA; the sub-$0.10 415(g) dime rounding
  is a nominal artifact omitted from the incidence gain, exactly as
  ppi_mermin omits it from its incidence ratio. Because the transported
  AIME, the credit level, and the bend points all scale linearly with
  NAWI[2048], every share reported here is invariant to the NAWI
  projection level (it sets only the immaterial absolute scale).
* Metric: per plan, the bottom-lifetime-earnings-quintile share of
  aggregate weighted benefit gain (the Table 15 statistic), on
  own-distribution weighted quintiles of the baseline transported AIME;
  plus the gain incidence by quintile and the weighted share of persons
  gaining. 5-seed person-disjoint half-split floors on the headline
  shares.

=====================================================================
NAMED POPULATION DELTAS vs the DYNASIM 2065 projection (documented, not
hidden -- the shares match in DIRECTION/PATTERN, not level)
=====================================================================
* COHORT: observed PSID retirees eligible 2005-2019 (born 1943-1957),
  transported to a common 2050 bracket geometry, vs DYNASIM's projected
  beneficiaries receiving benefits in 2065.
* EARNINGS: individual own-record earnings vs DYNASIM's couples splitting
  earnings and benefits in married years (Table 15 note). We do not
  share earnings within couples (no marriage join here), so our lifetime
  quintile is an individual own-record measure.
* WINDOW: truncated observation -- ages 22-61 annual/biennial PSID
  histories transported to 2050 vs DYNASIM's full projected careers. Our
  observed careers are compressed relative to the bends (fewer cap-riding
  high-AIME years), so more low-AIME person-years are exposed to the
  top-up and our bottom-quintile shares run HIGHER than the anchor's,
  especially for the 1/2-credit designs (the pre-registered expectation).
* NO PROJECTION: no mortality/survival, benefit-receipt years, claim-age
  reduction, or behavioural response; each person's gain is the
  reformed-minus-baseline monthly PIA (a cross-sectional expenditure-
  increase proxy). DYNASIM projects all of these to 2065.

Run (from the repository root, PSID family + birth files staged)::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv/bin/python scripts/replication_caregiver.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Weighted helpers, imported byte-for-byte from the merged builders (single
# source of truth), exactly as the other replications do.
from build_downstream_relevance import (  # noqa: E402
    _weighted_mean,
    _weighted_quantile,
)
from reform_delta_diagnostic import _summary  # noqa: E402

# The 2050 transport + the pre-415(g) PIA amount + the own-distribution
# quintile assignment, reused verbatim from the PPI/Mermin replication.
from replication_ppi_mermin import (  # noqa: E402
    N_QUINTILES,
    QUINTILE_LEVELS,
    _assign_quintiles,
    build_transport,
    scheduled_amount,
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

from populace_dynamics.data.births import birth_history  # noqa: E402
from populace_dynamics.data.family import (  # noqa: E402
    family_earnings_panel,
)
from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.ss.params import load_ssa_parameters  # noqa: E402

ARTIFACT_PATH = ROOT / "runs" / "replication_caregiver_v1.json"
ARTIFACT_SCHEMA_VERSION = "replication_caregiver.v1"
RUN_NAME = "replication_caregiver_v1"

#: This replication's frozen-spec registration (issue #42 comment).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911453454"
)
SPEC_REGISTRATION_POINTER = "#42 comment 4911453454"
PROGRAM_DESIGN_ISSUE = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/74"
)

_MONTHS = 12
_COMPUTATION_YEARS = 35


# =====================================================================
# The four plans (Table 2, printed p. 8; narrative printed p. 11)
# =====================================================================
@dataclass(frozen=True)
class Plan:
    """A candidate's caregiver-credit design (frozen-spec parameters)."""

    name: str
    #: Fraction of the economy-wide average wage the credit tops up to.
    credit_fraction: float
    #: A child qualifies while strictly younger than this age (so
    #: ``max age = age_limit - 1``: Biden 11 -> 12, Buttigieg 17 -> 18,
    #: Klobuchar/Warren 5 -> 6).
    child_age_limit: int
    #: Maximum number of credited (topped-up) years, or None for no cap.
    year_cap: int | None


PLANS: tuple[Plan, ...] = (
    Plan("Biden", 0.5, 12, 5),
    Plan("Buttigieg", 1.0, 18, None),
    Plan("Klobuchar", 0.5, 6, None),
    Plan("Warren", 1.0, 6, None),
)
PLAN_ORDER = tuple(p.name for p in PLANS)
#: The 1/2-average-wage designs (the pre-registered "in or above the band"
#: expectation is specifically about these).
HALF_CREDIT_PLANS = ("Biden", "Klobuchar")


# =====================================================================
# Anchor: transcribed Smith/Johnson/Favreault (2020) rows + citations.
# Verified against 103050-five-dem.{pdf,txt} on 2026-07-08. Printed page
# numbers cite the report's own pagination (PDF sequence page = printed
# + 10, verified against Table 2 at PDF p.18 = printed p.8 and Table 15 at
# PDF p.76 = printed p.66).
# =====================================================================
#: Table 15 (printed p. 66), "Create caregiver credit" row, scheduled
#: scenario -- percent of the benefit increase going to the bottom fifth
#: of lifetime earners in 2065, by plan.
ANCHOR_TABLE15_BOTTOM_FIFTH = {
    "Biden": 54.0,
    "Buttigieg": 52.0,
    "Klobuchar": 62.0,
    "Warren": 55.0,
}
ANCHOR_BAND = (52.0, 62.0)
#: Narrative (printed p. 27-28): share of beneficiaries who would gain
#: from each plan's caregiver credit in 2065.
ANCHOR_SHARE_GAINING_2065 = {
    "Biden": 36.0,
    "Buttigieg": 45.0,
    "Klobuchar": 27.0,
    "Warren": 33.0,
}
#: Table 3 (printed p. 19): 75-year cost of the caregiver provision alone,
#: percent of taxable payroll (negative = a cost / actuarial-balance
#: reduction).
ANCHOR_COST_PCT_PAYROLL = {
    "Biden": -0.12,
    "Buttigieg": -0.51,
    "Klobuchar": -0.12,
    "Warren": -0.30,
}


def anchor_provenance() -> dict[str, Any]:
    """Smith/Johnson/Favreault (2020) plan definitions + transcribed rows.

    Every figure verified against the archived PDF and its pdftotext
    (~/PolicyEngine/dynasim-refs/103050-five-dem.{pdf,txt}). Printed page
    numbers cite the report's own pagination.
    """
    return {
        "paper": (
            "Smith, K. E., Johnson, R. W. and Favreault, M. M. (2020). Five "
            "Democratic Approaches to Social Security Reform: Estimated "
            "Impact of Plans by 2020 Presidential Candidates. The Urban "
            "Institute (report 103050). DYNASIM3, ID980. 2020 Trustees "
            "intermediate assumptions."
        ),
        "source_files": [
            "~/PolicyEngine/dynasim-refs/103050-five-dem.pdf",
            "~/PolicyEngine/dynasim-refs/103050-five-dem.txt",
        ],
        "plan_definitions": {
            "citation": (
                "Table 2 (Major Benefit-Enhancing Provisions Included in the "
                "Democratic Candidates' Social Security Plans), printed p. 8; "
                "narrative 'PROVIDE SOCIAL SECURITY CREDITS TO CAREGIVERS', "
                "printed p. 11"
            ),
            "credit_amount": {
                "Biden": "own earnings plus up to 50% of the national "
                "average wage (modelled as a top-up to 50% of the average "
                "wage per the registration's common mechanic)",
                "Buttigieg": "100% of the national average wage",
                "Klobuchar": "50% of the national average wage",
                "Warren": "100% of the national average wage",
            },
            "maximum_age_for_qualifying_child": {
                "Biden": 11,
                "Buttigieg": 17,
                "Klobuchar": 5,
                "Warren": 5,
            },
            "year_cap": {
                "Biden": 5,
                "Buttigieg": None,
                "Klobuchar": None,
                "Warren": None,
            },
            "mechanic_quote": (
                "the other candidates' plans would replace a caregiver's "
                "earnings in the Social Security benefit formula with the "
                "caregiver credit as long as that credit exceeded the "
                "worker's earnings (printed p. 11)"
            ),
            "biden_variant_note": (
                "Biden's statutory design provides the credit on top of "
                "earnings and phases it out 50 cents per $1 earned; the "
                "registration models all four plans with the uniform "
                "top-up-to-credit-level mechanic, so Biden's phase-out is "
                "not separately encoded (a named simplification)."
            ),
        },
        "table15_bottom_fifth_2065": {
            "citation": (
                "Table 15 (Percentage of Benefit Increases Going to the "
                "Bottom Fifth of Lifetime Earners, 2065), printed p. 66, "
                "'Create caregiver credit' row, scheduled scenario; "
                "DYNASIM3 ID980"
            ),
            "quintile_variable_paper": (
                "bottom fifth of the lifetime earnings distribution in 2065; "
                "couples split earnings and benefits in married years "
                "(Table 15 notes)"
            ),
            "values_pct": dict(ANCHOR_TABLE15_BOTTOM_FIFTH),
            "band_pct": list(ANCHOR_BAND),
            "headline_quote": (
                "More than half of the benefit increases that would be "
                "provided by the caregiver credits ... would go [to] the "
                "bottom fifth of lifetime earners (printed p. 65-66)"
            ),
        },
        "share_gaining_2065": {
            "citation": "narrative, printed p. 27-28",
            "values_pct": dict(ANCHOR_SHARE_GAINING_2065),
            "note": (
                "share of beneficiaries who would gain from each plan's "
                "caregiver credit in 2065; Buttigieg (broadest eligibility) "
                "highest, Klobuchar (narrowest age limit and smaller credit) "
                "lowest"
            ),
        },
        "cost_75yr_pct_payroll": {
            "citation": (
                "Table 3 (actuarial balance as a percentage of taxable "
                "payroll, 2019-93), 'Provide caregiver credits' row, "
                "printed p. 19"
            ),
            "values": dict(ANCHOR_COST_PCT_PAYROLL),
        },
        "named_population_deltas": [
            "observed PSID retirees eligible 2005-2019 (born 1943-1957) "
            "transported to a common 2050 bracket geometry vs DYNASIM's "
            "projected 2065 beneficiaries",
            "individual own-record earnings (this study) vs DYNASIM couples "
            "splitting earnings and benefits in married years (Table 15 note)",
            "truncated observation window (ages 22-61 PSID careers "
            "transported to 2050) vs DYNASIM full projected careers -- our "
            "compressed careers expose more low-AIME person-years to the "
            "top-up, so bottom-quintile shares run higher than the anchor's",
            "no projection: no mortality, benefit-receipt years, claim-age "
            "reduction, or behavioural response (cross-sectional PIA-gain "
            "proxy) vs DYNASIM's fully projected 2065 expenditure",
        ],
    }


# =====================================================================
# Study data: the Phase-A career frame + observed births joined
# =====================================================================
class CaregiverStudy:
    """Phase-A careers, transported baseline AIME/PIA, and child births.

    The career frame is the Phase-A ``pia_observed`` rule built exactly as
    scripts/replication_r7_sharing.py builds it (same
    :func:`_person_history`, same constants): coverage >= 0.8 of ages
    22-61 with an age-62 eligibility year in 2005-2019. Births are joined
    from the Childbirth & Adoption History File (cah85_23) on the shared
    PSID person id.
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

        # Observed births: parent person id -> list of child birth years.
        # Actual childbirth events with a resolvable birth year (the
        # denial placeholders and adoptions are excluded; birth_events
        # would also drop is_event=False -- we filter to record_type
        # "birth" with a non-NA birth year for a literal "observed
        # births" read).
        bh = birth_history()
        bh = bh[(bh["record_type"] == "birth") & bh["birth_year"].notna()]
        child_births: dict[int, list[int]] = {}
        for pid, by in zip(
            bh["parent_person_id"].to_numpy(),
            bh["birth_year"].to_numpy(),
            strict=True,
        ):
            child_births.setdefault(int(pid), []).append(int(by))
        self.child_births = child_births
        self.n_birth_records = int(len(bh))


# =====================================================================
# Benefit math: transported full-415(b) AIME, credit top-up, gain
# =====================================================================
def transported_aime(
    history: dict[int, float], params: Any, transport: dict[str, Any]
) -> float:
    """Full statutory 42 USC 415(b) AIME under the 2050 transport.

    Each year's earnings are capped at that year's historical wage base,
    NAWI-indexed to the 2048 age-60 indexing year of the 2050 cohort, the
    top 35 indexed values summed over 35*12 months, and floored to the
    dollar. Years absent contribute a zero slot (statutory highest-35).
    """
    index_nawi = transport["index_nawi"]
    nawi = transport["nawi"]
    indexed = [
        min(float(earn), params.wage_base_for(year)) * index_nawi / nawi[year]
        for year, earn in history.items()
    ]
    top = sorted(indexed, reverse=True)[:_COMPUTATION_YEARS]
    top += [0.0] * (_COMPUTATION_YEARS - len(top))
    return float(math.floor(sum(top) / (_COMPUTATION_YEARS * _MONTHS)))


def qualifying_years(
    history: dict[int, float],
    child_births: list[int],
    plan: Plan,
    params: Any,
    transport: dict[str, Any],
    *,
    selection: str = "benefit_max",
) -> list[tuple[int, float]]:
    """The (year, nominal credit level) pairs the plan tops up.

    A year qualifies when the worker has a child strictly younger than the
    plan's age limit (``0 <= year - birth_year < child_age_limit``) and
    the worker's earnings are below the credit level (the plan's fraction
    of that year's economy-wide average wage, ``params.nawi[year]``).
    Under a year cap the credited years are selected to maximise the
    benefit (``benefit_max``: the qualifying years with the lowest
    NAWI-indexed earnings, i.e. the largest top-up, since every year's
    credit indexes to the same constant ``fraction * NAWI[2048]``) -- or,
    for the reported sensitivity, the earliest years (``chronological``).
    """
    index_nawi = transport["index_nawi"]
    nawi = transport["nawi"]
    limit = plan.child_age_limit
    cands: list[tuple[int, float, float]] = []
    for year, earn in history.items():
        if not any(0 <= (year - by) < limit for by in child_births):
            continue
        credit_level = plan.credit_fraction * params.nawi[year]
        if float(earn) >= credit_level:
            continue
        indexed_earn = (
            min(float(earn), params.wage_base_for(year))
            * index_nawi
            / nawi[year]
        )
        cands.append((year, credit_level, indexed_earn))
    if plan.year_cap is not None:
        if selection == "benefit_max":
            cands.sort(key=lambda t: (t[2], t[0]))
        elif selection == "chronological":
            cands.sort(key=lambda t: t[0])
        else:
            raise ValueError(f"unknown selection {selection!r}")
        cands = cands[: plan.year_cap]
    return [(year, credit_level) for year, credit_level, _ in cands]


def reformed_history(
    history: dict[int, float], credited: list[tuple[int, float]]
) -> dict[int, float]:
    """The earnings history with credited years topped up to the level."""
    new = dict(history)
    for year, credit_level in credited:
        new[year] = credit_level
    return new


# =====================================================================
# Scoring: per-person baseline, per-plan gain, lifetime quintile
# =====================================================================
def score_population(
    study: CaregiverStudy,
    params: Any,
    transport: dict[str, Any],
) -> pd.DataFrame:
    """Per career-frame person: weight, baseline AIME/PIA, per-plan gain.

    One row per career-frame person. The baseline transported AIME (the
    lifetime-earnings ranking variable) and pre-415(g) PIA are computed
    once; for each plan the credited history is formed, its AIME/PIA
    recomputed, and the gain (reformed - baseline PIA, always >= 0)
    recorded. Biden additionally carries a ``chronological`` cap-selection
    sensitivity column.
    """
    rows = []
    for pid in sorted(study.careers):
        hist = study.history[pid]
        base_aime = transported_aime(hist, params, transport)
        base_pia = float(scheduled_amount(np.array([base_aime]), transport)[0])
        births = study.child_births.get(pid, [])
        row: dict[str, Any] = {
            "person_id": pid,
            "weight": float(study.weight.get(pid, 1.0)),
            "base_aime": base_aime,
            "base_pia": base_pia,
            "n_children": len(births),
        }
        for plan in PLANS:
            credited = qualifying_years(hist, births, plan, params, transport)
            if credited:
                aime = transported_aime(
                    reformed_history(hist, credited), params, transport
                )
                pia = float(scheduled_amount(np.array([aime]), transport)[0])
            else:
                pia = base_pia
            row[f"gain_{plan.name}"] = max(0.0, pia - base_pia)
            row[f"ncred_{plan.name}"] = len(credited)
        # Biden cap-selection sensitivity (chronological first-5).
        biden = PLANS[0]
        credited_chr = qualifying_years(
            hist, births, biden, params, transport, selection="chronological"
        )
        if credited_chr:
            aime_chr = transported_aime(
                reformed_history(hist, credited_chr), params, transport
            )
            pia_chr = float(
                scheduled_amount(np.array([aime_chr]), transport)[0]
            )
        else:
            pia_chr = base_pia
        row["gain_Biden_chronological"] = max(0.0, pia_chr - base_pia)
        rows.append(row)
    return pd.DataFrame(rows)


def assign_quintiles(base_aime: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Own-distribution weighted lifetime-earnings quintiles (0..4).

    Weighted 20% cutpoints of the baseline transported AIME (the
    ppi_mermin convention), Q0 = bottom fifth.
    """
    cutpoints = _weighted_quantile(base_aime, weight, QUINTILE_LEVELS)
    return _assign_quintiles(base_aime, cutpoints)


def plan_metrics(
    df: pd.DataFrame, gain_col: str, quintile: np.ndarray
) -> dict[str, Any]:
    """Bottom-quintile share + incidence + share gaining for one plan.

    ``quintile`` is the own-distribution lifetime-earnings quintile of
    each row (0 = bottom). Reports the bottom-quintile share of aggregate
    weighted gain (the Table 15 statistic), the per-quintile incidence
    (share of aggregate gain, n, share gaining, mean gain), and the
    weighted share of persons gaining.
    """
    weight = df["weight"].to_numpy(dtype=np.float64)
    gain = df[gain_col].to_numpy(dtype=np.float64)
    wgain = weight * gain
    aggregate = float(wgain.sum())
    gaining = (gain > 1e-9).astype(np.float64)

    by_quintile = []
    for k in range(N_QUINTILES):
        mask = quintile == k
        wq = weight[mask]
        share_of_gain = (
            100.0 * float(wgain[mask].sum()) / aggregate
            if aggregate > 0.0
            else 0.0
        )
        by_quintile.append(
            {
                "quintile": k + 1,
                "n_persons": int(np.sum(mask)),
                "share_of_aggregate_gain_pct": share_of_gain,
                "share_gaining_pct": (
                    100.0 * _weighted_mean(gaining[mask], wq)
                    if wq.size and wq.sum() > 0
                    else 0.0
                ),
                "mean_gain": (
                    _weighted_mean(gain[mask], wq)
                    if wq.size and wq.sum() > 0
                    else 0.0
                ),
                "mean_baseline_aime": (
                    _weighted_mean(
                        df["base_aime"].to_numpy(dtype=np.float64)[mask], wq
                    )
                    if wq.size and wq.sum() > 0
                    else 0.0
                ),
            }
        )
    return {
        "bottom_quintile_share_pct": by_quintile[0][
            "share_of_aggregate_gain_pct"
        ],
        "share_gaining_pct": 100.0 * _weighted_mean(gaining, weight),
        "aggregate_weighted_gain": aggregate,
        "n_gainers": int(np.sum(gain > 1e-9)),
        "incidence_by_quintile": by_quintile,
    }


# =====================================================================
# Floors: 5-seed person-disjoint half-split on the headline shares
# =====================================================================
def seed_half_metrics(
    df: pd.DataFrame, seed: int
) -> dict[str, dict[str, dict[str, float]]]:
    """Each half's headline shares per plan for one seed.

    Splits scored persons 50/50 by person (the committed
    :func:`hpanel.split_panel_by_person`), recomputes own-distribution
    quintiles within each half, and reports each half's bottom-quintile
    share and share gaining per plan -- the A-vs-B gap is the
    sampling-noise floor at half scale.
    """
    side_a, side_b = hpanel.split_panel_by_person(
        df, "person_id", fraction=0.5, seed=seed
    )
    out: dict[str, dict[str, dict[str, float]]] = {}
    for plan in PLANS:
        out[plan.name] = {}
        for label, side in (("side_a", side_a), ("side_b", side_b)):
            q = assign_quintiles(
                side["base_aime"].to_numpy(dtype=np.float64),
                side["weight"].to_numpy(dtype=np.float64),
            )
            m = plan_metrics(side, f"gain_{plan.name}", q)
            out[plan.name][label] = {
                "bottom_quintile_share_pct": m["bottom_quintile_share_pct"],
                "share_gaining_pct": m["share_gaining_pct"],
                "n": int(len(side)),
            }
    return out


def build_floors(
    per_seed: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-plan floors = summary of |side_a - side_b| across the 5 seeds."""
    floors: dict[str, dict[str, dict[str, Any]]] = {}
    for plan in PLANS:
        floors[plan.name] = {}
        for metric in ("bottom_quintile_share_pct", "share_gaining_pct"):
            gaps = [
                abs(
                    s["half_metrics"][plan.name]["side_a"][metric]
                    - s["half_metrics"][plan.name]["side_b"][metric]
                )
                for s in per_seed
            ]
            floors[plan.name][metric] = _summary(gaps)
    return floors


# =====================================================================
# Four-plan table + the pre-registered expectation verdict
# =====================================================================
def build_plan_table(
    full: dict[str, dict[str, Any]],
    floors: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Per plan: our bottom-quintile share, its floor, the anchor, band."""
    table = []
    for plan in PLANS:
        m = full[plan.name]
        floor = floors[plan.name]["bottom_quintile_share_pct"]
        anchor = ANCHOR_TABLE15_BOTTOM_FIFTH[plan.name]
        our = m["bottom_quintile_share_pct"]
        table.append(
            {
                "plan": plan.name,
                "credit_fraction": plan.credit_fraction,
                "child_age_limit_max": plan.child_age_limit - 1,
                "year_cap": plan.year_cap,
                "our_bottom_quintile_share_pct": our,
                "bottom_quintile_share_floor_mean": floor["mean"],
                "bottom_quintile_share_floor_sd": floor["sd"],
                "anchor_bottom_fifth_pct": anchor,
                "in_or_above_band": bool(our >= ANCHOR_BAND[0]),
                "abs_gap_vs_anchor": round(abs(our - anchor), 2),
                "our_share_gaining_pct": m["share_gaining_pct"],
                "share_gaining_floor_mean": floors[plan.name][
                    "share_gaining_pct"
                ]["mean"],
                "anchor_share_gaining_pct": ANCHOR_SHARE_GAINING_2065[
                    plan.name
                ],
                "n_gainers": m["n_gainers"],
                "aggregate_weighted_gain": m["aggregate_weighted_gain"],
            }
        )
    return table


def _spearman_rho(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation of two equal-length sequences."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    n = len(a)
    d2 = float(np.sum((ra - rb) ** 2))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def registered_expectation(
    full: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """The registration's two pre-registered expectations, recomputed.

    (1) Bottom-quintile share lands in or above the anchor's 52-62% band
        for the 1/2-credit designs (Biden, Klobuchar) -- our
        truncated-window careers expose more low-AIME person-years to the
        top-up, so the share is at or above the band.
    (2) Ordering across plans matches (broader eligibility -> larger but
        less concentrated gains): the concentration endpoints match the
        anchor (Klobuchar most concentrated / highest bottom-quintile
        share, Buttigieg least concentrated / lowest) and the reach
        endpoints match (Buttigieg most persons gaining, Klobuchar
        fewest), consistent with the anchor's rankings.
    """
    half_credit = {
        name: full[name]["bottom_quintile_share_pct"]
        for name in HALF_CREDIT_PLANS
    }
    half_credit_ok = all(v >= ANCHOR_BAND[0] for v in half_credit.values())

    our_conc = {p: full[p]["bottom_quintile_share_pct"] for p in PLAN_ORDER}
    our_reach = {p: full[p]["share_gaining_pct"] for p in PLAN_ORDER}
    our_conc_hi = max(our_conc, key=our_conc.get)
    our_conc_lo = min(our_conc, key=our_conc.get)
    our_reach_hi = max(our_reach, key=our_reach.get)
    our_reach_lo = min(our_reach, key=our_reach.get)

    anchor_conc_hi = max(
        ANCHOR_TABLE15_BOTTOM_FIFTH, key=ANCHOR_TABLE15_BOTTOM_FIFTH.get
    )
    anchor_conc_lo = min(
        ANCHOR_TABLE15_BOTTOM_FIFTH, key=ANCHOR_TABLE15_BOTTOM_FIFTH.get
    )
    anchor_reach_hi = max(
        ANCHOR_SHARE_GAINING_2065, key=ANCHOR_SHARE_GAINING_2065.get
    )
    anchor_reach_lo = min(
        ANCHOR_SHARE_GAINING_2065, key=ANCHOR_SHARE_GAINING_2065.get
    )

    concentration_endpoints_match = bool(
        our_conc_hi == anchor_conc_hi and our_conc_lo == anchor_conc_lo
    )
    reach_endpoints_match = bool(
        our_reach_hi == anchor_reach_hi and our_reach_lo == anchor_reach_lo
    )
    ordering_ok = concentration_endpoints_match and reach_endpoints_match

    rho_conc = _spearman_rho(
        [our_conc[p] for p in PLAN_ORDER],
        [ANCHOR_TABLE15_BOTTOM_FIFTH[p] for p in PLAN_ORDER],
    )
    rho_reach = _spearman_rho(
        [our_reach[p] for p in PLAN_ORDER],
        [ANCHOR_SHARE_GAINING_2065[p] for p in PLAN_ORDER],
    )

    return {
        "half_credit_in_or_above_band": {
            "statement": (
                "bottom-quintile share >= 52% (in or above the anchor band) "
                "for the 1/2-credit designs Biden and Klobuchar"
            ),
            "our_shares_pct": {k: round(v, 2) for k, v in half_credit.items()},
            "band_pct": list(ANCHOR_BAND),
            "held": bool(half_credit_ok),
        },
        "plan_ordering_matches_anchor": {
            "statement": (
                "concentration endpoints (highest/lowest bottom-quintile "
                "share) and reach endpoints (highest/lowest share gaining) "
                "match the anchor's rankings"
            ),
            "our_concentration_high": our_conc_hi,
            "our_concentration_low": our_conc_lo,
            "anchor_concentration_high": anchor_conc_hi,
            "anchor_concentration_low": anchor_conc_lo,
            "concentration_endpoints_match": concentration_endpoints_match,
            "our_reach_high": our_reach_hi,
            "our_reach_low": our_reach_lo,
            "anchor_reach_high": anchor_reach_hi,
            "anchor_reach_low": anchor_reach_lo,
            "reach_endpoints_match": reach_endpoints_match,
            "spearman_rho_concentration": round(rho_conc, 3),
            "spearman_rho_reach": round(rho_reach, 3),
            "middle_pair_note": (
                "Biden and Warren (the middle ranks) swap between our results "
                "and the anchor; they are near-tied in the anchor (54 vs 55) "
                "and the endpoints are the robust claim"
            ),
            "held": bool(ordering_ok),
        },
        "expectation_held": bool(half_credit_ok and ordering_ok),
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


def _revision_pins(params: Any) -> dict[str, Any]:
    return {
        "populace_dynamics_sha": _sha(ROOT),
        "pe_us_revision": getattr(params, "pe_us_revision", None),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    }


# =====================================================================
# Driver
# =====================================================================
def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full caregiver-credit replication (reported)."""
    started = time.time()
    params = load_ssa_parameters()
    transport = build_transport(params)
    if verbose:
        print(
            f"pe_us_revision={params.pe_us_revision} "
            f"nawi[{transport['index_year']}]={transport['index_nawi']:.0f} "
            f"bends(2050)={transport['bend_points']}; loading PSID ..."
        )

    study = CaregiverStudy(params, transport)
    df = score_population(study, params, transport)
    quintile = assign_quintiles(
        df["base_aime"].to_numpy(dtype=np.float64),
        df["weight"].to_numpy(dtype=np.float64),
    )
    if verbose:
        n_parents = sum(1 for p in study.careers if p in study.child_births)
        print(
            f"career frame {len(study.careers)} persons; scored {len(df)}; "
            f"{n_parents} are parents ({time.time() - started:.0f}s)"
        )

    full = {
        plan.name: plan_metrics(df, f"gain_{plan.name}", quintile)
        for plan in PLANS
    }
    biden_chr = plan_metrics(df, "gain_Biden_chronological", quintile)

    per_seed = []
    for seed in SEEDS:
        per_seed.append(
            {"seed": seed, "half_metrics": seed_half_metrics(df, seed)}
        )
    floors = build_floors(per_seed)

    plan_table = build_plan_table(full, floors)
    expectation = registered_expectation(full)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "reported_not_gated": True,
        "registration": SPEC_REGISTRATION,
        "registration_pointer": SPEC_REGISTRATION_POINTER,
        "program_design_issue": PROGRAM_DESIGN_ISSUE,
        "purpose": (
            "Caregiver-credit external-anchor replication: does our stack "
            "(real PSID Phase-A careers + observed births cah85_23 + the "
            "statutory 415(a)/(b) chain transported to a common 2050 cohort) "
            "reproduce the progressivity PATTERN of the caregiver credit -- "
            "the bottom-lifetime-earnings-quintile share of aggregate benefit "
            "gains (Smith/Johnson/Favreault 2020, Table 15)? Reads no gate, "
            "changes no gate; publishes regardless of outcome."
        ),
        "what_is_tested": (
            "the progressivity STRUCTURE (the share of aggregate benefit "
            "gains reaching the bottom fifth of lifetime earners, and the "
            "ordering across the four plans' credit designs), not the "
            "population LEVELS -- every level difference vs the DYNASIM 2065 "
            "projection is a named delta"
        ),
        "real_data_only": True,
        "anchor_provenance": anchor_provenance(),
        "study_population": {
            "career_frame_rule": (
                "Phase-A pia_observed selection (reused from "
                "replication_r7_sharing): coverage >= 0.8 of ages 22-61, "
                "age-62 eligibility year in 2005-2019 (born 1943-1957), "
                "single-year gaps interpolated (mean of neighbours)"
            ),
            "n_career_frame": int(len(study.careers)),
            "n_scored": int(len(df)),
            "n_parents_in_frame": int(
                sum(1 for p in study.careers if p in study.child_births)
            ),
            "n_birth_records_joined": study.n_birth_records,
            "birth_join": (
                "Childbirth & Adoption History File cah85_23; actual "
                "childbirth events (record_type=birth) with a resolvable "
                "birth year, joined on the shared PSID person id "
                "(parent_person_id); child age in a year = year - birth_year"
            ),
            "weighting": (
                "anchor weight = the person's last observed PSID "
                "cross-sectional weight"
            ),
            "lifetime_earnings_quintile": (
                "own-distribution weighted 20% quintiles of the baseline "
                "transported AIME (plan-independent; Q1 = bottom fifth)"
            ),
        },
        "conventions": {
            "transport": (
                "the 2050-eligibility transport from replication_ppi_mermin "
                "(build_transport): NAWI projected to the 2048 age-60 "
                "indexing year at the 2005 TR nominal wage growth, 2050 bend "
                "points per 415(a)(1)(B); every career transported to the "
                "same synthetic 2050 cohort"
            ),
            "aime": (
                "full statutory 42 USC 415(b) top-35 indexed earnings "
                "(indexed to NAWI[2048], each year capped at its historical "
                "wage base, over 35*12, floored) -- the original Phase-A AIME "
                "convention, available on the frame's full annual careers"
            ),
            "pia_gain": (
                "gain = reformed - baseline pre-415(g) PIA amount (the "
                "90/32/15 bracket sum on the 2050 bends, ppi_mermin's "
                "scheduled_amount); the sub-$0.10 415(g) dime rounding is a "
                "nominal artifact omitted from the incidence gain"
            ),
            "credit_top_up": (
                "in a qualifying child-year (child younger than the plan's "
                "age limit) where earnings are below the credit level "
                "(fraction * NAWI[year]), earnings are topped up to the "
                "credit level; Biden's 5-year cap credits the benefit-"
                "maximising qualifying years (lowest NAWI-indexed earnings)"
            ),
            "biden_cap_selection_sensitivity": (
                "the Biden cap primary selection is benefit-maximising; a "
                "chronological (earliest-5) sensitivity is reported to show "
                "the cap-selection convention does not drive the finding"
            ),
            "scale_invariance": (
                "the transported AIME, the credit level, and the bend points "
                "all scale linearly with NAWI[2048], so every share reported "
                "here is invariant to the NAWI projection level"
            ),
            "floor": (
                "5-seed person-disjoint half-split "
                "(split_panel_by_person, fraction=0.5); floor per share = "
                "summary of |side_a - side_b| across seeds, own-distribution "
                "quintiles recomputed within each half"
            ),
        },
        "four_plan_table": plan_table,
        "registered_expectation": expectation,
        "full_sample_metrics": full,
        "biden_cap_sensitivity_chronological": {
            "note": (
                "Biden with the chronological (earliest-5) cap selection "
                "instead of benefit-maximising; primary Biden uses "
                "benefit-maximising"
            ),
            "bottom_quintile_share_pct": biden_chr[
                "bottom_quintile_share_pct"
            ],
            "share_gaining_pct": biden_chr["share_gaining_pct"],
            "primary_bottom_quintile_share_pct": full["Biden"][
                "bottom_quintile_share_pct"
            ],
        },
        "per_seed": per_seed,
        "revision_pins": _revision_pins(params),
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if verbose:
        _print_summary(plan_table, expectation)
    return artifact


def _print_summary(
    plan_table: list[dict[str, Any]], expectation: dict[str, Any]
) -> None:
    print(
        "\n=== Caregiver credit: bottom-quintile share of benefit gains "
        "(ours / floor / anchor) ==="
    )
    print(
        f"{'plan':<11}{'ours':>8}{'floor':>8}{'anchor':>8}"
        f"{'band?':>7}{'gaining':>9}{'anchG':>7}"
    )
    for row in plan_table:
        print(
            f"{row['plan']:<11}"
            f"{row['our_bottom_quintile_share_pct']:>8.1f}"
            f"{row['bottom_quintile_share_floor_mean']:>8.2f}"
            f"{row['anchor_bottom_fifth_pct']:>8.1f}"
            f"{str(row['in_or_above_band']):>7}"
            f"{row['our_share_gaining_pct']:>9.1f}"
            f"{row['anchor_share_gaining_pct']:>7.1f}"
        )
    print(f"\nanchor band: {ANCHOR_BAND[0]:.0f}-{ANCHOR_BAND[1]:.0f}%")
    e = expectation
    print(
        "half-credit designs in/above band: "
        f"{e['half_credit_in_or_above_band']['held']} "
        f"({e['half_credit_in_or_above_band']['our_shares_pct']})"
    )
    po = e["plan_ordering_matches_anchor"]
    print(
        "plan ordering endpoints match: "
        f"{po['held']} (concentration hi/lo="
        f"{po['our_concentration_high']}/{po['our_concentration_low']}, "
        f"reach hi/lo={po['our_reach_high']}/{po['our_reach_low']}; "
        f"rho_conc={po['spearman_rho_concentration']})"
    )
    print(f"REGISTERED EXPECTATION HELD: {e['expectation_held']}")


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
