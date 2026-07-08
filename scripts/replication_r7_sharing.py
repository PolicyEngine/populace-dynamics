"""Replication R7 (reported, not gated): earnings sharing on real PSID
couples vs Favreault & Steuerle (2007).

REPORTED, NOT GATED. This artifact reads no gate and changes no gate. It
is the earnings-sharing external-anchor replication of PolicyEngine/
populace-dynamics (issue #74, anchor 5): does our stack -- real PSID
marriage histories (the Marriage History File mh85_23) + real earnings
(the family panel) + the statutory chain (42 USC 415(a)/(b)) + the
spouse/survivor auxiliary-benefit plumbing (#80,
:mod:`populace_dynamics.household`) -- reproduce the *incidence
structure* of earnings sharing: who wins and who loses, by sex and
marital status?

Frozen spec: issue #42 comment 4911171806. Where this module and the
registration disagree, the registration wins. The anchor is Favreault,
M. M. and Steuerle, C. E. (2007), "Social Security Spouse and Survivor
Benefits for the Modern Family" (Urban Institute report 311436, DYNASIM3
runid 440v2): winners and losers under earnings-sharing packages by sex
x marital status (Table 3), for the 1960-80 birth cohorts evaluated in
2049. Package 1b (earnings sharing, no survivor benefit) is the primary
target; package 1a (survivor on the maximum shared vector) is run
descriptively; package 1c (self-financed survivor annuity) is documented
as not cleanly implementable (see :data:`PACKAGE_1C_NOT_IMPLEMENTED`).

=====================================================================
REAL COUPLES ONLY
=====================================================================
This diagnostic scores only *real* PSID couples: a study person from the
Phase-A career-selection frame whose spouse(s) also carry a computable
primary insurance amount from an observed earnings history. There is no
generated component, so no gate-2-passing model is required (registration:
"REAL COUPLES ONLY (no generated component, so no gate-2-passing model is
required for this diagnostic)"). The five seeds drive ONLY the real-vs-
real person-disjoint half-split floors on each share; the shares
themselves are deterministic (an expected claim-age reduction, not a
sampled claim age).

=====================================================================
THE GLOBAL BENEFIT INCREASE IS PART OF PACKAGE 1b (a modeling judgment)
=====================================================================
Each Favreault-Steuerle package is *approximately cost-neutral* by
construction: removing spouse (and, in 1b, survivor) benefits saves
money, which the authors return through a scalar increase to the PIA
formula factors -- 2.71 percent for 1a, 4.5 percent for 1b/1c (Table 2,
printed p.15; the discussion on p.19-20). This scalar is part of each
package's *cited definition* (it is literally in the package name:
"Earnings sharing ... 4.5 percent increase"), and it is directly visible
in the anchor's never-married Table-3 row: 97.4% (men) / 98.7% (women)
of never-married people show a "gain < 5%" under 1b -- the uniform +4.5%
bump, which has no other source (never-married people have no spouse to
share with). Omitting it would put every never-married person at "no
change", contradicting the anchor.

The primary package-1b figures therefore apply the +4.5% scalar. Because
the registration's enumerated mechanic (split 50/50, recompute AIME/PIA,
survivor per package) does not itemize the scalar, a scalar-off variant
(:data:`SENSITIVITY_NO_SCALAR`) is reported alongside for full
transparency. The scalar is the anchor's DYNASIM-calibrated value, NOT
re-derived for cost-neutrality in our population; the artifact records
our population's aggregate cost change under it (a named delta -- our
older, denser-career PSID sample is not the 2049 projection the scalar
was balanced on).

=====================================================================
Named population deltas vs the DYNASIM 2049 projection (documented, not
hidden -- the shares are expected to match in DIRECTION, not level)
=====================================================================
* COHORT: observed PSID retirees eligible 2005-2019 (born 1943-1957) vs
  DYNASIM's projected 1960-1980 cohorts evaluated in 2049. Our older
  cohorts have more single-earner couples, so more husbands lose heavily
  under sharing and fewer married men are the couple's lower earner (the
  DYNASIM married-men gains come from lower-earning husbands and from
  within-couple claim-timing, neither prevalent here).
* SELECTION: PSID long-stayers with coverage >= 0.8 of ages 22-61 (the
  Phase-A frame) vs DYNASIM's SIPP+PSID-calibrated synthetic population.
  Our coverage-0.8 women have strong own careers, so they rely LESS on
  survivor benefits and lose LESS when 1b removes them -- our widowed
  losses run below DYNASIM's.
* SPOUSE OBSERVATION: a study person is a coverage-0.8 long-stayer, but
  their spouse's earnings are whatever the family panel observed (often
  sparser). Sharing over a sparsely observed spouse understates what the
  higher earner shares INTO, biasing the higher earner's loss upward.
* NO DYNAMICS: no within-couple claim-age timing, no mortality/survival
  projection, no behavioural response; each person is evaluated at their
  own observed eligibility with the observed-era expected claim-age
  reduction. DYNASIM projects all of these to 2049.
* NO POVERTY / LIFETIME-RATIO REPLICATION: Table 4 (lifetime tax-benefit
  ratios) and Table 5 (poverty) are transcribed for provenance but not
  reproduced -- they need lifetime accumulation and household income the
  Phase-A own-benefit frame does not carry.

Run (from the repository root, PSID family + marriage files staged)::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv/bin/python scripts/replication_r7_sharing.py
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
# as scripts/replication_ppi_mermin.py does.
from build_downstream_relevance import _weighted_mean  # noqa: E402
from reform_delta_diagnostic import _summary  # noqa: E402

from populace_dynamics import claiming, household  # noqa: E402
from populace_dynamics.data.family import family_earnings_panel  # noqa: E402
from populace_dynamics.data.marriage import (  # noqa: E402
    marriage_episodes,
    marriage_history,
)
from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.ss import benefits  # noqa: E402
from populace_dynamics.ss.params import load_ssa_parameters  # noqa: E402

ARTIFACT_PATH = ROOT / "runs" / "replication_r7_sharing_v1.json"
ARTIFACT_SCHEMA_VERSION = "replication_r7_sharing.v1"
RUN_NAME = "replication_r7_sharing_v1"

#: This replication's frozen-spec registration (issue #42 comment).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911171806"
)
SPEC_REGISTRATION_POINTER = "#42 comment 4911171806"
PROGRAM_DESIGN_ISSUE = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/74"
)

# ---- Phase-A career-selection frame (the pia_observed_psid_v1 rule) -----
#: Eligibility (age-62) year window of the observed-career frame.
ELIGIBILITY_LO, ELIGIBILITY_HI = 2005, 2019
#: Career window: ages 22 to 61 (40 ages), the pia_observed convention.
AGE_LO, AGE_HI = 22, 61
N_CAREER_AGES = AGE_HI - AGE_LO + 1  # 40
#: Coverage floor over the 40 career ages (long-stayer selection).
COVERAGE_FLOOR = 0.8
#: Age-62 eligibility offset.
ELIGIBILITY_AGE = 62

# ---- Earnings-sharing packages (Favreault-Steuerle 2007, Table 2) -------
#: Global benefit increase (scalar on the PIA formula factors) that makes
#: each package approximately cost-neutral. Table 2 (printed p.15).
GLOBAL_INCREASE = {"1a": 0.0271, "1b": 0.045}
#: The primary package.
PRIMARY_PACKAGE = "1b"

# ---- Winner/loser thresholds (Table 3 buckets) --------------------------
#: The four reported threshold shares (registration: >=5% and >=20%).
THRESHOLDS = ("lose_ge_20", "lose_ge_5", "gain_ge_5", "gain_ge_20")
#: The nine Table-3 buckets, in the anchor's order.
BUCKETS = (
    "lose_ge_20",
    "lose_10_20",
    "lose_5_10",
    "lose_lt_5",
    "no_change",
    "gain_lt_5",
    "gain_5_10",
    "gain_10_20",
    "gain_ge_20",
)
SEXES = ("male", "female")
MARITAL = ("married", "divorced", "widowed", "never_married")
SEEDS = (0, 1, 2, 3, 4)

PACKAGE_1C_NOT_IMPLEMENTED = (
    "Package 1c (earnings sharing with a self-financed survivor benefit) "
    "requires people married at benefit entitlement to 'purchase' survivor "
    "insurance worth two-thirds of their own benefit through an "
    "actuarially fair, unisex-priced benefit reduction at a 4.6 percent "
    "real interest rate (Favreault-Steuerle 2007, Table 2, printed p.15). "
    "That pricing needs individual- and spouse-specific cohort survival "
    "probabilities and an annuity-factor method the paper describes only "
    "as 'roughly actuarially fair on a cohort basis' -- it states no "
    "annuity factors and no mortality table. Implementing it would inject "
    "an unvalidated cohort-mortality/annuity model beyond this reported "
    "diagnostic's scope, so 1c is transcribed (Table 3 rows recorded in "
    "the anchor provenance) but not scored, per the registration's "
    "'else document why not' clause."
)

# =====================================================================
# Anchor: transcribed Favreault-Steuerle (2007) target rows + citations.
# Verified against 311436-spouse-survivor-modern.{pdf,txt}. Order within
# each (sex) tuple follows BUCKETS: lose>=20, lose10-20, lose5-10, lose<5,
# no change, gain<5, gain5-10, gain10-20, gain>=20.
# =====================================================================
# Table 3 (winners/losers relative to current law scheduled, 2049), by
# package -> marital status -> sex -> the nine buckets.
DYNASIM_TABLE3: dict[str, dict[str, dict[str, tuple[float, ...]]]] = {
    "1a": {
        "married": {
            "male": (7.4, 9.4, 9.9, 16.7, 0.2, 25.0, 12.1, 9.0, 10.2),
            "female": (4.5, 6.0, 8.6, 17.6, 0.0, 28.2, 13.6, 10.0, 11.5),
        },
        "divorced": {
            "male": (5.6, 12.5, 12.5, 19.1, 0.4, 32.4, 7.5, 4.6, 5.4),
            "female": (3.0, 7.2, 6.0, 12.2, 0.6, 32.5, 12.9, 9.6, 16.2),
        },
        "widowed": {
            "male": (19.6, 29.9, 16.3, 12.3, 0.2, 11.7, 4.9, 3.4, 1.8),
            "female": (15.3, 24.2, 16.0, 16.0, 1.7, 15.2, 5.7, 4.0, 1.9),
        },
        "never_married": {
            "male": (0.0, 0.0, 0.0, 0.0, 2.3, 96.3, 0.9, 0.3, 0.3),
            "female": (0.0, 0.0, 0.0, 0.0, 4.1, 95.1, 0.4, 0.2, 0.2),
        },
    },
    "1b": {
        "married": {
            "male": (18.1, 19.9, 11.2, 10.3, 0.0, 11.1, 6.0, 6.5, 17.0),
            "female": (8.4, 7.6, 6.1, 6.6, 0.0, 11.0, 6.5, 9.7, 44.1),
        },
        "divorced": {
            "male": (7.6, 11.8, 10.1, 15.1, 0.1, 34.4, 9.8, 5.7, 5.6),
            "female": (9.2, 4.7, 3.7, 7.3, 0.0, 30.4, 16.3, 11.5, 16.9),
        },
        "widowed": {
            "male": (35.6, 28.6, 14.3, 10.0, 0.0, 7.2, 2.0, 1.7, 0.8),
            "female": (38.3, 17.4, 11.0, 10.2, 0.0, 12.0, 5.5, 3.9, 1.7),
        },
        "never_married": {
            "male": (0.0, 0.1, 0.0, 0.0, 0.0, 97.4, 2.0, 0.3, 0.3),
            "female": (0.0, 0.0, 0.0, 0.0, 0.0, 98.7, 0.9, 0.3, 0.2),
        },
    },
    "1c": {
        "married": {
            "male": (9.7, 23.7, 25.1, 14.3, 0.0, 10.3, 4.2, 3.4, 9.5),
            "female": (7.4, 24.9, 28.2, 14.7, 9.1, 9.1, 3.0, 2.7, 10.0),
        },
        "divorced": {
            "male": (6.0, 10.0, 9.9, 16.3, 0.0, 34.7, 9.9, 6.3, 7.0),
            "female": (3.2, 6.0, 5.2, 8.3, 0.0, 31.0, 16.6, 12.0, 17.8),
        },
        "widowed": {
            "male": (5.8, 9.7, 8.0, 8.3, 0.0, 8.8, 6.9, 13.2, 39.3),
            "female": (5.3, 8.0, 7.2, 7.7, 0.0, 13.5, 8.0, 11.9, 38.4),
        },
        "never_married": {
            "male": (0.0, 0.1, 0.0, 0.0, 0.0, 97.4, 2.0, 0.3, 0.3),
            "female": (0.0, 0.0, 0.0, 0.0, 0.0, 98.7, 0.9, 0.3, 0.2),
        },
    },
}

# Table 5 (poverty rate, percent below poverty in 2049), transcribed for
# provenance only (not reproduced -- needs household income + poverty
# thresholds the Phase-A own-benefit frame does not carry).
DYNASIM_TABLE5_POVERTY = {
    "citation": "Table 5, printed p.24 (percent of adult current-law "
    "scheduled OASDI beneficiaries below poverty in 2049)",
    "columns": ["current_law", "1a", "1b", "1c"],
    "rows": {
        "all_women": [5.43, 5.16, 6.01, 4.88],
        "married_spouse_beneficiary_women": [1.97, 1.14, 1.14, 1.14],
        "married_spouse_not_beneficiary_women": [3.73, 3.73, 3.71, 3.86],
        "never_married_women": [8.06, 7.42, 7.13, 7.13],
        "divorced_women": [7.83, 6.23, 7.10, 5.95],
        "widowed_women": [5.90, 6.31, 8.87, 5.47],
        "all_men": [4.89, 4.74, 4.89, 4.68],
        "all_people": [5.17, 4.96, 5.48, 4.79],
    },
    "headline": (
        "1b (no survivor) is the only earnings-sharing package that "
        "raises overall poverty (5.48% vs 5.17% current law), driven by "
        "widowed women (8.87% vs 5.90%); 1a and 1c reduce it.",
    ),
}


def _thresholds_from_buckets(b: dict[str, float]) -> dict[str, float]:
    """The four reported threshold shares from a nine-bucket dict."""
    return {
        "lose_ge_20": b["lose_ge_20"],
        "lose_ge_5": b["lose_ge_20"] + b["lose_10_20"] + b["lose_5_10"],
        "gain_ge_5": b["gain_5_10"] + b["gain_10_20"] + b["gain_ge_20"],
        "gain_ge_20": b["gain_ge_20"],
    }


def dynasim_cell(package: str, status: str, sex: str) -> dict[str, Any]:
    """DYNASIM Table-3 nine buckets and four thresholds for a cell."""
    vals = DYNASIM_TABLE3[package][status][sex]
    buckets = dict(zip(BUCKETS, vals, strict=True))
    return {
        "buckets": buckets,
        "thresholds": _thresholds_from_buckets(buckets),
    }


# =====================================================================
# Study data: careers, PIA/earnings supply, marriage episodes
# =====================================================================
def _person_history(sub: pd.DataFrame, birth_year: int) -> dict[int, float]:
    """Annual earnings over ages 22-61 with single-year gaps interpolated.

    ``sub`` is one person's family-panel rows. Earnings are summed per
    reference year (one row per year normally), restricted to the
    calendar years the person is aged 22-61, and single missing years
    *between two observed years* are filled with the mean of their
    neighbours (the pia_observed "single_year_gaps: interpolated (mean of
    neighbors)" convention -- this fills the biennial 1999+ gaps). Returns
    ``{year: earnings}`` over the covered years.
    """
    lo, hi = birth_year + AGE_LO, birth_year + AGE_HI
    series = (
        sub.loc[(sub["period"] >= lo) & (sub["period"] <= hi)]
        .groupby("period")["earnings"]
        .sum()
    )
    if series.empty:
        return {}
    full = pd.Series(
        index=range(int(series.index.min()), int(series.index.max()) + 1),
        dtype=float,
    )
    full.loc[series.index] = series.to_numpy()
    for year in list(full.index[full.isna()]):
        left, right = full.get(year - 1), full.get(year + 1)
        if (
            (year - 1) in full.index
            and (year + 1) in full.index
            and not np.isnan(left)
            and not np.isnan(right)
        ):
            full.loc[year] = (left + right) / 2.0
    covered = full.dropna()
    return {int(y): float(v) for y, v in covered.items()}


class StudyData:
    """Loaded PSID study inputs: histories, careers, PIA supply, marriage.

    All person-keyed maps use the shared PSID person id
    (``1968_interview * 1000 + person_number``). Built once (the load is
    the expensive step); every scoring pass reuses it.
    """

    def __init__(self, params: Any) -> None:
        self.params = params
        self.max_nawi_year = max(params.nawi)

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

        # Every person's earnings history (for sharing) once.
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

        # PIA supply: any person with positive earnings whose AIME is
        # indexable (age-60 year within the NAWI series) -- the "supplied
        # PIA from a certified earnings history" household.py models.
        self.pia: dict[int, float] = {}
        for pid, hist in self.history.items():
            b = self.birth_year[pid]
            if b + 60 > self.max_nawi_year:
                continue
            if any(v > 0 for v in hist.values()):
                aime = benefits.aime(hist, b, self.params)
                self.pia[pid] = benefits.pia(
                    aime, b + ELIGIBILITY_AGE, self.params
                )

        mh = marriage_history()
        self.episodes = marriage_episodes(mh)
        one = mh.drop_duplicates("person_id").set_index("person_id")
        self.sex = one["sex"].to_dict()
        self.last_known_status = one["last_known_status"].to_dict()
        self.ep_by_person = {
            int(p): g for p, g in self.episodes.groupby("person_id")
        }

    # ---- marital status at the evaluation (eligibility) age ----------
    def status_at_eligibility(self, pid: int) -> str:
        """Marital status at the person's age-62 eligibility year.

        Reconstructed from the marriage episodes: an active marriage at
        the eligibility year is ``married``; otherwise the most recent
        marriage that ended by then gives ``widowed`` (widowhood) or
        ``divorced`` (divorce/separation); no marriage by then is
        ``never_married``. Falls back to the file's ``last_known_status``
        when episode years are missing.
        """
        elig = self.birth_year[pid] + ELIGIBILITY_AGE
        episodes = self.ep_by_person.get(pid)
        if episodes is None or len(episodes) == 0:
            return "never_married"
        active, ended = [], []
        for row in episodes.itertuples(index=False):
            start = row.start_year
            if pd.isna(start):
                continue
            end = row.episode_end_year
            if int(start) <= elig and (pd.isna(end) or int(end) > elig):
                active.append(row)
            elif not pd.isna(end) and int(end) <= elig:
                ended.append(row)
        if active:
            return "married"
        if ended:
            last = max(ended, key=lambda r: int(r.episode_end_year))
            if last.how_ended == "widowhood":
                return "widowed"
            if last.how_ended in ("divorce", "separated"):
                return "divorced"
        fallback = self.last_known_status.get(pid)
        return {
            "married": "married",
            "widowed": "widowed",
            "divorced": "divorced",
            "separated": "divorced",
            "never_married": "never_married",
        }.get(fallback, "never_married")

    def relevant_episodes(self, pid: int) -> list[Any]:
        """Marriage episodes with a known start relevant to the career.

        Every actual marriage that began by the person's eligibility year
        (with a resolvable start year); these are the marriages over
        which earnings are shared and whose spouses must be computable.
        """
        episodes = self.ep_by_person.get(pid)
        if episodes is None:
            return []
        elig = self.birth_year[pid] + ELIGIBILITY_AGE
        out = []
        for row in episodes.itertuples(index=False):
            if pd.isna(row.start_year) or int(row.start_year) > elig:
                continue
            out.append(row)
        return out

    def both_spouse_covered(self, pid: int) -> bool:
        """Whether every relevant marriage's spouse has a computable PIA.

        Never-married people need no spouse and are trivially covered.
        For ever-married people, each relevant marriage must have a
        joinable spouse who is in the PIA supply (an observed, indexable
        earnings history) -- so the shared history is complete and the
        baseline auxiliary is scorable.
        """
        episodes = self.relevant_episodes(pid)
        if not episodes:
            return True
        for row in episodes:
            spouse = row.spouse_person_id
            if pd.isna(spouse) or int(spouse) not in self.pia:
                return False
        return True


# =====================================================================
# Benefit math: baseline and reformed monthly benefit for one person
# =====================================================================
def _expected_own_factor(study: StudyData, pid: int, sex: str) -> float:
    """Expected benefit-to-PIA factor over the observed-era claim-age
    distribution for this person's sex and eligibility year (the B2
    module's :func:`claiming.expected_reduction_factor`)."""
    b = study.birth_year[pid]
    elig = b + ELIGIBILITY_AGE
    try:
        return claiming.expected_reduction_factor(sex, elig, b, study.params)
    except Exception:
        return claiming.benefit_factor(ELIGIBILITY_AGE * 12, b, study.params)


def _claim_pmf(study: StudyData, pid: int, sex: str) -> dict[int, float]:
    b = study.birth_year[pid]
    try:
        return claiming.claim_age_pmf(sex, b + ELIGIBILITY_AGE)
    except Exception:
        return {ELIGIBILITY_AGE: 1.0}


def shared_history(
    study: StudyData, pid: int, own_hist: dict[int, float], elig: int
) -> dict[int, float]:
    """The person's earnings history after 50/50 earnings sharing (1a/1b).

    For each relevant marriage with a computable spouse, every year of the
    marriage's duration takes ``0.5 * (own + spouse)`` capped at that
    year's taxable maximum before sharing (Table 2: "Spouses share all
    earnings (capped at the taxable maximum prior to sharing) over the
    duration of marriage"); ongoing marriages share through the
    eligibility year. Years outside every marriage keep the person's own
    earnings. Callers pass only both-spouse-covered persons.
    """
    shared = dict(own_hist)
    for row in study.relevant_episodes(pid):
        spouse = row.spouse_person_id
        if pd.isna(spouse):
            continue
        sp = int(spouse)
        spouse_hist = study.history.get(sp, {})
        start = int(row.start_year)
        end = (
            int(row.episode_end_year)
            if not pd.isna(row.episode_end_year)
            else elig
        )
        for year in range(start, end + 1):
            if year not in own_hist and year not in spouse_hist:
                continue
            own_cap = min(
                own_hist.get(year, 0.0), study.params.wage_base_for(year)
            )
            sp_cap = min(
                spouse_hist.get(year, 0.0), study.params.wage_base_for(year)
            )
            shared[year] = 0.5 * (own_cap + sp_cap)
    return shared


def _current_spouse_pia(study: StudyData, pid: int, elig: int) -> float | None:
    for row in study.relevant_episodes(pid):
        end = row.episode_end_year
        if pd.isna(end) or int(end) > elig:
            spouse = row.spouse_person_id
            if not pd.isna(spouse) and int(spouse) in study.pia:
                return study.pia[int(spouse)]
    return None


def _last_widowhood_spouse(study: StudyData, pid: int, elig: int):
    widowhoods = [
        row
        for row in study.relevant_episodes(pid)
        if row.how_ended == "widowhood"
        and not pd.isna(row.episode_end_year)
        and int(row.episode_end_year) <= elig
    ]
    if not widowhoods:
        return None
    return max(widowhoods, key=lambda r: int(r.episode_end_year))


def baseline_benefit(
    study: StudyData, pid: int, sex: str, status: str, own_pia: float
) -> float:
    """Current-law monthly benefit: own + spousal top-up + survivor.

    * own reduced retirement benefit (PIA x expected claim-age factor);
    * plus the excess spouse's benefit if married with a computable
      spouse (42 USC 402(b)/(c), :func:`household.excess_spousal_benefit`,
      expected over the claim-age distribution);
    * OR the widow(er)'s benefit if widowhood is observed with a
      computable deceased spouse (42 USC 402(e)/(f),
      :func:`household.survivor_benefit_at_death`, expected over the
      survivor's and deceased's claim-age distributions -- the survivor
      is paid the larger of their own or the widow(er) amount).
    """
    b = study.birth_year[pid]
    elig = b + ELIGIBILITY_AGE
    factor = _expected_own_factor(study, pid, sex)
    own_reduced = own_pia * factor

    if status == "married":
        spouse_pia = _current_spouse_pia(study, pid, elig)
        if spouse_pia is None:
            return own_reduced
        pmf = _claim_pmf(study, pid, sex)
        excess = sum(
            prob
            * household.excess_spousal_benefit(
                own_pia, spouse_pia, age * 12, b, study.params
            )
            for age, prob in pmf.items()
        )
        return own_reduced + excess

    if status == "widowed":
        widow = _last_widowhood_spouse(study, pid, elig)
        if widow is None or int(widow.spouse_person_id) not in study.pia:
            return own_reduced
        deceased = int(widow.spouse_person_id)
        deceased_pia = study.pia[deceased]
        deceased_sex = study.sex.get(deceased, "male")
        deceased_b = study.birth_year.get(deceased, b)
        pmf_s = _claim_pmf(study, pid, sex)
        pmf_d = _claim_pmf(study, deceased, deceased_sex)
        widow_expected = 0.0
        for a_s, p_s in pmf_s.items():
            for a_d, p_d in pmf_d.items():
                sb = household.survivor_benefit_at_death(
                    own_pia,
                    deceased_pia,
                    a_s * 12,
                    a_d * 12,
                    b,
                    deceased_b,
                    study.params,
                )
                widow_expected += p_s * p_d * sb.widow_benefit
        return widow_expected

    return own_reduced


def reformed_benefit(
    study: StudyData,
    pid: int,
    sex: str,
    status: str,
    package: str,
    *,
    apply_scalar: bool = True,
) -> float:
    """Monthly benefit under an earnings-sharing package.

    Both packages split earnings 50/50 across marriage years, recompute
    the person's own AIME/PIA on the shared vector, remove all spouse
    benefits, and (optionally) apply the package's global benefit
    increase. Package 1b pays NO survivor benefit; package 1a pays a
    survivor benefit on the maximum shared vector (the deceased's shared
    record), so widowed people keep a survivor top-up under 1a.
    """
    b = study.birth_year[pid]
    elig = b + ELIGIBILITY_AGE
    factor = _expected_own_factor(study, pid, sex)
    scalar = 1.0 + GLOBAL_INCREASE[package] if apply_scalar else 1.0

    shared = shared_history(study, pid, study.history[pid], elig)
    shared_pia = benefits.pia(
        benefits.aime(shared, b, study.params), elig, study.params
    )
    own_reduced = shared_pia * scalar * factor

    if package == "1a" and status == "widowed":
        widow = _last_widowhood_spouse(study, pid, elig)
        if widow is not None and int(widow.spouse_person_id) in study.history:
            deceased = int(widow.spouse_person_id)
            deceased_shared = shared_history(
                study, deceased, study.history[deceased], elig
            )
            deceased_shared_pia = benefits.pia(
                benefits.aime(
                    deceased_shared, study.birth_year[deceased], study.params
                ),
                study.birth_year[deceased] + ELIGIBILITY_AGE,
                study.params,
            )
            deceased_sex = study.sex.get(deceased, "male")
            pmf_s = _claim_pmf(study, pid, sex)
            pmf_d = _claim_pmf(study, deceased, deceased_sex)
            widow_expected = 0.0
            for a_s, p_s in pmf_s.items():
                for a_d, p_d in pmf_d.items():
                    sb = household.survivor_benefit_at_death(
                        shared_pia * scalar,
                        deceased_shared_pia * scalar,
                        a_s * 12,
                        a_d * 12,
                        b,
                        study.birth_year[deceased],
                        study.params,
                    )
                    widow_expected += p_s * p_d * sb.widow_benefit
            return widow_expected

    return own_reduced


# =====================================================================
# Scoring: per-person deltas, then weighted shares by cell
# =====================================================================
def score_population(study: StudyData) -> pd.DataFrame:
    """Per-person baseline/reform benefits and sharing deltas.

    One row per scorable study person (career frame, both-spouse
    covered), with sex, marital status at eligibility, anchor weight,
    baseline benefit, and the reform benefit + delta for package 1b
    (with and without the global scalar) and package 1a.
    """
    rows = []
    for pid in sorted(study.careers):
        sex = study.sex.get(pid)
        if sex not in SEXES:
            continue
        status = study.status_at_eligibility(pid)
        if status not in MARITAL:
            continue
        if not study.both_spouse_covered(pid):
            continue
        b = study.birth_year[pid]
        own_pia = study.pia.get(pid)
        if own_pia is None:
            own_pia = benefits.pia(
                benefits.aime(study.history[pid], b, study.params),
                b + ELIGIBILITY_AGE,
                study.params,
            )
        baseline = baseline_benefit(study, pid, sex, status, own_pia)
        if baseline <= 0:
            continue
        reform_1b = reformed_benefit(study, pid, sex, status, "1b")
        reform_1b_ns = reformed_benefit(
            study, pid, sex, status, "1b", apply_scalar=False
        )
        reform_1a = reformed_benefit(study, pid, sex, status, "1a")
        rows.append(
            {
                "person_id": pid,
                "sex": sex,
                "status": status,
                "weight": float(study.weight.get(pid, 1.0)),
                "baseline": baseline,
                "delta_1b": (reform_1b - baseline) / baseline,
                "delta_1b_noscalar": (reform_1b_ns - baseline) / baseline,
                "delta_1a": (reform_1a - baseline) / baseline,
            }
        )
    return pd.DataFrame(rows)


def _bucket_of(delta: float) -> str:
    """The Table-3 bucket a benefit change falls in."""
    if delta <= -0.20:
        return "lose_ge_20"
    if delta <= -0.10:
        return "lose_10_20"
    if delta <= -0.05:
        return "lose_5_10"
    if delta < 0.0:
        return "lose_lt_5"
    if delta == 0.0:
        return "no_change"
    if delta < 0.05:
        return "gain_lt_5"
    if delta < 0.10:
        return "gain_5_10"
    if delta < 0.20:
        return "gain_10_20"
    return "gain_ge_20"


def _weighted_bucket_shares(
    deltas: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """Anchor-weighted percent in each of the nine buckets."""
    codes = np.array([_bucket_of(float(d)) for d in deltas])
    out = {}
    for bucket in BUCKETS:
        out[bucket] = 100.0 * _weighted_mean(
            (codes == bucket).astype(np.float64), weights
        )
    return out


def cell_shares(
    df: pd.DataFrame, delta_col: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """Weighted nine-bucket + four-threshold shares per (status, sex)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for status in MARITAL:
        for sex in SEXES:
            sub = df[(df["status"] == status) & (df["sex"] == sex)]
            if len(sub) == 0:
                out[(status, sex)] = {
                    "n_persons": 0,
                    "buckets": {b: 0.0 for b in BUCKETS},
                    "thresholds": {t: 0.0 for t in THRESHOLDS},
                }
                continue
            deltas = sub[delta_col].to_numpy(dtype=np.float64)
            weights = sub["weight"].to_numpy(dtype=np.float64)
            buckets = _weighted_bucket_shares(deltas, weights)
            out[(status, sex)] = {
                "n_persons": int(len(sub)),
                "buckets": buckets,
                "thresholds": _thresholds_from_buckets(buckets),
            }
    return out


# =====================================================================
# Floors: 5-seed person-disjoint half-split noise on each share
# =====================================================================
def seed_half_shares(
    df: pd.DataFrame, delta_col: str, seed: int
) -> dict[str, dict[str, dict[str, float]]]:
    """Threshold shares on each person-disjoint half for one seed.

    Splits the scored persons 50/50 by person (the committed
    :func:`hpanel.split_panel_by_person`), and reports each half's four
    threshold shares per (status, sex) cell -- the A-vs-B gap is the
    sampling-noise floor on the full-sample share at half scale.
    """
    side_a, side_b = hpanel.split_panel_by_person(
        df, "person_id", fraction=0.5, seed=seed
    )
    shares_a = cell_shares(side_a, delta_col)
    shares_b = cell_shares(side_b, delta_col)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for status in MARITAL:
        for sex in SEXES:
            key = f"{status}:{sex}"
            out[key] = {
                "side_a": shares_a[(status, sex)]["thresholds"],
                "side_b": shares_b[(status, sex)]["thresholds"],
                "n_a": shares_a[(status, sex)]["n_persons"],
                "n_b": shares_b[(status, sex)]["n_persons"],
            }
    return out


def build_floors(
    per_seed: list[dict[str, Any]], delta_col_key: str
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-cell, per-threshold floor = summary of |side_a - side_b|."""
    floors: dict[str, dict[str, dict[str, Any]]] = {}
    for status in MARITAL:
        for sex in SEXES:
            key = f"{status}:{sex}"
            floors[key] = {}
            for thr in THRESHOLDS:
                gaps = []
                for seed_row in per_seed:
                    cell = seed_row[delta_col_key][key]
                    gaps.append(abs(cell["side_a"][thr] - cell["side_b"][thr]))
                floors[key][thr] = _summary(gaps)
    return floors


# =====================================================================
# Three-way table (DYNASIM / ours / floor) + directional verdict
# =====================================================================
def build_three_way(
    package: str,
    shares: dict[tuple[str, str], dict[str, Any]],
    floors: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Per (status, sex): DYNASIM row, our shares, and the floors."""
    table = []
    for status in MARITAL:
        for sex in SEXES:
            cell = shares[(status, sex)]
            dyn = dynasim_cell(package, status, sex)
            key = f"{status}:{sex}"
            thr_rows = {}
            for thr in THRESHOLDS:
                our = cell["thresholds"][thr]
                floor = floors[key][thr]
                dynv = dyn["thresholds"][thr]
                thr_rows[thr] = {
                    "dynasim_pct": round(dynv, 2),
                    # Full precision so the live seed-0 pin is exact; the
                    # printed summary rounds for display.
                    "our_share_pct": our,
                    "abs_gap_vs_dynasim": round(abs(our - dynv), 2),
                    "floor_mean": round(floor["mean"], 2),
                    "floor_sd": round(floor["sd"], 2),
                    "our_exceeds_floor": bool(our > floor["mean"]),
                }
            table.append(
                {
                    "status": status,
                    "sex": sex,
                    "n_persons": cell["n_persons"],
                    "our_buckets": {
                        b: round(cell["buckets"][b], 2) for b in BUCKETS
                    },
                    "dynasim_buckets": dyn["buckets"],
                    "thresholds": thr_rows,
                }
            )
    return table


def directional_verdict(
    shares_1b: dict[tuple[str, str], dict[str, Any]],
    shares_1a: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """The registration's pre-registered directional expectations.

    (1) single-earner-couple husbands among the losers -> married men
        lose more than they gain (loss share >= 5% dominates);
    (2) divorced women among the gainers under sharing -> divorced women
        gain more than they lose under 1b;
    (3) widowed women gain (or lose far less) once a survivor benefit is
        restored -> widowed women's >=20% loss share is much smaller
        under 1a than 1b;
    (4) never-married rows near-zero by construction -> essentially no
        never-married winners or losers at the >=5% thresholds under 1b
        (the uniform +4.5% is below 5%).
    """
    mm = shares_1b[("married", "male")]["thresholds"]
    df_ = shares_1b[("divorced", "female")]["thresholds"]
    wf_1b = shares_1b[("widowed", "female")]["thresholds"]
    wf_1a = shares_1a[("widowed", "female")]["thresholds"]
    nm_m = shares_1b[("never_married", "male")]["thresholds"]
    nm_f = shares_1b[("never_married", "female")]["thresholds"]

    checks = {
        "married_men_are_losers": {
            "statement": "married men lose >= 5% more than they gain >= 5%",
            "lose_ge_5": round(mm["lose_ge_5"], 2),
            "gain_ge_5": round(mm["gain_ge_5"], 2),
            "held": bool(mm["lose_ge_5"] > mm["gain_ge_5"]),
        },
        "divorced_women_are_gainers": {
            "statement": "divorced women gain >= 5% more than they lose >= 5%",
            "gain_ge_5": round(df_["gain_ge_5"], 2),
            "lose_ge_5": round(df_["lose_ge_5"], 2),
            "held": bool(df_["gain_ge_5"] > df_["lose_ge_5"]),
        },
        "survivor_variant_helps_widowed_women": {
            "statement": (
                "widowed women's >=20% loss share is smaller under 1a "
                "(survivor restored) than under 1b (no survivor)"
            ),
            "widowed_women_lose_ge_20_1b": round(wf_1b["lose_ge_20"], 2),
            "widowed_women_lose_ge_20_1a": round(wf_1a["lose_ge_20"], 2),
            "held": bool(wf_1a["lose_ge_20"] < wf_1b["lose_ge_20"]),
        },
        "never_married_near_zero": {
            "statement": (
                "never-married >=5% winner and loser shares are near zero "
                "(< 5 pp) under 1b -- the uniform +4.5% is below 5%"
            ),
            "male_gain_ge_5": round(nm_m["gain_ge_5"], 2),
            "male_lose_ge_5": round(nm_m["lose_ge_5"], 2),
            "female_gain_ge_5": round(nm_f["gain_ge_5"], 2),
            "female_lose_ge_5": round(nm_f["lose_ge_5"], 2),
            "held": bool(
                max(
                    nm_m["gain_ge_5"],
                    nm_m["lose_ge_5"],
                    nm_f["gain_ge_5"],
                    nm_f["lose_ge_5"],
                )
                < 5.0
            ),
        },
    }
    checks["all_held"] = bool(all(c["held"] for c in checks.values()))
    return checks


def magnitude_check(
    three_way_1b: list[dict[str, Any]],
) -> dict[str, Any]:
    """Whether the largest cells land within +/-10pp of DYNASIM.

    The registration expects magnitudes within +/-10pp on the LARGEST
    cells. 'Largest' = cells whose DYNASIM threshold share is >= 20%.
    Reports each such cell's gap and the honest pass/fail (this is a
    reported diagnostic -- no tuning to hit the tolerance)."""
    large = []
    for row in three_way_1b:
        for thr, cell in row["thresholds"].items():
            if cell["dynasim_pct"] >= 20.0:
                large.append(
                    {
                        "cell": f"{row['status']}:{row['sex']}:{thr}",
                        "dynasim_pct": cell["dynasim_pct"],
                        "our_share_pct": cell["our_share_pct"],
                        "abs_gap": cell["abs_gap_vs_dynasim"],
                        "within_10pp": bool(
                            cell["abs_gap_vs_dynasim"] <= 10.0
                        ),
                    }
                )
    n_within = sum(c["within_10pp"] for c in large)
    return {
        "definition": "cells with a DYNASIM threshold share >= 20%",
        "n_large_cells": len(large),
        "n_within_10pp": n_within,
        "share_within_10pp": (
            round(n_within / len(large), 3) if large else None
        ),
        "cells": large,
    }


# =====================================================================
# Aggregate cost change (descriptive: how far from cost-neutral here)
# =====================================================================
def aggregate_cost_change(df: pd.DataFrame) -> dict[str, float]:
    """Weighted aggregate baseline vs 1b reform benefit (descriptive).

    The package scalar (+4.5%) was DYNASIM-calibrated for cost-neutrality
    in the 2049 projection, not re-derived here. This reports the weighted
    aggregate percent change in monthly benefit our population sees under
    it -- a named delta, not a target.
    """
    w = df["weight"].to_numpy(dtype=np.float64)
    baseline = df["baseline"].to_numpy(dtype=np.float64)
    reform = baseline * (1.0 + df["delta_1b"].to_numpy(dtype=np.float64))
    reform_ns = baseline * (
        1.0 + df["delta_1b_noscalar"].to_numpy(dtype=np.float64)
    )
    tot_base = float(np.sum(w * baseline))
    return {
        "weighted_aggregate_pct_change_1b": round(
            100.0 * (float(np.sum(w * reform)) - tot_base) / tot_base, 2
        ),
        "weighted_aggregate_pct_change_1b_noscalar": round(
            100.0 * (float(np.sum(w * reform_ns)) - tot_base) / tot_base, 2
        ),
        "note": (
            "the +4.5% scalar was DYNASIM-calibrated for 2049 "
            "cost-neutrality; applied to this observed 1943-57 sample it "
            "is not cost-neutral -- a named population delta"
        ),
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


def anchor_provenance() -> dict[str, Any]:
    """Favreault-Steuerle (2007) package definitions + transcribed tables.

    Every figure verified against the archived PDF and its pdftotext
    (~/PolicyEngine/dynasim-refs/311436-spouse-survivor-modern.{pdf,txt}).
    Printed page numbers cite the report's own pagination.
    """
    return {
        "paper": (
            "Favreault, M. M. and Steuerle, C. E. (2007). Social Security "
            "Spouse and Survivor Benefits for the Modern Family. The Urban "
            "Institute, Retirement Project Discussion Paper 07-01 (report "
            "311436). DYNASIM3, runid 440v2. Cost-balanced at 2050; 1960-80 "
            "birth cohorts evaluated in 2049."
        ),
        "source_files": [
            "~/PolicyEngine/dynasim-refs/311436-spouse-survivor-modern.pdf",
            "~/PolicyEngine/dynasim-refs/311436-spouse-survivor-modern.txt",
        ],
        "package_definitions": {
            "citation": "Table 2 (Details of the Social Security Packages "
            "Simulated), printed p.15; narrative printed p.14 and p.19",
            "earnings_sharing_core": (
                "Persons first entitled to OASDI in 2007 (born 1947+) share "
                "earnings retrospectively from 1961; spouses share all "
                "earnings capped at the taxable maximum prior to sharing "
                "over the full duration of each marriage (but no longer); "
                "covered quarters are recomputed on the shared vector; all "
                "spouse benefits are removed."
            ),
            "1a": (
                "Earnings sharing with modified survivors' benefits, global "
                "benefit increase of 2.71 percent. Survivor benefits are "
                "awarded on the maximum shared earnings vector among a "
                "worker's, the spouse's, and all former spouses' records."
            ),
            "1b": (
                "Earnings sharing with NO survivors' benefit, global benefit "
                "increase of 4.5 percent. Same as 1a except no survivor "
                "benefit whatsoever. THE PRIMARY TARGET."
            ),
            "1c": (
                "Earnings sharing with a self-financed survivor benefit, "
                "global benefit increase of 4.5 percent. Same as 1b except "
                "people married at entitlement purchase survivor insurance "
                "(2/3 of own benefit) through an actuarially fair, unisex, "
                "4.6 percent real annuity reduction."
            ),
            "global_increase_is_a_cost_offset": (
                "The scalar benefit increases (2.71% / 4.5%) return the "
                "savings from removing spouse/survivor benefits, making each "
                "package approximately cost-neutral; implemented as a scalar "
                "on the PIA formula factors (printed p.19-20)."
            ),
        },
        "table3_winners_losers_2049": {
            "citation": "Table 3 (Winners and Losers Relative to Current "
            "Law Scheduled among Current Law Beneficiaries under the Options "
            "in 2049, by Sex and Marital Status), printed p.19-20; DYNASIM3 "
            "runid 440v2",
            "bucket_order": list(BUCKETS),
            "packages": DYNASIM_TABLE3,
            "transcription": "verified cell-by-cell against the rendered "
            "PDF pages (PDF p.29-30). Rows are transcribed verbatim; most "
            "sum to 100.0-100.2 (the paper's 0.1-rounding).",
            "source_arithmetic_note": (
                "As printed, package-1c married women sum to 109.1, not "
                "100.0 (the paper's own 'All' row for that column reads "
                "100.0): the 'no change' cell is printed 9.1 where every "
                "other married earnings-sharing 'no change' cell (1a/1b/1c "
                "men) is 0.0, so the 9.1 is a printed typo for 0.0. Kept "
                "verbatim here (9.1) and flagged; package 1c is descriptive "
                "provenance only and is not scored, so it feeds no result."
            ),
            "narrative_quotes": {
                "divorced": "about 16 percent of [divorced women] "
                "experienced benefit increases of over 20 percent ... "
                "Divorced men are more likely to lose benefits (printed "
                "p.19)",
                "widowed_1b": "Fractions of widow(er)s with losses of "
                "greater than 20 percent increase to 36 percent of men and "
                "38 percent of women under this [1b] option (printed p.20)",
                "never_married": "Never married people receive small benefit "
                "increases (because of the benefit adjustment that "
                "compensates for the elimination of spouse and survivor "
                "benefits) (printed p.19)",
            },
        },
        "table5_poverty_2049": DYNASIM_TABLE5_POVERTY,
        "named_population_deltas": [
            "observed PSID retirees eligible 2005-2019 (born 1943-1957) vs "
            "DYNASIM projected 1960-1980 cohorts in 2049",
            "PSID coverage-0.8 long-stayers vs DYNASIM SIPP+PSID synthetic "
            "population (our women have strong own careers, so rely less on "
            "survivor benefits)",
            "spouse earnings observed on whatever PSID years exist (often "
            "sparser than the study person's coverage-0.8 career), biasing "
            "the higher earner's loss upward",
            "no within-couple claim-age timing, no mortality projection, no "
            "behavioural response (single expected claim-age reduction)",
            "package scalar is DYNASIM-calibrated (2.71%/4.5%), not "
            "re-derived for cost-neutrality in our sample",
            "poverty (Table 5) and lifetime tax-benefit ratios (Table 4) "
            "transcribed for provenance but not reproduced (need lifetime "
            "accumulation and household income)",
        ],
    }


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
    """Execute the full R7 earnings-sharing replication (reported)."""
    started = time.time()
    params = load_ssa_parameters()
    if verbose:
        print(f"pe_us_revision={params.pe_us_revision}; loading PSID ...")

    study = StudyData(params)
    n_career = len(study.careers)
    df = score_population(study)
    if verbose:
        print(
            f"career frame {n_career} persons; scored {len(df)} "
            f"(both-spouse covered) in {time.time() - started:.0f}s"
        )

    # Both-spouse coverage over the career frame, by marital status.
    coverage = _coverage_report(study)

    # Full-sample shares.
    shares_1b = cell_shares(df, "delta_1b")
    shares_1b_ns = cell_shares(df, "delta_1b_noscalar")
    shares_1a = cell_shares(df, "delta_1a")

    # Per-seed half-split shares for the floors.
    per_seed = []
    for seed in SEEDS:
        per_seed.append(
            {
                "seed": seed,
                "package_1b": seed_half_shares(df, "delta_1b", seed),
                "package_1a": seed_half_shares(df, "delta_1a", seed),
            }
        )
    floors_1b = build_floors(per_seed, "package_1b")
    floors_1a = build_floors(per_seed, "package_1a")

    three_way_1b = build_three_way("1b", shares_1b, floors_1b)
    three_way_1a = build_three_way("1a", shares_1a, floors_1a)

    verdict = directional_verdict(shares_1b, shares_1a)
    magnitude = magnitude_check(three_way_1b)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "reported_not_gated": True,
        "registration": SPEC_REGISTRATION,
        "registration_pointer": SPEC_REGISTRATION_POINTER,
        "program_design_issue": PROGRAM_DESIGN_ISSUE,
        "purpose": (
            "Earnings-sharing external-anchor replication: does our stack "
            "(real PSID marriage histories + real earnings + the statutory "
            "415(a)/(b) chain + the 402(b)/(c)/(e)/(f) auxiliary-benefit "
            "plumbing) reproduce the incidence structure of "
            "Favreault-Steuerle (2007) earnings sharing -- who wins and "
            "loses by sex x marital status (Table 3)? Reads no gate, "
            "changes no gate; publishes regardless of outcome."
        ),
        "what_is_tested": (
            "the incidence STRUCTURE (direction of winners/losers by sex x "
            "marital status), not the population LEVELS -- every level "
            "difference vs the DYNASIM 2049 projection is a named delta"
        ),
        "real_couples_only": True,
        "primary_package": PRIMARY_PACKAGE,
        "global_increase_decision": (
            "Package 1b applies the anchor's +4.5% global benefit increase "
            "(a scalar on the PIA formula factors) as part of its cited "
            "definition -- it is in the package name and is directly "
            "visible in the anchor's never-married Table-3 row (97-99% show "
            "a 'gain < 5%', the uniform bump). A scalar-off variant is "
            "reported in sensitivity_no_scalar. See the module docstring."
        ),
        "package_1c_not_implemented": PACKAGE_1C_NOT_IMPLEMENTED,
        "anchor_provenance": anchor_provenance(),
        "study_population": {
            "career_frame_rule": (
                "Phase-A pia_observed selection: coverage >= 0.8 of ages "
                "22-61, age-62 eligibility year in 2005-2019 (born "
                "1943-1957), single-year gaps interpolated (mean of "
                "neighbours); reconstructed here (the committed "
                "runs/pia_observed_psid_v1.json builder was not committed) "
                "and cross-checked to that artifact's ~1486 careers and "
                "mean PIA range"
            ),
            "n_career_frame": n_career,
            "n_scored": int(len(df)),
            "marital_status_classification": (
                "reconstructed from mh85_23 episodes at the age-62 "
                "eligibility year (active marriage -> married; else most "
                "recent ended marriage -> widowed/divorced; none -> "
                "never_married), with a last_known_status fallback"
            ),
            "both_spouse_coverage": coverage,
            "weighting": "anchor weight = the person's last observed PSID "
            "cross-sectional weight",
            "n_by_cell": {
                f"{status}:{sex}": shares_1b[(status, sex)]["n_persons"]
                for status in MARITAL
                for sex in SEXES
            },
        },
        "conventions": {
            "sharing": (
                "for each marriage with a computable spouse, every year of "
                "the marriage takes 0.5*(own + spouse) capped at that year's "
                "taxable maximum before sharing; non-marriage years keep own "
                "earnings; AIME/PIA recomputed on the shared vector (42 USC "
                "415(a)/(b))"
            ),
            "baseline": (
                "own reduced retirement benefit + excess spouse's benefit "
                "(402(b)/(c)) if married with a computable spouse, OR the "
                "widow(er)'s benefit (402(e)/(f), larger of own or survivor) "
                "if widowhood is observed with a computable deceased spouse"
            ),
            "claim_age_reduction": (
                "expected benefit-to-PIA factor over the observed-era "
                "(1998-2022) claim-age distribution for the person's sex and "
                "eligibility year (claiming.expected_reduction_factor); "
                "auxiliary benefits are the pmf-weighted expectation of the "
                "household.py scalar functions -- a deterministic expected "
                "reduction, not a sampled claim age"
            ),
            "global_increase": GLOBAL_INCREASE,
            "winner_loser_metric": (
                "(reform - baseline)/baseline bucketed into the Table-3 "
                "buckets; the reported thresholds are lose>=20%, lose>=5%, "
                "gain>=5%, gain>=20%; shares are anchor-weighted"
            ),
            "floor": (
                "5-seed person-disjoint half-split (split_panel_by_person, "
                "fraction=0.5); floor per share = summary of |side_a - "
                "side_b| across seeds -- the sampling-noise scale at half "
                "sample"
            ),
        },
        "package_1b_primary": {
            "three_way_by_cell": three_way_1b,
            "directional_verdict": verdict,
            "magnitude_check": magnitude,
            "aggregate_cost_change": aggregate_cost_change(df),
        },
        "package_1a_descriptive": {
            "note": (
                "survivor benefit on the maximum shared vector + 2.71% "
                "scalar; run descriptively (registration: 1a/1c descriptive "
                "if implementable)"
            ),
            "three_way_by_cell": three_way_1a,
        },
        "sensitivity_no_scalar": {
            "note": (
                "package 1b earnings sharing WITHOUT the +4.5% global "
                "increase -- isolates the pure redistribution; shifts every "
                "cell toward losses and moves never-married to 'no change'"
            ),
            "shares_by_cell": {
                f"{status}:{sex}": {
                    "n_persons": shares_1b_ns[(status, sex)]["n_persons"],
                    "buckets": {
                        b: round(shares_1b_ns[(status, sex)]["buckets"][b], 2)
                        for b in BUCKETS
                    },
                    "thresholds": {
                        t: round(
                            shares_1b_ns[(status, sex)]["thresholds"][t], 2
                        )
                        for t in THRESHOLDS
                    },
                }
                for status in MARITAL
                for sex in SEXES
            },
        },
        "per_seed": per_seed,
        "revision_pins": _revision_pins(params),
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if verbose:
        _print_summary(three_way_1b, verdict, magnitude, coverage)
    return artifact


def _coverage_report(study: StudyData) -> dict[str, Any]:
    """Both-spouse coverage over the career frame, by marital status."""
    by_status = {s: {"n": 0, "covered": 0} for s in MARITAL}
    n_total = 0
    n_covered = 0
    for pid in study.careers:
        if study.sex.get(pid) not in SEXES:
            continue
        status = study.status_at_eligibility(pid)
        if status not in MARITAL:
            continue
        covered = study.both_spouse_covered(pid)
        by_status[status]["n"] += 1
        n_total += 1
        if covered:
            by_status[status]["covered"] += 1
            n_covered += 1
    ep_cf = study.episodes[study.episodes["person_id"].isin(study.careers)]
    widowhood = ep_cf[ep_cf["how_ended"] == "widowhood"]
    widow_computable = sum(
        1
        for row in widowhood.itertuples(index=False)
        if not pd.isna(row.spouse_person_id)
        and int(row.spouse_person_id) in study.pia
    )
    return {
        "definition": "share of career-frame persons whose relevant "
        "marriage spouses all have a computable PIA (never-married "
        "trivially covered)",
        "n_career_frame_classified": n_total,
        "n_both_spouse_covered": n_covered,
        "both_spouse_coverage_share": (
            round(n_covered / n_total, 4) if n_total else 0.0
        ),
        "by_marital_status": {
            s: {
                "n": v["n"],
                "n_covered": v["covered"],
                "share": round(v["covered"] / v["n"], 4) if v["n"] else 0.0,
            }
            for s, v in by_status.items()
        },
        "n_widowhood_episodes_career": int(len(widowhood)),
        "n_widowhood_deceased_computable": int(widow_computable),
    }


def _print_summary(
    three_way_1b: list[dict[str, Any]],
    verdict: dict[str, Any],
    magnitude: dict[str, Any],
    coverage: dict[str, Any],
) -> None:
    print(
        "\n=== Package 1b winners/losers: DYNASIM / ours / floor (>=20% "
        "and >=5%) ==="
    )
    hdr = (
        f"{'group':<22}{'n':>5} "
        f"{'D:lose20':>9}{'O:lose20':>9}{'fl':>6}  "
        f"{'D:gain20':>9}{'O:gain20':>9}{'fl':>6}"
    )
    print(hdr)
    for row in three_way_1b:
        l20 = row["thresholds"]["lose_ge_20"]
        g20 = row["thresholds"]["gain_ge_20"]
        print(
            f"{row['status'] + '/' + row['sex']:<22}{row['n_persons']:>5} "
            f"{l20['dynasim_pct']:>9.1f}{l20['our_share_pct']:>9.1f}"
            f"{l20['floor_mean']:>6.1f}  "
            f"{g20['dynasim_pct']:>9.1f}{g20['our_share_pct']:>9.1f}"
            f"{g20['floor_mean']:>6.1f}"
        )
    print(
        f"\nboth-spouse coverage: "
        f"{coverage['both_spouse_coverage_share']:.3f} "
        f"({coverage['n_both_spouse_covered']}/"
        f"{coverage['n_career_frame_classified']})"
    )
    print("directional expectations held:")
    for name, check in verdict.items():
        if name == "all_held":
            continue
        print(f"  {name}: {check['held']}")
    print(f"  ALL HELD: {verdict['all_held']}")
    print(
        f"magnitude within +/-10pp on large cells: "
        f"{magnitude['n_within_10pp']}/{magnitude['n_large_cells']}"
    )


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
