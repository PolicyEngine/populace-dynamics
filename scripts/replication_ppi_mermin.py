"""Phase-A replication: progressive price indexing vs Mermin (2005).

REPORTED, NOT GATED. This artifact reads no gate and changes no gate. It
is the first external-anchor replication of PolicyEngine/populace-dynamics
(issue #74, phase A): can the certified earnings generator (candidate 11,
run 13) reproduce the *incidence structure* of a DYNASIM-published reform
analysis? The anchor is Mermin (2005), "The Effect of Benefit Reductions
on the Distribution of Social Security Benefits" (Urban Institute 411260,
DYNASIM3 run 432): benefits as a percent of scheduled for retired workers
aged 62-67 in 2050, by career-average-earnings quintile, under price
indexing (PI) and progressive price indexing (PPI) with a 30th-percentile
bend.

Frozen spec: issue #42 comment 4907444903. Where this module and the
registration disagree, the registration wins -- EXCEPT the one documented
deviation below, which the task coordinator directed and which this module
carries prominently in :data:`DEVIATION_FROM_REGISTRATION` and in the
artifact.

=====================================================================
DEVIATION FROM THE REGISTRATION (generator-domain support restriction)
=====================================================================
The registration specifies careers from the ``pia_observed_psid_v1``
selection (full observed careers, ages 22-61, back to reference year 1968)
and the full statutory 42 USC 415(b) AIME (indexed top-35). The certified
generator (candidate 11), however, is native to the gate-filtered panel:
ages 25-59, reference years 1998-2022, biennial. Its cell marginals, donor
pools, and participation gates are all fit on that window. Running it on
the fuller pia_observed support would require re-parameterizing the
generator -- bending a locked, gate-certified object.

Resolution (directed by the task coordinator): evaluate BOTH the real and
the generated careers on the COMMON gate-filtered support (ages 25-59,
periods 1998-2022, biennial), for the SAME persons, with ONE identical
career-average/AIME convention on that support for both sides -- the
committed PIA-proxy's biennial convention (top min(10, n) NAWI-indexed
biennial earnings over count*12*2), routed through the full statutory
415(a) bend-point formula at the 2050-eligibility transport parameters.
Because a faithful highest-35 AIME cannot be formed on a biennial
prime-age window (the committed ``not_full_415b`` disclaimer), the proxy
convention is the available identical-both-sides analogue.

This restriction applies SYMMETRICALLY to real and generated careers, so
the real-vs-generated comparison (the clean internal test -- does the
generator reproduce real incidence?) stays valid. The DYNASIM comparison
gains a NAMED population-concept delta on top of the registration's
existing deltas (observed 1940s-80s cohorts vs DYNASIM's 2050 projected
retirees; individual own-record career-average vs DYNASIM's spouse-shared
lifetime earnings; 2005 TR vintage): a TRUNCATED OBSERVATION WINDOW
(prime-age biennial PSID vs full 415(b) careers). The empirical
consequence, documented not hidden: our common-support AIME distribution
is COMPRESSED relative to bends (roughly 90% of careers fall below the
second statutory bend; almost none ride the taxable maximum), so our
incidence gradient is FLATTER and HIGHER than DYNASIM's, and the
registration's "PPI within +/-4pp of DYNASIM at quintiles 2-4"
expectation is NOT met -- by construction of the support restriction, not
by model failure. The three-way table reads: DYNASIM full careers vs our
common-support real vs our common-support generated.

Also deviated, as consequences of the above:
* Person selection: the pia_observed cohort (eligible 2005-2019, born
  1943-57) overlaps the gate window only at ages ~41-59 (thin, age-biased
  late-career support) AND is 30-45 years older than Mermin's 2050
  retirees. We instead apply the pia_observed *coverage* rule
  (coverage_floor 0.8, long-stayer) remapped to the gate window: gate
  persons with positive-earnings biennial coverage >= 0.8 over their
  observed in-window span AND >= 8 positive biennial observations. This
  honors pia_observed's dense-career selection on the common support and
  centers eligibility near 2030 (closest available to 2050). The
  fixed-window (ages 22-61) coverage cannot transfer because 1998-2022
  truncates younger cohorts, so a span-based coverage is used.
* AIME: the proxy biennial convention, not full 415(b) top-35 (see above).

Everything else is per the registration: the candidate-11 generation is
imported byte-for-byte (:func:`reform_delta_diagnostic.
fit_and_generate_candidate11`); the wedge is the 2005 Trustees intermediate
real-wage differential, NOT tuned to 67.8; the PPI bend is the weighted
30th percentile of each side's own AIME; the floor is the real-vs-real
ctx20 half-split; five locked seeds 0-4; one run in the gate venv.

=====================================================================
Provisions (from Mermin's own description, page-cited in the artifact)
=====================================================================
* PI (Mermin p.4, "Implementation of Indexing Proposals"): the bend points
  stay wage-indexed; the PIA formula factors (0.90/0.32/0.15) are reduced
  each year from 2012 by real-wage growth. Because all three factors scale
  by the SAME cumulative wedge W, the reformed PIA equals W times the
  scheduled PIA for every career -> the PI incidence ratio is W, quintile
  invariant by construction. W = the 2005 TR ultimate real-wage
  differential (1.1pp; nominal wage 3.9% over CPI 2.8%) compounded 2012 ->
  2050: W = (1.028/1.039)**(2050-2012). NOT tuned to DYNASIM's 67.8.
* PPI (Mermin p.4): a new bend at the 30th percentile of career-average
  earnings; factors below unchanged (wage-indexed), factors above scaled
  by the same wedge W, continuous at the bend. Here the 30th-percentile
  bend is the weighted 30th percentile of the study population's own AIME,
  computed per side.

Metric: the weighted-mean reformed/scheduled PIA ratio by own-distribution
career-average-earnings quintile (career-average annual earnings = AIME*12,
Mermin Figure 1 note), five numbers per provision, per side, per seed;
pooled across seeds; with real-vs-real person-disjoint half-split floors
per quintile (ctx20, 5 seeds).

415(g) note: the incidence ratio is formed from the pre-415(g) PIA amount
(the 90/32/15 bracket sum on the 415(b)-dollar-floored indexed AIME). The
sub-$0.10 415(g) dime rounding is omitted from the ratio only -- it is a
nominal artifact that would inject ~1e-5 noise into the by-construction-
flat PI ratio; it does not touch the incidence structure.

Run (from the repository root, PSID family files staged, gate venv --
candidate generation needs populace-fit)::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv-gate/bin/python scripts/replication_ppi_mermin.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Weighted-statistic helpers and the seed-stable pooling convention,
# imported byte-for-byte from the merged builders (single source of truth).
from build_downstream_relevance import (  # noqa: E402
    _weighted_mean,
    _weighted_quantile,
)

# The candidate-11 fit+generate and the empty-safe across-seed summary,
# imported byte-for-byte (defers populace.fit to call time, so this module
# imports without the gate venv).
from reform_delta_diagnostic import (  # noqa: E402
    _summary,
    fit_and_generate_candidate11,
)

# The locked protocol: seeds, the filter-first gate panel, and the
# person-disjoint 0.2 holdout/train split.
from run_gate1_baseline import SEEDS, load_filtered_panel  # noqa: E402

# The anchor rows (one per person: their chronologically last observed
# period) supply each person's anchor weight.
from run_gate1_candidate5b import anchor_rows  # noqa: E402

from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.ss.params import load_ssa_parameters  # noqa: E402

ARTIFACT_PATH = ROOT / "runs" / "replication_ppi_mermin_v1.json"
ARTIFACT_SCHEMA_VERSION = "replication_ppi_mermin.v1"
RUN_NAME = "replication_ppi_mermin_v1"

#: This replication's frozen-spec registration (issue #42 comment).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4907444903"
)
SPEC_REGISTRATION_POINTER = "#42 comment 4907444903"
#: The phase catalog / program-design context.
PROGRAM_DESIGN_ISSUE = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/74"
)
#: The reform-delta diagnostic (#72/#73) whose floor/artifact conventions
#: this mirrors and whose directional finding motivates the prediction.
REFORM_DELTA_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4906177609"
)

# ---- Transport + wedge constants (2005 Trustees intermediate) ----------
#: PI reform start (Mermin p.2 "Policy Scenarios": "Beginning in 2012").
PI_START_YEAR = 2012
#: The single synthetic eligibility year every career is transported to
#: (Mermin's headline cohort benchmark).
TRANSPORT_ELIGIBILITY_YEAR = 2050
#: Age-60 indexing year of the 2050-eligibility cohort (415(b) indexes to
#: the year the worker attains age 60 = eligibility - 2).
TRANSPORT_INDEX_YEAR = TRANSPORT_ELIGIBILITY_YEAR - 2
#: 2005 TR ultimate intermediate assumptions (also consistent with
#: Mermin fn.6's 2.8% current-law COLA). The real-wage differential is
#: additive: nominal average-wage growth = CPI + 1.1pp.
TR2005_CPI = 0.028
TR2005_REAL_WAGE_DIFF = 0.011
TR2005_WAGE_GROWTH = TR2005_CPI + TR2005_REAL_WAGE_DIFF
#: The statutory 1979-base bend-point dollar amounts (42 USC 415(a)(1)(B)).
_BASE_FIRST_BEND = 180.0
_BASE_SECOND_BEND = 1085.0
_BASE_NAWI_YEAR = 1977
#: Proxy AIME convention (committed downstream-relevance functional).
PROXY_TOP_N = 10
_MONTHS = 12
_BIENNIAL_SCALE = 2

# ---- Person selection (pia_observed coverage rule, gate-window remap) ---
COVERAGE_FLOOR = 0.8
MIN_POSITIVE_BIENNIAL_OBS = 8

# ---- The PPI 30th-percentile bend level ---------------------------------
PPI_BEND_PERCENTILE = 0.30
#: Own-distribution career-average-earnings quintiles (weighted, 20% each).
QUINTILE_LEVELS = np.array([0.2, 0.4, 0.6, 0.8])
N_QUINTILES = 5

# =====================================================================
# Anchor: transcribed Mermin (2005) target rows + citations (verbatim).
# Verified against 411260-benefit-reductions.{txt,pdf}; see the artifact
# provenance block. Registration expectations are logged, not tuned to.
# =====================================================================
#: Table 2 (retired workers 62-67 in 2050), PPI "Percent of Scheduled
#: Benefits" by shared-lifetime-income quintile, lowest -> highest.
DYNASIM_PPI_BY_QUINTILE = (98.7, 90.4, 81.3, 75.7, 71.7)
DYNASIM_PPI_ALL = 80.4
#: Table 2, PI row (essentially flat).
DYNASIM_PI_BY_QUINTILE = (67.8, 67.8, 67.9, 67.8, 67.8)
DYNASIM_PI_ALL = 67.8
#: Table 2, scheduled mean annual benefit (2005 dollars) by quintile.
DYNASIM_SCHEDULED_MEAN_2005USD = (9200, 13900, 18100, 21900, 25600)
DYNASIM_SCHEDULED_MEAN_ALL_2005USD = 17700
#: Table 1, 75-year OASDI financial effect (percent of taxable payroll).
DYNASIM_TABLE1_PAYROLL_PCT = {
    "scheduled_deficit": -1.69,
    "price_indexing": 0.68,
    "progressive_price_indexing": -0.14,
}


# =====================================================================
# Transport parameters (NAWI projected to the 2048 indexing year)
# =====================================================================
def build_transport(params: Any) -> dict[str, Any]:
    """Wedge, projected NAWI, and 2050 bend points for the transport.

    The pinned policyengine-us NAWI series ends before the 2048 indexing
    year (its tail is already Trustees projections). Per the registration
    fallback, NAWI is extended past its last entry at the stated 2005 TR
    nominal average-wage growth (CPI 2.8% + 1.1pp = 3.9%), and the 2050
    bend points are derived per 42 USC 415(a)(1)(B) from the projected AWI
    at the 2048 index year. Because the transported AIME and the bend
    points BOTH scale linearly with NAWI[2048], every incidence RATIO is
    invariant to the projection level -- the projection only sets the
    (immaterial) absolute scale.
    """
    nawi = dict(params.nawi)
    last_year = max(nawi)
    for year in range(last_year + 1, TRANSPORT_INDEX_YEAR + 1):
        nawi[year] = nawi[last_year] * (1.0 + TR2005_WAGE_GROWTH) ** (
            year - last_year
        )
    index_nawi = nawi[TRANSPORT_INDEX_YEAR]
    ratio = index_nawi / params.nawi[_BASE_NAWI_YEAR]
    first_bend = float(round(_BASE_FIRST_BEND * ratio))
    second_bend = float(round(_BASE_SECOND_BEND * ratio))
    n_wedge_years = TRANSPORT_ELIGIBILITY_YEAR - PI_START_YEAR
    wedge = (1.0 + TR2005_CPI) ** n_wedge_years / (
        1.0 + TR2005_WAGE_GROWTH
    ) ** n_wedge_years
    return {
        "nawi": nawi,
        "index_year": TRANSPORT_INDEX_YEAR,
        "index_nawi": float(index_nawi),
        "nawi_last_realized_or_projected_in_pe_us": int(last_year),
        "eligibility_year": TRANSPORT_ELIGIBILITY_YEAR,
        "bend_points": (first_bend, second_bend),
        "pia_factors": tuple(float(f) for f in params.pia_factors),
        "wedge": float(wedge),
        "n_wedge_years": int(n_wedge_years),
    }


# =====================================================================
# Benefit math (transported proxy AIME -> scheduled / PI / PPI amount)
# =====================================================================
def transported_person_aime(
    periods: np.ndarray,
    earnings: np.ndarray,
    params: Any,
    transport: dict[str, Any],
) -> float:
    """Transported AIME (proxy convention) for one career.

    Committed proxy convention (build_downstream_relevance), transported to
    the 2050 cohort: positive-earnings periods only; cap each at that
    year's historical wage base (415(b)); NAWI-index each to the 2048
    indexing year (nawi[2048]/nawi[year]); average the top min(10, n)
    indexed values over count*12*2 (biennial scale); floor to the dollar
    (415(b)). Zero positive observations -> 0.0.
    """
    pos = earnings > 0
    if not np.any(pos):
        return 0.0
    nawi = transport["nawi"]
    index_nawi = transport["index_nawi"]
    yrs = periods[pos].astype(int)
    earn = earnings[pos].astype(np.float64)
    indexed = np.empty(earn.size, dtype=np.float64)
    for i in range(earn.size):
        year = int(yrs[i])
        capped = min(float(earn[i]), params.wage_base_for(year))
        indexed[i] = capped * index_nawi / nawi[year]
    count = min(PROXY_TOP_N, indexed.size)
    top = np.sort(indexed)[::-1][:count]
    aime = float(np.sum(top)) / (count * _MONTHS * _BIENNIAL_SCALE)
    return float(math.floor(aime))


def scheduled_amount(
    aime: np.ndarray, transport: dict[str, Any]
) -> np.ndarray:
    """Pre-415(g) scheduled PIA amount (90/32/15 over the 2050 bends)."""
    b1, b2 = transport["bend_points"]
    f1, f2, f3 = transport["pia_factors"]
    aime = np.asarray(aime, dtype=np.float64)
    return (
        f1 * np.minimum(aime, b1)
        + f2 * np.clip(np.minimum(aime, b2) - b1, 0.0, None)
        + f3 * np.clip(aime - b2, 0.0, None)
    )


def price_indexed_amount(
    aime: np.ndarray, transport: dict[str, Any]
) -> np.ndarray:
    """PI PIA amount: all factors scaled by the cumulative wedge W.

    All three factors scale by the same W, so this equals W * scheduled
    for every career -> the PI/scheduled ratio is W, quintile invariant.
    """
    return transport["wedge"] * scheduled_amount(aime, transport)


def progressive_price_indexed_amount(
    aime: np.ndarray, bend30: float, transport: dict[str, Any]
) -> np.ndarray:
    """PPI PIA amount: below bend30 unchanged, above scaled by W, continuous.

    ``bend30`` is the weighted 30th percentile of the side's own AIME.
    Below the bend the scheduled (wage-indexed) amount is kept; above it
    the marginal amount is scaled by the wedge W; continuous at the bend.
    """
    aime = np.asarray(aime, dtype=np.float64)
    below = np.minimum(aime, bend30)
    sched_below = scheduled_amount(below, transport)
    sched_all = scheduled_amount(aime, transport)
    return sched_below + transport["wedge"] * (sched_all - sched_below)


# =====================================================================
# Person selection: pia_observed coverage rule remapped to the gate window
# =====================================================================
def coverage_selected_persons(panel: pd.DataFrame) -> set[int]:
    """Gate-window persons meeting the remapped pia_observed coverage rule.

    A person is selected iff, over their observed in-window biennial span,
    the fraction of biennial slots with positive earnings is at least
    :data:`COVERAGE_FLOOR` AND they have at least
    :data:`MIN_POSITIVE_BIENNIAL_OBS` positive biennial observations. This
    is the pia_observed selection (coverage_floor 0.8, long-stayer)
    remapped from its fixed ages-22-61 window to the person's observed
    span in the gate window (1998-2022 truncates younger cohorts, so a
    fixed-window coverage cannot transfer).
    """
    selected: set[int] = set()
    ordered = panel.sort_values(["person_id", "period"])
    for pid, sub in ordered.groupby("person_id"):
        per = sub["period"].to_numpy()
        earn = sub["earnings"].to_numpy()
        span_slots = (int(per.max()) - int(per.min())) // _BIENNIAL_SCALE + 1
        n_pos = int(np.sum(earn > 0))
        if (
            span_slots > 0
            and n_pos / span_slots >= COVERAGE_FLOOR
            and n_pos >= MIN_POSITIVE_BIENNIAL_OBS
        ):
            selected.add(int(pid))
    return selected


# =====================================================================
# Per-side quintile ratios (real / generated / a floor half)
# =====================================================================
def _assign_quintiles(aime: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    """Own-distribution quintile 0..4 by AIME (searchsorted on cutpoints)."""
    q = np.searchsorted(cutpoints, aime, side="right")
    return np.clip(q, 0, N_QUINTILES - 1).astype(int)


def side_metrics(
    panel: pd.DataFrame,
    person_ids: set[int],
    weight_of: dict[int, float],
    params: Any,
    transport: dict[str, Any],
) -> dict[str, Any]:
    """PI/PPI reform-over-scheduled ratios by own AIME quintile, one side.

    Restricts ``panel`` to ``person_ids``, computes each career's
    transported AIME and anchor weight, forms the side's own weighted
    30th-percentile PPI bend and weighted quintile cutpoints, and reports
    the weighted-mean PI and PPI reform/scheduled ratios (percent) per
    quintile and overall. Careers with zero scheduled amount (AIME 0) are
    excluded from the ratio means (undefined percent of scheduled) and
    counted separately.
    """
    sub = panel[panel["person_id"].isin(person_ids)]
    aime_list: list[float] = []
    weight_list: list[float] = []
    for pid, g in sub.groupby("person_id", sort=True):
        aime_list.append(
            transported_person_aime(
                g["period"].to_numpy(),
                g["earnings"].to_numpy(dtype=np.float64),
                params,
                transport,
            )
        )
        weight_list.append(float(weight_of[int(pid)]))
    aime = np.array(aime_list, dtype=np.float64)
    weight = np.array(weight_list, dtype=np.float64)

    bend30 = float(_weighted_quantile(aime, weight, np.array([0.30]))[0])
    cutpoints = _weighted_quantile(aime, weight, QUINTILE_LEVELS)
    quintile = _assign_quintiles(aime, cutpoints)

    sched = scheduled_amount(aime, transport)
    pi_amt = price_indexed_amount(aime, transport)
    ppi_amt = progressive_price_indexed_amount(aime, bend30, transport)
    positive = sched > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        pi_ratio = np.where(positive, pi_amt / sched, np.nan)
        ppi_ratio = np.where(positive, ppi_amt / sched, np.nan)

    quintiles = []
    for k in range(N_QUINTILES):
        mask = (quintile == k) & positive
        if not np.any(mask):
            quintiles.append(
                {"quintile": k + 1, "n_persons": 0, "n_positive": 0}
            )
            continue
        w = weight[mask]
        quintiles.append(
            {
                "quintile": k + 1,
                "n_persons": int(np.sum(quintile == k)),
                "n_positive": int(np.sum(mask)),
                "pi_ratio_pct": 100.0 * _weighted_mean(pi_ratio[mask], w),
                "ppi_ratio_pct": 100.0 * _weighted_mean(ppi_ratio[mask], w),
                "mean_aime": _weighted_mean(aime[mask], w),
                "mean_scheduled_amount": _weighted_mean(sched[mask], w),
            }
        )
    all_mask = positive
    return {
        "n_persons": int(len(aime)),
        "n_zero_scheduled_excluded": int(np.sum(~positive)),
        "bend30_aime": bend30,
        "quintile_cutpoints_aime": [float(c) for c in cutpoints],
        "overall_pi_ratio_pct": (
            100.0 * _weighted_mean(pi_ratio[all_mask], weight[all_mask])
            if np.any(all_mask)
            else None
        ),
        "overall_ppi_ratio_pct": (
            100.0 * _weighted_mean(ppi_ratio[all_mask], weight[all_mask])
            if np.any(all_mask)
            else None
        ),
        "quintiles": quintiles,
    }


def _quintile_vector(side: dict[str, Any], key: str) -> list[float | None]:
    """Pull the five quintile values of ``key`` (None where a bin empty)."""
    out: list[float | None] = []
    for q in side["quintiles"]:
        out.append(q.get(key) if q.get("n_positive", 0) > 0 else None)
    return out


# =====================================================================
# Per-seed measurement (real-vs-generated and the real-vs-real floor)
# =====================================================================
def measure_realgen_seed(
    holdout: pd.DataFrame,
    candidate: pd.DataFrame,
    selected: set[int],
    weight_of: dict[int, float],
    params: Any,
    transport: dict[str, Any],
) -> dict[str, Any]:
    """Real-holdout vs generated quintile ratios for one seed.

    Both sides are the SAME holdout persons (intersected with the
    coverage-selected set) on their observed periods; only earnings differ
    (candidate anchor held at real earnings), so the comparison is
    person-aligned but each side uses its OWN AIME distribution for the
    bend and the quintiles (own-distribution incidence).
    """
    persons = set(int(p) for p in holdout["person_id"].unique()) & selected
    return {
        "n_selected_holdout_persons": len(persons),
        "real": side_metrics(holdout, persons, weight_of, params, transport),
        "generated": side_metrics(
            candidate, persons, weight_of, params, transport
        ),
    }


def measure_floor_seed(
    seed: int,
    selected_panel: pd.DataFrame,
    weight_of: dict[int, float],
    params: Any,
    transport: dict[str, Any],
) -> dict[str, Any]:
    """Real-vs-real quintile-ratio floor for one seed (ctx20 half-split).

    The build_pia_proxy_floor ctx20 construction on the coverage-selected
    panel: draw 40% of persons (fraction=0.4, seed=1000+s), halve it
    person-disjointly (fraction=0.5, seed=s), giving two DISJOINT
    ~20%-of-selected real samples A and B, each pushed through the same
    per-side quintile-ratio machinery (own bend, own quintiles). The
    A-vs-B per-quintile gap is the noise scale for the real-vs-generated
    per-quintile gap at the same ~20%-of-selected scale.
    """
    forty, _ = hpanel.split_panel_by_person(
        selected_panel, "person_id", fraction=0.4, seed=1000 + seed
    )
    side_a, side_b = hpanel.split_panel_by_person(
        forty, "person_id", fraction=0.5, seed=seed
    )
    a_ids = set(int(p) for p in side_a["person_id"].unique())
    b_ids = set(int(p) for p in side_b["person_id"].unique())
    return {
        "n_persons_side_a": len(a_ids),
        "n_persons_side_b": len(b_ids),
        "side_a": side_metrics(side_a, a_ids, weight_of, params, transport),
        "side_b": side_metrics(side_b, b_ids, weight_of, params, transport),
    }


# =====================================================================
# Pooling across seeds + the three-way comparison table
# =====================================================================
def _pool_quintile_ratio(
    rows: list[dict[str, Any]], side: str, key: str
) -> list[dict[str, Any]]:
    """Per-quintile across-seed summary of one side's ratio (drops None)."""
    pooled = []
    for k in range(N_QUINTILES):
        vals = [
            v
            for r in rows
            if (v := _quintile_vector(r[side], key)[k]) is not None
        ]
        pooled.append(_summary(vals))
    return pooled


def _abs_gap_floor(per_seed_gaps: list[float | None]) -> dict[str, Any]:
    """Pool per-seed signed gaps into abs and signed scales (mirrors #72)."""
    kept = [float(v) for v in per_seed_gaps if v is not None]
    return {
        "per_seed_signed": kept,
        "abs": _summary([abs(v) for v in kept]),
        "signed": _summary(kept),
    }


def _per_seed_gap(
    rows: list[dict[str, Any]],
    side_hi: str,
    side_lo: str,
    key: str,
    k: int,
) -> list[float | None]:
    """Per-seed ``side_hi - side_lo`` gap of quintile ``k``'s ``key``."""
    gaps: list[float | None] = []
    for r in rows:
        hi = _quintile_vector(r[side_hi], key)[k]
        lo = _quintile_vector(r[side_lo], key)[k]
        gaps.append(hi - lo if (hi is not None and lo is not None) else None)
    return gaps


def build_three_way(
    realgen_rows: list[dict[str, Any]],
    floor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """The PPI three-way comparison + PI scalars + directional verdict.

    For each quintile: DYNASIM published, pooled real, pooled generated,
    the real-vs-real floor scale, the real-vs-generated gap scale, and the
    within-floor / directional flags. PI scalars: pooled real, pooled
    generated, the wedge-implied W, and DYNASIM's 67.8.
    """
    real_ppi = _pool_quintile_ratio(realgen_rows, "real", "ppi_ratio_pct")
    gen_ppi = _pool_quintile_ratio(realgen_rows, "generated", "ppi_ratio_pct")
    ppi_table = []
    for k in range(N_QUINTILES):
        realgen_gap = _abs_gap_floor(
            _per_seed_gap(
                realgen_rows, "generated", "real", "ppi_ratio_pct", k
            )
        )
        floor_gap = _abs_gap_floor(
            _per_seed_gap(floor_rows, "side_a", "side_b", "ppi_ratio_pct", k)
        )
        realgen_scale = realgen_gap["abs"]["mean"]
        floor_scale = floor_gap["abs"]["mean"]
        # Signed pooled gap (generated - real); >0 means generated is cut
        # LESS than real (generated ratio higher).
        signed_mean = realgen_gap["signed"]["mean"]
        ppi_table.append(
            {
                "quintile": k + 1,
                "dynasim_pct": DYNASIM_PPI_BY_QUINTILE[k],
                "real_pooled": real_ppi[k],
                "generated_pooled": gen_ppi[k],
                "realgen_gap": realgen_gap,
                "floor_gap": floor_gap,
                "realgen_scale": realgen_scale,
                "floor_scale": floor_scale,
                "generated_cut_smaller_than_real": bool(signed_mean > 0.0),
                "gap_exceeds_floor": bool(realgen_scale > floor_scale),
            }
        )

    # Directional prediction (registration): generated top-quintile cut
    # smaller than real, and the real-vs-generated gap exceeds its floor at
    # Q5 only (not Q1-Q3).
    q5 = ppi_table[4]
    q1_q3_within_floor = all(
        not ppi_table[k]["gap_exceeds_floor"] for k in (0, 1, 2)
    )
    directional = {
        "prediction": (
            "generated top-quintile (Q5) cut smaller than real "
            "(under-concentrated cap-riding careers -> too-low top AIME -> "
            "less exposure above the bend); real-vs-generated gap exceeds "
            "its floor at Q5 only, not at Q1-Q3"
        ),
        "q5_generated_cut_smaller": q5["generated_cut_smaller_than_real"],
        "q5_gap_exceeds_floor": q5["gap_exceeds_floor"],
        "q1_q3_gaps_within_floor": q1_q3_within_floor,
        "prediction_held": bool(
            q5["generated_cut_smaller_than_real"]
            and q5["gap_exceeds_floor"]
            and q1_q3_within_floor
        ),
    }

    # PI scalars: pooled real/generated (both should equal the wedge W by
    # construction), the wedge-implied scalar, and DYNASIM's 67.8.
    real_pi = _pool_quintile_ratio(realgen_rows, "real", "pi_ratio_pct")
    gen_pi = _pool_quintile_ratio(realgen_rows, "generated", "pi_ratio_pct")
    return {
        "ppi_by_quintile": ppi_table,
        "ppi_all": {
            "dynasim_pct": DYNASIM_PPI_ALL,
            "note": (
                "our common-support 'all' is the weighted mean over careers "
                "of the PPI ratio; recorded in per_seed overall_ppi_ratio_pct"
            ),
        },
        "pi_scalars": {
            "real_pooled_mean_pct": _summary(
                [
                    row["real"]["overall_pi_ratio_pct"]
                    for row in realgen_rows
                    if row["real"]["overall_pi_ratio_pct"] is not None
                ]
            ),
            "generated_pooled_mean_pct": _summary(
                [
                    row["generated"]["overall_pi_ratio_pct"]
                    for row in realgen_rows
                    if row["generated"]["overall_pi_ratio_pct"] is not None
                ]
            ),
            "real_by_quintile_pooled": real_pi,
            "generated_by_quintile_pooled": gen_pi,
            "wedge_implied_scalar_pct": None,  # filled by caller (needs W)
            "dynasim_pct": DYNASIM_PI_ALL,
        },
        "directional_prediction": directional,
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


def _gates_amendment_state() -> dict[str, Any]:
    """Gate-1 lock + ratified-amendment state, parsed from gates.yaml."""
    doc = yaml.safe_load((ROOT / "gates.yaml").read_text())
    gate1 = doc["gates"]["gate_1"]
    thresholds = gate1.get("thresholds", {})
    history = gate1.get("amendment_history", []) or []
    rules = gate1.get("amendment_rules", {}) or {}
    return {
        "gate_1_locked": bool(thresholds.get("locked", False)),
        "amendments_ratified": [
            {"id": a.get("id"), "ratified": a.get("ratified")} for a in history
        ],
        "amendment_rules": sorted(rules.keys()),
    }


def _revision_pins(params: Any) -> dict[str, Any]:
    import sklearn

    return {
        "populace_dynamics_sha": _sha(ROOT),
        "pe_us_revision": getattr(params, "pe_us_revision", None),
        "sklearn_version": str(sklearn.__version__),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "gates_yaml": _gates_amendment_state(),
    }


def anchor_provenance() -> dict[str, Any]:
    """Mermin (2005) transcribed target rows + page/section citations.

    Every row verified against runs of the archived PDF and its pdftotext
    (~/PolicyEngine/dynasim-refs/411260-benefit-reductions.{pdf,txt}) on
    2026-07-07. Page numbers give the printed page (PDF page = printed +2).
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
        "pi_mechanics": {
            "citation": "printed p.2 (Policy Scenarios) and p.4 "
            "(Implementation of Indexing Proposals)",
            "quote_start": (
                "Beginning in 2012, initial benefit growth is indexed to "
                "prices instead of wages."
            ),
            "quote_mechanism": (
                "The indexing proposals continue to index the bend points "
                "to the growth in wages but reduce the growth of initial "
                "benefits by reducing the formula factors. Price indexing "
                "reduces all of the formula factors each year by the growth "
                "in inflation-adjusted wages."
            ),
            "encoding": (
                "all PIA formula factors scaled by the cumulative wedge W "
                "from 2012 to the 2050 eligibility year; bend points remain "
                "wage-indexed; PI/scheduled = W (quintile invariant)"
            ),
        },
        "ppi_mechanics": {
            "citation": "printed p.2 (Policy Scenarios) and p.4",
            "quote": (
                "Progressive price indexing adds a new bend point at the "
                "30th percentile of career average earnings. Factors beyond "
                "the new bend point are reduced such that starting benefits "
                "for workers who always earn the maximum amount covered by "
                "Social Security remain constant over time in "
                "inflation-adjusted dollars."
            ),
            "encoding": (
                "new bend at the weighted 30th percentile of the study "
                "population's own AIME; factors below unchanged; factors "
                "above scaled by the same wedge W; continuous at the bend"
            ),
            "reference_population_gap": (
                "the paper does NOT name the population defining the 30th "
                "percentile; we use each side's own AIME distribution"
            ),
        },
        "population_and_quintile": {
            "citation": "Table 2 title + notes (PDF p.16); p.3 "
            "(Methodology) and fn.9",
            "population": (
                "retired workers aged 62-67 in 2050, own-record benefits "
                "only, N=5351"
            ),
            "quintile_variable_paper": (
                "SHARED lifetime income quintile: a worker's entire earnings "
                "in single years plus half of both spouses' earnings in "
                "married years, wage-indexed like AIME (p.3, fn.9)"
            ),
            "career_average_definition": (
                "career average annual earnings = average indexed monthly "
                "earnings * 12 (Figure 1 note, PDF p.14)"
            ),
            "phase_a_analogue": (
                "Phase A has no marriage/survivorship (gate-C territory), so "
                "the quintile variable here is the INDIVIDUAL own-record "
                "career-average earnings (AIME*12), a named population delta "
                "vs the paper's spouse-shared measure"
            ),
        },
        "table2_retired_workers_62_67_in_2050": {
            "citation": "Table 2, PDF p.16 (Percent of Scheduled Benefits)",
            "quintile_order": "lowest -> highest shared-lifetime-income",
            "scheduled_mean_2005usd": {
                "by_quintile": list(DYNASIM_SCHEDULED_MEAN_2005USD),
                "all": DYNASIM_SCHEDULED_MEAN_ALL_2005USD,
            },
            "price_indexing_pct": {
                "by_quintile": list(DYNASIM_PI_BY_QUINTILE),
                "all": DYNASIM_PI_ALL,
            },
            "progressive_price_indexing_pct": {
                "by_quintile": list(DYNASIM_PPI_BY_QUINTILE),
                "all": DYNASIM_PPI_ALL,
            },
        },
        "table1_75yr_payroll_effect_pct": {
            "citation": "Table 1, PDF p.15 (percent of taxable payroll, "
            "CBO 2005 scoring)",
            "values": DYNASIM_TABLE1_PAYROLL_PCT,
        },
        "wedge_source": {
            "paper_states_number": False,
            "citation": "printed p.3 (2005 TR intermediate assumptions); "
            "fn.6 (2.8% current-law COLA)",
            "fallback": (
                "the paper states no numeric real-wage differential; we use "
                "the 2005 OASDI Trustees Report intermediate ULTIMATE "
                "real-wage differential of 1.1 percentage points (nominal "
                "average-wage growth 3.9% over CPI 2.8%), consistent with "
                "Mermin fn.6's 2.8% COLA. Not tuned to DYNASIM's 67.8."
            ),
            "wedge_formula": "W = (1.028/1.039)**(2050-2012)",
        },
        "named_population_deltas": [
            "observed PSID prime-age biennial careers (this study) vs "
            "DYNASIM's full projected 2050 careers (Mermin)",
            "individual own-record career-average earnings (this study) vs "
            "spouse-shared lifetime earnings quintiles (Mermin)",
            "truncated observation window: gate-filtered ages 25-59, "
            "1998-2022 biennial (this study) vs full 415(b) careers (Mermin)",
            "2005 Trustees vintage wedge applied to a single 2050-transport "
            "cohort (this study) vs Mermin's 62-67-in-2050 mixed-eligibility "
            "cohort (mean eligibility ~2047)",
        ],
    }


DEVIATION_FROM_REGISTRATION = (
    "Careers and AIME deviate from the registration by task-coordinator "
    "direction, because the certified generator (candidate 11) is native to "
    "the gate-filtered panel (ages 25-59, periods 1998-2022, biennial) and "
    "cannot be run on the fuller pia_observed support without "
    "re-parameterizing a locked, gate-certified object. Both real and "
    "generated careers are therefore evaluated on the COMMON gate-filtered "
    "support, for the same persons, with one identical AIME convention (the "
    "committed PIA-proxy biennial convention through the full statutory "
    "415(a) formula at the 2050 transport). The restriction is SYMMETRIC, "
    "so the real-vs-generated comparison stays clean; the DYNASIM "
    "comparison gains a named truncated-observation-window population "
    "delta. Person selection uses the pia_observed coverage rule "
    "(coverage_floor 0.8, long-stayer) remapped to a span-based coverage "
    "on the gate window (the pia_observed cohort, eligible 2005-2019, "
    "overlaps the gate window only at ages ~41-59 and is 30-45 years older "
    "than Mermin's 2050 retirees). Empirical consequence, documented: our "
    "common-support AIME distribution is compressed relative to bends "
    "(~90% below the second bend), so the incidence gradient is flatter and "
    "higher than DYNASIM's and the registration's +/-4pp-at-Q2-Q4 "
    "expectation is not met -- by construction of the support restriction, "
    "not by model failure."
)


# =====================================================================
# Driver
# =====================================================================
def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full Phase-A PI/PPI replication (reported, not gated)."""
    started = time.time()

    params = load_ssa_parameters()
    transport = build_transport(params)
    if verbose:
        print(
            f"pe_us_revision={params.pe_us_revision} "
            f"wedge W={transport['wedge']:.5f} "
            f"bends(2050)={transport['bend_points']} "
            f"nawi[{transport['index_year']}]={transport['index_nawi']:.0f}"
        )

    panel = load_filtered_panel()
    all_anchor = anchor_rows(panel)
    weight_of = dict(
        zip(
            all_anchor["person_id"].to_numpy(),
            all_anchor["weight"].to_numpy(),
            strict=True,
        )
    )
    selected = coverage_selected_persons(panel)
    selected_panel = panel[panel["person_id"].isin(selected)].reset_index(
        drop=True
    )
    if verbose:
        print(
            f"gate panel: {len(panel)} rows, {panel.person_id.nunique()} "
            f"persons; coverage-selected {len(selected)} persons"
        )

    realgen_rows: list[dict[str, Any]] = []
    floor_rows: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        t0 = time.time()
        holdout, candidate = fit_and_generate_candidate11(
            seed, panel, all_anchor
        )
        realgen = measure_realgen_seed(
            holdout, candidate, selected, weight_of, params, transport
        )
        floor = measure_floor_seed(
            seed, selected_panel, weight_of, params, transport
        )
        realgen_rows.append(realgen)
        floor_rows.append(floor)
        per_seed.append({"seed": seed, **realgen, "floor": floor})
        if verbose:
            rp = [
                round(q.get("ppi_ratio_pct", float("nan")), 1)
                for q in realgen["real"]["quintiles"]
            ]
            gp = [
                round(q.get("ppi_ratio_pct", float("nan")), 1)
                for q in realgen["generated"]["quintiles"]
            ]
            print(
                f"seed {seed}: n_sel_holdout="
                f"{realgen['n_selected_holdout_persons']} "
                f"real_PPI={rp} gen_PPI={gp} "
                f"PI_real={realgen['real']['overall_pi_ratio_pct']:.2f} "
                f"({time.time() - t0:.0f}s)"
            )

    three_way = build_three_way(realgen_rows, floor_rows)
    three_way["pi_scalars"]["wedge_implied_scalar_pct"] = (
        100.0 * transport["wedge"]
    )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "reported_not_gated": True,
        "registration": SPEC_REGISTRATION,
        "registration_pointer": SPEC_REGISTRATION_POINTER,
        "program_design_issue": PROGRAM_DESIGN_ISSUE,
        "reform_delta_reference": REFORM_DELTA_REGISTRATION,
        "purpose": (
            "Phase-A external-anchor replication: can the certified "
            "earnings generator (candidate 11) reproduce the incidence "
            "structure of Mermin (2005)'s price-indexing and "
            "progressive-price-indexing tables (percent of scheduled by "
            "career-average-earnings quintile)? Reads no gate, changes no "
            "gate; publishes regardless of outcome."
        ),
        "deviation_from_registration": DEVIATION_FROM_REGISTRATION,
        "anchor_provenance": anchor_provenance(),
        "transport_and_conventions": {
            "eligibility_year": transport["eligibility_year"],
            "index_year": transport["index_year"],
            "index_nawi": transport["index_nawi"],
            "nawi_last_in_pe_us": transport[
                "nawi_last_realized_or_projected_in_pe_us"
            ],
            "nawi_projection": (
                "NAWI extended past its last pe-us entry to the 2048 "
                "indexing year at the 2005 TR nominal average-wage growth "
                "(CPI 2.8% + 1.1pp = 3.9%); 2050 bend points derived per "
                "415(a)(1)(B) from the projected AWI. Every incidence RATIO "
                "is invariant to the projection level (AIME and bends both "
                "scale with NAWI[2048])."
            ),
            "bend_points_2050": list(transport["bend_points"]),
            "pia_factors": list(transport["pia_factors"]),
            "wedge": transport["wedge"],
            "wedge_years": transport["n_wedge_years"],
            "wedge_formula": "(1.028/1.039)**(2050-2012)",
            "aime_convention": (
                "committed PIA-proxy biennial convention (cap at historical "
                "wage base; NAWI-index to 2048; top min(10,n) over "
                "count*12*2; 415(b) dollar floor)"
            ),
            "ppi_bend": (
                "weighted 30th percentile of each side's own transported "
                "AIME"
            ),
            "quintile_variable": (
                "own-distribution career-average earnings (AIME*12); "
                "weighted 20% quintiles"
            ),
            "ratio_415g_note": (
                "incidence ratio uses the pre-415(g) PIA amount; the "
                "sub-$0.10 dime rounding is omitted from the ratio only"
            ),
        },
        "person_selection": {
            "rule": (
                "pia_observed coverage rule remapped to the gate window: "
                "positive-earnings biennial coverage >= 0.8 over the "
                "person's observed in-window span AND >= 8 positive biennial "
                "observations"
            ),
            "coverage_floor": COVERAGE_FLOOR,
            "min_positive_biennial_obs": MIN_POSITIVE_BIENNIAL_OBS,
            "n_selected": len(selected),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "common_support": "gate-filtered ages 25-59, 1998-2022 biennial",
            "generation": (
                "candidate 11 (run 13) fit on the seed's train complement, "
                "generated on the holdout support -- imported byte-for-byte "
                "from reform_delta_diagnostic.fit_and_generate_candidate11"
            ),
            "floor": (
                "real-vs-real ctx20 on the coverage-selected panel "
                "(fraction=0.4 seed=1000+s, then fraction=0.5 seed=s); two "
                "disjoint ~20%-of-selected halves, per-quintile A-vs-B gap"
            ),
            "metric": (
                "weighted-mean reform/scheduled PIA ratio (percent) by own "
                "AIME quintile, per provision (PI, PPI), per side (real, "
                "generated), per seed; pooled across seeds; floors per "
                "quintile"
            ),
        },
        "three_way_comparison": three_way,
        "per_seed": per_seed,
        "revision_pins": _revision_pins(params),
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if verbose:
        _print_summary(three_way, transport)
    return artifact


def _print_summary(
    three_way: dict[str, Any], transport: dict[str, Any]
) -> None:
    print(
        "\n=== PPI three-way (DYNASIM / real / generated), % of scheduled ==="
    )
    print(
        f"{'Q':>2} {'DYNASIM':>8} {'real':>8} {'gen':>8} "
        f"{'|gen-real|':>10} {'floor':>8} {'>floor':>7}"
    )
    for row in three_way["ppi_by_quintile"]:
        print(
            f"{row['quintile']:>2} {row['dynasim_pct']:>8.1f} "
            f"{row['real_pooled']['mean']:>8.1f} "
            f"{row['generated_pooled']['mean']:>8.1f} "
            f"{row['realgen_scale']:>10.2f} {row['floor_scale']:>8.2f} "
            f"{str(row['gap_exceeds_floor']):>7}"
        )
    pi = three_way["pi_scalars"]
    print(
        f"\nPI scalars: real={pi['real_pooled_mean_pct']['mean']:.2f} "
        f"gen={pi['generated_pooled_mean_pct']['mean']:.2f} "
        f"wedge={pi['wedge_implied_scalar_pct']:.2f} "
        f"DYNASIM={pi['dynasim_pct']}"
    )
    d = three_way["directional_prediction"]
    print(f"directional prediction held: {d['prediction_held']}")


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
