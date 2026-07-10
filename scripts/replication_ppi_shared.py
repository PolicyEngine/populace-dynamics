"""Shared-earnings PPI replication: Mermin (2005)'s exact quintile concept.

REPORTED, NOT GATED. This artifact reads no gate and changes no gate. It
is the shared-lifetime-earnings companion to the Phase-A PI/PPI
replication (:mod:`scripts.replication_ppi_mermin`). Phase A could only
run the INDIVIDUAL own-record analogue of Mermin (2005)'s incidence table
-- it had no marriage histories, so it ranked retired workers by their own
career-average earnings, a named population delta versus the paper's
SHARED lifetime-earnings quintiles. With the marriage histories validated
(C0, gate-2a), that exact concept is now computable on REAL couples, and
this module closes the delta: it reports Mermin's own-record PI/PPI benefit
cuts by SHARED-earnings quintile, exactly as the paper does.

Frozen spec: issue #42 comment 4931009783. Where this module and the
registration disagree, the registration wins.

=====================================================================
REAL DATA ONLY (the ratified tranche map gates generated couples)
=====================================================================
Per the freshly ratified gate-2 tranche map (``gates.yaml``
``gate_2 ... provision_class_coverage.marriage_x_earnings_joint``, #112),
the marriage x earnings joint -- WHO marries WHOM as a function of
earnings -- is "NOT COVERED ... Declared the separate UNLOCKED tranche
2c_marriage_earnings_joint (reported-not-gated), not silence". A
GENERATED-couples version of this diagnostic (synthetic spouses drawn
jointly with earnings) would require that unlocked tranche 2c. It is not
available, so this diagnostic runs on REAL PSID couples only: a study
person from the Phase-A career frame whose marriage-history spouses all
carry a computable earnings record (the R7 join). There is no generated
component and no gate-2-passing model is required; the five seeds drive
ONLY the real-vs-real person-disjoint half-split floors. The shared
measure itself is deterministic.

The same tranche map (gates.yaml certification_scope) records that
OWN-RECORD benefit levels are OUTSIDE tranches 2a/2b/2c -- already
certified (#74's EH + SF surfaces), not household-tranche territory. The
benefit ratio here is exactly that own-record surface, so it needs no
tranche; only the shared RANKING touches marriage, and it uses real
couples. The unlocked 2c would be needed only to GENERATE couples.

=====================================================================
Mermin's shared lifetime earnings (page-cited; the RANKING variable only)
=====================================================================
Mermin (2005), printed page 3 (Methodology), with footnote 9
(~/PolicyEngine/dynasim-refs/411260-benefit-reductions.{pdf,txt}):

    "To classify individuals by lifetime income the analysis uses shared
    lifetime earnings. The earnings stream used to calculate shared
    lifetime earnings includes a worker's entire earnings in years he or
    she is single and half of the earnings of both the worker and the
    worker's spouse in years he or she is married." (printed p.3)

    footnote 9: "This report uses shared lifetime earnings to better
    classify the well-being of individuals who share resources with their
    spouses. For instance, when using shared lifetime earnings a
    nonworking spouse of a high-wage worker is not classified as
    low-income. Similar to Social Security's average indexed monthly
    earnings, earnings are wage-indexed to equate the same relative
    earnings over time." (printed p.3)

Two facts from that text pin the construction:

* The shared stream is the COUPLE MEAN in married years (half of both
  spouses' earnings = ``0.5 * (own + spouse)``) and OWN earnings when
  single. Built here on the R7 couple-join machinery verbatim
  (:meth:`StudyData.relevant_episodes` for the marriage years and the
  joinable spouse, :attr:`StudyData.history` for the spouse's earnings).
* "Similar to ... average indexed monthly earnings ... wage-indexed" ties
  the shared stream to the AIME convention, which CAPS each year at the
  taxable maximum and wage-indexes it. So the cap is applied to the couple
  mean (cap-AFTER-averaging): the raw couple mean is formed here, and the
  Phase-A transport (:func:`transported_person_aime`, imported verbatim)
  caps each year at that year's wage base (415(b)) and NAWI-indexes it.
  This differs from Favreault-Steuerle (2007)'s "capped at the taxable
  maximum PRIOR to sharing" convention used in R7 -- a different paper;
  on this compressed common support the cap almost never binds, so the
  two conventions are numerically indistinguishable here.

Crucially, in Mermin the shared measure is ONLY the well-being ranking
variable (footnote 9). Benefits are the retired worker's OWN-record
benefit (Table 2 is "retired workers ... own-record benefits only"). So
the PI/PPI percent-of-scheduled is computed on OWN AIME with the PPI bend
at the 30th percentile of OWN AIME -- the Phase-A benefit math VERBATIM --
and only the reporting QUINTILE changes from own to shared. The individual
version and the shared version therefore share the identical per-person
benefit ratio; the sole difference is the ranking. That is exactly what
the three-way table isolates: individual (own quintile) vs shared (shared
quintile) vs the Mermin anchor.

=====================================================================
Named population deltas vs Mermin's DYNASIM 2050 table (documented)
=====================================================================
Closing the individual-vs-shared delta leaves the SAME residual deltas the
Phase-A run named (this is its shared sibling on the same common support):

* TRUNCATED OBSERVATION WINDOW: gate-filtered ages 25-59, 1998-2022
  biennial PSID careers pushed through the committed PIA-proxy convention
  (top min(10,n) NAWI-indexed biennial earnings), not full 415(b) top-35
  careers. The common-support AIME distribution is compressed relative to
  the bends (most careers below the second bend), so the incidence
  gradient is FLATTER and HIGHER than DYNASIM's -- the upper quintiles stay
  compressed by construction of the support restriction (named, not
  chased).
* OBSERVED COHORTS: PSID retired workers observed in 1998-2022 (eligible
  near 2005-2030) transported to a single 2050-eligibility cohort, vs
  Mermin's projected 62-67-in-2050 retirees.
* COMMON-SUPPORT RESTRICTION: the identical gate-filtered support and
  proxy-AIME convention as the Phase-A run, so the individual and shared
  versions differ ONLY in the ranking variable -- a clean internal
  contrast.

Run (from the repository root, PSID family + marriage files staged)::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv/bin/python scripts/replication_ppi_shared.py
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Weighted-statistic helpers and the seed-stable pooling convention,
# imported byte-for-byte from the merged builders (single source of truth).
from build_downstream_relevance import (  # noqa: E402
    _weighted_mean,
    _weighted_quantile,
)
from reform_delta_diagnostic import _summary  # noqa: E402

# Phase-A transport + PI/PPI provision encodings + the career frame, all
# imported VERBATIM (the registration: "Phase-A 2050 transport and
# provision encodings verbatim").
from replication_ppi_mermin import (  # noqa: E402
    DYNASIM_PI_ALL,
    DYNASIM_PI_BY_QUINTILE,
    DYNASIM_PPI_ALL,
    DYNASIM_PPI_BY_QUINTILE,
    DYNASIM_SCHEDULED_MEAN_2005USD,
    DYNASIM_SCHEDULED_MEAN_ALL_2005USD,
    N_QUINTILES,
    PPI_BEND_PERCENTILE,
    QUINTILE_LEVELS,
    _assign_quintiles,
    build_transport,
    coverage_selected_persons,
    price_indexed_amount,
    progressive_price_indexed_amount,
    scheduled_amount,
    transported_person_aime,
)

# The R7 couple-join machinery, imported VERBATIM (the registration: "the
# R7 join machinery verbatim").
from replication_r7_sharing import (  # noqa: E402
    ELIGIBILITY_AGE,
    MARITAL,
    StudyData,
)
from run_gate1_baseline import SEEDS, load_filtered_panel  # noqa: E402
from run_gate1_candidate5b import anchor_rows  # noqa: E402

from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.ss.params import load_ssa_parameters  # noqa: E402

ARTIFACT_PATH = ROOT / "runs" / "replication_ppi_shared_v1.json"
ARTIFACT_SCHEMA_VERSION = "replication_ppi_shared.v1"
RUN_NAME = "replication_ppi_shared_v1"

#: This replication's frozen-spec registration (issue #42 comment).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4931009783"
)
SPEC_REGISTRATION_POINTER = "#42 comment 4931009783"
#: The phase catalog / program-design context.
PROGRAM_DESIGN_ISSUE = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/74"
)
#: The Phase-A individual-version replication whose delta this closes.
PHASE_A_INDIVIDUAL_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4907444903"
)
PHASE_A_ARTIFACT = ROOT / "runs" / "replication_ppi_mermin_v1.json"
#: The R7 earnings-sharing replication whose couple-join machinery this
#: reuses.
R7_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911171806"
)
#: The ratified tranche map that gates generated couples on the unlocked
#: marriage x earnings joint tranche.
TRANCHE_MAP_ISSUE = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/112"
)


# =====================================================================
# Shared-earnings construction (Mermin's couple mean; R7 join machinery)
# =====================================================================
def shared_earnings_on_periods(
    study: StudyData,
    pid: int,
    periods: np.ndarray,
    own_earnings: np.ndarray,
    elig: int,
) -> np.ndarray:
    """Mermin shared earnings aligned to a person's observed periods.

    Reuses the R7 join machinery verbatim
    (:meth:`StudyData.relevant_episodes` for each marriage's span and
    joinable spouse, :attr:`StudyData.history` for the spouse's earnings).
    For every observed period inside a marriage span the value becomes the
    COUPLE MEAN ``0.5 * (own + spouse)`` (Mermin p.3: "half of the earnings
    of both the worker and the worker's spouse in years he or she is
    married"); periods outside every marriage keep OWN earnings ("a
    worker's entire earnings in years he or she is single"). The couple
    mean is of RAW earnings -- the Phase-A transport caps each year at the
    taxable maximum (Mermin fn.9 "wage-indexed like AIME"), so the cap is
    applied AFTER averaging. Ongoing marriages share through the
    eligibility year. Callers pass only both-spouse-covered persons.
    """
    year_to_spouse: dict[int, int] = {}
    for row in study.relevant_episodes(pid):
        spouse = row.spouse_person_id
        if pd.isna(spouse):
            continue
        start = row.start_year
        if pd.isna(start):
            continue
        end = (
            int(row.episode_end_year)
            if not pd.isna(row.episode_end_year)
            else elig
        )
        for year in range(int(start), end + 1):
            year_to_spouse[year] = int(spouse)

    shared = np.array(own_earnings, dtype=np.float64).copy()
    for i in range(len(periods)):
        year = int(periods[i])
        spouse_id = year_to_spouse.get(year)
        if spouse_id is None:
            continue
        spouse_earn = study.history.get(spouse_id, {}).get(year, 0.0)
        shared[i] = 0.5 * (float(own_earnings[i]) + float(spouse_earn))
    return shared


def select_study_persons(panel: pd.DataFrame, study: StudyData) -> list[int]:
    """Phase-A career frame intersected with marriage-history joinable.

    The Phase-A career frame is :func:`coverage_selected_persons` on the
    gate panel (the registration's "Phase-A career frame"). A person is
    joinable iff they carry marriage-history data and every relevant
    marriage's spouse has a computable earnings record
    (:meth:`StudyData.both_spouse_covered`, the R7 join -- never-married
    persons are trivially covered).
    """
    career_frame = coverage_selected_persons(panel)
    out: list[int] = []
    for pid in sorted(career_frame):
        pid = int(pid)
        if pid not in study.birth_year or pid not in study.sex:
            continue
        if study.both_spouse_covered(pid):
            out.append(pid)
    return out


def build_person_frame(
    study: StudyData,
    panel: pd.DataFrame,
    person_ids: list[int],
    weight_of: dict[int, float],
    params: Any,
    transport: dict[str, Any],
) -> pd.DataFrame:
    """One row per study person: own and shared transported AIME + weight.

    Both AIMEs use the SAME observed periods and OWN earnings from the gate
    panel (so the own AIME matches the Phase-A individual run exactly); the
    shared AIME replaces marriage-year earnings with the couple mean. The
    per-person AIMEs are computed once here and reused for the full sample
    and every half-split floor (the AIME does not depend on the seed).
    """
    ids = set(person_ids)
    sub = panel[panel["person_id"].isin(ids)]
    rows: list[dict[str, float]] = []
    for pid, g in sub.groupby("person_id", sort=True):
        pid = int(pid)
        periods = g["period"].to_numpy()
        own_earn = g["earnings"].to_numpy(dtype=np.float64)
        elig = int(study.birth_year[pid]) + ELIGIBILITY_AGE
        own_aime = transported_person_aime(
            periods, own_earn, params, transport
        )
        shared_earn = shared_earnings_on_periods(
            study, pid, periods, own_earn, elig
        )
        shared_aime = transported_person_aime(
            periods, shared_earn, params, transport
        )
        rows.append(
            {
                "person_id": pid,
                "own_aime": own_aime,
                "shared_aime": shared_aime,
                "weight": float(weight_of[pid]),
            }
        )
    return pd.DataFrame(rows)


# =====================================================================
# Per-quintile own-record PI/PPI ratios, grouped by own AND by shared AIME
# =====================================================================
def _quintile_rows(
    quintile: np.ndarray,
    pi_ratio: np.ndarray,
    ppi_ratio: np.ndarray,
    own_aime: np.ndarray,
    shared_aime: np.ndarray,
    sched: np.ndarray,
    weight: np.ndarray,
    positive: np.ndarray,
) -> list[dict[str, Any]]:
    """Weighted-mean own-record PI/PPI ratios (percent) per quintile bin.

    Careers with a zero scheduled amount (own AIME 0) are excluded from the
    ratio means (undefined percent of scheduled) and counted separately.
    """
    rows: list[dict[str, Any]] = []
    for k in range(N_QUINTILES):
        in_bin = quintile == k
        mask = in_bin & positive
        if not np.any(mask):
            rows.append(
                {
                    "quintile": k + 1,
                    "n_persons": int(np.sum(in_bin)),
                    "n_positive": 0,
                }
            )
            continue
        w = weight[mask]
        rows.append(
            {
                "quintile": k + 1,
                "n_persons": int(np.sum(in_bin)),
                "n_positive": int(np.sum(mask)),
                "pi_ratio_pct": 100.0 * _weighted_mean(pi_ratio[mask], w),
                "ppi_ratio_pct": 100.0 * _weighted_mean(ppi_ratio[mask], w),
                "mean_own_aime": _weighted_mean(own_aime[mask], w),
                "mean_shared_aime": _weighted_mean(shared_aime[mask], w),
                "mean_scheduled_amount": _weighted_mean(sched[mask], w),
            }
        )
    return rows


def measure_population(
    frame: pd.DataFrame, transport: dict[str, Any]
) -> dict[str, Any]:
    """Own-record PI/PPI ratios by own quintile AND by shared quintile.

    The benefit math is the Phase-A encoding VERBATIM on OWN AIME: the PPI
    bend is the weighted 30th percentile of OWN AIME, and the reform /
    scheduled ratio is a per-person property of OWN AIME. It is reported
    two ways -- grouped by OWN-AIME quintile (``by_own_quintile``, the
    Phase-A individual version) and by SHARED-AIME quintile
    (``by_shared_quintile``, Mermin's shared ranking). The per-person ratio
    is identical in both; only the ranking differs, exactly as Mermin
    reports own-record benefits by shared-lifetime-income quintile.
    """
    own_aime = frame["own_aime"].to_numpy(dtype=np.float64)
    shared_aime = frame["shared_aime"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)

    bend30 = float(
        _weighted_quantile(own_aime, weight, np.array([PPI_BEND_PERCENTILE]))[
            0
        ]
    )
    sched = scheduled_amount(own_aime, transport)
    pi_amt = price_indexed_amount(own_aime, transport)
    ppi_amt = progressive_price_indexed_amount(own_aime, bend30, transport)
    positive = sched > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        pi_ratio = np.where(positive, pi_amt / sched, np.nan)
        ppi_ratio = np.where(positive, ppi_amt / sched, np.nan)

    own_cut = _weighted_quantile(own_aime, weight, QUINTILE_LEVELS)
    shared_cut = _weighted_quantile(shared_aime, weight, QUINTILE_LEVELS)
    own_q = _assign_quintiles(own_aime, own_cut)
    shared_q = _assign_quintiles(shared_aime, shared_cut)

    common = (pi_ratio, ppi_ratio, own_aime, shared_aime, sched, weight)
    all_mask = positive
    return {
        "n_persons": int(len(own_aime)),
        "n_zero_scheduled_excluded": int(np.sum(~positive)),
        "bend30_own_aime": bend30,
        "own_quintile_cutpoints_aime": [float(c) for c in own_cut],
        "shared_quintile_cutpoints_aime": [float(c) for c in shared_cut],
        "n_reshuffled_own_vs_shared": int(np.sum(own_q != shared_q)),
        "weighted_share_reshuffled": (
            float(
                _weighted_mean((own_q != shared_q).astype(np.float64), weight)
            )
        ),
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
        "by_own_quintile": _quintile_rows(own_q, *common, positive),
        "by_shared_quintile": _quintile_rows(shared_q, *common, positive),
    }


def measure_floor_seed(
    frame: pd.DataFrame, seed: int, transport: dict[str, Any]
) -> dict[str, Any]:
    """Real-vs-real quintile-ratio floor for one seed (50/50 half-split).

    Splits the study persons person-disjointly 50/50 (the committed
    :func:`hpanel.split_panel_by_person`, fraction=0.5) and measures each
    half's own- and shared-quintile ratios (each on its OWN half
    distribution's bend and cutpoints). The per-quintile A-vs-B gap is the
    sampling-noise scale at half sample.
    """
    side_a, side_b = hpanel.split_panel_by_person(
        frame, "person_id", fraction=0.5, seed=seed
    )
    return {
        "n_persons_side_a": int(side_a["person_id"].nunique()),
        "n_persons_side_b": int(side_b["person_id"].nunique()),
        "side_a": measure_population(side_a, transport),
        "side_b": measure_population(side_b, transport),
    }


# =====================================================================
# Three-way table (individual / shared / anchor) + floors + expectations
# =====================================================================
def _vec(side: dict[str, Any], grouping: str, key: str) -> list[float | None]:
    """The five quintile values of ``key`` for a grouping (None if empty)."""
    return [
        q.get(key) if q.get("n_positive", 0) > 0 else None
        for q in side[grouping]
    ]


def _pooled_floor(
    per_seed: list[dict[str, Any]], grouping: str, key: str, k: int
) -> dict[str, Any]:
    """Pooled |side_a - side_b| scale at quintile ``k`` for a grouping."""
    gaps: list[float] = []
    for r in per_seed:
        a = _vec(r["side_a"], grouping, key)[k]
        b = _vec(r["side_b"], grouping, key)[k]
        if a is not None and b is not None:
            gaps.append(abs(a - b))
    return _summary(gaps)


def build_three_way(
    full: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per quintile: anchor (Mermin), individual (own q), shared (shared q).

    Records the shared-vs-individual movement, each side's absolute gap to
    the Mermin anchor, whether shared moved CLOSER to the anchor, and the
    per-quintile half-split floors for both groupings.
    """
    ind = _vec(full, "by_own_quintile", "ppi_ratio_pct")
    shr = _vec(full, "by_shared_quintile", "ppi_ratio_pct")
    table: list[dict[str, Any]] = []
    for k in range(N_QUINTILES):
        anchor = DYNASIM_PPI_BY_QUINTILE[k]
        ind_gap = abs(ind[k] - anchor)
        shr_gap = abs(shr[k] - anchor)
        shr_floor = _pooled_floor(
            per_seed, "by_shared_quintile", "ppi_ratio_pct", k
        )
        table.append(
            {
                "quintile": k + 1,
                "anchor_dynasim_ppi_pct": anchor,
                "individual_ppi_pct": ind[k],
                "shared_ppi_pct": shr[k],
                "shared_minus_individual_pp": shr[k] - ind[k],
                "individual_abs_gap_vs_anchor_pp": ind_gap,
                "shared_abs_gap_vs_anchor_pp": shr_gap,
                "shared_closer_to_anchor": bool(shr_gap < ind_gap),
                "abs_shared_minus_individual_pp": abs(shr[k] - ind[k]),
                "regroup_exceeds_shared_floor": bool(
                    abs(shr[k] - ind[k]) > shr_floor["mean"]
                ),
                "individual_floor_pp": _pooled_floor(
                    per_seed, "by_own_quintile", "ppi_ratio_pct", k
                ),
                "shared_floor_pp": shr_floor,
            }
        )
    return {
        "ppi_by_quintile": table,
        "ppi_all": {
            "anchor_dynasim_pct": DYNASIM_PPI_ALL,
            "individual_overall_pct": full["overall_ppi_ratio_pct"],
            "shared_overall_pct": full["overall_ppi_ratio_pct"],
            "note": (
                "individual and shared share the identical per-person "
                "own-record ratio, so the population 'all' PPI ratio is the "
                "same for both -- only the by-quintile grouping differs"
            ),
        },
    }


def build_pi_scalars(
    full: dict[str, Any], transport: dict[str, Any]
) -> dict[str, Any]:
    """PI scalars: quintile-invariant == the wedge W, by construction."""
    return {
        "individual_by_quintile_pct": _vec(
            full, "by_own_quintile", "pi_ratio_pct"
        ),
        "shared_by_quintile_pct": _vec(
            full, "by_shared_quintile", "pi_ratio_pct"
        ),
        "overall_pct": full["overall_pi_ratio_pct"],
        "wedge_implied_scalar_pct": 100.0 * transport["wedge"],
        "anchor_dynasim_pct": DYNASIM_PI_ALL,
        "anchor_dynasim_by_quintile": list(DYNASIM_PI_BY_QUINTILE),
        "note": (
            "PI scales all PIA factors by the same wedge W, so the "
            "PI/scheduled ratio is W for every career, quintile invariant "
            "under either ranking (Mermin's PI row is likewise flat at "
            f"{DYNASIM_PI_ALL})"
        ),
    }


def build_expectations(table: list[dict[str, Any]]) -> dict[str, Any]:
    """The registration's pre-registered expectations (logged, not tuned).

    (1) the shared-quintile gradient moves CLOSER to Mermin's levels than
        the individual version at Q1-Q2; (2) monotonicity holds; (3) Q1
        within 3pp of the anchor's 98.7 (the individual version sat at
        100.0); (4) the upper quintiles remain compressed by the support
        restriction (named, not chased).
    """
    shr = [row["shared_ppi_pct"] for row in table]
    ind = [row["individual_ppi_pct"] for row in table]
    monotone = all(shr[k] >= shr[k + 1] - 1e-9 for k in range(N_QUINTILES - 1))
    closer_q1 = table[0]["shared_closer_to_anchor"]
    closer_q2 = table[1]["shared_closer_to_anchor"]
    q1_within_3pp = abs(shr[0] - DYNASIM_PPI_BY_QUINTILE[0]) <= 3.0
    top_gap = table[N_QUINTILES - 1]["shared_abs_gap_vs_anchor_pp"]
    return {
        "prediction": (
            "the shared-quintile PPI gradient moves CLOSER to Mermin's "
            "levels than the Phase-A individual version at Q1-Q2 (spousal "
            "sharing pulls low-own-earnings wives of higher earners up the "
            "distribution, so shared Q1 picks up some above-bend careers "
            "the individual Q1 lacked); monotonicity holds; Q1 within 3pp "
            "of the anchor's 98.7 (the individual version sat at 100.0); "
            "the upper quintiles remain compressed by the truncated-window "
            "support restriction (named, not chased)"
        ),
        "anchor_q1_ppi_pct": DYNASIM_PPI_BY_QUINTILE[0],
        "individual_q1_ppi_pct": ind[0],
        "shared_q1_ppi_pct": shr[0],
        "shared_q1_within_3pp_of_anchor": bool(q1_within_3pp),
        "shared_closer_at_q1": bool(closer_q1),
        "shared_closer_at_q2": bool(closer_q2),
        "shared_closer_at_q1_q2": bool(closer_q1 and closer_q2),
        "shared_ppi_monotone_nonincreasing": bool(monotone),
        "top_quintile_shared_abs_gap_vs_anchor_pp": top_gap,
        "top_quintile_still_compressed": bool(top_gap > 0.0),
        "all_core_expectations_held": bool(
            q1_within_3pp and closer_q1 and closer_q2 and monotone
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


def _gates_amendment_state() -> dict[str, Any]:
    """Gate-1 lock + gate-2 tranche-scope state, parsed from gates.yaml."""
    doc = yaml.safe_load((ROOT / "gates.yaml").read_text())
    gate1 = doc["gates"]["gate_1"]
    thresholds = gate1.get("thresholds", {})
    history = gate1.get("amendment_history", []) or []
    gate2 = doc["gates"].get("gate_2", {})
    coverage = (
        gate2.get("thresholds", {})
        .get("scope", {})
        .get("provision_class_coverage", {})
    )
    if not coverage:
        coverage = gate2.get("scope", {}).get("provision_class_coverage", {})
    return {
        "gate_1_locked": bool(thresholds.get("locked", False)),
        "amendments_ratified": [
            {"id": a.get("id"), "ratified": a.get("ratified")} for a in history
        ],
        "gate_2_marriage_x_earnings_joint_scope": coverage.get(
            "marriage_x_earnings_joint"
        ),
    }


def _revision_pins(params: Any) -> dict[str, Any]:
    return {
        "populace_dynamics_sha": _sha(ROOT),
        "pe_us_revision": getattr(params, "pe_us_revision", None),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "gates_yaml": _gates_amendment_state(),
    }


def anchor_provenance() -> dict[str, Any]:
    """Mermin (2005) shared-earnings definition + Table 2 (page-cited).

    The definition and the target rows are verified against the archived
    PDF and its pdftotext
    (~/PolicyEngine/dynasim-refs/411260-benefit-reductions.{pdf,txt}). The
    Table 2 target rows reuse the constants transcribed and page-cited in
    the Phase-A run (imported here so there is one source of truth).
    """
    return {
        "paper": (
            "Mermin, G. B. T. (2005). The Effect of Benefit Reductions on "
            "the Distribution of Social Security Benefits. Urban Institute "
            "report 411260. DYNASIM3, Runid 432. 2005 Trustees intermediate "
            "assumptions."
        ),
        "source_files": [
            "~/PolicyEngine/dynasim-refs/411260-benefit-reductions.pdf",
            "~/PolicyEngine/dynasim-refs/411260-benefit-reductions.txt",
        ],
        "shared_lifetime_earnings_definition": {
            "citation": "printed page 3 (Methodology) and footnote 9",
            "quote": (
                "To classify individuals by lifetime income the analysis "
                "uses shared lifetime earnings. The earnings stream used to "
                "calculate shared lifetime earnings includes a worker's "
                "entire earnings in years he or she is single and half of "
                "the earnings of both the worker and the worker's spouse in "
                "years he or she is married."
            ),
            "footnote_9_quote": (
                "This report uses shared lifetime earnings to better "
                "classify the well-being of individuals who share resources "
                "with their spouses. For instance, when using shared "
                "lifetime earnings a nonworking spouse of a high-wage worker "
                "is not classified as low-income. Similar to Social "
                "Security's average indexed monthly earnings, earnings are "
                "wage-indexed to equate the same relative earnings over "
                "time."
            ),
            "encoding": (
                "couple mean 0.5*(own + spouse) in married years, own "
                "earnings when single; the raw couple mean is wage-indexed "
                "and capped at the taxable maximum by the Phase-A transport "
                "(cap-after-averaging, per fn.9's 'like AIME'); the shared "
                "measure is the RANKING variable only -- benefits stay "
                "own-record"
            ),
            "capping_convention_note": (
                "Mermin's fn.9 ties the shared stream to AIME (cap at the "
                "taxable maximum, wage-indexed), applied to the couple mean "
                "-> cap AFTER averaging. This differs from "
                "Favreault-Steuerle (2007)'s 'capped ... PRIOR to sharing' "
                "(the R7 convention). On this compressed common support the "
                "cap almost never binds, so the choice is numerically "
                "immaterial here."
            ),
        },
        "population_and_quintile": {
            "citation": "Table 2 title + notes (PDF p.16); p.3 (Methodology)",
            "population": (
                "retired workers aged 62-67 in 2050, own-record benefits "
                "only, N=5351 -- so the PI/PPI percent-of-scheduled is an "
                "OWN-record quantity, reported by SHARED-lifetime-income "
                "quintile"
            ),
            "career_average_definition": (
                "career average annual earnings = average indexed monthly "
                "earnings * 12 (Figure 1 note, PDF p.14); the shared measure "
                "ranks on shared AIME (equivalently shared AIME*12)"
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
        "named_population_deltas": [
            "truncated observation window: gate-filtered ages 25-59, "
            "1998-2022 biennial PSID careers through the committed "
            "PIA-proxy convention (this study) vs full 415(b) careers "
            "(Mermin) -- compresses the AIME distribution, flattening and "
            "raising the incidence gradient, so the upper quintiles stay "
            "compressed by construction",
            "observed cohorts: PSID retired workers observed 1998-2022 "
            "transported to a single 2050-eligibility cohort (this study) "
            "vs Mermin's projected 62-67-in-2050 retirees",
            "common-support restriction: the identical gate-filtered support "
            "and proxy-AIME convention as the Phase-A run, so the individual "
            "and shared versions differ ONLY in the ranking variable",
            "the individual-vs-shared quintile-concept delta the Phase-A "
            "run named is CLOSED here: the shared measure is computed "
            "exactly (spouse-shared during marriage on real couples)",
        ],
    }


def _phase_a_full_population_reference() -> dict[str, Any]:
    """Phase-A full-population individual PPI vector, read (not hardcoded).

    Read from the committed Phase-A artifact when present; a labeled
    reference for the individual version on its LARGER full population
    (before the marriage-joinable intersection). None if unavailable.
    """
    if not PHASE_A_ARTIFACT.exists():
        return {"available": False}
    doc = json.loads(PHASE_A_ARTIFACT.read_text())
    rows = doc["three_way_comparison"]["ppi_by_quintile"]
    return {
        "available": True,
        "source": "runs/replication_ppi_mermin_v1.json",
        "note": (
            "the Phase-A individual version on its full population (larger "
            "than this run's marriage-joinable intersection); shown for "
            "context, not as the three-way individual column (which is "
            "recomputed here on the identical study population)"
        ),
        "individual_ppi_by_quintile_pct": [
            row["real_pooled"]["mean"] for row in rows
        ],
        "individual_pi_overall_pct": doc["three_way_comparison"]["pi_scalars"][
            "real_pooled_mean_pct"
        ]["mean"],
    }


def _join_report(panel: pd.DataFrame, study: StudyData) -> dict[str, Any]:
    """Marriage-joinability over the Phase-A career frame, by status."""
    career_frame = coverage_selected_persons(panel)
    by_status = {s: {"n": 0, "joinable": 0} for s in MARITAL}
    n_total = 0
    n_joinable = 0
    for pid in career_frame:
        pid = int(pid)
        if pid not in study.birth_year or pid not in study.sex:
            continue
        status = study.status_at_eligibility(pid)
        if status not in by_status:
            continue
        joinable = study.both_spouse_covered(pid)
        by_status[status]["n"] += 1
        n_total += 1
        if joinable:
            by_status[status]["joinable"] += 1
            n_joinable += 1
    return {
        "definition": (
            "share of Phase-A career-frame persons whose relevant "
            "marriage spouses all carry a computable earnings record "
            "(never-married trivially joinable)"
        ),
        "n_career_frame_classified": n_total,
        "n_marriage_joinable": n_joinable,
        "marriage_joinable_share": (
            round(n_joinable / n_total, 4) if n_total else 0.0
        ),
        "by_marital_status": {
            s: {
                "n": v["n"],
                "n_joinable": v["joinable"],
                "share": (round(v["joinable"] / v["n"], 4) if v["n"] else 0.0),
            }
            for s, v in by_status.items()
        },
    }


def _ever_married_counts(
    study: StudyData, person_ids: list[int]
) -> dict[str, int]:
    """How many study persons share earnings (ever married) vs own-only."""
    ever = 0
    for pid in person_ids:
        if study.relevant_episodes(int(pid)):
            ever += 1
    return {
        "n_study": len(person_ids),
        "n_ever_married_sharing": ever,
        "n_never_married_own_only": len(person_ids) - ever,
    }


# =====================================================================
# Driver
# =====================================================================
def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the shared-earnings PI/PPI replication (reported, not gated)."""
    started = time.time()

    params = load_ssa_parameters()
    transport = build_transport(params)
    if verbose:
        print(
            f"pe_us_revision={params.pe_us_revision} "
            f"wedge W={transport['wedge']:.5f} "
            f"bends(2050)={transport['bend_points']}"
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

    if verbose:
        print("loading PSID marriage histories + family earnings ...")
    study = StudyData(params)

    person_ids = select_study_persons(panel, study)
    frame = build_person_frame(
        study, panel, person_ids, weight_of, params, transport
    )
    if verbose:
        print(
            f"gate panel: {panel.person_id.nunique()} persons; "
            f"study population (career frame ∩ marriage joinable): "
            f"{len(person_ids)}"
        )

    full = measure_population(frame, transport)

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        per_seed.append(
            {"seed": seed, **measure_floor_seed(frame, seed, transport)}
        )

    three_way = build_three_way(full, per_seed)
    pi_scalars = build_pi_scalars(full, transport)
    expectations = build_expectations(three_way["ppi_by_quintile"])

    join_report = _join_report(panel, study)
    ever_married = _ever_married_counts(study, person_ids)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "reported_not_gated": True,
        "registration": SPEC_REGISTRATION,
        "registration_pointer": SPEC_REGISTRATION_POINTER,
        "program_design_issue": PROGRAM_DESIGN_ISSUE,
        "phase_a_individual_registration": PHASE_A_INDIVIDUAL_REGISTRATION,
        "r7_join_machinery_registration": R7_REGISTRATION,
        "purpose": (
            "Shared-earnings companion to the Phase-A PI/PPI replication: "
            "report Mermin (2005)'s own-record price-indexing and "
            "progressive-price-indexing benefit cuts by SHARED "
            "lifetime-earnings quintile -- the paper's exact quintile "
            "concept -- computed on REAL PSID couples, closing the Phase-A "
            "individual-vs-shared named delta. Reads no gate, changes no "
            "gate; publishes regardless of outcome."
        ),
        "real_data_only_scope": {
            "real_data_only": True,
            "tranche_map_issue": TRANCHE_MAP_ISSUE,
            "statement": (
                "The ratified gate-2 tranche map (gates.yaml "
                "gate_2 ... provision_class_coverage.marriage_x_earnings_"
                "joint, #112) declares the marriage x earnings joint -- who "
                "marries whom as a function of earnings -- NOT COVERED and "
                "the separate UNLOCKED tranche 2c_marriage_earnings_joint "
                "(reported-not-gated). A generated-couples version of this "
                "diagnostic (synthetic spouses drawn jointly with earnings) "
                "would require that unlocked tranche 2c, so this diagnostic "
                "runs on REAL PSID couples only: a Phase-A career-frame "
                "person whose marriage-history spouses all carry a "
                "computable earnings record (the R7 join). No generated "
                "component; the five seeds drive only the real-vs-real "
                "half-split floors. The tranche map further records "
                "(certification_scope) that OWN-RECORD benefit levels are "
                "OUTSIDE tranches 2a/2b/2c (already certified), and the "
                "benefit ratio here is exactly that own-record surface -- "
                "only the shared RANKING touches marriage, and it uses real "
                "couples."
            ),
            "gate_2_marriage_x_earnings_joint_scope": (
                _gates_amendment_state()[
                    "gate_2_marriage_x_earnings_joint_scope"
                ]
            ),
        },
        "design": (
            "The benefit reform is the Phase-A encoding VERBATIM on OWN "
            "AIME (PPI bend at the weighted 30th percentile of own AIME); "
            "the shared lifetime earnings is Mermin's RANKING variable "
            "only. So the individual and shared versions share the "
            "identical per-person own-record PI/PPI ratio and differ ONLY "
            "in the reporting quintile (own-AIME quintile vs shared-AIME "
            "quintile), exactly as Mermin reports own-record benefits by "
            "shared-lifetime-income quintile."
        ),
        "anchor_provenance": anchor_provenance(),
        "transport_and_conventions": {
            "eligibility_year": transport["eligibility_year"],
            "index_year": transport["index_year"],
            "index_nawi": transport["index_nawi"],
            "bend_points_2050": list(transport["bend_points"]),
            "pia_factors": list(transport["pia_factors"]),
            "wedge": transport["wedge"],
            "wedge_years": transport["n_wedge_years"],
            "wedge_formula": "(1.028/1.039)**(2050-2012)",
            "aime_convention": (
                "committed PIA-proxy biennial convention (cap at the "
                "historical wage base; NAWI-index to 2048; top min(10,n) "
                "over count*12*2; 415(b) dollar floor) -- imported from the "
                "Phase-A transport VERBATIM"
            ),
            "ppi_bend": (
                "weighted 30th percentile of OWN transported AIME (the "
                "benefit-relevant distribution); identical for the "
                "individual and shared versions"
            ),
            "shared_measure": (
                "couple mean 0.5*(own + spouse) in marriage years (R7 join "
                "machinery), own earnings when single; ranked by shared "
                "transported AIME; cap-after-averaging via the transport"
            ),
            "quintile_variable_individual": (
                "own-distribution transported AIME; weighted 20% quintiles"
            ),
            "quintile_variable_shared": (
                "shared-distribution transported AIME; weighted 20% "
                "quintiles"
            ),
        },
        "study_population": {
            "career_frame_rule": (
                "Phase-A coverage_selected_persons on the gate panel "
                "(positive-earnings biennial coverage >= 0.8 over the "
                "person's observed in-window span AND >= 8 positive "
                "biennial observations)"
            ),
            "n_gate_panel_persons": int(panel.person_id.nunique()),
            "n_career_frame": int(len(coverage_selected_persons(panel))),
            "n_study": len(person_ids),
            "marriage_joinability": join_report,
            "ever_married_sharing": ever_married,
            "n_reshuffled_own_vs_shared": full["n_reshuffled_own_vs_shared"],
            "weighted_share_reshuffled": full["weighted_share_reshuffled"],
            "weighting": (
                "anchor weight = the person's chronologically last observed "
                "gate-panel cross-sectional weight (the Phase-A anchor)"
            ),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "common_support": "gate-filtered ages 25-59, 1998-2022 biennial",
            "floor": (
                "real-vs-real 50/50 person-disjoint half-split "
                "(split_panel_by_person, fraction=0.5) on the study "
                "population; per-quintile A-vs-B gap on each grouping, "
                "pooled over 5 seeds"
            ),
            "metric": (
                "weighted-mean own-record reform/scheduled PIA ratio "
                "(percent) by quintile, grouped by own AIME (individual) "
                "and by shared AIME (shared); deterministic full sample; "
                "floors per quintile"
            ),
        },
        "three_way_comparison": {
            "columns": "individual (own quintile) / shared (shared quintile) "
            "/ anchor (Mermin DYNASIM), percent of scheduled",
            "ppi_by_quintile": three_way["ppi_by_quintile"],
            "ppi_all": three_way["ppi_all"],
            "pi_scalars": pi_scalars,
            "pre_registered_expectations": expectations,
            "phase_a_full_population_reference": (
                _phase_a_full_population_reference()
            ),
        },
        "full_sample": full,
        "per_seed": per_seed,
        "revision_pins": _revision_pins(params),
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if verbose:
        _print_summary(three_way["ppi_by_quintile"], pi_scalars, expectations)
    return artifact


def _print_summary(
    table: list[dict[str, Any]],
    pi_scalars: dict[str, Any],
    expectations: dict[str, Any],
) -> None:
    print(
        "\n=== PPI three-way (individual / shared / anchor), % of scheduled "
        "==="
    )
    print(
        f"{'Q':>2} {'indiv':>8} {'shared':>8} {'anchor':>8} "
        f"{'shr-ind':>8} {'shrFloor':>9} {'closer':>7}"
    )
    for row in table:
        print(
            f"{row['quintile']:>2} {row['individual_ppi_pct']:>8.2f} "
            f"{row['shared_ppi_pct']:>8.2f} "
            f"{row['anchor_dynasim_ppi_pct']:>8.1f} "
            f"{row['shared_minus_individual_pp']:>8.2f} "
            f"{row['shared_floor_pp']['mean']:>9.2f} "
            f"{str(row['shared_closer_to_anchor']):>7}"
        )
    print(
        f"\nPI: individual/shared flat at "
        f"{pi_scalars['overall_pct']:.2f} "
        f"(wedge {pi_scalars['wedge_implied_scalar_pct']:.2f}); "
        f"anchor {pi_scalars['anchor_dynasim_pct']}"
    )
    print("pre-registered expectations:")
    print(
        f"  Q1 within 3pp of 98.7: "
        f"{expectations['shared_q1_within_3pp_of_anchor']} "
        f"(shared Q1 = {expectations['shared_q1_ppi_pct']:.2f}, "
        f"individual Q1 = {expectations['individual_q1_ppi_pct']:.2f})"
    )
    print(
        f"  shared closer at Q1-Q2: "
        f"{expectations['shared_closer_at_q1_q2']}"
    )
    print(
        f"  shared monotone: "
        f"{expectations['shared_ppi_monotone_nonincreasing']}"
    )
    print(
        f"  ALL CORE EXPECTATIONS HELD: "
        f"{expectations['all_core_expectations_held']}"
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
