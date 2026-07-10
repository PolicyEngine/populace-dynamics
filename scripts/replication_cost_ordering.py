"""Cost-ordering synthesis (reported, not gated): do the committed
provision encodings agree with the anchors on sign and relative magnitude?

REPORTED, NOT GATED. This artifact reads no gate and changes no gate. It
synthesises the five committed external-anchor replications of
PolicyEngine/populace-dynamics (issue #74): every provision encoding
already committed --

    * Mermin (2005) price indexing (PI), progressive price indexing (PPI),
      COLA -0.4pp, and NRA->70 -- the Mermin quartet;
    * the four Smith/Johnson/Favreault (2020) caregiver-credit plans
      (Biden, Buttigieg, Klobuchar, Warren);
    * Favreault-Steuerle (2007) earnings sharing (package 1b)

-- is scored ONCE on a single common frame, and the resulting aggregate
cost deltas are tested ORDINALLY against the anchors' published cost
columns: Mermin's 75-year payroll effects (Table 1) and the Five-Approaches
actuarial-balance A-table (Table 3). No level matching is claimed: our
frame is observed completed careers under the Phase-A 2050 transport, not a
Trustees-assumption projection (Phase E), so units differ by construction
-- signs and orderings are the transportable content.

Frozen spec: issue #42 comment 4931034068. The registration wins over any
disagreement.

=====================================================================
THE COMMON FRAME (frozen spec: "the Phase-A transport population and
baseline exactly as scripts/replication_ppi_mermin.py")
=====================================================================
The canonical Phase-A frame -- the one the NRA/COLA (replication_mermin_
rows) and caregiver (replication_caregiver) replications both build
identically, and on which every benefit-side encoding runs verbatim:

    * population: the Phase-A career-selection frame (the
      ``pia_observed_psid_v1`` rule) -- real PSID careers with coverage
      >= 0.8 of ages 22-61 and an age-62 eligibility year in 2005-2019
      (born 1943-1957), built by :class:`replication_caregiver.
      CaregiverStudy` / :class:`replication_mermin_rows.MerminStudy` from
      ``family_earnings_panel`` and r7's ``_person_history`` (byte-for-byte
      the same code);
    * baseline: the pre-415(g) scheduled PIA (the 90/32/15 bracket sum on
      the 2050 bends), ``replication_ppi_mermin.scheduled_amount`` -- the
      SCHEDULED (not payable) denominator per #74 protocol note 1;
    * AIME: the full statutory 42 USC 415(b) top-35 transported AIME
      (``replication_caregiver.transported_aime``) at the common 2050
      transport (``replication_ppi_mermin.build_transport``);
    * common support: persons with a resolvable sex (the Marriage History
      File join the NRA/COLA claim distribution needs) -- a NAMED
      common-support restriction, not a model choice.

ppi_mermin's own biennial gate panel + biennial-proxy AIME is its
*documented deviation*, forced only by the candidate-11 generator (native
to the gate panel). This synthesis uses NO generator -- it scores real
completed careers -- so it uses the canonical full-415(b) Phase-A frame,
exactly the frame ppi_mermin says it had to deviate from. PI and PPI are
pure functions of the AIME, so ppi_mermin's committed
:func:`price_indexed_amount` / :func:`progressive_price_indexed_amount`
run on this frame's full-415(b) AIME unchanged.

=====================================================================
THE AGGREGATE COST DELTA (frozen spec)
=====================================================================
For each committed provision encoding, reused verbatim from its committed
replication script::

    aggregate cost delta = (sum w * reform_benefit
                            - sum w * scheduled_benefit)
                           / (sum w * scheduled_benefit)

at the common evaluation ages (retired workers 62-67 in 2050, Mermin's
headline cohort), with 5-seed person-disjoint half-split floors. The
scheduled benefit is the current-law scheduled benefit as each committed
encoding defines it:

    * PI / PPI / caregiver: the scheduled PIA (the claim-age factor
      cancels -- these provisions do not alter it), reform = the committed
      reform PIA (PI = W * PIA; PPI = the 30th-percentile-bend amount;
      caregiver = the credited-history PIA);
    * NRA->70 / COLA-0.4pp: the claim-adjusted scheduled benefit
      (PIA * expected claim-age factor -- the claim factor IS the reform's
      mechanism here), reform = the committed reform benefit (NRA: the
      FRA-70 factor; COLA: the reduced-COLA receipt stream over 62-67).

Both conventions measure "% change vs current-law scheduled benefit" -- the
transportable ordinal content. Absolute levels are not compared. The
committed encodings are imported, never re-derived.

=====================================================================
EARNINGS SHARING (R7): excluded from the common frame, scored on its own
=====================================================================
Earnings sharing is intrinsically a couples provision: its committed
encoding (:mod:`replication_r7_sharing`) needs marriage episodes, spouse
PIA records, the 402(b)/(c)/(e)/(f) auxiliary-benefit plumbing, each
person's ACTUAL eligibility year (not the 2050 transport), and a package
global-benefit-increase scalar. None of that lives on the single-person
2050-transport common frame, so per the frozen spec's "record it as
excluded-with-reason rather than adapting it" clause it is EXCLUDED from
the common-frame aggregate. Its committed :func:`aggregate_cost_change` is
run on its own committed couples frame and recorded DESCRIPTIVELY for T1
(the anchor's packages are cost-balanced by design, so the anchor gives no
strict aggregate sign -- the registration's "recorded descriptively if the
anchor gives none").

=====================================================================
TESTS + PRE-REGISTERED FORECASTS
=====================================================================
* T1 sign agreement (forecast: 100%): savings provisions negative,
  caregiver plans positive.
* T2 Kendall tau, Mermin quartet vs its 75-yr payroll ordering
  (forecast: tau = 1.0).
* T3 Kendall tau, four caregiver plans vs their A-table cost ordering
  (forecast: tau >= 0.8; at most one adjacent swap).

Reported, not gated: one run, publishes regardless of outcome. The
orchestrator grades the forecasts on #42; this module does not.

Run (from the repository root, PSID family + marriage + birth files
staged)::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv/bin/python scripts/replication_cost_ordering.py
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
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Weighted helpers + the seed-stable across-seed summary, imported
# byte-for-byte from the merged builders (single source of truth), exactly
# as every other replication does.
import replication_caregiver as _cg  # noqa: E402
import replication_mermin_rows as _mr  # noqa: E402
import replication_ppi_mermin as _ppi  # noqa: E402
import replication_r7_sharing as _r7  # noqa: E402
from build_downstream_relevance import _weighted_quantile  # noqa: E402
from reform_delta_diagnostic import _summary  # noqa: E402

from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.ss.params import load_ssa_parameters  # noqa: E402

# Every committed provision encoding + its anchor, reused VERBATIM (bound
# to a local name, never re-derived). The 2050 transport + PI/PPI reforms
# from the PPI/Mermin replication; the NRA/COLA reforms + Mermin Table 1
# anchor from the Mermin-rows replication; the four caregiver-credit
# encodings + the Table 3 A-table anchor from the caregiver replication;
# the earnings-sharing aggregate from the r7 replication (own couples
# frame).
build_transport = _ppi.build_transport
price_indexed_amount = _ppi.price_indexed_amount
progressive_price_indexed_amount = _ppi.progressive_price_indexed_amount

CaregiverStudy = _cg.CaregiverStudy
caregiver_score = _cg.score_population
PLANS = _cg.PLANS
CAREGIVER_ANCHOR_A_TABLE = _cg.ANCHOR_COST_PCT_PAYROLL

MerminStudy = _mr.MerminStudy
mermin_score = _mr.score_population
Survival = _mr.Survival
_coefficient_table = _mr._coefficient_table
cola_pct = _mr.cola_pct
MERMIN_ANCHOR_PAYROLL = _mr.ANCHOR_TABLE1_PAYROLL_PCT

R7StudyData = _r7.StudyData
r7_score = _r7.score_population
aggregate_cost_change = _r7.aggregate_cost_change
GLOBAL_INCREASE = _r7.GLOBAL_INCREASE
SEEDS = _r7.SEEDS

ARTIFACT_PATH = ROOT / "runs" / "replication_cost_ordering_v1.json"
ARTIFACT_SCHEMA_VERSION = "replication_cost_ordering.v1"
RUN_NAME = "replication_cost_ordering_v1"

SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4931034068"
)
SPEC_REGISTRATION_POINTER = "#42 comment 4931034068"
PROGRAM_DESIGN_ISSUE = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/74"
)

# =====================================================================
# Anchor cost columns (transcribed from the archived Urban PDFs, verified
# against pdftotext per #74 protocol note 3 on 2026-07-09; the Mermin and
# caregiver constants are imported from the committed replications and the
# transcription re-verified below).
# =====================================================================
#: Mermin (2005), Urban 411260, Table 1 ("Annual Mean Social Security
#: Benefits at Ages 62 to 67, by Policy Scenario"), "75-year deficit/surplus
#: (percent of taxable payroll)" row. Column headers verified: Price
#: indexing +0.68; Progressive price indexing -0.14; Reduced cost of living
#: adjustment -1.12; Normal retirement age raised to 70 -0.5. Scheduled
#: deficit -1.69; payable 0. PDF p.15; narrative printed p.5-6 (the 1.69pp
#: balance, the COLA "reduces the deficit by about a third", PI "a surplus
#: of 0.68 percentage points"). A positive entry improves solvency = a
#: bigger benefit cut, so the ordering by savings is PI > PPI > NRA > COLA.
MERMIN_PAYROLL_PCT = {
    "price_indexing": 0.68,
    "progressive_price_indexing": -0.14,
    "reduced_cola": MERMIN_ANCHOR_PAYROLL["reduced_cola"],  # -1.12
    "nra_raised_to_70": MERMIN_ANCHOR_PAYROLL["nra_raised_to_70"],  # -0.5
}
MERMIN_SCHEDULED_DEFICIT_PCT = MERMIN_ANCHOR_PAYROLL["scheduled_deficit"]
#: Same Mermin Table 1, "Percent of Scheduled Benefits" 2050 row -- a direct
#: percent-of-scheduled cross-check column (verified from the PDF): PI 69.3,
#: PPI 81.8, reduced-COLA 98.3, NRA 85.2 (scheduled 100.0, payable 78.4).
MERMIN_PCT_SCHEDULED_2050 = {
    "price_indexing": 69.3,
    "progressive_price_indexing": 81.8,
    "reduced_cola": 98.3,
    "nra_raised_to_70": 85.2,
}
MERMIN_ANCHOR_CITE = (
    "Mermin (2005), Urban Institute 411260, DYNASIM3 Runid 432, Table 1 "
    "('Annual Mean Social Security Benefits at Ages 62 to 67, by Policy "
    "Scenario'), '75-year deficit/surplus (percent of taxable payroll)' "
    "row; verified against 411260-benefit-reductions.{txt,pdf} PDF p.15, "
    "narrative printed p.5-6"
)
#: Smith/Johnson/Favreault (2020), Urban 103050, Table 3 (actuarial balance
#: as a percentage of taxable payroll, 2019-93), "Provide caregiver
#: credits" row: Biden -0.12, Buttigieg -0.51, Klobuchar -0.12, Warren
#: -0.30 (the Sanders column is blank -- no caregiver credit). Negative =
#: a cost / actuarial-balance reduction, so a bigger benefit expansion is a
#: more-negative anchor entry. Printed p.19 (PDF p.29).
CAREGIVER_ANCHOR_CITE = (
    "Smith/Johnson/Favreault (2020), Urban Institute 103050, DYNASIM3 "
    "ID980, Table 3 (actuarial balance as a percentage of taxable payroll, "
    "2019-93), 'Provide caregiver credits' row; verified against "
    "103050-five-dem.{txt,pdf} printed p.19 (PDF p.29)"
)
#: Favreault-Steuerle (2007), Urban 311436, Table 3; the earnings-sharing
#: packages are cost-balanced by design (each carries a DYNASIM-calibrated
#: global benefit increase for ~2049 cost neutrality), so the anchor states
#: no strict aggregate cost sign.
R7_ANCHOR_CITE = (
    "Favreault-Steuerle (2007), Urban Institute 311436, DYNASIM3 runid "
    "440v2; earnings-sharing packages cost-balanced by design (package 1b "
    "carries a +4.5% global benefit increase calibrated for 2049 cost "
    "neutrality) -- no strict aggregate sign stated"
)

#: The common evaluation ages (Mermin's headline retired-worker cohort).
COMMON_EVALUATION_AGES = "retired workers 62-67 in 2050"

#: Provision -> the units-differ note carried in every per-provision row.
_SCHED_PIA_NOTE = (
    "delta vs the scheduled PIA (claim-age factor cancels -- this provision "
    "does not alter it); % change, not a level"
)
_CLAIM_ADJ_NOTE = (
    "delta vs the claim-adjusted scheduled benefit (PIA * expected claim-age "
    "factor -- the claim-age factor is this provision's mechanism); % change "
    "at ages 62-67, not a level"
)
_CAREGIVER_NOTE = (
    "delta vs the scheduled PIA (benefit GAIN from the credited-history "
    "PIA); cross-sectional PIA-gain proxy, no projection/DI interaction"
)


# =====================================================================
# The common frame
# =====================================================================
def build_common_frame(
    params: Any, transport: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The shared Phase-A frame: one row per sex-resolvable career.

    Builds the caregiver and Mermin studies (identical career frames), runs
    each committed :func:`score_population` verbatim, and inner-joins on the
    shared PSID person id. Columns: ``person_id``, ``weight``, ``base_aime``
    (full-415(b)), ``base_pia`` (scheduled PIA), ``sex``, ``elig_year``,
    ``nra_baseline_factor`` / ``nra_reform_factor`` (the NRA claim-age
    factors), and ``gain_<plan>`` for the four caregiver plans. The inner
    join to the sex-resolvable set is the NAMED common-support restriction.
    """
    cg_study = CaregiverStudy(params, transport)
    cg_df = caregiver_score(cg_study, params, transport)
    mr_study = MerminStudy(params, transport)
    mr_df = mermin_score(mr_study, params, transport)

    common = cg_df.merge(
        mr_df[
            [
                "person_id",
                "sex",
                "elig_year",
                "nra_baseline_factor",
                "nra_reform_factor",
            ]
        ],
        on="person_id",
        how="inner",
    ).reset_index(drop=True)

    meta = {
        "n_careers_caregiver": int(len(cg_df)),
        "n_careers_mermin_sex_resolvable": int(len(mr_df)),
        "n_common_frame": int(len(common)),
        "weight_sum": float(common["weight"].to_numpy(float).sum()),
    }
    return common, meta


# =====================================================================
# Per-provision (weighted reform, weighted scheduled) on any subframe
# =====================================================================
def _mermin_pairs(
    sub: pd.DataFrame,
    transport: dict[str, Any],
    survival: Survival,
) -> dict[str, tuple[float, float]]:
    """(weighted reform benefit, weighted scheduled benefit) per Mermin
    provision on ``sub``, using each committed reform encoding verbatim."""
    w = sub["weight"].to_numpy(np.float64)
    aime = sub["base_aime"].to_numpy(np.float64)
    base_pia = sub["base_pia"].to_numpy(np.float64)
    f67 = sub["nra_baseline_factor"].to_numpy(np.float64)
    f70 = sub["nra_reform_factor"].to_numpy(np.float64)

    sched_pia_sum = float(np.sum(w * base_pia))

    # PI: all factors scaled by the wedge W -> reform = W * scheduled PIA.
    pi_reform = float(np.sum(w * price_indexed_amount(aime, transport)))

    # PPI: 30th-percentile bend = the subframe's own weighted 30th pctile.
    bend30 = float(_weighted_quantile(aime, w, np.array([0.30]))[0])
    ppi_reform = float(
        np.sum(w * progressive_price_indexed_amount(aime, bend30, transport))
    )

    # NRA->70: scheduled = PIA * f67 (claim-adjusted current law), reform =
    # PIA * f70. The claim factor is the mechanism, so it does not cancel.
    nra_sched = float(np.sum(w * base_pia * f67))
    nra_reform = float(np.sum(w * base_pia * f70))

    # COLA -0.4pp: the committed coefficient encoding, dollar-weighted by
    # the scheduled PIA over the common evaluation ages 62-67. A/B are the
    # reduced-COLA receipt ratio mass and the scheduled mass per person.
    coeffs = _coefficient_table(sub, survival)
    a_coef = np.array(
        [
            coeffs[(r.sex, int(r.elig_year))]["62_67"][0]
            for r in sub.itertuples(index=False)
        ],
        dtype=np.float64,
    )
    b_coef = np.array(
        [
            coeffs[(r.sex, int(r.elig_year))]["62_67"][1]
            for r in sub.itertuples(index=False)
        ],
        dtype=np.float64,
    )
    cola_reform = float(np.sum(w * base_pia * a_coef))
    cola_sched = float(np.sum(w * base_pia * b_coef))

    return {
        "price_indexing": (pi_reform, sched_pia_sum),
        "progressive_price_indexing": (ppi_reform, sched_pia_sum),
        "nra_raised_to_70": (nra_reform, nra_sched),
        "reduced_cola": (cola_reform, cola_sched),
    }


def _caregiver_pairs(sub: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """(weighted reform benefit, weighted scheduled PIA) per caregiver plan
    on ``sub``: reform = scheduled PIA + committed benefit gain."""
    w = sub["weight"].to_numpy(np.float64)
    base_pia = sub["base_pia"].to_numpy(np.float64)
    sched = float(np.sum(w * base_pia))
    out: dict[str, tuple[float, float]] = {}
    for plan in PLANS:
        gain = sub[f"gain_{plan.name}"].to_numpy(np.float64)
        out[f"caregiver_{plan.name}"] = (
            sched + float(np.sum(w * gain)),
            sched,
        )
    return out


def _all_pairs(
    sub: pd.DataFrame,
    transport: dict[str, Any],
    survival: Survival,
) -> dict[str, tuple[float, float]]:
    pairs = _mermin_pairs(sub, transport, survival)
    pairs.update(_caregiver_pairs(sub))
    return pairs


def _deltas_from_pairs(
    pairs: dict[str, tuple[float, float]],
) -> dict[str, float]:
    return {
        name: (reform - sched) / sched if sched > 0 else float("nan")
        for name, (reform, sched) in pairs.items()
    }


# =====================================================================
# 5-seed person-disjoint half-split floors
# =====================================================================
def build_floors(
    common: pd.DataFrame,
    transport: dict[str, Any],
    survival: Survival,
) -> dict[str, dict[str, Any]]:
    """Per provision, the person-disjoint half-split A-vs-B delta gap over
    the five locked seeds (the committed half-split floor convention:
    ``split_panel_by_person`` fraction=0.5, seed=s). The abs-gap mean is the
    sampling-noise scale of the aggregate cost delta at half sample."""
    names = list(_all_pairs(common, transport, survival).keys())
    per_seed_gap: dict[str, list[float]] = {n: [] for n in names}
    for seed in SEEDS:
        side_a, side_b = hpanel.split_panel_by_person(
            common, "person_id", fraction=0.5, seed=seed
        )
        da = _deltas_from_pairs(_all_pairs(side_a, transport, survival))
        db = _deltas_from_pairs(_all_pairs(side_b, transport, survival))
        for n in names:
            per_seed_gap[n].append(da[n] - db[n])
    return {
        n: {
            "per_seed_signed_gap": [float(v) for v in per_seed_gap[n]],
            "abs": _summary([abs(v) for v in per_seed_gap[n]]),
            "signed": _summary(per_seed_gap[n]),
        }
        for n in names
    }


# =====================================================================
# The three ordinal tests
# =====================================================================
MERMIN_QUARTET = (
    "price_indexing",
    "progressive_price_indexing",
    "nra_raised_to_70",
    "reduced_cola",
)
CAREGIVER_QUARTET = tuple(f"caregiver_{p.name}" for p in PLANS)


def t1_sign_agreement(deltas: dict[str, float]) -> dict[str, Any]:
    """Sign agreement: savings provisions negative, caregiver plans
    positive. R7 is descriptive (handled by the caller)."""
    checks = []
    for name in MERMIN_QUARTET:
        checks.append(
            {
                "provision": name,
                "delta": deltas[name],
                "expected_sign": "negative",
                "ok": bool(deltas[name] < 0.0),
            }
        )
    for name in CAREGIVER_QUARTET:
        checks.append(
            {
                "provision": name,
                "delta": deltas[name],
                "expected_sign": "positive",
                "ok": bool(deltas[name] > 0.0),
            }
        )
    n_ok = sum(1 for c in checks if c["ok"])
    return {
        "checks": checks,
        "n_ok": n_ok,
        "n_total": len(checks),
        "pct_agreement": round(100.0 * n_ok / len(checks), 1),
        "forecast_pct": 100.0,
        "forecast_met": bool(n_ok == len(checks)),
    }


def t2_mermin_tau(deltas: dict[str, float]) -> dict[str, Any]:
    """Kendall tau, Mermin quartet: our benefit-reduction magnitude vs the
    anchor 75-yr payroll deficit-reduction. Both aligned so that a bigger
    reduction is a bigger number, so perfect agreement is tau = +1."""
    provs = list(MERMIN_QUARTET)
    our_reduction = [-deltas[p] for p in provs]
    anchor_reduction = [
        MERMIN_PAYROLL_PCT[p] - MERMIN_SCHEDULED_DEFICIT_PCT for p in provs
    ]
    tau, _ = kendalltau(our_reduction, anchor_reduction)
    # Cross-check against the anchor's direct "percent of scheduled" column.
    our_pct_sched = [100.0 * (1.0 + deltas[p]) for p in provs]
    anchor_pct_sched = [MERMIN_PCT_SCHEDULED_2050[p] for p in provs]
    tau_xcheck, _ = kendalltau(
        [-v for v in our_pct_sched], [-v for v in anchor_pct_sched]
    )
    return {
        "provisions": provs,
        "our_delta": {p: deltas[p] for p in provs},
        "our_benefit_reduction": {
            p: r for p, r in zip(provs, our_reduction, strict=True)
        },
        "anchor_payroll_pct": {p: MERMIN_PAYROLL_PCT[p] for p in provs},
        "anchor_deficit_reduction_pct": {
            p: r for p, r in zip(provs, anchor_reduction, strict=True)
        },
        "our_order_by_reduction": sorted(provs, key=lambda p: deltas[p]),
        "anchor_order_by_reduction": sorted(
            provs, key=lambda p: -MERMIN_PAYROLL_PCT[p]
        ),
        "kendall_tau": float(tau),
        "kendall_tau_pct_scheduled_xcheck": float(tau_xcheck),
        "alignment": (
            "both series are benefit-reduction magnitudes (bigger = bigger "
            "cut), so perfect agreement is tau = +1"
        ),
        "forecast_tau": 1.0,
        "forecast_met": bool(tau >= 1.0 - 1e-9),
    }


def t3_caregiver_tau(deltas: dict[str, float]) -> dict[str, Any]:
    """Kendall tau, four caregiver plans: our benefit-gain vs the anchor
    A-table cost magnitude. Aligned so a bigger gain / bigger cost is a
    bigger number, so perfect agreement is tau = +1 (tau-b handles the
    Biden/Klobuchar anchor tie at -0.12)."""
    plans = [p.name for p in PLANS]
    our_gain = [deltas[f"caregiver_{p}"] for p in plans]
    anchor_cost_mag = [-CAREGIVER_ANCHOR_A_TABLE[p] for p in plans]
    tau, _ = kendalltau(our_gain, anchor_cost_mag)
    return {
        "plans": plans,
        "our_gain_delta": {p: deltas[f"caregiver_{p}"] for p in plans},
        "anchor_cost_pct_payroll": {
            p: CAREGIVER_ANCHOR_A_TABLE[p] for p in plans
        },
        "our_order_by_gain": sorted(
            plans, key=lambda p: -deltas[f"caregiver_{p}"]
        ),
        "anchor_order_by_cost": sorted(
            plans, key=lambda p: CAREGIVER_ANCHOR_A_TABLE[p]
        ),
        "kendall_tau": float(tau),
        "alignment": (
            "our benefit-gain vs anchor cost magnitude (bigger = more "
            "costly); perfect agreement is tau = +1; tau-b handles the "
            "Biden/Klobuchar anchor tie at -0.12"
        ),
        "forecast_tau_min": 0.8,
        "forecast_met": bool(tau >= 0.8),
    }


# =====================================================================
# Earnings sharing (R7): committed encoding on its own couples frame
# =====================================================================
def earnings_sharing_descriptive(params: Any) -> dict[str, Any]:
    """R7 aggregate cost change on its committed couples frame (excluded
    from the single-person common frame; recorded descriptively for T1)."""
    study = R7StudyData(params)
    df = r7_score(study)
    agg = aggregate_cost_change(df)
    delta_1b = agg["weighted_aggregate_pct_change_1b"] / 100.0
    delta_1b_ns = agg["weighted_aggregate_pct_change_1b_noscalar"] / 100.0
    return {
        "provision": "earnings_sharing_1b",
        "family": "favreault_steuerle_2007",
        "common_frame": False,
        "excluded_from_common_frame": True,
        "exclusion_reason": (
            "intrinsically a couples provision: the committed encoding "
            "needs marriage episodes, spouse PIA records, the 402(b)/(c)/"
            "(e)/(f) auxiliary-benefit plumbing, each person's ACTUAL "
            "eligibility year (not the 2050 transport), and a package "
            "global-benefit-increase scalar -- none on the single-person "
            "2050-transport common frame. Per the frozen spec, recorded as "
            "excluded-with-reason and scored on its own committed couples "
            "frame instead of adapting it."
        ),
        "n_scored_couples_frame": int(len(df)),
        "our_delta_1b": delta_1b,
        "our_delta_1b_pct": agg["weighted_aggregate_pct_change_1b"],
        "our_delta_1b_noscalar": delta_1b_ns,
        "our_delta_1b_noscalar_pct": agg[
            "weighted_aggregate_pct_change_1b_noscalar"
        ],
        "package_global_increase": GLOBAL_INCREASE["1b"],
        "anchor_cite": R7_ANCHOR_CITE,
        "t1_role": "descriptive",
        "t1_note": (
            "the anchor's earnings-sharing packages are cost-balanced by "
            "design (a DYNASIM-calibrated global increase for 2049 cost "
            "neutrality), so no strict aggregate sign is stated; on our "
            "observed 1943-57 sample package 1b is a NET REDUCTION "
            f"({agg['weighted_aggregate_pct_change_1b']}% with the "
            "calibrated scalar, "
            f"{agg['weighted_aggregate_pct_change_1b_noscalar']}% without) "
            "-- a named population delta, recorded not chased"
        ),
    }


# =====================================================================
# Per-provision rows for the artifact
# =====================================================================
def _provision_rows(
    deltas: dict[str, float],
    floors: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in MERMIN_QUARTET:
        is_ppi_pi = name in (
            "price_indexing",
            "progressive_price_indexing",
        )
        rows.append(
            {
                "provision": name,
                "family": "mermin_savings",
                "in_common_frame": True,
                "our_delta": deltas[name],
                "our_delta_pct": round(100.0 * deltas[name], 3),
                "our_pct_of_scheduled": round(100.0 * (1.0 + deltas[name]), 2),
                "floor_abs_mean": floors[name]["abs"]["mean"],
                "floor_abs_max": floors[name]["abs"]["max"],
                "anchor_value_payroll_pct": MERMIN_PAYROLL_PCT[name],
                "anchor_pct_of_scheduled_2050": MERMIN_PCT_SCHEDULED_2050[
                    name
                ],
                "anchor_units": (
                    "75-yr OASDI effect, % of taxable payroll (positive = "
                    "improves solvency = bigger benefit cut)"
                ),
                "anchor_cite": MERMIN_ANCHOR_CITE,
                "units_differ_note": (
                    _SCHED_PIA_NOTE if is_ppi_pi else _CLAIM_ADJ_NOTE
                ),
                "expected_sign": "negative",
                "sign_ok": bool(deltas[name] < 0.0),
                "in_T2": True,
            }
        )
    for plan in PLANS:
        name = f"caregiver_{plan.name}"
        rows.append(
            {
                "provision": name,
                "family": "caregiver_credit",
                "in_common_frame": True,
                "plan_design": {
                    "credit_fraction_of_avg_wage": plan.credit_fraction,
                    "child_age_limit": plan.child_age_limit,
                    "year_cap": plan.year_cap,
                },
                "our_delta": deltas[name],
                "our_delta_pct": round(100.0 * deltas[name], 3),
                "floor_abs_mean": floors[name]["abs"]["mean"],
                "floor_abs_max": floors[name]["abs"]["max"],
                "anchor_value_payroll_pct": CAREGIVER_ANCHOR_A_TABLE[
                    plan.name
                ],
                "anchor_units": (
                    "75-yr actuarial-balance cost, % of taxable payroll "
                    "(negative = a cost = a bigger benefit expansion)"
                ),
                "anchor_cite": CAREGIVER_ANCHOR_CITE,
                "units_differ_note": _CAREGIVER_NOTE,
                "expected_sign": "positive",
                "sign_ok": bool(deltas[name] > 0.0),
                "in_T3": True,
            }
        )
    return rows


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


def _named_deltas() -> list[str]:
    return [
        "no long-run projection: observed COMPLETED PSID careers under the "
        "Phase-A 2050 transport, not a Trustees-assumption Phase-E "
        "projection -- so units differ (our benefit-side % change vs the "
        "anchor's 75-yr % of taxable payroll); signs and orderings are the "
        "transportable content, absolute levels are not compared",
        "no revenue side: benefit-side provisions only (the donut cap and "
        "payroll-rate rows are out of scope until Phase E)",
        "observed cohorts: PSID retirees born 1943-1957 (eligible "
        "2005-2019) transported to a common 2050 bracket geometry vs "
        "DYNASIM's projected 2050/2065 beneficiaries",
        "common-support restriction: the aggregate runs on the "
        "sex-resolvable Phase-A careers (the Marriage History join the "
        "NRA/COLA claim distribution needs)",
        "PPI/NRA ordering driver (the T2 swap): our observed careers are "
        "compressed relative to the bends (most AIME below the second "
        "bend), so progressive price indexing -- which bites only ABOVE the "
        "30th-percentile bend -- cuts LESS than the uniform NRA actuarial "
        "reduction on our frame, flipping PPI and NRA versus DYNASIM's "
        "fuller projected careers (documented in the ppi_mermin artifact)",
        "caregiver frame omits DI interactions (the registration's named "
        "T3 delta) and models Biden's phase-out as the uniform "
        "top-up-to-credit mechanic",
        "NRA/COLA baseline = the claim-adjusted scheduled benefit (the "
        "claim factor is the reform mechanism); PI/PPI/caregiver baseline = "
        "the scheduled PIA (the claim factor cancels) -- both are '% change "
        "vs current-law scheduled', per-provision units-differ note carried",
        "COLA aggregate is dollar-weighted (scheduled PIA) over the common "
        "evaluation ages 62-67; the committed per-capita cola_pct barely "
        "differs because the COLA ratio is PIA-independent",
        "earnings sharing excluded from the single-person common frame "
        "(intrinsically a couples provision) and scored on its own "
        "committed couples frame; recorded descriptively for T1",
    ]


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the cost-ordering synthesis (reported, not gated)."""
    started = time.time()

    params = load_ssa_parameters()
    transport = build_transport(params)
    survival = Survival()

    common, meta = build_common_frame(params, transport)
    if verbose:
        print(
            f"common frame: {meta['n_common_frame']} sex-resolvable "
            f"careers (caregiver {meta['n_careers_caregiver']}, mermin "
            f"{meta['n_careers_mermin_sex_resolvable']}); "
            f"wedge W={transport['wedge']:.5f}"
        )

    pairs = _all_pairs(common, transport, survival)
    deltas = _deltas_from_pairs(pairs)
    floors = build_floors(common, transport, survival)

    # The committed per-capita COLA percent-of-scheduled, for cross-check.
    coeffs = _coefficient_table(common, survival)
    cola_pct_committed_62_67 = cola_pct(common, coeffs, "62_67")
    cola_pct_committed_80_85 = cola_pct(common, coeffs, "80_85")

    t1 = t1_sign_agreement(deltas)
    t2 = t2_mermin_tau(deltas)
    t3 = t3_caregiver_tau(deltas)

    r7 = earnings_sharing_descriptive(params)

    rows = _provision_rows(deltas, floors)

    if verbose:
        for r in rows:
            print(
                f"  {r['provision']:26s} delta {r['our_delta']:+.4f} "
                f"floor {r['floor_abs_mean']:.4f} "
                f"anchor {r['anchor_value_payroll_pct']:+.2f}"
            )
        print(
            f"  earnings_sharing_1b        delta "
            f"{r7['our_delta_1b']:+.4f} (couples frame, descriptive)"
        )
        print(
            f"T1 sign agreement {t1['pct_agreement']}% "
            f"(forecast {t1['forecast_pct']}%) | "
            f"T2 tau {t2['kendall_tau']:.4f} (forecast "
            f"{t2['forecast_tau']}) | "
            f"T3 tau {t3['kendall_tau']:.4f} (forecast >= "
            f"{t3['forecast_tau_min']})"
        )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "reported_not_gated": True,
        "registration": SPEC_REGISTRATION,
        "registration_pointer": SPEC_REGISTRATION_POINTER,
        "program_design_issue": PROGRAM_DESIGN_ISSUE,
        "purpose": (
            "Cost-ordering synthesis: score every committed provision "
            "encoding ONCE on a single common Phase-A frame and test the "
            "aggregate cost deltas ORDINALLY against the anchors' published "
            "cost columns (Mermin 75-yr payroll effects; Five-Approaches "
            "A-table). Signs and orderings are the transportable content; "
            "no level matching is claimed. Reads no gate, changes no gate; "
            "publishes regardless of outcome."
        ),
        "grading_note": (
            "the orchestrator grades the pre-registered forecasts on #42; "
            "this artifact reports outcomes vs forecasts and does not grade"
        ),
        "common_frame": {
            "description": (
                "the canonical Phase-A career frame (family_earnings_panel "
                "+ r7 _person_history + full-415(b) transported_aime + "
                "ppi_mermin build_transport/scheduled_amount), intersected "
                "with the sex-resolvable set -- the frame the NRA/COLA and "
                "caregiver replications both build identically"
            ),
            "population": (
                "Phase-A pia_observed rule: coverage >= 0.8 of ages 22-61, "
                "age-62 eligibility year 2005-2019 (born 1943-1957)"
            ),
            "baseline": (
                "scheduled PIA (pre-415(g) 90/32/15 on the 2050 bends), "
                "replication_ppi_mermin.scheduled_amount -- SCHEDULED (not "
                "payable) denominator per #74 protocol note 1"
            ),
            "aime_convention": (
                "full statutory 42 USC 415(b) top-35 transported AIME "
                "(replication_caregiver.transported_aime)"
            ),
            "common_evaluation_ages": COMMON_EVALUATION_AGES,
            "wedge": transport["wedge"],
            "bend_points_2050": list(transport["bend_points"]),
            "ppi_deviation_note": (
                "ppi_mermin's biennial gate panel + biennial-proxy AIME is "
                "its documented generator-forced deviation; this synthesis "
                "uses no generator, so PI/PPI (pure AIME functions) run on "
                "the canonical full-415(b) frame the anchor tables target"
            ),
            **meta,
        },
        "encoding_reuse": {
            "price_indexing": (
                "replication_ppi_mermin.price_indexed_amount (verbatim)"
            ),
            "progressive_price_indexing": (
                "replication_ppi_mermin.progressive_price_indexed_amount "
                "(verbatim; 30th-pctile bend = the frame's own weighted "
                "30th percentile)"
            ),
            "nra_raised_to_70": (
                "replication_mermin_rows.score_population NRA factors "
                "(expected_nra_factors / benefit_factor_at_fra, verbatim)"
            ),
            "reduced_cola": (
                "replication_mermin_rows._coefficient_table / "
                "cola_group_coefficients (verbatim), dollar-weighted over "
                "62-67"
            ),
            "caregiver_plans": (
                "replication_caregiver.score_population gains "
                "(qualifying_years / reformed_history / transported_aime, "
                "verbatim)"
            ),
            "earnings_sharing": (
                "replication_r7_sharing.aggregate_cost_change on "
                "score_population (verbatim, own couples frame -- excluded "
                "from the common frame)"
            ),
        },
        "anchor_provenance": {
            "mermin_payroll_pct": MERMIN_PAYROLL_PCT,
            "mermin_scheduled_deficit_pct": MERMIN_SCHEDULED_DEFICIT_PCT,
            "mermin_pct_of_scheduled_2050": MERMIN_PCT_SCHEDULED_2050,
            "mermin_cite": MERMIN_ANCHOR_CITE,
            "caregiver_a_table_cost_pct_payroll": dict(
                CAREGIVER_ANCHOR_A_TABLE
            ),
            "caregiver_cite": CAREGIVER_ANCHOR_CITE,
            "reverified_pdftotext": (
                "each anchor number re-verified against pdftotext output on "
                "2026-07-09 per #74 protocol note 3"
            ),
        },
        "provisions": rows,
        "floors": {
            "convention": (
                "per provision, the person-disjoint half-split A-vs-B "
                "aggregate-cost-delta gap over the 5 locked seeds "
                "(split_panel_by_person fraction=0.5, seed=s); the abs-gap "
                "mean is the sampling-noise scale at half sample. PI's floor "
                "is exactly 0 (a uniform scalar W-1); NRA/COLA floors are "
                "~1e-4 (the actuarial factors are near population-invariant)"
            ),
            "per_provision": floors,
        },
        "earnings_sharing": r7,
        "cola_cross_checks": {
            "committed_per_capita_cola_pct_62_67": (cola_pct_committed_62_67),
            "committed_per_capita_cola_pct_80_85": (cola_pct_committed_80_85),
            "dollar_weighted_delta_62_67": deltas["reduced_cola"],
            "note": (
                "the T2 COLA delta is dollar-weighted over 62-67; the "
                "committed per-capita cola_pct is reported for cross-check "
                "(they agree because the COLA ratio is PIA-independent). "
                "80-85 is a secondary, later-life COLA compounding row"
            ),
        },
        "tests": {
            "T1_sign_agreement": t1,
            "T2_mermin_kendall_tau": t2,
            "T3_caregiver_kendall_tau": t3,
        },
        "results_vs_forecasts": {
            "T1": {
                "forecast": "100% sign agreement",
                "result_pct": t1["pct_agreement"],
                "met": t1["forecast_met"],
            },
            "T2": {
                "forecast": "kendall tau = 1.0",
                "result_tau": t2["kendall_tau"],
                "met": t2["forecast_met"],
            },
            "T3": {
                "forecast": "kendall tau >= 0.8 (<= one adjacent swap)",
                "result_tau": t3["kendall_tau"],
                "met": t3["forecast_met"],
            },
        },
        "named_deltas": _named_deltas(),
        "commit": _sha(ROOT),
        "pe_us_revision": getattr(params, "pe_us_revision", None),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return artifact


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serialisable: {type(obj)!r}")


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, indent=1, default=_json_default) + "\n"
    )
    print(f"\nwrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
