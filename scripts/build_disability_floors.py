"""Build the M4 disability floors: PSID work-limitation hazards, the
DI->retirement conversion validation vs SSA 6.B5.1, and the noise floor
(roadmap #113, M4).

REPORTED ANCHOR, NOT A GATE RUN. Like the mortality and claiming floors,
this reads no gate and decides nothing on its own; it is committed
evidence pinned by a reproduction test. It is the data + floors basis a
future DI gate (roadmap #113 M4) would derive pre-registered thresholds
from -- the ceremony (floors -> thresholds -> adversarial referee round
-> verification -> ratification) comes AFTER, exactly as the mortality
foundation records. The proposed thresholds are recorded here
(``proposed_thresholds_note``) explicitly marked NOT RATIFIED.

Three products, all on the PSID self-reported-disability panel
(:mod:`populace_dynamics.data.disability`):

(a) **DI-status hazards.** Weighted per-interval incidence (not-disabled
    -> disabled), recovery (disabled -> not-disabled) and disabled
    prevalence by age band x sex, from grid-adjacent (<=2 yr) wave pairs.

(b) **DI->retirement conversion validation.** The PSID conversion analog
    (disabled -> retired transitions among retired-entries at the
    FRA-crossing ages) against the SSA *Annual Statistical Supplement*
    Table 6.B5.1 disability-conversion column (the archived reference the
    claiming module already committed), by sex, with every concept delta
    NAMED and the ratio REPORTED, never calibrated.

(c) **Internal noise floor.** A person-disjoint 50/50 half-split (seeds
    0-4) of the same band x sex hazards, giving the across-seed mean/sd
    of the ``|log rate ratio|`` between two independent real halves -- the
    sd basis a future gate would turn into ``round(mean + k*sd)``
    thresholds, with a min-event reliability partition (gate-eligible vs
    report-only).

The named concept delta between PSID work-limitation and SSA DI is
carried in full (``concept_deltas``) and the exact SSA administrative
series a future gate needs are named (``wanted_ssa_tables``) -- the
orchestrator browser-fetches those separately (ssa.gov 403s programmatic
fetches).

Run from the repository root with the PSID individual file staged::

    .venv/bin/python scripts/build_disability_floors.py

It needs no populace-fit and no policyengine-us checkout (real-vs-real
and real-vs-archived-SSA only).
"""

from __future__ import annotations

import datetime as dt
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from populace_dynamics import artifacts, claiming
from populace_dynamics.data import deaths, disability
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "m4_disability_v1.json"
REFERENCE_REL = "data/external/ssa_claim_ages_2023supplement.json"
SCHEMA_VERSION = "m4_disability.v1"
RUN = "m4_disability_v1"

SEEDS = (0, 1, 2, 3, 4)

#: Minimum transition events (weaker half, worst seed) for a cell to be
#: proposed gate-eligible -- the transition analog of the mortality
#: floor's 20-death NCHS reliability floor. Cells below it are
#: report-only (denominator/numerator fragile).
MIN_EVENTS_FOR_GATE = 20

#: Reference entitlement years at which the 6.B5.1 conversion share is
#: pulled out for the headline comparison (earliest, mid, recent, last).
CONVERSION_REF_YEARS = (1998, 2010, 2019, 2022)


# --------------------------------------------------------------------------
# Noise floor over the reference moments
# --------------------------------------------------------------------------
def measure_seed_halfsplit(
    seed: int, panel: disability.DisabilityPanel
) -> dict[str, Any]:
    """One seed: split persons 50/50, reference moments on each half.

    Returns, per moment cell, the two halves' rates and event counts and
    the ``|ln(rate_A / rate_B)|`` floor statistic (``None`` when either
    half has a zero rate -- numerator-fragile, undefined ratio).
    """
    ids_frame = panel.person_years[["person_id"]].drop_duplicates()
    left, right = hpanel.split_panel_by_person(
        ids_frame, "person_id", fraction=0.5, seed=seed
    )
    ids_a = set(left["person_id"])
    ids_b = set(right["person_id"])
    mom_a = disability.reference_moments(panel, ids_a)
    mom_b = disability.reference_moments(panel, ids_b)

    cells: dict[str, Any] = {}
    for key in mom_a:
        a = mom_a[key]
        b = mom_b[key]
        r_a, r_b = a["rate"], b["rate"]
        defined = r_a > 0 and r_b > 0
        cells[key] = {
            "rate_a": float(r_a),
            "rate_b": float(r_b),
            "n_events_a": int(a["n_events"]),
            "n_events_b": int(b["n_events"]),
            "log_ratio_abs": (
                float(abs(np.log(r_a / r_b))) if defined else None
            ),
            "pct_diff_abs": (
                float(abs(r_a - r_b) / r_b * 100.0) if defined else None
            ),
        }
    return {
        "seed": seed,
        "n_persons_side_a": len(ids_a),
        "n_persons_side_b": len(ids_b),
        "cells": cells,
    }


def _summary(values: list[float]) -> dict[str, Any]:
    """Mean/sd/min/max (ddof=1) and raw values -- the floor convention
    shared with the other ``runs/`` artifacts."""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seeds": int(arr.size),
        "values": [float(v) for v in arr],
    }


def pool_internal_floor(
    per_seed: list[dict[str, Any]], keys: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Across-seed floor per cell, plus the reliability partition."""
    noise_floor: dict[str, Any] = {}
    stability: dict[str, Any] = {}
    for key in keys:
        log_ratios = [s["cells"][key]["log_ratio_abs"] for s in per_seed]
        pct_diffs = [s["cells"][key]["pct_diff_abs"] for s in per_seed]
        min_events = min(
            min(s["cells"][key]["n_events_a"], s["cells"][key]["n_events_b"])
            for s in per_seed
        )
        defined_seeds = sum(v is not None for v in log_ratios)
        stability[key] = {
            "defined_seeds": defined_seeds,
            "n_seeds": len(per_seed),
            "min_events_either_half": int(min_events),
            "gate_eligible": (
                defined_seeds == len(per_seed)
                and min_events >= MIN_EVENTS_FOR_GATE
            ),
        }
        if defined_seeds == len(per_seed):
            block = _summary([float(v) for v in log_ratios])
            block["pct_diff_abs"] = _summary([float(v) for v in pct_diffs])
            noise_floor[key] = block
    return noise_floor, stability


# --------------------------------------------------------------------------
# DI -> retirement conversion validation vs 6.B5.1
# --------------------------------------------------------------------------
def conversion_validation(
    moments: dict[str, dict[str, float]],
    ref: claiming.ClaimAgeReference,
) -> dict[str, Any]:
    """PSID conversion analog vs the archived SSA 6.B5.1 column, by sex.

    Reports the PSID transition-based conversion analog and the disabled
    prevalence approaching FRA next to the administrative 6.B5.1
    disability-conversion share -- as a ratio, with the concept delta
    named. This is the external anchor of the M4 foundation and is a
    SHAPE/ratio report, exactly as the mortality anchor reports the PSID
    undercount; nothing here is calibrated toward a match.
    """
    admin_years = ref.years()
    admin_series = {
        sex: {
            y: ref.row(sex, y)["categories"]["disability_conversion"]
            for y in admin_years
        }
        for sex in disability.SEXES
    }

    by_sex: dict[str, Any] = {}
    for sex in disability.SEXES:
        psid_conv = (
            moments[f"conversion.retired_from_disabled|{sex}"]["rate"] * 100.0
        )
        # Disabled prevalence at the FRA-approaching band, for context.
        prev_60_66 = moments[f"prevalence.60-66|{sex}"]["rate"] * 100.0
        admin = admin_series[sex]
        admin_mean = float(np.mean(list(admin.values())))
        admin_ref = {str(y): admin[y] for y in CONVERSION_REF_YEARS}
        by_sex[sex] = {
            "psid_conversion_analog_pct": round(psid_conv, 3),
            "psid_conversion_n_events": moments[
                f"conversion.retired_from_disabled|{sex}"
            ]["n_events"],
            "psid_conversion_n_retired_entries": moments[
                f"conversion.retired_from_disabled|{sex}"
            ]["n_at_risk"],
            "psid_disabled_prevalence_60_66_pct": round(prev_60_66, 3),
            "admin_6b51_conversion_share_pct": {
                "mean_1998_2022": round(admin_mean, 3),
                "by_reference_year": admin_ref,
            },
            "ratio_psid_analog_to_admin_mean": (
                round(psid_conv / admin_mean, 4) if admin_mean else None
            ),
        }

    return {
        "target": {
            "source": "SSA Annual Statistical Supplement 2023, Table 6.B5.1",
            "column": "disability conversions (footnote b)",
            "definition": (
                "Disabled-worker beneficiaries who reach FRA and whose "
                "benefit automatically converts to a retired-worker "
                "benefit, expressed as a percent of that year's "
                "retired-worker awards. An ADMINISTRATIVE award flow "
                "among disability-insured workers -- NOT a self-report."
            ),
            "reference_artifact": REFERENCE_REL,
            "years_available": [admin_years[0], admin_years[-1]],
        },
        "psid_analog": {
            "definition": (
                "Among grid-adjacent PSID wave pairs transitioning INTO "
                "self-reported 'retired' employment status at ages "
                f"{disability.CONVERSION_WINDOW[0]}-"
                f"{disability.CONVERSION_WINDOW[1]}, the weighted share "
                "whose origin state was self-reported 'permanently "
                "disabled' -- a disabled -> retired transition read as a "
                "conversion. A SELF-REPORTED-LABOR-FORCE transition, "
                "pooled over covered waves 1982-2023."
            ),
            "window_ages": list(disability.CONVERSION_WINDOW),
        },
        "by_sex": by_sex,
        "interpretation": (
            "The PSID analog runs well below the 6.B5.1 administrative "
            "share (ratio far below 1). This is EXPECTED and REPORTED, "
            "not corrected: see concept_deltas. The two share a structure "
            "(conversions over retirement entries) but differ in "
            "denominator (all self-reported retirements vs new "
            "retired-worker awards), measurement (self-reported "
            "labor-force relabeling vs administrative auto-conversion), "
            "and censoring (a biennial gap misses a disabled->retired "
            "step that resolves within two years). The anchor certifies "
            "co-movement and ordering, never a level match."
        ),
    }


# --------------------------------------------------------------------------
# Named concept deltas and the wanted SSA tables
# --------------------------------------------------------------------------
def concept_deltas() -> list[dict[str, str]]:
    """The PSID-work-limitation vs SSA-DI deltas, named, never conflated."""
    return [
        {
            "name": "definition (self-report vs adjudication)",
            "delta": (
                "PSID EMPLOYMENT STATUS == 5 is the respondent's own "
                "'permanently disabled' labor-force status; an SSA DI "
                "award is a medical-vocational adjudication of inability "
                "to engage in substantial gainful activity. Different "
                "constructs, not two measures of one thing."
            ),
        },
        {
            "name": "population (all adults vs insured workers)",
            "delta": (
                "The PSID series covers all adults; DI awards go only to "
                "disability-insured workers below FRA (20/40 recent-work "
                "test). PSID includes never-insured and non-worker "
                "disability the DI program never sees."
            ),
        },
        {
            "name": "severity threshold",
            "delta": (
                "Self-reported 'permanently disabled' has no SGA/duration "
                "screen; DI requires a medically determinable impairment "
                "expected to last >=12 months or result in death. PSID "
                "captures milder and shorter limitation."
            ),
        },
        {
            "name": "transience (recovery churn)",
            "delta": (
                "Interval recovery from self-reported disability is "
                "25-50% in these data -- respondents cycle in and out and "
                "relabel toward 'retired' near FRA. SSA DI recovery is "
                "~1%/yr. The PSID recovery hazard is therefore NOT a DI "
                "termination rate and is never equated to one."
            ),
        },
        {
            "name": "conversion denominator",
            "delta": (
                "The 6.B5.1 conversion share is conversions over "
                "retired-worker AWARDS; the PSID analog is disabled-> "
                "retired transitions over all self-reported retirement "
                "entries. Different denominators, so the levels differ "
                "even where the concept lines up."
            ),
        },
        {
            "name": "timing / biennial censoring",
            "delta": (
                "A self-report can precede or follow an award, and the "
                "PSID grid is biennial: a disabled->retired step that "
                "resolves within a 2-year gap is observed only at its "
                "endpoints, so onset/conversion timing is bounded to the "
                "interval, not dated within it."
            ),
        },
        {
            "name": "period pooling",
            "delta": (
                "PSID hazards pool covered waves 1982-2023 against a "
                "single-era SSA column; DI incidence and prevalence have "
                "strong secular trends (the 1990s-2010s rise and later "
                "decline), a named period-concept delta a future gate "
                "must window."
            ),
        },
    ]


def wanted_ssa_tables() -> list[dict[str, Any]]:
    """Exact SSA administrative series a future M4 gate needs.

    The orchestrator browser-fetches these (ssa.gov 403s programmatic
    fetches, so they are NOT fetched here). Each names the series, the
    fields, and why the M4 hazards need it. 6.B5.1 is already in hand
    (the conversion column, validated above); the rest are WANTED.
    """
    return [
        {
            "series": "SSA Annual Statistical Report on the SSDI Program",
            "table": "Table 4 (disabled-worker awards) + the "
            "age-sex-adjusted incidence-rate series (awards per 1,000 "
            "disability-insured)",
            "fields": "disabled-worker awards and incidence rate by age "
            "band and sex, by year",
            "needed_for": "the INCIDENCE hazard external anchor "
            "(not-disabled -> DI onset), the administrative counterpart "
            "to incidence_cells",
            "status": "WANTED",
        },
        {
            "series": "SSA Annual Statistical Report on the SSDI Program",
            "table": "Table 50 (benefit terminations by reason)",
            "fields": "disabled-worker terminations split by recovery, "
            "death, and conversion-to-retirement, by age and sex",
            "needed_for": "the RECOVERY hazard anchor -- the true DI "
            "recovery rate that the self-reported PSID recovery churn "
            "must be separated from",
            "status": "WANTED",
        },
        {
            "series": "OASDI Trustees Report",
            "table": "Table V.C5 (disability incidence rates) and Table "
            "V.C6 (disability termination rates), by age and sex",
            "fields": "gross/net incidence and termination (death, "
            "recovery, conversion) rates by age and sex",
            "needed_for": "the assumption-layer targets a projection "
            "(M6) aligns the DI hazards to; the Trustees' own age-sex "
            "rate surfaces",
            "status": "WANTED",
        },
        {
            "series": "SSA Annual Statistical Supplement",
            "table": "Table 5.D (disabled workers in current-payment "
            "status by age and sex) / Table 6.C",
            "fields": "disabled-worker beneficiary counts by age and sex "
            "(the prevalence stock)",
            "needed_for": "the PREVALENCE anchor -- the administrative "
            "disabled-worker prevalence the PSID self-reported "
            "prevalence is compared against",
            "status": "WANTED",
        },
        {
            "series": "SSA Annual Statistical Supplement",
            "table": "Table 4.C2 (disability-insured workers)",
            "fields": "count of disability-insured workers by age and "
            "sex (the incidence-rate denominator)",
            "needed_for": "converting PSID onset counts to an "
            "insured-population incidence rate comparable to SSA's",
            "status": "WANTED",
        },
        {
            "series": "SSA Annual Statistical Supplement",
            "table": "Table 6.B5.1 (disability-conversion column)",
            "fields": "conversions as a percent of retired-worker awards "
            "by sex and year, 1998-2022",
            "needed_for": "the CONVERSION validation (done above)",
            "status": "IN HAND (archived; validated in "
            "conversion_validation)",
        },
    ]


# --------------------------------------------------------------------------
# Proposed thresholds note (NOT RATIFIED)
# --------------------------------------------------------------------------
def proposed_thresholds_note(stability: dict[str, Any]) -> str:
    gate_eligible = sorted(
        k for k, v in stability.items() if v["gate_eligible"]
    )
    report_only = sorted(
        k for k, v in stability.items() if not v["gate_eligible"]
    )
    return (
        "PROPOSED VALIDATION STANDARD FOR THE M4 DISABILITY COMPONENT -- "
        "NOT RATIFIED.\n\n"
        "This is a proposal for the future gate ceremony (roadmap #113 "
        "M4), recorded so the standard exists BEFORE any scored use. It "
        "changes no locked value and no model reads it. Ratification "
        "requires the full lock ceremony: floors -> thresholds with "
        "machine-bound derivations -> an adversarial referee round -> "
        "verification -> maintainer ratification by merge. The k values "
        "below are PLACEHOLDERS for that ceremony to set.\n\n"
        "STATISTIC. Per age band x sex, the weighted per-interval PSID "
        "transition rate (incidence not-disabled->disabled; recovery "
        "disabled->not-disabled) and disabled prevalence from this "
        "artifact, and a synthetic candidate's same-band rate built the "
        "same way. The candidate-vs-PSID discrepancy is scored as the "
        "absolute log rate ratio |ln(r_candidate / r_PSID)| per cell.\n\n"
        "INTERNAL FLOOR. The person-disjoint 50/50 half-split |log rate "
        "ratio| floor in internal_noise_floor.noise_floor_seeds_0_4 "
        "(mean/sd across seeds 0-4). A proposed per-cell threshold is "
        "round(mean + k*sd) at the shared derivation convention "
        "(tests/test_gates_derivations.py), with a PROPOSED k in [2, 3] "
        "for the ceremony to fix.\n\n"
        "GATE-ELIGIBLE VS REPORT-ONLY CELLS. Gate only cells whose "
        "real-vs-real floor is defined on every seed AND whose weaker "
        "half carries at least "
        f"{MIN_EVENTS_FOR_GATE} transition events on the worst seed: "
        f"{gate_eligible}. Report-only (numerator-fragile): "
        f"{report_only}.\n\n"
        "EXTERNAL ANCHOR. The PSID self-reported series is NOT the SSA "
        "DI program (see concept_deltas): the anchor must NOT gate a "
        "level match to any SSA DI rate -- that would reject reality. "
        "The proposed external standard is a SHAPE check: (i) incidence "
        "monotone non-decreasing in age up to the pre-FRA bands; (ii) "
        "prevalence rising with age to the 50-59 band; (iii) the "
        "conversion analog co-moving with the 6.B5.1 column in sign and "
        "sex-ordering, with the level ratio reported as a named concept "
        "delta, never calibrated. The exact SSA series a level-anchored "
        "version would need are named in wanted_ssa_tables (browser-"
        "fetched separately; ssa.gov 403s programmatic fetches).\n\n"
        "BASELINE CONVENTION. DI feeds the payable-baseline composition "
        "and the COLA/FRA interactions (roadmap #113 M4 unlocks); any "
        "scored reform must state scheduled vs payable. This note fixes "
        "none of that; it fixes the component's estimation/validation "
        "standard only."
    )


# --------------------------------------------------------------------------
# Provenance + driver
# --------------------------------------------------------------------------
def _git_sha(cwd: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode()
            .strip()
        )
    except Exception:
        return None


def run(verbose: bool = True) -> dict[str, Any]:
    started = time.time()

    status = disability.read_disability_status()
    death_records = deaths.read_death_records()
    panel = disability.build_disability_panel(status, death_records)
    coding = disability.verify_employment_status_codes()

    moments = disability.reference_moments(panel)
    keys = sorted(moments)

    per_seed = [measure_seed_halfsplit(s, panel) for s in SEEDS]
    noise_floor, stability = pool_internal_floor(per_seed, keys)

    ref = claiming.load_claim_age_reference()
    conversion = conversion_validation(moments, ref)

    n_persons = int(panel.person_years["person_id"].nunique())
    waves = sorted(int(w) for w in status["period"].unique())
    if verbose:
        print(
            f"panel: {len(panel.person_years)} person-years, "
            f"{n_persons} persons, {len(panel.pairs)} pairs"
        )
        for key in keys:
            fl = noise_floor.get(key)
            sd = "NA" if fl is None else f"{fl['sd']:.3f}"
            print(
                f"  {key:>42}: rate={moments[key]['rate'] * 100:6.2f}% "
                f"n_ev={moments[key]['n_events']:5d} floor_sd={sd}"
            )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run": RUN,
        "reported_anchor_not_gated": True,
        "component": "disability: DI incidence/recovery hazards + "
        "DI->retirement conversion (roadmap #113, M4)",
        "purpose": (
            "The M4 disability foundation: weighted PSID self-reported "
            "work-limitation incidence/recovery/prevalence hazards by "
            "age band x sex, the DI->retirement conversion analog "
            "validated against the archived SSA 6.B5.1 conversion column "
            "(concept delta named, undercount reported honestly), and "
            "the person-disjoint half-split noise floor a future DI gate "
            "would derive thresholds from. Reads no gate and changes no "
            "gate on its own; the pre-registered ceremony comes after. "
            "See proposed_thresholds_note (NOT RATIFIED)."
        ),
        "data": {
            "psid_micro_basis": (
                "populace_dynamics.data.disability.read_disability_status "
                "-- the cross-year individual-file EMPLOYMENT STATUS "
                "recode (code 5 = 'Permanently disabled'), a "
                "SELF-REPORTED labor-force status, joined to "
                "person-constant sex (ER32000) from "
                "populace_dynamics.data.deaths"
            ),
            "n_persons": n_persons,
            "n_person_years": int(len(panel.person_years)),
            "n_transition_pairs": int(len(panel.pairs)),
            "waves_covered": waves,
            "wave_coverage_note": (
                "EMPLOYMENT STATUS is in 20 individual-file waves: 1982, "
                "1983, then 1993-1997 (annual) and 1999-2023 (biennial). "
                "1968-1981 and 1984-1992 carry no individual-file "
                "employment-status recode under this label. 1982-1983 are "
                "isolated snapshots (no adjacent wave within 2 years) so "
                "they add prevalence person-years but almost no "
                "transition pairs."
            ),
            "employment_status_code_verification": {
                "method": (
                    "disability.verify_employment_status_codes: each "
                    "covered wave's EMPLOYMENT STATUS variable resolved to "
                    "its SAS VALUE format; code 5 asserted 'disabled', "
                    "code 1 asserted 'working'"
                ),
                "n_waves_verified": len(coding),
                "example_2023": coding.get(2023),
            },
            "disability_state": (
                "valid self-reported status = EMPLOYMENT STATUS in 1-8 "
                "(code 0 'Inap.' and 9 'NA/DK' excluded); disabled = code "
                "5; retired = code 4 (conversion analog only)"
            ),
        },
        "hazard_construction": {
            "unit": "grid-adjacent wave-pair transition",
            "rule": (
                "For each person, consecutive observed waves (w, w') with "
                "w'-w <= 2 form one transition carrying age and weight at "
                "w; incidence = not-disabled->disabled over not-disabled "
                "exposure; recovery = disabled->not-disabled over "
                "disabled exposure; prevalence = disabled share of valid "
                "person-years; all banded by age-at-w x sex, weighted by "
                "the start-wave cross-sectional individual weight."
            ),
            "interval_note": (
                "Rates are per observation interval; the grid mixes 1- "
                "and 2-year steps, so each cell carries its "
                "exposure-weighted mean_interval_yr and annualization is "
                "a documented future-gate decision, not applied here."
            ),
            "age_bands": [
                disability.band_label(lo, hi)
                for lo, hi in disability.DI_AGE_BANDS
            ],
            "sexes": list(disability.SEXES),
            "max_interval_years": disability.MAX_INTERVAL,
        },
        "reference_moments": moments,
        "conversion_validation": conversion,
        "internal_noise_floor": {
            "method": (
                "person-disjoint 50/50 half-split "
                "(populace_dynamics.harness.panel.split_panel_by_person, "
                "fraction=0.5, seeds 0-4) of the same band x sex "
                "reference moments; floor statistic is |ln(r_A / r_B)| "
                "between the two independent real halves"
            ),
            "seeds": list(SEEDS),
            "min_events_for_gate": MIN_EVENTS_FOR_GATE,
            "noise_floor_seeds_0_4": noise_floor,
            "cell_stability": stability,
            "per_seed": per_seed,
        },
        "concept_deltas": concept_deltas(),
        "wanted_ssa_tables": wanted_ssa_tables(),
        "proposed_thresholds_note": proposed_thresholds_note(stability),
        "revision_pins": {
            "populace_dynamics_sha": _git_sha(ROOT),
            "claim_age_reference_schema": ref.schema_version,
            "artifact_schema_version": SCHEMA_VERSION,
        },
        "build": {
            "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "built_by": "scripts/build_disability_floors.py",
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return artifact


def main() -> None:
    artifact = run(verbose=True)
    artifacts.write_new(ARTIFACT_PATH, artifact)
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
