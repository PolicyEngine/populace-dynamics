"""Build the cross-wave job-ID consistency check (C3 §6 pre-lock).

REQUIRED PRE-LOCK ARTIFACT for the seam ruling (#230 §6; ADR 0004
referee item 7; #214's concept-delta 5, previously an UNVERIFIED
ASSUMPTION). Question: are SIPP ``EJB`` job IDs longitudinally
consistent across the pu2022 -> pu2023 file boundary, or partly
reassigned — in which case part of the measured 9.45% Dec->Jan seam
separation rate would be a linkage artifact rather than seam-bunched
real separations?

Design — three bounds, none assuming what they test:

(a) **Gross consistency**: the share of December-held jobs whose ID
    survives into January at all. Wholesale per-wave reassignment
    would put this near zero; the #214 artifact already implies
    ~90.6%, so gross reassignment is bounded by the seam rate
    itself.

(b) **The re-key signature**: among Dec->Jan *separations* (no
    common ID), the share where the person holds a January job that
    matches the vanished December job on industry code AND class of
    worker AND monthly earnings within 20% (|log ratio| < 0.1823) —
    the profile of the same employer continuing under a new ID.
    Genuine job-to-job moves can also match by coincidence, so the
    identical signature is computed for *within-wave* separations
    (pooled month-pairs inside each file), whose IDs are known-good
    under dependent interviewing. The EXCESS of the seam signature
    over the within-wave baseline is the upper bound on the ID
    artifact among employed-next-month separators.

(c) **The structural bound**: seam separations decompose into exits
    to nonemployment (no January job exists, so no new ID could
    have been issued — these CANNOT be ID artifacts) versus
    separations-to-employment. Only the latter can hide re-keying,
    so the E->E share caps the artifact regardless of (b).

Verdict rule: the ruling's conditional check PASSES if the implied
ID-artifact share of the seam rate — excess re-key signature applied
to the E->E component — is under 15% of the measured seam rate;
between 15% and 30% the seam figures carry a correction band; above
30% the #214 ruling returns to the referee.

PROVENANCE OF THIS RULE (corrected 2026-07-19, review of #235).
Earlier revisions of this docstring described the rule as
"pre-registered here". That claim is not supported by the record and
is withdrawn:

  * the rule and the result land in a single commit (87788eb); no
    earlier commit, issue comment, or ADR fixes the 15/30 bands.
    #230's body conditions on this check without naming a threshold.
  * a first run of this check returned PASS_WITH_CORRECTION_BAND
    against a 6.06% conditioned rate. The estimator was then changed
    (inner-join -> person-month universe) and re-run to PASS. The
    fix is believed correct on its merits, but it means a verdict
    was observed before the committed estimator existed.

The accurate description is DISCLOSED RE-ANALYSIS AFTER A DISCOVERED
DEFECT, not pre-registration. #230 section 6 should cite it as such.

OPEN (referee, C3): the 15%/30% bands have no derivation on record.
Every other bar in this repo is derived from a noise floor. These
were chosen by the author. They need either a derivation or separate
ratification before this artifact can carry the seam ruling.

OPEN (referee, C3): the operative scoring population is unregistered
and the verdict depends on it -- see ``scoring_population_sensitivity``
in the artifact. This choice MUST be made by the referee round and
recorded here. It cannot be settled by whoever reads the numbers
first without reproducing the defect this file documents.

Usage::

    python scripts/build_crosswave_jobid_check.py

writes ``runs/crosswave_jobid_check_draft_v0.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from populace_dynamics.data import sipp_jobs  # noqa: E402

FILE_YEARS = (2022, 2023)
EARN_LOG_TOL = abs(np.log(0.8))  # earnings within 20%
ARTIFACT = REPO / "runs/crosswave_jobid_check_draft_v0.json"


def _verdict_for(share: float) -> str:
    """Apply the 15/30 bands to an artifact share.

    Factored out so the same rule can be reported against both
    candidate scoring populations without either being privileged.
    """
    if share < 0.15:
        return "PASS"
    if share <= 0.30:
        return "PASS_WITH_CORRECTION_BAND"
    return "REFER_BACK"


def person_month_presence(year: int) -> pd.DataFrame:
    """All person-months in the file (employed or not)."""
    import os

    data_dir = Path(
        os.environ.get(
            "POPULACE_DYNAMICS_SIPP_DIR",
            str(Path("~/PolicyEngine/sipp-data").expanduser()),
        )
    ).expanduser()
    for suffix in (".csv", ".csv.gz"):
        path = data_dir / f"pu{year}{suffix}"
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"pu{year}.csv[.gz] not staged")
    raw = pd.read_csv(
        path,
        sep="|",
        usecols=["SSUID", "PNUM", "MONTHCODE"],
        dtype={"SSUID": "string"},
    )
    raw["person_id"] = raw["SSUID"].astype(str) + "-" + raw["PNUM"].astype(str)
    return raw[["person_id", "MONTHCODE"]].rename(
        columns={"MONTHCODE": "month"}
    )


def month_frame(job_months: pd.DataFrame) -> pd.DataFrame:
    """Per person-month: job set plus per-job attribute map."""
    jm = job_months.copy()
    jm["attrs"] = list(
        zip(
            jm["job_id"],
            jm["industry"].astype(str),
            jm["clwrk"],
            jm["earnings"],
            strict=True,
        )
    )
    return (
        jm.groupby(["person_id", "month"])
        .agg(jobs=("job_id", frozenset), attrs=("attrs", list))
        .reset_index()
    )


def _rekey_match(lost, new_jobs) -> bool:
    """Does any new job match a lost job's employer profile?

    KNOWN BIAS, not sensitivity-tested (review of #235). A missing
    value on class-of-worker or earnings does not disqualify a match:
    the ``pd.notna`` guards mean a NaN falls through to ``return
    True``. Missingness is therefore scored as agreement, inflating
    the re-key signature. This matters only if item non-response
    differs across the file boundary -- which is exactly the boundary
    under test, so it cannot be assumed away.

    ``EARN_LOG_TOL`` (20%) is likewise unmotivated and untested; it
    moves the seam and within-wave signatures non-proportionally.

    Both are left AS-IS deliberately: changing them changes the
    committed numbers, and re-running requires the staged SIPP
    microdata. Registered here as C3 sensitivity work.
    """
    _, ind, clwrk, earn = lost
    for _, n_ind, n_clwrk, n_earn in new_jobs:
        if n_ind != ind:
            continue
        if pd.notna(clwrk) and pd.notna(n_clwrk) and n_clwrk != clwrk:
            continue
        if (
            pd.notna(earn)
            and pd.notna(n_earn)
            and earn > 0
            and n_earn > 0
            and abs(np.log(n_earn / earn)) > EARN_LOG_TOL
        ):
            continue
        return True
    return False


def separation_decomposition(
    current: pd.DataFrame,
    following: pd.DataFrame,
    present_next: set,
) -> dict:
    """Decompose separations between two adjacent person-months.

    ``present_next`` is the set of person_ids in the panel next
    month (employed or not): a person present with no jobs is an
    exit to nonemployment; a person absent left the sample and is
    excluded from the denominator entirely.
    """
    current = current[current["person_id"].isin(present_next)]
    merged = current.merge(
        following,
        on="person_id",
        suffixes=("", "_n"),
        how="left",
    )
    merged["jobs_n"] = merged["jobs_n"].apply(
        lambda x: x if isinstance(x, frozenset) else frozenset()
    )
    merged["attrs_n"] = merged["attrs_n"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    jobs_held = jobs_kept = 0
    lost_to_nonemp = lost_to_emp = lost_rekey_sig = 0
    for row in merged.itertuples(index=False):
        kept_ids = row.jobs & row.jobs_n
        jobs_held += len(row.jobs)
        jobs_kept += len(kept_ids)
        new_jobs = [a for a in row.attrs_n if a[0] not in row.jobs]
        for lost in row.attrs:
            if lost[0] in kept_ids:
                continue
            if not row.jobs_n:
                lost_to_nonemp += 1
                continue
            lost_to_emp += 1
            if _rekey_match(lost, new_jobs):
                lost_rekey_sig += 1
    separations = jobs_held - jobs_kept
    return {
        "jobs_held": jobs_held,
        "separations": separations,
        "sep_rate": round(separations / jobs_held, 4),
        "to_nonemployment": lost_to_nonemp,
        "to_employment": lost_to_emp,
        "rekey_signature": lost_rekey_sig,
        "rekey_signature_share_of_seps": (
            round(lost_rekey_sig / separations, 4) if separations else None
        ),
    }


def build() -> dict:
    frames = {
        year: sipp_jobs.read_sipp_job_months(year) for year in FILE_YEARS
    }
    months = {year: month_frame(frames[year]) for year in FILE_YEARS}
    presence = {year: person_month_presence(year) for year in FILE_YEARS}

    # Within-wave baseline: pooled adjacent month-pairs in each file.
    within = {
        "jobs_held": 0,
        "separations": 0,
        "to_nonemployment": 0,
        "to_employment": 0,
        "rekey_signature": 0,
    }
    for year in FILE_YEARS:
        mf = months[year]
        pres = presence[year]
        for month in range(1, 12):
            cur = mf[mf["month"] == month]
            nxt = mf[mf["month"] == month + 1].drop(columns="month")
            present_next = set(pres[pres["month"] == month + 1]["person_id"])
            d = separation_decomposition(cur, nxt, present_next)
            for key in within:
                within[key] += d[key]
    within["sep_rate"] = round(within["separations"] / within["jobs_held"], 4)
    within["rekey_signature_share_of_seps"] = round(
        within["rekey_signature"] / within["separations"], 4
    )

    # Across-wave: Dec (pu2022, ref Dec 2021) -> Jan (pu2023).
    dec = months[FILE_YEARS[0]]
    dec = dec[dec["month"] == 12]
    jan = months[FILE_YEARS[1]]
    jan = jan[jan["month"] == 1].drop(columns="month")
    present_jan = set(
        presence[FILE_YEARS[1]][presence[FILE_YEARS[1]]["month"] == 1][
            "person_id"
        ]
    )
    seam = separation_decomposition(dec, jan, present_jan)

    # The bound: excess re-key signature at the seam over the
    # within-wave baseline, applied to seam separations.
    seam_ee_sig_share = (
        seam["rekey_signature"] / seam["to_employment"]
        if seam["to_employment"]
        else 0.0
    )
    within_ee_sig_share = (
        within["rekey_signature"] / within["to_employment"]
        if within["to_employment"]
        else 0.0
    )
    excess_sig_ee = max(0.0, seam_ee_sig_share - within_ee_sig_share)
    ee_share_of_seps = seam["to_employment"] / seam["separations"]
    implied_artifact_share = round(excess_sig_ee * ee_share_of_seps, 4)
    ee_cap = round(ee_share_of_seps, 4)

    verdict = _verdict_for(implied_artifact_share)

    return {
        "artifact": "crosswave_jobid_check",
        "version": "draft_v0",
        "status": (
            "DRAFT - pre-lock artifact for the #230 section-6 seam "
            "ruling. NOT pre-registered: the verdict rule and the "
            "result were committed together (87788eb), and a first "
            "run returned a different verdict before the estimator "
            "was corrected. Accurate label: disclosed re-analysis "
            "after a discovered defect. See the module docstring."
        ),
        "issue": "230",
        "question": (
            "are EJB job IDs longitudinally consistent across the "
            "pu2022->pu2023 boundary, or is part of the 9.45% seam "
            "separation rate a re-keying (linkage) artifact?"
        ),
        "within_wave_baseline": within,
        "across_wave_seam": seam,
        "rekey_signature_definition": (
            "a vanished job whose person holds a next-month job "
            "matching it on industry code, class of worker, and "
            "earnings within 20% (|log ratio| < 0.1823); computed "
            "identically at the seam and within-wave, so the "
            "within-wave share is the coincidental-match baseline"
        ),
        "bounds": {
            # NOTE: 1 - sep_rate is the arithmetic complement of the
            # seam rate, i.e. a definitional identity, NOT evidence.
            # It would take this same value if every seam separation
            # were a re-key. Retained as context, relabelled so it
            # cannot be read as a bound. (Review of #235.)
            "gross_id_survival_identity": round(1 - seam["sep_rate"], 4),
            "excess_rekey_signature_share_of_seam_seps": (
                implied_artifact_share
            ),
            "structural_ee_cap_share_of_seam_seps": ee_cap,
        },
        # Both scoring populations, so the referee can see that the
        # verdict depends on which one is operative. Disclosure only:
        # this file does NOT choose between them.
        "scoring_population_sensitivity": {
            "ee_only_excess_share": round(excess_sig_ee, 4),
            "ee_only_verdict": _verdict_for(excess_sig_ee),
            "scaled_to_all_seps_excess_share": implied_artifact_share,
            "scaled_to_all_seps_verdict": verdict,
            "note": (
                "The E->E population is the one in which re-keying "
                "can occur at all, and scores ABOVE the 15% bar. The "
                "scaled figure multiplies it by the E->E share of "
                "separations (structural_ee_cap) and scores below. "
                "Which is operative is unregistered -- see the OPEN "
                "items in the module docstring."
            ),
        },
        "verdict_rule": (
            "PASS if excess re-key share < 15% of seam separations; "
            "PASS_WITH_CORRECTION_BAND if 15-30%; REFER_BACK if "
            ">30%"
        ),
        "verdict_bar_provenance": (
            "OPEN - no derivation on record; author-chosen, not "
            "floor-derived. Requires ratification (review of #235)."
        ),
        "verdict": verdict,
        "verdict_is_conditional_on_scoring_population": True,
    }


def main() -> None:
    artifact = build()
    ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT}")
    print("within-wave:", artifact["within_wave_baseline"])
    print("seam:", artifact["across_wave_seam"])
    print("bounds:", artifact["bounds"])
    print("VERDICT:", artifact["verdict"])


if __name__ == "__main__":
    main()
