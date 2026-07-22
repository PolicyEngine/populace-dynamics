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

Verdict rule (author-proposed, UNRATIFIED — see the artifact's
status field): under 15% implied ID-artifact share PASSES; 15-30%
carries a correction band; above 30% the #214 ruling returns to the
referee. Two scoring populations are reported (E->E-only and
all-separations) and the operative one is a referee item, as is the
bar itself. This artifact is a DISCLOSED RE-ANALYSIS, not a
pre-registration: the first committed estimator had a population
defect (documented in the status field) and the corrected estimator
re-ran after a verdict had been observed.

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


def _rekey_match(lost, new_jobs, strict: bool = False) -> bool:
    """Does any new job match a lost job's employer profile?

    Default (main) matching treats a missing field as compatible;
    ``strict=True`` treats any missing industry/class/earnings on
    either side as a mismatch (the sensitivity variant for the
    NaN-matching caveat).
    """
    _, ind, clwrk, earn = lost
    for _, n_ind, n_clwrk, n_earn in new_jobs:
        if n_ind != ind:
            continue
        if pd.isna(clwrk) or pd.isna(n_clwrk):
            if strict:
                continue
        elif n_clwrk != clwrk:
            continue
        if pd.isna(earn) or pd.isna(n_earn) or not earn > 0 or not n_earn > 0:
            if strict:
                continue
        elif abs(np.log(n_earn / earn)) > EARN_LOG_TOL:
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
    lost_rekey_sig_strict = 0
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
            if _rekey_match(lost, new_jobs, strict=True):
                lost_rekey_sig_strict += 1
    separations = jobs_held - jobs_kept
    return {
        "jobs_held": jobs_held,
        "separations": separations,
        "sep_rate": round(separations / jobs_held, 4),
        "to_nonemployment": lost_to_nonemp,
        "to_employment": lost_to_emp,
        "rekey_signature": lost_rekey_sig,
        "rekey_signature_strict": lost_rekey_sig_strict,
        "rekey_signature_share_of_seps": (
            round(lost_rekey_sig / separations, 4) if separations else None
        ),
    }


def _input_pins() -> dict:
    """sha256 + size of the staged pu files consumed."""
    import hashlib as _h
    import os

    data_dir = Path(
        os.environ.get(
            "POPULACE_DYNAMICS_SIPP_DIR",
            str(Path("~/PolicyEngine/sipp-data").expanduser()),
        )
    ).expanduser()
    pins = {}
    for year in FILE_YEARS:
        for suffix in (".csv", ".csv.gz"):
            p = data_dir / f"pu{year}{suffix}"
            if p.exists():
                digest = _h.sha256()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b""):
                        digest.update(chunk)
                pins[p.name] = {
                    "sha256": digest.hexdigest(),
                    "bytes": p.stat().st_size,
                }
                break
    return pins


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
        "rekey_signature_strict": 0,
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
    # within-wave baseline. Two defensible scoring populations exist
    # and the verdict differs between them, so BOTH are reported and
    # the operative choice is a referee item, not an author choice.
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
    share_ee_population = round(excess_sig_ee, 4)
    share_all_separations = round(excess_sig_ee * ee_share_of_seps, 4)
    ee_cap = round(ee_share_of_seps, 4)

    # Point-estimate uncertainty: binomial SEs on the two signature
    # shares, propagated to the excess (independent samples), and a
    # one-sided 95% upper bound per population.
    import math

    se_seam = math.sqrt(
        seam_ee_sig_share * (1 - seam_ee_sig_share) / seam["to_employment"]
    )
    se_within = math.sqrt(
        within_ee_sig_share
        * (1 - within_ee_sig_share)
        / within["to_employment"]
    )
    se_excess = math.sqrt(se_seam**2 + se_within**2)
    upper_ee = round(excess_sig_ee + 1.645 * se_excess, 4)
    upper_all = round(
        (excess_sig_ee + 1.645 * se_excess) * ee_share_of_seps, 4
    )

    def band(x: float) -> str:
        if x < 0.15:
            return "PASS"
        if x <= 0.30:
            return "PASS_WITH_CORRECTION_BAND"
        return "REFER_BACK"

    strict_seam = (
        seam["rekey_signature_strict"] / seam["to_employment"]
        if seam["to_employment"]
        else 0.0
    )
    strict_within = (
        within["rekey_signature_strict"] / within["to_employment"]
        if within["to_employment"]
        else 0.0
    )
    strict_excess = max(0.0, strict_seam - strict_within)
    strict_variant = {
        "note": (
            "missing industry/class/earnings treated as MISMATCH "
            "(main variant treats missing as compatible)"
        ),
        "seam_signature_share": round(strict_seam, 4),
        "within_signature_share": round(strict_within, 4),
        "excess_ee_population": round(strict_excess, 4),
        "excess_all_separations": round(strict_excess * ee_share_of_seps, 4),
    }

    return {
        "artifact": "crosswave_jobid_check",
        "version": "draft_v1",
        "status": (
            "DRAFT - pre-lock artifact for the #230 section-6 seam "
            "ruling. DISCLOSED RE-ANALYSIS, not pre-registration: "
            "the first committed estimator (inner-join population, "
            "conditioned 6.06% seam rate) returned "
            "PASS_WITH_CORRECTION_BAND; a population defect (exits "
            "to nonemployment silently dropped, contradicting the "
            "documented design) was found and fixed, and the "
            "corrected estimator re-ran. Both runs are disclosed "
            "here; the 15/30 bands are author-proposed and "
            "UNRATIFIED (referee item), as is the operative scoring "
            "population."
        ),
        "issue": "230",
        "inputs": _input_pins(),
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
            "gross_id_survival_identity": {
                "value": round(1 - seam["sep_rate"], 4),
                "note": (
                    "definitional identity (1 - seam sep rate), NOT "
                    "evidence — it would be unchanged if every seam "
                    "separation were a re-key; retained only as "
                    "context"
                ),
            },
            "excess_rekey_share_ee_population": share_ee_population,
            "excess_rekey_share_all_separations": share_all_separations,
            "one_sided_95_upper_ee_population": upper_ee,
            "one_sided_95_upper_all_separations": upper_all,
            "structural_ee_cap_share_of_seam_seps": ee_cap,
        },
        "verdict_rule": (
            "author-proposed, UNRATIFIED (referee item): PASS if "
            "excess re-key share < 15%; PASS_WITH_CORRECTION_BAND "
            "if 15-30%; REFER_BACK if >30%. The operative scoring "
            "population (E->E separations only, arguably the "
            "conservative reading since re-keying is a "
            "within-continuing-employment phenomenon, vs all seam "
            "separations, since E->N separations cannot be ID "
            "artifacts) is ALSO a referee item — the verdict "
            "differs between them."
        ),
        "verdict_by_population": {
            "ee_population": band(share_ee_population),
            "all_separations": band(share_all_separations),
            "operative": "REFEREE",
        },
        "caveats": {
            "composition_mismatch": (
                "the within-wave coincidence baseline has a "
                "different separation mix (E->N share "
                f"{within['to_nonemployment'] / within['separations']:.3f}"
                " within-wave vs "
                f"{seam['to_nonemployment'] / seam['separations']:.3f}"
                " at the seam)"
            ),
            "nan_matching": (
                "the re-key signature treats missing "
                "industry/class/earnings as matching (pd.notna "
                "guards), biasing the signature upward where item "
                "nonresponse differs across the boundary; the "
                "strict variant below treats missing as mismatch"
            ),
            "seam_denominator": (
                "person presence at the seam is keyed on SSUID+PNUM "
                "- the same cross-file linkage under test; a person "
                "whose ID re-keyed would leave the denominator as a "
                "sample leaver rather than appear as a separation, "
                "so person-level re-keying is NOT bounded by this "
                "artifact (jobs_held 10,828 at the seam vs ~17,500 "
                "per within-wave pair reflects sample rotation plus "
                "any such loss)"
            ),
        },
        "strict_nan_variant": strict_variant,
    }


def main() -> None:
    artifact = build()
    ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT}")
    print("within-wave:", artifact["within_wave_baseline"])
    print("seam:", artifact["across_wave_seam"])
    print("bounds:", artifact["bounds"])
    print("strict variant:", artifact["strict_nan_variant"])
    print("VERDICT BY POPULATION:", artifact["verdict_by_population"])


if __name__ == "__main__":
    main()
