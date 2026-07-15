"""Build the DRAFT SIPP-vs-J2J seam-reconciliation artifact (#192).

REPORTED ANCHOR, NOT A GATE RUN — and explicitly a DRAFT: no
thresholds, C3 not locked. This is the reconciliation run the #192
protocol rules require before E2/E4 thresholds lock ("commit a
documented SIPP-vs-J2J rate-reconciliation run and decide ex ante
which source is truth for rates vs persistence structure"). The
groundwork numbers were first posted to #192 (comment 4982442068);
this script commits the reproducible version and adds the E->N leg
the groundwork lacked.

Three measurements, every concept delta NAMED, none adjusted away:

(a) **Within-wave monthly job separation** per SIPP file: a job held
    in month m and absent in m+1, among persons present in the
    panel in both months (presence from the person-month universe,
    so exits to nonemployment COUNT as separations — unlike the
    groundwork, which conditioned on employed-both-months).
(b) **Across-wave separation** at the wave boundary: December of
    one file's reference year to January of the next file's,
    linking persons on ``SSUID``-``PNUM`` and the wave-consistent
    ``EJB`` job ids. Seam bias concentrates here.
(c) **The J2J benchmark**: national all-industry main-job
    separations (``MSep/MainB``) by quarter from the committed
    extract, converted to a monthly equivalent
    ``1 - (1 - q)**(1/3)``.

Named concept deltas that remain: J2J counts jobs (person-employer
pairs) in UI-covered non-federal employment and separations of the
*main* job; SIPP here counts all jobs including self-employment
(JBORSE 2/3 excludable in later cuts) per job-month held. J2J
"quarterly separation" is not literally three independent monthly
draws, so the monthly equivalent is an approximation stated as such.

Usage::

    python scripts/build_seam_reconciliation.py

writes ``runs/seam_reconciliation_draft_v0.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from populace_dynamics.data import sipp_jobs  # noqa: E402

FILE_YEARS = (2022, 2023)
J2J_YEARS = (2021, 2022)
J2J_EXTRACT = REPO / "data/external/j2j_us_firmsize_sector_2015on.csv"
ARTIFACT = REPO / "runs/seam_reconciliation_draft_v0.json"


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


def within_wave(year: int) -> pd.DataFrame:
    """Monthly job-separation rates, presence-conditioned only."""
    job_months = sipp_jobs.read_sipp_job_months(year)
    presence = person_month_presence(year)
    present = set(zip(presence["person_id"], presence["month"], strict=True))
    jobs = (
        job_months.groupby(["person_id", "month"])["job_id"]
        .agg(frozenset)
        .reset_index()
    )
    jobs_next = {
        (p, m): j
        for p, m, j in zip(
            jobs["person_id"], jobs["month"], jobs["job_id"], strict=True
        )
    }
    rows = []
    for month in range(1, 12):
        held = kept = 0
        month_slice = jobs[jobs["month"] == month]
        for person, js in zip(
            month_slice["person_id"], month_slice["job_id"], strict=True
        ):
            if (person, month + 1) not in present:
                continue  # left the sample, not a separation
            held += len(js)
            kept += len(js & jobs_next.get((person, month + 1), frozenset()))
        rows.append(
            {
                "month_pair": f"{month}->{month + 1}",
                "jobs_held": held,
                "jobs_kept": kept,
                "sep_rate": round(1 - kept / held, 4),
            }
        )
    return pd.DataFrame(rows)


def across_wave() -> dict:
    """Dec (first file) -> Jan (second file), sample-present both."""
    first = sipp_jobs.read_sipp_job_months(FILE_YEARS[0])
    second = sipp_jobs.read_sipp_job_months(FILE_YEARS[1])
    presence_second = person_month_presence(FILE_YEARS[1])
    present_jan = set(
        presence_second[presence_second["month"] == 1]["person_id"]
    )
    dec = (
        first[first["month"] == 12]
        .groupby("person_id")["job_id"]
        .agg(frozenset)
    )
    jan = (
        second[second["month"] == 1]
        .groupby("person_id")["job_id"]
        .agg(frozenset)
    )
    held = kept = persons = 0
    for person, js in dec.items():
        if person not in present_jan:
            continue
        persons += 1
        held += len(js)
        kept += len(js & jan.get(person, frozenset()))
    return {
        "persons_linked": persons,
        "jobs_held": held,
        "jobs_kept": kept,
        "sep_rate": round(1 - kept / held, 4),
    }


def j2j_benchmark() -> list[dict]:
    j2j = pd.read_csv(J2J_EXTRACT, dtype={"industry": str})
    national = j2j[(j2j.industry == "00") & (j2j.year.isin(J2J_YEARS))]
    out = (
        national.groupby(["year", "quarter"])
        .agg(MainB=("MainB", "sum"), MSep=("MSep", "sum"))
        .reset_index()
    )
    out["q_sep_rate"] = (out.MSep / out.MainB).round(4)
    out["monthly_equivalent"] = (
        1 - (1 - out.MSep / out.MainB) ** (1 / 3)
    ).round(4)
    return out[
        ["year", "quarter", "q_sep_rate", "monthly_equivalent"]
    ].to_dict("records")


def build() -> dict:
    within = {
        str(year): within_wave(year).to_dict("records") for year in FILE_YEARS
    }
    seam = across_wave()
    bench = j2j_benchmark()
    within_means = {
        year: round(float(pd.DataFrame(rows)["sep_rate"].mean()), 4)
        for year, rows in within.items()
    }
    return {
        "artifact": "seam_reconciliation",
        "version": "draft_v0",
        "status": "DRAFT - NOT RATIFIED; C3 not locked; no thresholds",
        "issue": "192",
        "first_reported": (
            "PolicyEngine/populace-dynamics#192 comment 4982442068 "
            "(groundwork conditioned on employed-both-months; this "
            "artifact adds the E->N leg via the person-month "
            "universe)"
        ),
        "within_wave_monthly_separation": within,
        "within_wave_means": within_means,
        "across_wave_dec_to_jan": seam,
        "j2j_national_benchmark": bench,
        "concept_deltas": [
            "J2J counts jobs (person-employer pairs) in UI-covered "
            "non-federal employment; SIPP counts all jobs incl. "
            "self-employment per job-month held",
            "J2J MSep is main-job separations; SIPP here counts "
            "every held job",
            "the monthly equivalent 1-(1-q)^(1/3) treats a quarter "
            "as three independent monthly draws — an approximation",
            "persons leaving the SIPP sample are excluded from the "
            "denominator, not counted as separations",
        ],
        "proposed_ruling_note": (
            "NOT RATIFIED: as the plan proposed ex ante, J2J is "
            "truth for rate LEVELS and SIPP for persistence "
            "STRUCTURE; phase-1 hazards estimated seam-aware "
            "(wave-frequency with within-wave interpolation). "
            "Ratification belongs to the C3 referee round."
        ),
    }


def main() -> None:
    artifact = build()
    ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT}")
    print("within-wave means:", artifact["within_wave_means"])
    print("across-wave:", artifact["across_wave_dec_to_jan"])


if __name__ == "__main__":
    main()
