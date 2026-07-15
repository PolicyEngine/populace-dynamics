"""Build DRAFT SIPP job-spell noise floors for gates E4/E5 (#192).

REPORTED ANCHOR, NOT A GATE RUN — and explicitly a DRAFT: C3 (the
employer gate block) has not locked, no thresholds are proposed
here, and nothing below is ratified. Like the disability floors,
this commits the person-disjoint half-vs-half sampling-noise floor
that pre-registered E4/E5 thresholds would later be derived from,
so the floor-building method is on the record before any candidate
model exists (issue #192 protocol: floors -> thresholds -> referee
round -> one-shot runs).

Moments, computed on the SIPP job-month panel
(:mod:`populace_dynamics.data.sipp_jobs`, reference year 2022 from
the pu2023 file):

(a) **E4 retention pairs.** Among persons employed in consecutive
    reference months m and m+1, the weighted share retaining at
    least one employer (same within-panel ``EJB`` job id in both
    months), by age band x sex. This is the month-frequency analog
    of the plan's "2-window employer-retention persistence".

(b) **E5 attachment runs.** The weighted distribution of maximal
    same-employer run lengths (1-12 months within the reference
    year, from :func:`job_spells`), by age band — the view that
    catches chained-model persistence understatement.

(c) **The floor.** For seeds 0-4, persons are split into two
    disjoint halves; each cell's rate is computed on both halves
    and the across-seed mean and sd of ``|log(rate_a / rate_b)|``
    is the sampling-noise floor for that cell, exactly the
    disability-floor convention. Cells with fewer than 200
    unweighted persons per half are reported but flagged thin.

Seam caveat (pre-registered on #192): SIPP transitions bunch at
interview seams, and both halves share the seam structure, so this
floor cannot see seam bias — the seam-vs-J2J reconciliation run is
a separate, required artifact before E2/E4 thresholds lock.

Usage::

    python scripts/build_sipp_spell_floors.py

writes ``runs/sipp_spell_floors_draft_v0.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from populace_dynamics.data import sipp_jobs  # noqa: E402

YEAR = 2023
SEEDS = (0, 1, 2, 3, 4)
AGE_BANDS = ((16, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 99))
THIN_CELL_PERSONS = 200

ARTIFACT = Path(__file__).resolve().parents[1] / (
    "runs/sipp_spell_floors_draft_v0.json"
)


def _age_band(age: pd.Series) -> pd.Series:
    bins = [AGE_BANDS[0][0] - 1, *[hi for _, hi in AGE_BANDS]]
    labels = [f"{lo}_{hi}" for lo, hi in AGE_BANDS]
    return pd.cut(age, bins=bins, labels=labels)


def _half(person_id: pd.Series, seed: int) -> pd.Series:
    """Deterministic person-disjoint half assignment."""

    def bucket(pid: str) -> int:
        digest = hashlib.sha256(f"{seed}:{pid}".encode()).digest()
        return digest[0] & 1

    return person_id.map(bucket)


def retention_frame(job_months: pd.DataFrame) -> pd.DataFrame:
    """One row per person x consecutive-month pair, employed both."""
    person_months = (
        job_months.groupby(["person_id", "month"])
        .agg(
            jobs=("job_id", frozenset),
            age=("age", "first"),
            sex=("sex", "first"),
            weight=("weight", "first"),
        )
        .reset_index()
    )
    nxt = person_months.copy()
    nxt["month"] -= 1
    pairs = person_months.merge(
        nxt,
        on=["person_id", "month"],
        suffixes=("", "_next"),
    )
    pairs["retained"] = [
        bool(a & b) for a, b in zip(pairs.jobs, pairs.jobs_next, strict=True)
    ]
    pairs["age_band"] = _age_band(pairs["age"])
    return pairs[pairs["age_band"].notna()]


def run_length_frame(job_months: pd.DataFrame) -> pd.DataFrame:
    """One row per person: longest same-employer run in the year."""
    spells = sipp_jobs.job_spells(job_months)
    person_attrs = job_months.groupby("person_id").agg(
        age=("age", "first"), weight=("weight", "first")
    )
    longest = spells.groupby("person_id")["n_months"].max()
    out = person_attrs.join(longest).dropna(subset=["n_months"])
    out["age_band"] = _age_band(out["age"])
    return out[out["age_band"].notna()].reset_index()


def _weighted_rate(frame: pd.DataFrame, flag: str) -> float:
    total = frame["weight"].sum()
    return float((frame["weight"] * frame[flag]).sum() / total)


def floors_for_retention(pairs: pd.DataFrame) -> dict:
    cells = {}
    for (band, sex), cell in pairs.groupby(["age_band", "sex"], observed=True):
        gaps = []
        halves_n = []
        for seed in SEEDS:
            half = _half(cell["person_id"], seed)
            a = cell[half == 0]
            b = cell[half == 1]
            halves_n.append(
                min(a["person_id"].nunique(), b["person_id"].nunique())
            )
            ra, rb = _weighted_rate(a, "retained"), _weighted_rate(
                b, "retained"
            )
            gaps.append(abs(np.log(ra / rb)))
        cells[f"{band}|sex{int(sex)}"] = {
            "rate": round(_weighted_rate(cell, "retained"), 4),
            "pairs_unweighted": int(len(cell)),
            "floor_abs_log_ratio_mean": round(float(np.mean(gaps)), 5),
            "floor_abs_log_ratio_sd": round(float(np.std(gaps)), 5),
            "thin": bool(min(halves_n) < THIN_CELL_PERSONS),
        }
    return cells


def floors_for_runs(runs: pd.DataFrame) -> dict:
    runs = runs.assign(long_run=runs["n_months"] >= 12)
    cells = {}
    for band, cell in runs.groupby("age_band", observed=True):
        gaps = []
        halves_n = []
        for seed in SEEDS:
            half = _half(cell["person_id"], seed)
            a, b = cell[half == 0], cell[half == 1]
            halves_n.append(min(len(a), len(b)))
            ra, rb = _weighted_rate(a, "long_run"), _weighted_rate(
                b, "long_run"
            )
            gaps.append(abs(np.log(ra / rb)))
        cells[str(band)] = {
            "full_year_run_share": round(_weighted_rate(cell, "long_run"), 4),
            "persons_unweighted": int(len(cell)),
            "floor_abs_log_ratio_mean": round(float(np.mean(gaps)), 5),
            "floor_abs_log_ratio_sd": round(float(np.std(gaps)), 5),
            "thin": bool(min(halves_n) < THIN_CELL_PERSONS),
        }
    return cells


def build() -> dict:
    job_months = sipp_jobs.read_sipp_job_months(YEAR)
    pairs = retention_frame(job_months)
    runs = run_length_frame(job_months)
    return {
        "artifact": "sipp_spell_floors",
        "version": "draft_v0",
        "status": "DRAFT - NOT RATIFIED; C3 not locked; no thresholds",
        "issue": "192",
        "source": f"pu{YEAR} (reference year {YEAR - 1})",
        "method": (
            "person-disjoint sha256 half-splits, seeds 0-4; per-cell "
            "|log(rate_a/rate_b)| mean/sd across seeds; weighted by "
            "WPFINWGT"
        ),
        "seam_caveat": (
            "both halves share SIPP seam structure; the seam-vs-J2J "
            "reconciliation run is a separate required artifact "
            "before thresholds lock"
        ),
        "e4_retention_by_age_sex": floors_for_retention(pairs),
        "e5_runs_by_age": floors_for_runs(runs),
    }


def main() -> None:
    artifact = build()
    ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT}")
    e4 = artifact["e4_retention_by_age_sex"]
    print(f"E4 cells: {len(e4)}; example:", next(iter(e4.items())))


if __name__ == "__main__":
    main()
