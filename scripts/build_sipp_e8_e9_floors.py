"""Build DRAFT SIPP floors for gates E8 and E9 (#192).

REPORTED ANCHOR, NOT A GATE RUN — and explicitly a DRAFT: C3 has
not locked, no thresholds are proposed. Completes Workstream A's
floor battery (E3 tenure and E4/E5 spells are committed siblings;
E10 needs no floor — it is pass/fail on the locked PSID gates).

Moments, on the pu2023 file (reference year 2022), restricted to
persons observed in the panel for all 12 reference months so spell
durations are not right-censored by sample exit (the restriction is
recorded, not hidden):

(a) **E8 — nonemployment spell durations** (the zero-spell battery
    analog). A person-month is nonemployed when the person is in
    the panel that month with no active EJB job. Among persons with
    at least one employed month, maximal nonemployment runs are
    collapsed; moments per age band: the weighted share of persons
    with any nonemployment spell, and the weighted share of those
    spells lasting >= 3 months.

(b) **E9 — earnings change by transition type** (the layering-
    coherence moment). Consecutive-month person transitions are
    classified stay (a common job id), j2j (employed both months,
    no common id), exit (employed -> nonemployed), entry
    (nonemployed -> employed). For stay and j2j, where both months'
    known earnings totals are positive, the moment is the weighted
    median and IQR of log(earn_{m+1} / earn_m). Exit/entry carry
    rates, not earnings changes (their change is to/from zero by
    construction).

(c) **The floor**: person-disjoint sha256 half-splits, seeds 0-4;
    for rates the |log rate ratio| between halves, for medians/IQRs
    the absolute gap in log-points; mean/sd across seeds. Cells
    under 200 unweighted persons per half are flagged thin.

Thin-flag units (recorded for honesty across the floor battery):
the E8 thin flag counts **rows** per half, which equal persons
because the E8 frame has one row per person; the E9
earnings-change thin flag counts **rows** per half, which are
consecutive-month transition *pairs* (a person can contribute up
to 11), not persons — unlike the E4 spell floor, which counts
distinct persons. All compare against the same
``THIN_CELL_PERSONS = 200``.

Seam caveat: identical to the E4/E5 floors — both halves share
SIPP's seam structure, so these floors cannot see seam bias; the
reconciliation artifact (#214) carries that measurement.

Usage::

    python scripts/build_sipp_e8_e9_floors.py

writes ``runs/sipp_e8_e9_floors_draft_v0.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from populace_dynamics.data import sipp_jobs  # noqa: E402

YEAR = 2023
SEEDS = (0, 1, 2, 3, 4)
AGE_BANDS = ((16, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 99))
THIN_CELL_PERSONS = 200

ARTIFACT = REPO / "runs/sipp_e8_e9_floors_draft_v0.json"


def _reader_commit() -> str:
    """Last commit touching the SIPP reader in effect for this run."""
    import subprocess

    try:
        return subprocess.run(
            [
                "git",
                "log",
                "-1",
                "--format=%H",
                "--",
                "src/populace_dynamics/data/sipp_jobs.py",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _age_band(age: pd.Series) -> pd.Series:
    bins = [AGE_BANDS[0][0] - 1, *[hi for _, hi in AGE_BANDS]]
    labels = [f"{lo}_{hi}" for lo, hi in AGE_BANDS]
    return pd.cut(age, bins=bins, labels=labels)


def _half(person_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{person_id}".encode()).digest()
    return digest[0] & 1


def _person_month_universe(year: int) -> pd.DataFrame:
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
        usecols=["SSUID", "PNUM", "MONTHCODE", "WPFINWGT", "TAGE"],
        dtype={"SSUID": "string"},
    )
    raw["person_id"] = raw["SSUID"].astype(str) + "-" + raw["PNUM"].astype(str)
    return raw.rename(
        columns={"MONTHCODE": "month", "WPFINWGT": "weight", "TAGE": "age"}
    )[["person_id", "month", "weight", "age"]]


def build_panel(year: int) -> pd.DataFrame:
    """Person x month grid for full-year persons, with jobs/earnings."""
    universe = _person_month_universe(year)
    counts = universe.groupby("person_id")["month"].nunique()
    full_year = set(counts[counts == 12].index)
    universe = universe[universe["person_id"].isin(full_year)]

    job_months = sipp_jobs.read_sipp_job_months(year)
    jobs = (
        job_months.groupby(["person_id", "month"])
        .agg(jobs=("job_id", frozenset), earn=("earnings", "sum"))
        .reset_index()
    )
    panel = universe.merge(jobs, on=["person_id", "month"], how="left")
    panel["jobs"] = panel["jobs"].apply(
        lambda x: x if isinstance(x, frozenset) else frozenset()
    )
    panel["employed"] = panel["jobs"].map(len) > 0
    return panel.sort_values(["person_id", "month"])


def e8_person_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Per person: any-nonemployment flag and longest N-spell."""
    rows = []
    for person, grp in panel.groupby("person_id", sort=False):
        employed = grp.sort_values("month")["employed"].to_numpy()
        if not employed.any():
            continue  # never employed: outside the E8 universe
        runs = []
        run = 0
        for e in employed:
            if not e:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)
        rows.append(
            {
                "person_id": person,
                "age": grp["age"].iloc[0],
                "weight": grp["weight"].iloc[0],
                "any_nonemp": bool(runs),
                "long_nonemp": bool(runs and max(runs) >= 3),
            }
        )
    out = pd.DataFrame(rows)
    out["age_band"] = _age_band(out["age"])
    return out[out["age_band"].notna()]


def e9_transition_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Person x month-pair transitions with earnings changes."""
    nxt = panel[["person_id", "month", "jobs", "earn", "employed"]].copy()
    nxt["month"] -= 1
    pairs = panel.merge(nxt, on=["person_id", "month"], suffixes=("", "_n"))

    def classify(row) -> str:
        if row["employed"] and row["employed_n"]:
            return "stay" if row["jobs"] & row["jobs_n"] else "j2j"
        if row["employed"] and not row["employed_n"]:
            return "exit"
        if not row["employed"] and row["employed_n"]:
            return "entry"
        return "neither"

    pairs["transition"] = pairs.apply(classify, axis=1)
    pairs = pairs[pairs["transition"] != "neither"].copy()
    both_known = (
        pairs["earn"].notna()
        & pairs["earn_n"].notna()
        & (pairs["earn"] > 0)
        & (pairs["earn_n"] > 0)
    )
    pairs["log_change"] = np.nan
    mask = both_known & pairs["transition"].isin(("stay", "j2j"))
    pairs.loc[mask, "log_change"] = np.log(
        pairs.loc[mask, "earn_n"] / pairs.loc[mask, "earn"]
    )
    pairs["age_band"] = _age_band(pairs["age"])
    return pairs[pairs["age_band"].notna()]


def _weighted_quantile(values, weights, q: float) -> float:
    order = np.argsort(values, kind="stable")
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    cum = np.cumsum(weights) - 0.5 * weights
    cum /= weights.sum()
    return float(np.interp(q, cum, values))


def _rate_floor(cell: pd.DataFrame, flag: str) -> dict:
    gaps, halves_n = [], []
    for seed in SEEDS:
        half = cell["person_id"].map(lambda p, s=seed: _half(p, s))
        a, b = cell[half == 0], cell[half == 1]
        halves_n.append(min(len(a), len(b)))
        ra = float((a["weight"] * a[flag]).sum() / a["weight"].sum())
        rb = float((b["weight"] * b[flag]).sum() / b["weight"].sum())
        gaps.append(abs(np.log(ra / rb)) if ra > 0 and rb > 0 else np.nan)
    gaps = [g for g in gaps if not np.isnan(g)]
    return {
        "floor_abs_log_ratio_mean": round(float(np.mean(gaps)), 5),
        "floor_abs_log_ratio_sd": round(float(np.std(gaps)), 5),
        "thin": bool(min(halves_n) < THIN_CELL_PERSONS),
    }


def e8_floors(persons: pd.DataFrame) -> dict:
    cells = {}
    for band, cell in persons.groupby("age_band", observed=True):
        w = cell["weight"]
        cells[str(band)] = {
            "any_nonemp_share": round(
                float((w * cell["any_nonemp"]).sum() / w.sum()), 4
            ),
            "long_nonemp_share": round(
                float((w * cell["long_nonemp"]).sum() / w.sum()), 4
            ),
            "persons_unweighted": int(len(cell)),
            "any_nonemp": _rate_floor(cell, "any_nonemp"),
            "long_nonemp": _rate_floor(cell, "long_nonemp"),
        }
    return cells


def e9_floors(pairs: pd.DataFrame) -> dict:
    out: dict = {"transition_rates": {}, "earnings_change": {}}
    monthly = pairs.groupby("transition")["weight"].sum()
    total = monthly.sum()
    out["transition_rates"] = {
        t: round(float(v / total), 4) for t, v in monthly.items()
    }
    for kind in ("stay", "j2j"):
        cell = pairs[
            (pairs["transition"] == kind) & pairs["log_change"].notna()
        ]
        values = cell["log_change"].to_numpy(dtype=float)
        weights = cell["weight"].to_numpy(dtype=float)
        med = _weighted_quantile(values, weights, 0.5)
        iqr = _weighted_quantile(values, weights, 0.75) - _weighted_quantile(
            values, weights, 0.25
        )
        med_gaps, iqr_gaps, halves_n = [], [], []
        for seed in SEEDS:
            half = cell["person_id"].map(lambda p, s=seed: _half(p, s))
            a, b = cell[half == 0], cell[half == 1]
            halves_n.append(min(len(a), len(b)))

            def q(frame, qq):
                return _weighted_quantile(
                    frame["log_change"].to_numpy(dtype=float),
                    frame["weight"].to_numpy(dtype=float),
                    qq,
                )

            med_gaps.append(abs(q(a, 0.5) - q(b, 0.5)))
            iqr_gaps.append(
                abs((q(a, 0.75) - q(a, 0.25)) - (q(b, 0.75) - q(b, 0.25)))
            )
        out["earnings_change"][kind] = {
            "median_log_change": round(med, 4),
            "iqr_log_change": round(iqr, 4),
            "pairs_unweighted": int(len(cell)),
            "floor_abs_median_gap": {
                "mean": round(float(np.mean(med_gaps)), 5),
                "sd": round(float(np.std(med_gaps)), 5),
            },
            "floor_abs_iqr_gap": {
                "mean": round(float(np.mean(iqr_gaps)), 5),
                "sd": round(float(np.std(iqr_gaps)), 5),
            },
            "thin": bool(min(halves_n) < THIN_CELL_PERSONS),
        }
    return out


def build() -> dict:
    panel = build_panel(YEAR)
    persons = e8_person_frame(panel)
    pairs = e9_transition_frame(panel)
    return {
        "artifact": "sipp_e8_e9_floors",
        "version": "draft_v0",
        "status": "DRAFT - NOT RATIFIED; C3 not locked; no thresholds",
        "issue": "192",
        "source": f"pu{YEAR} (reference year {YEAR - 1}), persons "
        "observed all 12 reference months (censoring-free draft "
        "restriction, recorded)",
        "method": (
            "person-disjoint sha256 half-splits, seeds 0-4; rates "
            "floored on |log rate ratio|, earnings-change medians/"
            "IQRs on absolute gaps in log-points; weighted by "
            "WPFINWGT"
        ),
        "seam_caveat": (
            "identical to the E4/E5 floors: half-splits share SIPP's "
            "seam structure; #214 carries the seam measurement"
        ),
        "stay_median_heaping_caveat": (
            "within-job SIPP monthly earnings are mostly constant "
            "across a wave (dependent-interview reporting), so the "
            "stay-transition median log-change heaps at exactly 0 "
            "and its half-vs-half floor is degenerate (0.0) - the "
            "same failure class as the tenure quantile heaping; "
            "E9-stay thresholds should be stated on the IQR or a "
            "distributional distance, not the median"
        ),
        "thin_flag_units": {
            "e8_nonemployment_by_age": (
                "rows per half, equal to persons (one row per "
                "person in the E8 frame) vs THIN_CELL_PERSONS=200"
            ),
            "e9_transitions.earnings_change": (
                "rows per half = consecutive-month transition "
                "pairs, not persons (a person can contribute up to "
                "11) vs THIN_CELL_PERSONS=200"
            ),
        },
        "sipp_jobs_reader_commit": _reader_commit(),
        "e8_nonemployment_by_age": e8_floors(persons),
        "e9_transitions": e9_floors(pairs),
    }


def main() -> None:
    artifact = build()
    ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT}")
    print("e9 transition mix:", artifact["e9_transitions"]["transition_rates"])
    stay = artifact["e9_transitions"]["earnings_change"]["stay"]
    print(
        "stay: median log-change",
        stay["median_log_change"],
        "floor",
        stay["floor_abs_median_gap"]["mean"],
    )


if __name__ == "__main__":
    main()
