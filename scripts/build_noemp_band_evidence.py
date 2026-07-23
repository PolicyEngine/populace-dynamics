"""Build the NOEMP band-label evidence artifact (issue #192).

REPORTED ANCHOR, NOT A GATE RUN. Like the mortality/claiming/
disability floors, this reads no gate and decides nothing on its
own; it is committed evidence pinned by a reproduction test. It
records the empirical basis for the IC2 banding decision's treatment
of CPS ASEC firm size: **the 2019+ data dictionaries' relabeling of
NOEMP codes 2/3 (from 10-49 / 50-99 to 10-24 / 25-99) never
happened in the instrument.**

Two products:

(a) **The discontinuity test.** Weighted NOEMP code shares among
    ``WKSWORK > 0`` workers from the published public-use files on
    both sides of the supposed 2018/2019 re-binning (2017 converted
    from the fixed-width ``asec2017_pubuse.dat`` at the dictionary
    positions; 2019/2024 from the published CSVs). A genuine re-bin
    of a 39-integer band (10-49) into a 15-integer band (10-24)
    would roughly halve code 2's share and double code 3's; the
    artifact records the observed deltas (under one percentage
    point across seven years).

(b) **The SUSB cross-check.** The administrative enterprise-size
    employment distribution puts 50-99 employment near 7.5% and
    25-99 near 15%; NOEMP code 3's observed ~7% share matches the
    former and excludes the latter. Reference values are recorded
    with their source note, not calibrated against.

Inputs are the staged ASEC person files under
``~/PolicyEngine/asec-data`` (``POPULACE_DYNAMICS_ASEC_DIR``), as
documented in the ASEC reader (#204). Sources: Census ASEC
public-use files and data dictionaries at
www2.census.gov/programs-surveys/cps/datasets; SUSB annual data
tables (2021 employment by enterprise size). Method and finding
first posted to issue #192 (2026-07-15).

Usage::

    python scripts/build_noemp_band_evidence.py

writes ``runs/noemp_band_evidence_v1.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

YEARS = (2017, 2019, 2024)

#: Dictionary labels for codes 2/3 as published per year — the
#: conflicting documentation this artifact adjudicates.
DICTIONARY_LABELS = {
    2017: {"2": "10-49", "3": "50-99"},
    2019: {"2": "10-24", "3": "25-99"},
    2024: {"2": "10-24", "3": "25-99"},
}

#: SUSB 2021 employment shares by enterprise size (annual data
#: tables; reference points only, never calibrated against).
SUSB_REFERENCE = {
    "50_99_share_pct": 7.5,
    "25_99_share_pct": 15.0,
    "note": (
        "SUSB annual tables, 2021 employment by enterprise "
        "employment size; 50-99 near 7.5% of employment, 25-99 "
        "near 15%. NOEMP code 3's observed ~7% share matches the "
        "50-99 reading and excludes the 25-99 reading."
    ),
}

ARTIFACT = Path(__file__).resolve().parents[1] / (
    "runs/noemp_band_evidence_v1.json"
)


def _data_dir() -> Path:
    env = os.environ.get("POPULACE_DYNAMICS_ASEC_DIR")
    if env:
        return Path(env).expanduser()
    return Path("~/PolicyEngine/asec-data").expanduser()


def code_shares(year: int, data_dir: Path) -> dict:
    """Weighted NOEMP code shares among WKSWORK > 0 workers."""
    path = data_dir / f"pppub{year % 100:02d}.csv"
    raw = pd.read_csv(
        path, usecols=lambda c: c in {"NOEMP", "WKSWORK", "MARSUPWT"}
    )
    workers = raw[raw["WKSWORK"] > 0]
    weights = workers.groupby("NOEMP")["MARSUPWT"].sum() / 100.0
    weights = weights[weights.index > 0]
    shares = weights / weights.sum() * 100.0
    return {
        "workers_unweighted": int(len(workers)),
        "workers_weighted_millions": round(weights.sum() / 1e6, 1),
        "share_pct_by_code": {
            str(int(code)): round(float(share), 2)
            for code, share in shares.items()
        },
    }


def build() -> dict:
    data_dir = _data_dir()
    by_year = {str(year): code_shares(year, data_dir) for year in YEARS}

    def share(year: int, code: int) -> float:
        return by_year[str(year)]["share_pct_by_code"][str(code)]

    max_delta = max(
        abs(share(a, code) - share(b, code))
        for code in (2, 3)
        for a, b in ((2017, 2019), (2019, 2024))
    )
    return {
        "artifact": "noemp_band_evidence",
        "version": 1,
        "issue": "192",
        "finding": (
            "The 2019+ ASEC data dictionaries relabel NOEMP codes "
            "2/3 from 10-49/50-99 to 10-24/25-99, but the weighted "
            "code shares are continuous across the supposed "
            "2018/2019 re-binning and match SUSB only under the "
            "10-49/50-99 reading: the relabeling is documentary, "
            "not an instrument change. IPUMS FIRMSIZE inherits the "
            "mislabel for 2019+ samples."
        ),
        "dictionary_labels_by_year": DICTIONARY_LABELS,
        "share_pct_by_year": by_year,
        "max_adjacent_delta_pct_codes_2_3": round(max_delta, 2),
        "expected_delta_if_rebinned_pct": (
            "code 2 roughly halves (~7 pp drop) and code 3 roughly "
            "doubles (~7 pp rise)"
        ),
        "susb_reference": SUSB_REFERENCE,
        "sources": {
            "asec_files": (
                "www2.census.gov/programs-surveys/cps/datasets/"
                "{year}/march (2017 converted from "
                "asec2017_pubuse.dat at the dictionary positions; "
                "2019/2024 from the published person CSVs)"
            ),
            "first_reported": (
                "PolicyEngine/populace-dynamics#192, 2026-07-15"
            ),
        },
    }


def main() -> None:
    artifact = build()
    ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT}")
    print(
        "max adjacent delta on codes 2/3:",
        artifact["max_adjacent_delta_pct_codes_2_3"],
        "pp",
    )


if __name__ == "__main__":
    main()
