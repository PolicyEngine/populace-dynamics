"""Fetch employer-firm calibration/gate target extracts (workstream B).

Week-1 target pipeline for the employer-firm extension (issue #192,
`docs/adr/0003-employer-firm-extension.md`): downloads the pinned
public Census/LEHD source files and writes small, tidy aggregate
extracts to ``data/external/`` (never raw microdata; every committed
file stays well under 1 MB). Provenance for each committed extract is
documented in ``data/external/employer_firm_target_sources.md`` and
must be updated whenever this script's pins change.

Sources (all keyless HTTPS GETs of published aggregate files):

* **SUSB 2022** -- US enterprise-size x NAICS-sector employment/firm
  counts, from the annual "U.S. & states, NAICS, detailed employment
  sizes" table. Feeds gate E1 and the phase-0 SUSB calibration.
* **BDS 2022 (fz)** -- economy-wide firm-size time series 1978-2022 of
  establishments, job creation/destruction. Feeds E12's reallocation
  references and phase-2 firm-dynamics shapes.
* **QWI R2026Q1 (us, sa x fs, sector)** -- national hires/separations/
  employment/mean-earnings by firm size x NAICS sector, collapsed to
  the all-sex all-age margin, 2015Q1 on. Feeds E2/E7 calibration
  targets. NOTE: QWI counts *jobs*, not persons (each person-employer
  pair in a quarter is a separate job); see the provenance note.
  The Census data API now requires a key (probed 2026-07-14:
  ``api.census.gov`` 302s to ``missing_key.html``), so the LEHD
  public-use CSVs are the pinned keyless source instead.
* **J2J R2026Q1 (us, no demographics x fs, sector)** -- national
  job-to-job hire/separation flows by firm size x NAICS sector,
  2015Q2 on. Feeds E11's national margin.

Run from the repository root::

    .venv/bin/python scripts/fetch_employer_firm_targets.py

Re-running is idempotent: raw files are cached in a temp directory,
verified against the pinned sha256 digests below, and the extracts
are rewritten deterministically.
"""

from __future__ import annotations

import gzip
import hashlib
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external"

RETRIEVED = "2026-07-14"

#: Pinned raw source files: url -> sha256 of the download performed on
#: RETRIEVED. A digest mismatch means the agency re-issued the file;
#: re-pin deliberately (and update the provenance note).
SOURCES: dict[str, tuple[str, str]] = {
    "susb_us_state_naics_detailedsizes_2022.txt": (
        "https://www2.census.gov/programs-surveys/susb/tables/2022/"
        "us_state_naics_detailedsizes_2022.txt",
        "f07eef547763a7a21cb453bf6f4119f768f5939dc499a65850a532ea" "74489ccf",
    ),
    "bds2022_fz.csv": (
        "https://www2.census.gov/programs-surveys/bds/tables/"
        "time-series/2022/bds2022_fz.csv",
        "61e0e624b00ef3876a503fdaf8c1550d9aaa14070c8962dbd76ec500" "560ae86b",
    ),
    "qwi_us_sa_fs_gn_ns_op_u.csv.gz": (
        "https://lehd.ces.census.gov/data/qwi/latest_release/us/"
        "qwi_us_sa_fs_gn_ns_op_u.csv.gz",
        "76f147615d796eb0ba93401107762e1c67667a25c468b2ee4ca07e07" "d9eabd85",
    ),
    "j2j_us_d_fs_gn_ns_oslp_u.csv.gz": (
        "https://lehd.ces.census.gov/data/j2j/latest_release/us/j2j/"
        "j2j_us_d_fs_gn_ns_oslp_u.csv.gz",
        "abdd573d414d66f864828952501cee0a6ef6c8db88cb789da38af1a3" "c9d55c6f",
    ),
    "label_firmsize.csv": (
        "https://lehd.ces.census.gov/data/schema/latest/" "label_firmsize.csv",
        "29dfd8fed594be600c6c554b4cb27bd590c45da549c30e32824cea42" "48dffe1f",
    ),
}

#: First year kept in the QWI/J2J quarterly extracts (keeps each
#: committed file small while covering the post-2014 SIPP-panel era).
LEHD_START_YEAR = 2015

#: LEHD firm-size code -> label (pinned from label_firmsize.csv).
FIRMSIZE_LABELS = {
    0: "All Firm Sizes",
    1: "0-19 Employees",
    2: "20-49 Employees",
    3: "50-249 Employees",
    4: "250-499 Employees",
    5: "500+ Employees",
}

QWI_MEASURES = [
    "Emp",
    "EmpEnd",
    "EmpS",
    "EmpTotal",
    "HirA",
    "Sep",
    "HirAS",
    "SepS",
    "TurnOvrS",
    "FrmJbGn",
    "FrmJbLs",
    "EarnS",
    "EarnHirAS",
    "EarnSepS",
]

J2J_MEASURES = [
    "MainB",
    "MainE",
    "MHire",
    "MSep",
    "EEHire",
    "EESep",
    "AQHire",
    "AQSep",
    "J2JHire",
    "J2JSep",
    "NEHire",
    "ENSep",
]


def fetch(name: str, cache_dir: Path) -> Path:
    """Download (or reuse) a pinned raw file and verify its sha256."""
    url, expected = SOURCES[name]
    path = cache_dir / name
    if not path.exists():
        print(f"downloading {url}")
        req = urllib.request.Request(
            url, headers={"User-Agent": "populace-dynamics fetch script"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            path.write_bytes(resp.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"{name}: sha256 {digest} != pinned {expected}. The source "
            "was re-issued; re-pin deliberately and update the "
            "provenance note."
        )
    return path


def build_susb(cache_dir: Path) -> None:
    """US x NAICS-sector x detailed enterprise-size class extract."""
    raw = pd.read_csv(
        fetch("susb_us_state_naics_detailedsizes_2022.txt", cache_dir),
        dtype=str,
        encoding="latin-1",
    )
    us = raw[raw["STATE"] == "00"]
    # Sector-level NAICS: 2-digit codes and the published ranges
    # (31-33 etc.), plus the "--" all-sectors total.
    sector = us[us["NAICS"].str.match(r"^(\d\d(-\d\d)?|--)$")].copy()
    out = pd.DataFrame(
        {
            "naics_sector": sector["NAICS"],
            "naics_description": sector["NAICSDSCR"],
            "entrsize_code": sector["ENTRSIZE"],
            "entrsize_label": sector["ENTRSIZEDSCR"]
            .str.replace(r"^\d+:\s*", "", regex=True)
            .str.strip(),
            "firms": pd.to_numeric(sector["FIRM"]),
            "establishments": pd.to_numeric(sector["ESTB"]),
            "employment": pd.to_numeric(sector["EMPL"]),
            "employment_noise_flag": sector["EMPLFL_N"],
            "annual_payroll_kusd": pd.to_numeric(sector["PAYR"]),
        }
    )
    out.to_csv(OUT_DIR / "susb_us_sector_size_2022.csv", index=False)
    print(f"susb_us_sector_size_2022.csv: {len(out)} rows")


def build_bds(cache_dir: Path) -> None:
    """Economy-wide firm-size time series (subset of bds2022_fz)."""
    raw = pd.read_csv(fetch("bds2022_fz.csv", cache_dir))
    cols = [
        "year",
        "fsize",
        "firms",
        "estabs",
        "emp",
        "denom",
        "estabs_entry",
        "estabs_entry_rate",
        "estabs_exit",
        "estabs_exit_rate",
        "job_creation",
        "job_creation_rate",
        "job_destruction",
        "job_destruction_rate",
        "net_job_creation",
        "net_job_creation_rate",
        "reallocation_rate",
        "firmdeath_firms",
        "firmdeath_emp",
    ]
    out = raw[cols].copy()
    out.to_csv(OUT_DIR / "bds_us_firm_size_1978_2022.csv", index=False)
    print(f"bds_us_firm_size_1978_2022.csv: {len(out)} rows")


def _firmsize_label(codes: pd.Series) -> pd.Series:
    return codes.map(FIRMSIZE_LABELS)


def build_qwi(cache_dir: Path) -> None:
    """National QWI by firm size x sector, all-sex all-age margin.

    The raw file is sex x age x firm size; the committed extract keeps
    only the sex=0 (all) / agegrp=A00 (all) margin -- a pure row
    filter, no re-aggregation -- from LEHD_START_YEAR on.
    """
    path = fetch("qwi_us_sa_fs_gn_ns_op_u.csv.gz", cache_dir)
    with gzip.open(path, "rt") as fh:
        raw = pd.read_csv(fh, low_memory=False)
    fs = raw["firmsize"].astype(str)
    keep = raw[
        (raw["sex"] == 0)
        & (raw["agegrp"] == "A00")
        & (raw["year"] >= LEHD_START_YEAR)
        & fs.isin(["1", "2", "3", "4", "5"])
        & (raw["ind_level"].isin(["A", "S"]))  # NAICS sector
    ].copy()
    keep["firmsize"] = keep["firmsize"].astype(int)
    id_cols = ["year", "quarter", "industry", "firmsize"]
    flag_cols = [f"s{m}" for m in QWI_MEASURES]
    out = keep[id_cols + QWI_MEASURES + flag_cols].copy()
    out.insert(4, "firmsize_label", _firmsize_label(out["firmsize"]))
    out = out.sort_values(id_cols).reset_index(drop=True)
    out.to_csv(
        OUT_DIR / "qwi_us_firmsize_sector_2015on.csv",
        index=False,
        float_format="%.10g",
    )
    print(f"qwi_us_firmsize_sector_2015on.csv: {len(out)} rows")


def build_j2j(cache_dir: Path) -> None:
    """National J2J flows by firm size x sector (no demographics).

    NAICS 92 (Public Administration) is dropped: firm size is not
    defined for public-sector employers (LEHD publishes it as "N"),
    and the residual 92 x firm-size cells are single-digit noise.
    This matches the SUSB/QWI private-employer universe.
    """
    path = fetch("j2j_us_d_fs_gn_ns_oslp_u.csv.gz", cache_dir)
    with gzip.open(path, "rt") as fh:
        raw = pd.read_csv(fh, low_memory=False)
    fs = raw["firmsize"].astype(str)
    keep = raw[
        (raw["year"] >= LEHD_START_YEAR)
        & fs.isin(["1", "2", "3", "4", "5"])
        & (raw["ind_level"].isin(["A", "S"]))
        & (raw["industry"].astype(str) != "92")
    ].copy()
    keep["firmsize"] = keep["firmsize"].astype(int)
    id_cols = ["year", "quarter", "industry", "firmsize"]
    flag_cols = [f"s{m}" for m in J2J_MEASURES]
    out = keep[id_cols + J2J_MEASURES + flag_cols].copy()
    out.insert(4, "firmsize_label", _firmsize_label(out["firmsize"]))
    out = out.sort_values(id_cols).reset_index(drop=True)
    out.to_csv(
        OUT_DIR / "j2j_us_firmsize_sector_2015on.csv",
        index=False,
        float_format="%.10g",
    )
    print(f"j2j_us_firmsize_sector_2015on.csv: {len(out)} rows")


def main() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "employer_firm_raw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_susb(cache_dir)
    build_bds(cache_dir)
    build_qwi(cache_dir)
    build_j2j(cache_dir)
    for f in sorted(OUT_DIR.glob("*_us_*2015on.csv")) + [
        OUT_DIR / "susb_us_sector_size_2022.csv",
        OUT_DIR / "bds_us_firm_size_1978_2022.csv",
    ]:
        size = f.stat().st_size
        assert size < 1_000_000, f"{f.name} is {size} bytes (>1 MB)"
        print(f"{f.name}: {size} bytes")


if __name__ == "__main__":
    main()
