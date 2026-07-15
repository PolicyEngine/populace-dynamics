"""Label-verified loaders for the employer-firm target extracts.

Follows the :mod:`populace_dynamics.uk.ons_rates` pattern: the pinned
aggregate extracts are committed at ``data/external/`` (built by
``scripts/fetch_employer_firm_targets.py``; provenance in
``data/external/employer_firm_target_sources.md``), so no download
happens at load time. Each loader validates the expected schema and
basic accounting identities so a silently changed extract fails
loudly.

Unit reminders (provenance note has the details):

* QWI/J2J cells count **jobs**, not persons — a person with two
  employers in a quarter appears twice. Calibrating primary-job-only
  person spells to these cells needs the pre-registered job-to-person
  adjustment (ADR 0003).
* QWI ``EarnS`` is a **mean** monthly earnings measure (QWI never
  publishes medians); rates derived here are per-job per-quarter.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

__all__ = [
    "load_susb_sector_size",
    "load_bds_firm_size",
    "load_qwi_firmsize_sector",
    "load_j2j_firmsize_sector",
]

EXTERNAL_DIR = Path(__file__).resolve().parents[3] / "data" / "external"
SUSB_PATH = EXTERNAL_DIR / "susb_us_sector_size_2022.csv"
BDS_PATH = EXTERNAL_DIR / "bds_us_firm_size_1978_2022.csv"
QWI_PATH = EXTERNAL_DIR / "qwi_us_firmsize_sector_2015on.csv"
J2J_PATH = EXTERNAL_DIR / "j2j_us_firmsize_sector_2015on.csv"

#: SUSB 2022 US total-employment pin (all sectors, ENTRSIZE 01),
#: verified against the published table at fetch time.
SUSB_TOTAL_EMPLOYMENT_2022 = 135_748_407
SUSB_TOTAL_FIRMS_2022 = 6_395_635

#: LEHD detail firm-size codes (1..5; 0 is the all-sizes margin).
LEHD_DETAIL_FIRMSIZES = {1, 2, 3, 4, 5}

_SUSB_COLUMNS = [
    "naics_sector",
    "naics_description",
    "entrsize_code",
    "entrsize_label",
    "firms",
    "establishments",
    "employment",
    "employment_noise_flag",
    "annual_payroll_kusd",
]

_BDS_FSIZE_CATEGORIES = [
    "a) 1 to 4",
    "b) 5 to 9",
    "c) 10 to 19",
    "d) 20 to 99",
    "e) 100 to 499",
    "f) 500 to 999",
    "g) 1000 to 2499",
    "h) 2500 to 4999",
    "i) 5000 to 9999",
    "j) 10000+",
]

#: Full BDS extract column list (verified against the committed
#: extract). Pinned in full — not just the first few — so a corrupted
#: or reordered tail column fails loudly instead of coercing to NaN
#: that reads as expected suppression.
_BDS_COLUMNS = [
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


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Committed extract not found at {path}; run "
            "scripts/fetch_employer_firm_targets.py."
        )
    return pd.read_csv(path, dtype={"entrsize_code": str})


def load_susb_sector_size(path: str | None = None) -> pd.DataFrame:
    """SUSB 2022 US enterprise-size x NAICS-sector table (tidy).

    Includes the Total rows (``naics_sector == "--"``,
    ``entrsize_code == "01"``) and the <20/<500 subtotal classes; use
    :data:`populace_dynamics.firms.banding.SUSB_SUBTOTAL_CODES` to
    drop non-detail rows.

    Returns a fresh copy on each call: the parse/validate work is
    cached, but callers get their own frame so an in-place edit cannot
    leak into a later load.
    """
    return _load_susb_sector_size(path).copy()


@lru_cache(maxsize=1)
def _load_susb_sector_size(path: str | None = None) -> pd.DataFrame:
    df = _read(Path(path) if path is not None else SUSB_PATH)
    if list(df.columns) != _SUSB_COLUMNS:
        raise ValueError(f"SUSB extract columns changed: {list(df.columns)}")
    total = df[(df["naics_sector"] == "--") & (df["entrsize_code"] == "01")]
    if len(total) != 1:
        raise ValueError("SUSB extract lacks a unique US total row.")
    if int(total["employment"].iloc[0]) != SUSB_TOTAL_EMPLOYMENT_2022:
        raise ValueError(
            "SUSB US total employment does not match the pinned value "
            f"{SUSB_TOTAL_EMPLOYMENT_2022}."
        )
    if int(total["firms"].iloc[0]) != SUSB_TOTAL_FIRMS_2022:
        raise ValueError(
            "SUSB US total firm count does not match the pinned value "
            f"{SUSB_TOTAL_FIRMS_2022}."
        )
    return df


def load_bds_firm_size(path: str | None = None) -> pd.DataFrame:
    """BDS economy-wide firm-size time series, 1978-2022.

    Rates (``*_rate`` columns) are DHS rates per 100 employees, as
    published. ``(D)``/``(X)``-style suppression markers, if any ever
    appear, surface as NaN via coercion. Returns a fresh copy on each
    call (see :func:`load_susb_sector_size`).
    """
    return _load_bds_firm_size(path).copy()


@lru_cache(maxsize=1)
def _load_bds_firm_size(path: str | None = None) -> pd.DataFrame:
    df = _read(Path(path) if path is not None else BDS_PATH)
    if list(df.columns) != _BDS_COLUMNS:
        raise ValueError(f"BDS extract columns changed: {list(df.columns)}")
    if sorted(df["fsize"].unique()) != _BDS_FSIZE_CATEGORIES:
        raise ValueError(
            f"BDS fsize categories changed: {sorted(df['fsize'].unique())}"
        )
    if int(df["year"].min()) != 1978 or int(df["year"].max()) != 2022:
        raise ValueError("BDS extract year range changed.")
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_lehd(path: Path, measures: list[str], name: str) -> pd.DataFrame:
    df = _read(path)
    id_cols = ["year", "quarter", "industry", "firmsize", "firmsize_label"]
    missing = [c for c in id_cols + measures if c not in df.columns]
    if missing:
        raise ValueError(f"{name} extract is missing columns {missing}.")
    if set(df["firmsize"].unique()) != LEHD_DETAIL_FIRMSIZES:
        raise ValueError(
            f"{name} firmsize codes changed: "
            f"{sorted(df['firmsize'].unique())}"
        )
    if not df["quarter"].isin([1, 2, 3, 4]).all():
        raise ValueError(f"{name} quarter values out of range.")
    dupes = df.duplicated(["year", "quarter", "industry", "firmsize"])
    if dupes.any():
        raise ValueError(f"{name} extract has duplicate cells.")
    return df


def load_qwi_firmsize_sector(path: str | None = None) -> pd.DataFrame:
    """National QWI by firm size x NAICS sector, 2015Q1 on (jobs).

    All-sex all-age margin only. Adds derived per-job quarterly rates
    ``hire_rate`` (HirA / EmpTotal) and ``separation_rate``
    (Sep / EmpTotal). ``industry == "00"`` is the all-industry margin.
    Returns a fresh copy on each call (see
    :func:`load_susb_sector_size`).
    """
    return _load_qwi_firmsize_sector(path).copy()


@lru_cache(maxsize=1)
def _load_qwi_firmsize_sector(path: str | None = None) -> pd.DataFrame:
    measures = ["Emp", "EmpEnd", "EmpS", "EmpTotal", "HirA", "Sep", "EarnS"]
    df = _load_lehd(
        Path(path) if path is not None else QWI_PATH, measures, "QWI"
    )
    if (df[["Emp", "EmpTotal", "HirA", "Sep"]] < 0).any().any():
        raise ValueError("QWI counts must be non-negative.")
    if (df["EarnS"] <= 0).any():
        raise ValueError("QWI EarnS must be positive.")
    # Missing values are allowed only where LEHD flags them as not
    # computable (flag -1: e.g. separations in the final released
    # quarter need the following quarter's data) or suppressed
    # (flag 5: e.g. preliminary-quarter firm-size cells).
    for m in measures:
        missing = df[m].isna()
        if not df.loc[missing, f"s{m}"].isin([-1, 5]).all():
            raise ValueError(f"QWI {m} has unexplained missing cells.")
    df = df.copy()
    # Guard the denominator (a firm-size x sector cell can have zero
    # total employment); mirrors the J2J loader's ``.where`` convention
    # so an empty cell yields NaN, not inf.
    base = df["EmpTotal"].where(df["EmpTotal"] > 0)
    df["hire_rate"] = df["HirA"] / base
    df["separation_rate"] = df["Sep"] / base
    for col in ("hire_rate", "separation_rate"):
        observed = df[col].dropna()
        if not observed.between(0.0, 1.0).all():
            raise ValueError(f"QWI derived {col} outside [0, 1].")
    return df


def load_j2j_firmsize_sector(path: str | None = None) -> pd.DataFrame:
    """National J2J flows by firm size x NAICS sector, 2015Q1 on.

    No demographic detail. Adds derived per-job quarterly rates
    ``j2j_hire_rate`` (J2JHire / MainB) and ``j2j_separation_rate``
    (J2JSep / MainB). Returns a fresh copy on each call (see
    :func:`load_susb_sector_size`).
    """
    return _load_j2j_firmsize_sector(path).copy()


@lru_cache(maxsize=1)
def _load_j2j_firmsize_sector(path: str | None = None) -> pd.DataFrame:
    measures = [
        "MainB",
        "MainE",
        "MHire",
        "MSep",
        "EEHire",
        "EESep",
        "J2JHire",
        "J2JSep",
        "NEHire",
        "ENSep",
    ]
    df = _load_lehd(
        Path(path) if path is not None else J2J_PATH, measures, "J2J"
    )
    if (df[measures] < 0).any().any():
        raise ValueError("J2J counts must be non-negative.")
    for m in measures:
        missing = df[m].isna()
        if not df.loc[missing, f"s{m}"].isin([-1, 5]).all():
            raise ValueError(f"J2J {m} has unexplained missing cells.")
    df = df.copy()
    base = df["MainB"].where(df["MainB"] > 0)
    df["j2j_hire_rate"] = df["J2JHire"] / base
    df["j2j_separation_rate"] = df["J2JSep"] / base
    for col in ("j2j_hire_rate", "j2j_separation_rate"):
        observed = df[col].dropna()
        if not observed.between(0.0, 1.0).all():
            raise ValueError(f"J2J derived {col} outside [0, 1].")
    return df
