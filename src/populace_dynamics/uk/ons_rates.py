"""ONS mortality and fertility rate loaders for the UK panel pipeline.

Ported from archived
`policyengine-uk-data#346 <https://github.com/PolicyEngine/policyengine-uk-data/pull/346>`_
(``targets/sources/ons_mortality.py`` + ``ons_fertility.py``) per
populace#148 / ADR 0002. The pinned ONS workbooks are committed at
``data/external/`` (public ONS data), so no download happens at load
time; the pinned source URLs are recorded here for provenance and
refresh.

- **Mortality**: single-year ``qx`` from the ONS UK National Life
  Tables (3-year rolling periods, 1980-2023 release), by age x sex.
- **Fertility**: age-specific fertility rates from Table 10 of the
  ONS *Births in England and Wales: registrations* workbook (per
  1,000 women by 5-year band, expanded uniformly to single-year ages
  and converted to per-woman-per-year probabilities).

Both feed :func:`populace_dynamics.uk.demographic_ageing.age_dataset`
directly.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

__all__ = [
    "load_ons_life_tables",
    "get_mortality_rates",
    "get_mortality_rates_unisex",
    "load_ons_fertility_rates",
    "get_fertility_rates",
]

EXTERNAL_DIR = Path(__file__).resolve().parents[3] / "data" / "external"
LIFE_TABLES_PATH = EXTERNAL_DIR / "ons_national_life_tables.xlsx"
ASFR_PATH = EXTERNAL_DIR / "ons_asfr.xlsx"

# Pinned source URLs (provenance; refresh is an intentional,
# reviewable step):
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/datasets/nationallifetablesunitedkingdomreferencetables
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/livebirths/datasets/birthsinenglandandwalesbirthregistrations

MALE = "MALE"
FEMALE = "FEMALE"

_ASFR_SHEET = "Table_10"
_ASFR_HEADER_ROW = 5  # zero-indexed

DEFAULT_COUNTRY = "England, Wales and Elsewhere"
MOTHER_PARENT = "Mother"

# Open-ended bands ("40 and over") are capped at low + span - 1 so an
# open band does not massively over-state ASFR at 45+ (40-44 accounts
# for ~95% of 40+ births in recent UK data).
_OPEN_BAND_SPAN = 5


# ---------------------------------------------------------------------
# Mortality — ONS National Life Tables
# ---------------------------------------------------------------------


def _parse_period_sheet(raw: pd.DataFrame, period: str) -> pd.DataFrame:
    """Parse one rolling-period sheet into long-format (age, sex, qx).

    Each period sheet has, after a header block, two side-by-side
    tables: columns 0-5 are Males (age, mx, qx, lx, dx, ex), column 6
    a blank separator, columns 7-12 Females with the same fields.
    """
    ages = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    mask = ages.notna() & ages.between(0, 120)
    data = raw[mask].reset_index(drop=True)
    if data.empty:
        raise ValueError(f"Sheet {period!r} contained no numeric age rows.")

    male = pd.DataFrame(
        {
            "period": period,
            "sex": MALE,
            "age": pd.to_numeric(data.iloc[:, 0]).astype(int),
            "qx": pd.to_numeric(data.iloc[:, 2], errors="coerce"),
        }
    )
    female = pd.DataFrame(
        {
            "period": period,
            "sex": FEMALE,
            "age": pd.to_numeric(data.iloc[:, 7]).astype(int),
            "qx": pd.to_numeric(data.iloc[:, 9], errors="coerce"),
        }
    )
    long = pd.concat([male, female], ignore_index=True)
    return long.dropna(subset=["qx"]).reset_index(drop=True)


def _iter_period_sheets(xls: pd.ExcelFile):
    """Yield sheet names that look like rolling periods, e.g. 2022-2024."""
    for name in xls.sheet_names:
        parts = name.strip().split("-")
        if len(parts) != 2:
            continue
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if 1900 < start <= end < 2100:
            yield name, start, end


@lru_cache(maxsize=1)
def load_ons_life_tables(path: str | None = None) -> pd.DataFrame:
    """Return every ONS NLT rolling period in one long-format frame.

    Columns: ``period`` (``"YYYY-YYYY"``), ``period_start``,
    ``period_end``, ``sex`` (``"MALE"``/``"FEMALE"``), ``age``
    (0-100), ``qx``.
    """
    xlsx_path = Path(path) if path is not None else LIFE_TABLES_PATH
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"ONS National Life Tables workbook not found at {xlsx_path}."
        )

    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    frames = []
    for name, start, end in _iter_period_sheets(xls):
        raw = pd.read_excel(
            xls, sheet_name=name, header=None, engine="openpyxl"
        )
        long = _parse_period_sheet(raw, name)
        long["period_start"] = start
        long["period_end"] = end
        frames.append(long)

    if not frames:
        raise ValueError(
            "ONS NLT workbook contained no recognisable period sheets."
        )

    return pd.concat(frames, ignore_index=True)


def _resolve_period(tables: pd.DataFrame, year: int | str | None) -> str:
    """Return the period covering ``year``: exact match preferred,
    else the most recent period ending before it; ``None`` picks the
    most recent overall."""
    if year is None:
        latest = tables.sort_values("period_end").iloc[-1]
        return str(latest["period"])

    if isinstance(year, str) and "-" in year:
        if (tables["period"] == year).any():
            return year
        raise KeyError(f"Period {year!r} not present in ONS NLT.")

    y = int(year)
    covers = tables[
        (tables["period_start"] <= y) & (tables["period_end"] >= y)
    ]
    if not covers.empty:
        return str(covers.sort_values("period_end").iloc[-1]["period"])

    earlier = tables[tables["period_end"] < y]
    if not earlier.empty:
        return str(earlier.sort_values("period_end").iloc[-1]["period"])

    raise KeyError(
        f"No ONS NLT period covers {y}; earliest is "
        f"{int(tables['period_start'].min())}."
    )


def get_mortality_rates(
    year: int | str | None = None,
    *,
    tables: pd.DataFrame | None = None,
) -> dict[str, dict[int, float]]:
    """Return ``{sex: {age: qx}}`` for the period covering ``year``.

    ``year`` may be a calendar year (2024), an explicit period label
    (``"2022-2024"``), or ``None`` for the most recent period. Age
    keys run 0-100 inclusive.
    """
    if tables is None:
        tables = load_ons_life_tables()
    period = _resolve_period(tables, year)
    sub = tables[tables["period"] == period]
    out: dict[str, dict[int, float]] = {MALE: {}, FEMALE: {}}
    for sex, group in sub.groupby("sex"):
        out[str(sex)] = {
            int(a): float(q)
            for a, q in zip(group["age"], group["qx"], strict=True)
        }
    return out


def get_mortality_rates_unisex(
    year: int | str | None = None,
    *,
    male_share: float = 0.5,
    tables: pd.DataFrame | None = None,
) -> dict[int, float]:
    """Return ``{age: qx}`` averaged across sexes."""
    rates = get_mortality_rates(year, tables=tables)
    ages = sorted(set(rates[MALE]) | set(rates[FEMALE]))
    return {
        a: male_share * rates[MALE].get(a, 0.0)
        + (1 - male_share) * rates[FEMALE].get(a, 0.0)
        for a in ages
    }


# ---------------------------------------------------------------------
# Fertility — ONS ASFR (births registrations, Table 10)
# ---------------------------------------------------------------------


def _parse_age_band(label: str) -> tuple[int, int] | None:
    """Inclusive (low, high) single-year ages for an ONS band label.

    Unknown labels (e.g. "All ages") return ``None``.
    """
    s = str(label).strip()
    if not s or s.lower() in {"all ages", "nan"}:
        return None

    m = re.match(r"under\s*(\d+)", s, flags=re.IGNORECASE)
    if m:
        high = int(m.group(1)) - 1
        # Fertility is negligible below 15; constrain to the
        # conventional start of the fertility window.
        return (15, high)

    m = re.match(r"(\d+)\s*to\s*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    m = re.match(r"(\d+)\s*and\s*over", s, flags=re.IGNORECASE)
    if m:
        low = int(m.group(1))
        return (low, low + _OPEN_BAND_SPAN - 1)

    return None


@lru_cache(maxsize=1)
def load_ons_fertility_rates(path: str | None = None) -> pd.DataFrame:
    """Return a long-format frame of ASFR by year / country / band.

    Columns: ``year``, ``country``, ``age_low``, ``age_high``,
    ``rate_per_1000``. Only ``Parent == "Mother"`` rows are kept.
    """
    xlsx_path = Path(path) if path is not None else ASFR_PATH
    if not xlsx_path.exists():
        raise FileNotFoundError(f"ONS ASFR workbook not found at {xlsx_path}.")

    raw = pd.read_excel(
        xlsx_path,
        sheet_name=_ASFR_SHEET,
        header=_ASFR_HEADER_ROW,
        engine="openpyxl",
    )

    rate_col = next(
        (c for c in raw.columns if "fertility rate" in str(c).lower()),
        None,
    )
    if rate_col is None:
        raise ValueError(
            f"Could not locate the ASFR column in {xlsx_path.name!r} "
            f"(header row {_ASFR_HEADER_ROW}). Columns: {list(raw.columns)}"
        )

    mothers = raw[raw["Parent"] == MOTHER_PARENT].copy()
    mothers["_band"] = mothers["Age group (years)"].map(_parse_age_band)
    mothers = mothers.dropna(subset=["_band"])

    low = mothers["_band"].map(lambda t: t[0]).astype(int)
    high = mothers["_band"].map(lambda t: t[1]).astype(int)
    rate = pd.to_numeric(mothers[rate_col], errors="coerce")

    out = pd.DataFrame(
        {
            "year": pd.to_numeric(mothers["Year"], errors="coerce").astype(
                "Int64"
            ),
            "country": mothers["Country"].astype(str),
            "age_low": low.values,
            "age_high": high.values,
            "rate_per_1000": rate.values,
        }
    )
    out = out.dropna(subset=["year", "rate_per_1000"]).reset_index(drop=True)
    out["year"] = out["year"].astype(int)
    return out


def _resolve_year(tables: pd.DataFrame, year: int | None) -> int:
    """Best available year: ``None`` picks the latest; a missing
    explicit year falls back to the latest earlier year."""
    if year is None:
        return int(tables["year"].max())

    y = int(year)
    available = set(tables["year"].astype(int))
    if y in available:
        return y

    earlier = [a for a in available if a < y]
    if earlier:
        return max(earlier)

    raise KeyError(
        f"No ONS ASFR data for {y}; earliest available is "
        f"{min(available)}."
    )


def get_fertility_rates(
    year: int | None = None,
    *,
    country: str = DEFAULT_COUNTRY,
    tables: pd.DataFrame | None = None,
) -> dict[int, float]:
    """Return ``{age: probability}`` for a single calendar year.

    Expands each 5-year band uniformly to single-year ages and
    converts per-1,000 rates to per-woman-per-year probabilities.
    """
    if tables is None:
        tables = load_ons_fertility_rates()
    picked_year = _resolve_year(tables, year)

    sub = tables[
        (tables["year"] == picked_year) & (tables["country"] == country)
    ]
    if sub.empty:
        raise KeyError(
            f"No ASFR rows for country={country!r} in {picked_year}. "
            f"Available: {sorted(tables['country'].unique())}."
        )

    # Narrower bands overwrite wider overlapping ones (e.g. "40 to 44"
    # over "40 and over") via ascending iteration order.
    sub_narrow = sub.sort_values(by="age_high", ascending=True)

    out: dict[int, float] = {}
    for _, row in sub_narrow.iterrows():
        lo, hi = int(row["age_low"]), int(row["age_high"])
        rate = float(row["rate_per_1000"]) / 1_000.0
        for age in range(lo, hi + 1):
            out[age] = rate
    return out
