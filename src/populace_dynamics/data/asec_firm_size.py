"""CPS ASEC firm-size (NOEMP) records for the employer extension.

NOEMP is the worker-reported count of employees "at all locations
where this employer operates" for the **longest job held last
calendar year** (universe ``WKSWORK > 0``), and is the plan's
designated firm-size training label (issue #192; loader scope in
issue #193). The variable keeps the identical code domain ``0:6`` in
every year while codes 2 and 3 silently change meaning across two
band regimes, so a year-blind read mis-bands those codes with no
error. This reader hard-codes the dictionary-adjudicated map per
year and refuses years it has not verified.

Band regimes, verified against every year's Census public-use data
dictionary at www2.census.gov/programs-surveys/cps/datasets
(read 2026-07-14):

* 2011-2018 (``asec2011_pubuse.dd.txt`` ... ``08ASEC2018_Data_
  Dict_Full.txt``): 1 under 10, **2 = 10-49, 3 = 50-99**,
  4 = 100-499, 5 = 500-999, 6 = 1000+.
* 2019-2025 (``06_ASEC_2019-Data_Dictionary_Full.pdf`` ...
  ``asec2025_ddl_pub_full.pdf``): 1 under 10, **2 = 10-24,
  3 = 25-99**, 4 = 100-499, 5 = 500-999, 6 = 1000+.

Note the 2019+ regime cannot resolve a 50-employee cut (it falls
inside 25-99), while 2011-2018 can — the C2 banding decision on
issue #192 consumes the tabulations this module emits.

Alongside the band, each record carries the fields the calibration
side needs to reason about universes: ``LJCW`` (longest-job class of
worker — SUSB/QWI targets exclude government and self-employment),
longest-job industry (detailed ``INDUSTRY`` and major-group
``WEIND``), ``WKSWORK``, the ``I_NOEMP`` allocation flag, and the
ASEC supplement weight ``MARSUPWT``.

Staging: like the PSID readers, raw microdata stays out of the
repository. Person files are staged as CSVs with original Census
variable names (the published ``asecpub{yy}csv.zip`` person file for
2017+; earlier years converted from the fixed-width ``.dat`` with the
year's ``.dd`` layout) under ``~/PolicyEngine/asec-data`` as
``pppub{yy}.csv`` (or ``.csv.gz``), overridable via the
``POPULACE_DYNAMICS_ASEC_DIR`` environment variable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

__all__ = [
    "ASEC_FIRM_SIZE_YEARS",
    "CLASS_OF_WORKER_LABELS",
    "NOEMP_BANDS_2011_2018",
    "NOEMP_BANDS_2019_PLUS",
    "band_regime",
    "firm_size_tabulation",
    "noemp_band_map",
    "read_asec_firm_size",
]

#: NOEMP code -> band label, 2011-2018 dictionaries (code 0 is NIU).
NOEMP_BANDS_2011_2018: dict[int, str] = {
    1: "under_10",
    2: "10_49",
    3: "50_99",
    4: "100_499",
    5: "500_999",
    6: "1000_plus",
}

#: NOEMP code -> band label, 2019-2025 dictionaries (code 0 is NIU).
NOEMP_BANDS_2019_PLUS: dict[int, str] = {
    1: "under_10",
    2: "10_24",
    3: "25_99",
    4: "100_499",
    5: "500_999",
    6: "1000_plus",
}

#: Survey years whose dictionaries the band maps were verified
#: against; ``noemp_band_map`` refuses anything else.
ASEC_FIRM_SIZE_YEARS: tuple[int, ...] = tuple(range(2011, 2026))

#: LJCW code -> label, from the 2024 dictionary ("longest job class
#: of worker"; 5/6 are self-employed incorporated yes/no-or-farm).
CLASS_OF_WORKER_LABELS: dict[int, str] = {
    1: "private",
    2: "federal",
    3: "state",
    4: "local",
    5: "self_employed_incorporated",
    6: "self_employed_unincorporated",
    7: "without_pay",
}

#: Raw person-file columns the reader requires.
_REQUIRED_COLUMNS = (
    "PERIDNUM",
    "NOEMP",
    "I_NOEMP",
    "LJCW",
    "INDUSTRY",
    "WEIND",
    "WKSWORK",
    "MARSUPWT",
)

_DATA_DIR_ENV = "POPULACE_DYNAMICS_ASEC_DIR"
_DEFAULT_DATA_DIR = Path("~/PolicyEngine/asec-data").expanduser()

_README_POINTER = (
    "stage Census ASEC person files as pppub{yy}.csv[.gz] under "
    "~/PolicyEngine/asec-data (or POPULACE_DYNAMICS_ASEC_DIR)"
)

#: A pppub filename encodes its survey year as two digits; used to
#: refuse a path/year mismatch instead of silently mis-banding.
_PPPUB_YEAR_RE = re.compile(r"pppub(\d{2})\.csv(\.gz)?$", re.I)


def band_regime(year: int) -> str:
    """Return the band-regime key (``"2011_2018"``/``"2019_plus"``).

    Raises:
        ValueError: If ``year`` has no dictionary-verified band map.
    """
    if year not in ASEC_FIRM_SIZE_YEARS:
        raise ValueError(
            f"No dictionary-verified NOEMP band map for ASEC {year}; "
            f"supported years are {ASEC_FIRM_SIZE_YEARS[0]}-"
            f"{ASEC_FIRM_SIZE_YEARS[-1]}. Extend the module only "
            "with that year's Census data dictionary in hand."
        )
    return "2011_2018" if year <= 2018 else "2019_plus"


def noemp_band_map(year: int) -> dict[int, str]:
    """Return the NOEMP code -> band label map for a survey year."""
    if band_regime(year) == "2011_2018":
        return dict(NOEMP_BANDS_2011_2018)
    return dict(NOEMP_BANDS_2019_PLUS)


def _resolve_data_dir(data_dir: Path | None) -> Path:
    """Resolve the ASEC data directory from arg, env var, default."""
    if data_dir is not None:
        return Path(data_dir).expanduser()
    env_value = os.environ.get(_DATA_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser()
    return _DEFAULT_DATA_DIR


def _resolve_person_path(year: int, data_dir: Path) -> Path:
    """Locate the staged person file for ``year`` under ``data_dir``."""
    stem = f"pppub{year % 100:02d}"
    for suffix in (".csv", ".csv.gz"):
        candidate = data_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {stem}.csv[.gz] under {data_dir}; {_README_POINTER}."
    )


def _check_path_year(path: Path, year: int) -> None:
    """Refuse a pppub filename whose year digits contradict ``year``."""
    match = _PPPUB_YEAR_RE.search(path.name)
    if match is None:
        return
    file_year = 2000 + int(match.group(1))
    if file_year != year:
        raise ValueError(
            f"{path.name} is an ASEC {file_year} person file but "
            f"year={year} was requested; the NOEMP band regimes "
            "differ across years, so the mismatch would silently "
            "mis-band codes 2-3."
        )


def _domain_error(year: int, column: str, bad: pd.Series) -> ValueError:
    values = ", ".join(str(v) for v in sorted(bad.unique())[:8])
    return ValueError(
        f"ASEC {year} {column} contains out-of-dictionary value(s) "
        f"[{values}] on {len(bad)} row(s); refusing to band a file "
        "that does not match the year's data dictionary."
    )


def read_asec_firm_size(
    year: int,
    *,
    path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Read one ASEC year's longest-job firm-size records.

    Args:
        year: ASEC survey year (the job attributes describe the
            longest job of calendar year ``year - 1``).
        path: Explicit person-file CSV. When the filename carries
            pppub year digits they must agree with ``year``.
        data_dir: Staging directory (default resolution: explicit
            argument, then ``POPULACE_DYNAMICS_ASEC_DIR``, then
            ``~/PolicyEngine/asec-data``).

    Returns:
        One row per person in the NOEMP universe (``WKSWORK > 0``,
        i.e. worked last calendar year), with columns ``person_id``,
        ``year``, ``income_year``, ``band_regime``, ``noemp``,
        ``firm_size_band``, ``noemp_allocated``, ``ljcw``,
        ``class_of_worker``, ``industry_major``,
        ``industry_detailed``, ``wkswork``, and ``weight``
        (``MARSUPWT``, persons).

    Raises:
        ValueError: On an unsupported year, a path/year mismatch,
            missing required columns, or any value outside the
            year's dictionary domain (NOEMP outside 0-6, LJCW
            outside 0-7, or a nonzero NOEMP off the ``WKSWORK > 0``
            universe).
        FileNotFoundError: If no staged person file can be found.
    """
    bands = noemp_band_map(year)
    if path is not None:
        person_path = Path(path).expanduser()
        if not person_path.exists():
            raise FileNotFoundError(
                f"ASEC person file not found: {person_path}"
            )
    else:
        person_path = _resolve_person_path(year, _resolve_data_dir(data_dir))
    _check_path_year(person_path, year)

    raw = pd.read_csv(person_path, usecols=None, low_memory=False)
    missing = sorted(set(_REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(
            f"{person_path.name} is missing required column(s) "
            f"{missing}; expected original Census ASEC person-file "
            "variable names."
        )
    raw = raw.loc[:, list(_REQUIRED_COLUMNS)].copy()

    for column, low, high in (
        ("NOEMP", 0, 6),
        ("LJCW", 0, 7),
        ("WKSWORK", 0, 52),
    ):
        values = pd.to_numeric(raw[column], errors="coerce")
        bad = raw[column][values.isna() | (values < low) | (values > high)]
        if len(bad):
            raise _domain_error(year, column, bad)
        raw[column] = values.astype(int)

    off_universe = raw[(raw["NOEMP"] > 0) & (raw["WKSWORK"] == 0)]
    if len(off_universe):
        raise ValueError(
            f"ASEC {year}: {len(off_universe)} row(s) carry a nonzero "
            "NOEMP outside the WKSWORK > 0 universe; the file does "
            "not match the dictionary's universe statement."
        )

    universe = raw[raw["WKSWORK"] > 0].reset_index(drop=True)
    return pd.DataFrame(
        {
            "person_id": universe["PERIDNUM"].astype(str),
            "year": year,
            "income_year": year - 1,
            "band_regime": band_regime(year),
            "noemp": universe["NOEMP"],
            "firm_size_band": universe["NOEMP"].map(
                lambda code: bands.get(int(code), "niu")
            ),
            "noemp_allocated": pd.to_numeric(universe["I_NOEMP"]) > 0,
            "ljcw": universe["LJCW"],
            "class_of_worker": universe["LJCW"].map(
                lambda code: CLASS_OF_WORKER_LABELS.get(int(code), "niu")
            ),
            "industry_major": pd.to_numeric(universe["WEIND"]).astype(int),
            "industry_detailed": pd.to_numeric(universe["INDUSTRY"]).astype(
                int
            ),
            "wkswork": universe["WKSWORK"],
            "weight": pd.to_numeric(universe["MARSUPWT"]).astype(float),
        }
    )


def firm_size_tabulation(
    records: pd.DataFrame,
    by: tuple[str, ...] = (
        "year",
        "band_regime",
        "firm_size_band",
        "class_of_worker",
    ),
) -> pd.DataFrame:
    """Weighted firm-size tabulation — the C2 evidence artifact.

    Args:
        records: Output of :func:`read_asec_firm_size` (one or more
            years concatenated).
        by: Grouping columns.

    Returns:
        One row per group with ``weighted_persons`` (MARSUPWT sum),
        ``unweighted_n``, and ``allocated_share`` (weighted share of
        the group whose NOEMP was allocated/edited), sorted by the
        grouping columns.
    """
    missing = sorted(
        (set(by) | {"weight", "noemp_allocated"}) - set(records.columns)
    )
    if missing:
        raise ValueError(
            f"records is missing column(s) {missing}; pass the "
            "output of read_asec_firm_size."
        )
    working = records.assign(
        _allocated_weight=records["weight"] * records["noemp_allocated"]
    )
    grouped = working.groupby(list(by), sort=True)
    out = grouped.agg(
        weighted_persons=("weight", "sum"),
        unweighted_n=("weight", "size"),
        _allocated_weight=("_allocated_weight", "sum"),
    ).reset_index()
    out["allocated_share"] = out["_allocated_weight"] / out["weighted_persons"]
    return out.drop(columns="_allocated_weight")
