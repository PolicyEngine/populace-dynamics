"""CPS January job-tenure supplement records (issue #198).

The biennial Displaced Worker / Employee Tenure / Occupational
Mobility supplement asks "How long have you been working
CONTINUOUSLY ... for your present employer?" — the **current job at
the January interview**. That reference period deliberately differs
from ASEC firm size (the *preceding calendar year's longest job*);
the mismatch is pre-registered in ADR 0003 and the supplement asks
no firm-size question, so this loader supplies the tenure margin
only. It is the reference data for gate E3 (tenure P25/P50/P75 by
age band).

The primary variable is Census's edited recode ``PTST1TN`` —
"recode for employer tenure, expressed in years, with two implied
decimals", built from the raw items ``PEST1A``/``PEST1B``/``PEST3``
with Census's own usability rules (amount 1-99, exact months 1-35,
and **age minus tenure >= 14 years**), topcoded at 3100 (31.00
years). Negative codes are nonresponse: -1 NIU, -2 don't know,
-3 refused, -9 no response. Using the recode rather than
reconstructing from the raw trio is deliberate: the edit rules live
with Census, not with us. The supplement weight is ``PWTENWGT``
(four implied decimals) — not the basic monthly weight and not
``PWSUPWGT`` (the displaced-worker weight).

Supported years and dictionary provenance (each verified from the
Census technical documentation, read 2026-07-14): 2020, 2022, 2024
via ``cpsjan20.pdf`` / ``cpsjan22.pdf`` / ``cpsjan24.pdf`` at
www2.census.gov/programs-surveys/cps/techdocs — identical PTST1TN
semantics in all three. These are the supplement years Census
publishes natively as CSV (``jan{yy}pub.csv``); 2018 and earlier
exist only as fixed-width ``.dat`` and enter later via a documented
conversion, the same posture as pre-2017 ASEC in the NOEMP loader.
The 1983-2018 historical series has wording/bracket breaks and is a
separate, documented extension (the ``family.py`` pre-1994
precedent).

Two traps documented for the next reader:

* ``HETENURE``/``HXTENURE`` on the same file are *housing* tenure
  (own/rent), not job tenure.
* ``HRHHID`` is a 15-digit household id: like ASEC ``PERIDNUM`` it
  must never pass through a numeric dtype (int64 overflow /
  float64 rounding), so all id components read as strings.

Staging: raw supplement person files stay out of the repository,
staged as published (``jan{yy}pub.csv`` or ``.csv.gz``) under
``~/PolicyEngine/cps-tenure-data``, overridable via the
``POPULACE_DYNAMICS_CPS_TENURE_DIR`` environment variable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "CPS_TENURE_YEARS",
    "DEFAULT_AGE_BANDS",
    "PEIO1COW_LABELS",
    "PTST1TN_NONRESPONSE",
    "PTST1TN_TOPCODE",
    "read_cps_tenure",
    "tenure_tabulation",
]

#: Supplement years whose technical documentation the reader was
#: verified against; anything else is refused.
CPS_TENURE_YEARS: tuple[int, ...] = (2020, 2022, 2024)

#: PTST1TN nonresponse codes (dictionary: NIU / don't know /
#: refused / no response).
PTST1TN_NONRESPONSE: dict[int, str] = {
    -1: "niu",
    -2: "dont_know",
    -3: "refused",
    -9: "no_response",
}

#: PTST1TN topcode: 3100 = 31.00 years.
PTST1TN_TOPCODE = 3100

#: Census's usability rule baked into the PTST1TN recode.
_MIN_AGE_MINUS_TENURE = 14

#: PEIO1COW (class of worker, main job) code -> label, from the
#: January 2024 dictionary.
PEIO1COW_LABELS: dict[int, str] = {
    1: "federal",
    2: "state",
    3: "local",
    4: "private_for_profit",
    5: "private_nonprofit",
    6: "self_employed_incorporated",
    7: "self_employed_unincorporated",
    8: "without_pay",
}

#: Default E3 age bands (BLS tenure-release convention).
DEFAULT_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (16, 19),
    (20, 24),
    (25, 34),
    (35, 44),
    (45, 54),
    (55, 64),
    (65, 200),
)

#: Raw person-file columns the reader requires. Id components are
#: strings (see module docstring).
_REQUIRED_COLUMNS = (
    "HRHHID",
    "HRHHID2",
    "PULINENO",
    "PTST1TN",
    "PWTENWGT",
    "PRTAGE",
    "PESEX",
    "GESTFIPS",
    "PEMLR",
    "PEIO1COW",
)

_STRING_COLUMNS = {"HRHHID": "string", "HRHHID2": "string"}

_DATA_DIR_ENV = "POPULACE_DYNAMICS_CPS_TENURE_DIR"
_DEFAULT_DATA_DIR = Path("~/PolicyEngine/cps-tenure-data").expanduser()

_README_POINTER = (
    "stage Census CPS January supplement person files as "
    "jan{yy}pub.csv[.gz] under ~/PolicyEngine/cps-tenure-data "
    "(or POPULACE_DYNAMICS_CPS_TENURE_DIR)"
)

#: A jan{yy}pub filename encodes its supplement year; used to refuse
#: a path/year mismatch.
_JAN_YEAR_RE = re.compile(r"jan(\d{2})pub\.csv(\.gz)?$", re.I)


def _resolve_data_dir(data_dir: Path | None) -> Path:
    """Resolve the staging directory from arg, env var, default."""
    if data_dir is not None:
        return Path(data_dir).expanduser()
    env_value = os.environ.get(_DATA_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser()
    return _DEFAULT_DATA_DIR


def _resolve_person_path(year: int, data_dir: Path) -> Path:
    """Locate the staged supplement file for ``year``."""
    stem = f"jan{year % 100:02d}pub"
    for suffix in (".csv", ".csv.gz"):
        candidate = data_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {stem}.csv[.gz] under {data_dir}; {_README_POINTER}."
    )


def _check_year(year: int) -> None:
    if year not in CPS_TENURE_YEARS:
        raise ValueError(
            f"No dictionary-verified tenure-supplement map for "
            f"{year}; supported years are {CPS_TENURE_YEARS}. The "
            "supplement is biennial (even years) and this reader "
            "accepts only years whose technical documentation was "
            "verified — extend it with the year's cpsjan{yy}.pdf "
            "in hand."
        )


def _check_path_year(path: Path, year: int) -> None:
    """Refuse a jan{yy}pub filename that contradicts ``year``."""
    match = _JAN_YEAR_RE.search(path.name)
    if match is None:
        return
    file_year = 2000 + int(match.group(1))
    if file_year != year:
        raise ValueError(
            f"{path.name} is a January {file_year} supplement file "
            f"but year={year} was requested."
        )


def _domain_error(year: int, column: str, bad: pd.Series) -> ValueError:
    values = ", ".join(sorted(map(str, bad.unique()))[:8])
    return ValueError(
        f"CPS January {year} {column} contains out-of-dictionary "
        f"value(s) [{values}] on {len(bad)} row(s); refusing a file "
        "that does not match the year's technical documentation."
    )


def read_cps_tenure(
    year: int,
    *,
    path: str | Path | None = None,
    data_dir: str | Path | None = None,
    include_nonresponse: bool = False,
) -> pd.DataFrame:
    """Read one January supplement's employer-tenure records.

    Args:
        year: Supplement year (biennial; the tenure question refers
            to the current job at the January interview).
        path: Explicit person-file CSV. A ``jan{yy}pub`` filename
            must agree with ``year``.
        data_dir: Staging directory (explicit argument, then
            ``POPULACE_DYNAMICS_CPS_TENURE_DIR``, then
            ``~/PolicyEngine/cps-tenure-data``).
        include_nonresponse: When True, also return rows whose
            ``PTST1TN`` is a nonresponse code, with
            ``tenure_years`` NaN and ``response`` naming the code —
            so exclusions are countable, never silent. Default
            False returns usable responses only.

    Returns:
        One row per person with a usable tenure response (plus
        nonresponse rows when requested): ``person_id`` (string,
        HRHHID+HRHHID2+line), ``year``, ``tenure_years``
        (``PTST1TN / 100``), ``tenure_topcoded`` (31.00-year
        topcode), ``response``, ``age``, ``sex``, ``state_fips``,
        ``pemlr``, ``class_of_worker_code``, ``class_of_worker``,
        and ``weight`` (``PWTENWGT / 10_000``, persons).

    Raises:
        ValueError: On an unsupported year, path/year mismatch,
            missing columns, out-of-dictionary domains, a usable
            tenure response outside the employed universe
            (``PEMLR`` 1-2), or a violation of the recode's
            age-minus-tenure >= 14 rule.
        FileNotFoundError: If no staged file can be found.
    """
    _check_year(year)
    if path is not None:
        person_path = Path(path).expanduser()
        if not person_path.exists():
            raise FileNotFoundError(
                f"CPS January person file not found: {person_path}"
            )
    else:
        person_path = _resolve_person_path(year, _resolve_data_dir(data_dir))
    _check_path_year(person_path, year)

    required = set(_REQUIRED_COLUMNS)
    try:
        raw = pd.read_csv(
            person_path,
            usecols=lambda column: column in required,
            dtype=_STRING_COLUMNS,
        )
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"{person_path.name} is empty; {_README_POINTER}."
        ) from None
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(
            f"{person_path.name} is missing required column(s) "
            f"{missing}; expected original Census CPS person-file "
            "variable names."
        )

    # The id components get the same refuse-on-mismatch treatment as
    # the coded columns: a blank id would concatenate into a NaN
    # person_id that vanishes from downstream groupbys.
    for column in ("HRHHID", "HRHHID2"):
        ids = raw[column]
        blank = ids.isna() | (ids.astype(str).str.strip() == "")
        if blank.any():
            raise ValueError(
                f"CPS January {year}: {int(blank.sum())} row(s) have "
                f"a blank {column}; refusing a file with unusable "
                "person ids."
            )

    valid_tenure = set(PTST1TN_NONRESPONSE) | {*range(0, PTST1TN_TOPCODE + 1)}
    for column, check, cast in (
        ("PTST1TN", lambda v: ~v.isin(sorted(valid_tenure)), int),
        ("PWTENWGT", lambda v: (v < 0) | ~np.isfinite(v), float),
        ("PULINENO", lambda v: (v < 1) | (v > 99), int),
        ("PRTAGE", lambda v: (v < 0) | (v > 90), int),
        ("PESEX", lambda v: ~v.isin((1, 2)), int),
        ("GESTFIPS", lambda v: (v < 1) | (v > 56), int),
        ("PEMLR", lambda v: ~v.isin((-1, 1, 2, 3, 4, 5, 6, 7)), int),
        ("PEIO1COW", lambda v: ~v.isin((-1, *PEIO1COW_LABELS)), int),
    ):
        values = pd.to_numeric(raw[column], errors="coerce")
        bad = raw[column][
            values.isna()
            | check(values)
            | ((values != values.round()) if cast is int else False)
        ]
        if len(bad):
            raise _domain_error(year, column, bad)
        raw[column] = values.astype(cast)

    duplicated = raw.duplicated(["HRHHID", "HRHHID2", "PULINENO"])
    if duplicated.any():
        raise ValueError(
            f"CPS January {year}: {int(duplicated.sum())} duplicated "
            "person id(s) (HRHHID+HRHHID2+PULINENO); refusing a file "
            "with a corrupted person-id column."
        )

    usable = raw["PTST1TN"] >= 0

    off_universe = raw[usable & ~raw["PEMLR"].isin((1, 2))]
    if len(off_universe):
        raise ValueError(
            f"CPS January {year}: {len(off_universe)} row(s) carry "
            "a usable PTST1TN outside the employed universe "
            "(PEMLR 1-2); the file does not match the recode's "
            "universe."
        )

    tenure_years = raw["PTST1TN"].where(usable) / 100.0
    age_minus_tenure = raw["PRTAGE"] - tenure_years
    violations = raw[usable & (age_minus_tenure < _MIN_AGE_MINUS_TENURE)]
    if len(violations):
        raise ValueError(
            f"CPS January {year}: {len(violations)} row(s) violate "
            "the recode's age-minus-tenure >= "
            f"{_MIN_AGE_MINUS_TENURE} usability rule; the file does "
            "not match the year's technical documentation."
        )

    keep = raw if include_nonresponse else raw[usable]
    keep = keep.reset_index(drop=True)
    keep_usable = keep["PTST1TN"] >= 0
    return pd.DataFrame(
        {
            "person_id": (
                keep["HRHHID"].astype(str)
                + "-"
                + keep["HRHHID2"].astype(str)
                + "-"
                + keep["PULINENO"].astype(str)
            ),
            "year": year,
            "tenure_years": keep["PTST1TN"].where(keep_usable) / 100.0,
            "tenure_topcoded": keep["PTST1TN"] == PTST1TN_TOPCODE,
            "response": np.where(
                keep_usable,
                "usable",
                keep["PTST1TN"].map(PTST1TN_NONRESPONSE),
            ),
            "age": keep["PRTAGE"],
            "sex": keep["PESEX"],
            "state_fips": keep["GESTFIPS"],
            "pemlr": keep["PEMLR"],
            "class_of_worker_code": keep["PEIO1COW"],
            "class_of_worker": keep["PEIO1COW"].map(
                lambda code: PEIO1COW_LABELS.get(int(code), "niu")
            ),
            "weight": keep["PWTENWGT"] / 10_000.0,
        }
    )


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: float
) -> float:
    """Weighted quantile with linear interpolation on cumulative
    weight midpoints (deterministic, matching common survey
    practice). Zero-weight observations must be excluded by the
    caller: as interpolation knots they would distort the path."""
    if weights.sum() <= 0:
        raise ValueError("weighted quantile needs positive weight")
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cum = np.cumsum(weights) - 0.5 * weights
    cum /= weights.sum()
    return float(np.interp(q, cum, values))


def tenure_tabulation(
    records: pd.DataFrame,
    age_bands: tuple[tuple[int, int], ...] = DEFAULT_AGE_BANDS,
    by: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Weighted tenure quantiles by age band — the E3 evidence.

    Args:
        records: Output of :func:`read_cps_tenure`. Nonresponse rows
            (NaN ``tenure_years``), zero-weight rows (supplement
            nonrespondents carry weight 0, and a zero-weight knot
            would distort the interpolated quantiles), and ages
            outside every band are all excluded — so
            ``unweighted_n`` sums can be smaller than
            ``len(records)`` by design.
        age_bands: Inclusive ``(lo, hi)`` age bands. Bands must be
            sorted and non-overlapping; gaps are allowed (ages in a
            gap are excluded).
        by: Extra grouping columns (e.g. ``("sex",)`` or
            ``("class_of_worker",)``) on top of year and age band.

    Returns:
        One row per year x age band (x ``by``) with ``p25``,
        ``p50``, ``p75`` of ``tenure_years``, ``weighted_persons``,
        ``unweighted_n``, and ``topcoded_share``. Empty input
        yields an empty frame with these columns.

    Raises:
        ValueError: On missing columns or malformed ``age_bands``
            (reversed or overlapping).
    """
    needed = {"year", "tenure_years", "age", "weight", "tenure_topcoded"}
    missing = sorted((needed | set(by)) - set(records.columns))
    if missing:
        raise ValueError(
            f"records is missing column(s) {missing}; pass the "
            "output of read_cps_tenure."
        )
    for lo, hi in age_bands:
        if lo > hi:
            raise ValueError(f"age band ({lo}, {hi}) is reversed.")
    intervals = pd.IntervalIndex.from_tuples(list(age_bands), closed="both")
    if intervals.is_overlapping:
        raise ValueError(f"age bands {age_bands} overlap.")
    usable = records[
        records["tenure_years"].notna() & (records["weight"] > 0)
    ].copy()
    cuts = pd.cut(usable["age"], bins=intervals)
    usable["age_band"] = cuts.map(
        lambda i: f"{int(i.left)}_{int(i.right)}", na_action="ignore"
    )
    usable = usable[usable["age_band"].notna()]

    rows = []
    group_columns = ["year", "age_band", *by]
    for keys, group in usable.groupby(group_columns, sort=True, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = group["tenure_years"].to_numpy(dtype=float)
        weights = group["weight"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "p25": _weighted_quantile(values, weights, 0.25),
                "p50": _weighted_quantile(values, weights, 0.50),
                "p75": _weighted_quantile(values, weights, 0.75),
                "weighted_persons": float(weights.sum()),
                "unweighted_n": int(len(group)),
                "topcoded_share": float(
                    (weights * group["tenure_topcoded"].to_numpy()).sum()
                    / weights.sum()
                ),
            }
        )
    columns = [
        "year",
        "age_band",
        *by,
        "p25",
        "p50",
        "p75",
        "weighted_persons",
        "unweighted_n",
        "topcoded_share",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]
