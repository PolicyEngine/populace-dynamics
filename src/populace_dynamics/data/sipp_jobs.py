"""SIPP job-level monthly records and IC1-preview spells (issue #200).

The 2014-redesign SIPP public-use files are the employer-firm plan's
primary label panel (#192): one row per person-month (``SSUID`` x
``PNUM`` x ``MONTHCODE``), carrying up to seven job slots
(``EJB1_*`` ... ``EJB7_*``). ``EJB{n}_JOBID`` is "a unique
identifier for a job which is consistent across waves" — the
within-panel employer-attachment key that phase-1 transition hazards
rest on. ``EJB{n}_EMPSIZE`` measures **establishment** size at the
worker's location (the redesign dropped the all-locations question),
so it is the IC2 proxy-chain input, never firm size (ADR 0003;
``firms/banding.py``).

Every variable this reader touches was verified against the Census
SIPP variable API (``api.census.gov/data/{year}/sipp/variables``)
for **2022 and 2023**, with identical labels in both years (read
2026-07-14). The API base does not exist for 2021 and earlier; those
files — and the 2014 panel's wave files — are documented extensions
requiring their own dictionary verification, the same posture as the
ASEC and tenure readers.

Format facts that bite (from Census's own
``2023_sipp_python_input_example.py``): the pu files are
**pipe-delimited**, and Census instructs reading with ``usecols``
because the full file does not fit in memory. ``SSUID`` is a
scrambled composite identifier and reads as a **string** (the ASEC
``PERIDNUM`` dtype lesson); ``TJB{n}_IND``/``TJB{n}_OCC`` are
string-typed in the API schema.

``job_spells`` collapses maximal consecutive-month runs per
(person, job id) into spell rows whose shape mirrors the IC1 spell
schema. It is still labeled **IC1-preview**, but for a narrower
reason than when it was written: ADR 0003 is now Accepted and IC1
is frozen, so what remains preview-grade is this collapse's own
coverage (single ``ref_year`` only — cross-year spell linkage
raises rather than guessing), not the schema's status. The output
also serves as Workstream B's generator for IC1-conforming fixture
files. Attribute changes inside a spell
(class of worker, industry, establishment size) are surfaced via
``attributes_constant`` — never silently averaged.

Staging: raw pu files stay out of the repository, staged as
``pu{year}.csv`` (or ``.csv.gz``) under ``~/PolicyEngine/sipp-data``,
overridable via ``POPULACE_DYNAMICS_SIPP_DIR``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "SIPP_JOB_YEARS",
    "MAX_JOB_SLOTS",
    "CLWRK_LABELS",
    "JBORSE_LABELS",
    "read_sipp_job_months",
    "job_spells",
]

#: File years verified against the Census SIPP variable API.
SIPP_JOB_YEARS: tuple[int, ...] = (2022, 2023)

#: The redesign carries up to seven job slots per person-month.
MAX_JOB_SLOTS = 7

#: EJB{n}_CLWRK code -> label (Census SIPP variable API, 2022/2023).
CLWRK_LABELS: dict[int, str] = {
    1: "federal",
    2: "active_duty_military",
    3: "state",
    4: "local",
    5: "private_for_profit",
    6: "private_nonprofit",
    7: "self_employed_incorporated",
    8: "self_employed_unincorporated",
}

#: EJB{n}_JBORSE code -> label (work-arrangement type).
JBORSE_LABELS: dict[int, str] = {
    1: "employer",
    2: "self_employed",
    3: "other",
}

#: EJB{n}_EMPSIZE valid codes (establishment size; banding semantics
#: live in firms/banding.py, verified on #195).
_EMPSIZE_CODES = frozenset(range(1, 9))

_MISSING = -9
_MISSING_ID = -999

#: Person-month columns required from every file.
_PERSON_COLUMNS = (
    "SSUID",
    "PNUM",
    "MONTHCODE",
    "SWAVE",
    "WPFINWGT",
    "TAGE",
    "ESEX",
)

#: Per-slot column templates.
_SLOT_TEMPLATES = (
    "EJB{n}_JOBID",
    "EJB{n}_BMONTH",
    "EJB{n}_EMONTH",
    "EJB{n}_CLWRK",
    "EJB{n}_JBORSE",
    "EJB{n}_EMPSIZE",
    "TJB{n}_IND",
    "TJB{n}_MSUM",
)

_STRING_COLUMNS = ("SSUID",)

_DATA_DIR_ENV = "POPULACE_DYNAMICS_SIPP_DIR"
_DEFAULT_DATA_DIR = Path("~/PolicyEngine/sipp-data").expanduser()

_README_POINTER = (
    "stage SIPP public-use files as pu{year}.csv[.gz] under "
    "~/PolicyEngine/sipp-data (or POPULACE_DYNAMICS_SIPP_DIR); "
    "the files are pipe-delimited as published"
)

_PU_YEAR_RE = re.compile(r"pu(\d{4})\.csv(\.gz)?$", re.I)


def _slot_columns(slot: int) -> list[str]:
    return [template.format(n=slot) for template in _SLOT_TEMPLATES]


def _resolve_data_dir(data_dir: Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir).expanduser()
    env_value = os.environ.get(_DATA_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser()
    return _DEFAULT_DATA_DIR


def _resolve_pu_path(year: int, data_dir: Path) -> Path:
    for suffix in (".csv", ".csv.gz"):
        candidate = data_dir / f"pu{year}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No pu{year}.csv[.gz] under {data_dir}; {_README_POINTER}."
    )


def _check_year(year: int) -> None:
    if year not in SIPP_JOB_YEARS:
        raise ValueError(
            f"No API-verified SIPP job-variable map for {year}; "
            f"supported file years are {SIPP_JOB_YEARS}. Extend the "
            "reader only with that year's variable metadata "
            "verified (the Census SIPP variable API covers 2022+; "
            "earlier files need their own dictionary pass)."
        )


def _check_path_year(path: Path, year: int) -> None:
    match = _PU_YEAR_RE.search(path.name)
    if match is None:
        return
    file_year = int(match.group(1))
    if file_year != year:
        raise ValueError(
            f"{path.name} is a SIPP {file_year} file but "
            f"year={year} was requested."
        )


def _domain_error(year: int, column: str, bad: pd.Series) -> ValueError:
    values = ", ".join(sorted(map(str, bad.unique()))[:8])
    return ValueError(
        f"SIPP {year} {column} contains out-of-dictionary value(s) "
        f"[{values}] on {len(bad)} row(s); refusing a file that "
        "does not match the API-verified variable metadata."
    )


def _validate_codes(
    year: int,
    frame: pd.DataFrame,
    column: str,
    active: pd.Series,
    allowed: frozenset[int],
    *,
    allow_missing_code: int = _MISSING,
) -> pd.Series:
    """Coerce a coded column, enforcing its domain on active rows.

    Inactive job slots are structurally empty (NaN) and exempt via the
    ``active`` mask. On an active slot the *only* excused non-domain
    values are the explicit ``-9``/``-999`` sentinels; a blank cell and
    an unparseable string both coerce to NaN and must NOT be waved
    through as item nonresponse — they refuse the file, matching the
    person-column path in :func:`read_sipp_job_months`. (Otherwise a
    mistyped or corrupt code on an active job would silently map to
    ``"missing"`` instead of being caught.)
    """
    values = pd.to_numeric(frame[column], errors="coerce")
    sentinel = values.isin((allow_missing_code, _MISSING_ID))
    ok = sentinel | values.isin(sorted(allowed))
    bad = frame[column][active & ~ok]
    if len(bad):
        raise _domain_error(year, column, bad)
    return values


def read_sipp_job_months(
    year: int,
    *,
    path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Read one SIPP file's job-level person-month records.

    Args:
        year: SIPP file year (the reference months are calendar
            year ``year - 1``).
        path: Explicit pu-file path; a ``pu{year}`` filename must
            agree with ``year``.
        data_dir: Staging directory (explicit argument, then
            ``POPULACE_DYNAMICS_SIPP_DIR``, then
            ``~/PolicyEngine/sipp-data``).

    Returns:
        One row per person-month-job (job slots whose
        ``EJB{n}_JOBID`` is present and not -999): ``person_id``
        (``SSUID-PNUM``, string), ``year``, ``ref_year``, ``month``,
        ``wave``, ``job_slot``, ``job_id``, ``bmonth``/``emonth``,
        ``clwrk``/``class_of_worker``, ``jborse``/
        ``work_arrangement``, ``empsize_code`` (establishment
        size — passed through **raw**: valid codes 1-8, the -9
        sentinel for item nonresponse on employer jobs, and NaN for
        self-employment/other arrangements whose establishment size
        is structurally NIU; downstream consumers such as
        ``firms/banding.py`` must expect that mix), ``industry``
        (string, passed through **unvalidated** — a full
        Census-industry allow-list is impractical, so sentinels can
        appear here and flow into ``job_spells``'s modal logic),
        ``earnings`` (monthly, NaN when -999), ``earnings_share``
        and ``top_earner`` within the person-month, ``age``,
        ``sex``, ``weight``.

    Raises:
        ValueError: On an unsupported year, path/year mismatch,
            missing columns, or out-of-dictionary values on active
            job slots (month codes, class-of-worker, work
            arrangement, establishment size, spell edge months,
            non-positive job ids, negative earnings or weights).
        FileNotFoundError: If no staged file can be found.
    """
    _check_year(year)
    if path is not None:
        pu_path = Path(path).expanduser()
        if not pu_path.exists():
            raise FileNotFoundError(f"SIPP pu file not found: {pu_path}")
    else:
        pu_path = _resolve_pu_path(year, _resolve_data_dir(data_dir))
    _check_path_year(pu_path, year)

    header = pd.read_csv(pu_path, sep="|", nrows=0)
    present = set(header.columns)
    missing_person = sorted(set(_PERSON_COLUMNS) - present)
    if missing_person:
        raise ValueError(
            f"{pu_path.name} is missing required column(s) "
            f"{missing_person}; expected the pipe-delimited Census "
            "SIPP public-use layout."
        )
    slots = [
        n for n in range(1, MAX_JOB_SLOTS + 1) if f"EJB{n}_JOBID" in present
    ]
    if not slots:
        raise ValueError(
            f"{pu_path.name} carries no EJB job slots; expected at "
            "least EJB1_JOBID."
        )
    incomplete = {
        column
        for n in slots
        for column in _slot_columns(n)
        if column not in present
    }
    if incomplete:
        raise ValueError(
            f"{pu_path.name} is missing job-slot column(s) "
            f"{sorted(incomplete)} for slots {slots}."
        )

    wanted = set(_PERSON_COLUMNS) | {
        column for n in slots for column in _slot_columns(n)
    }
    raw = pd.read_csv(
        pu_path,
        sep="|",
        usecols=lambda column: column in wanted,
        dtype={column: "string" for column in _STRING_COLUMNS}
        | {f"TJB{n}_IND": "string" for n in slots},
    )

    # Structural keys must identify the person-month; the API lists
    # -9/-999 sentinels even for these, but a row whose identity is
    # missing is unusable, so sentinels here refuse the file. The
    # bounds are deliberately conservative supersets of the published
    # ranges (2023 API: PNUM 101-499, SWAVE 1-4) so that later panel
    # years cannot be false-refused.
    for column, low, high in (
        ("MONTHCODE", 1, 12),
        ("PNUM", 101, 9999),
        ("SWAVE", 1, 99),
    ):
        values = pd.to_numeric(raw[column], errors="coerce")
        bad = raw[column][values.isna() | (values < low) | (values > high)]
        if len(bad):
            raise _domain_error(year, column, bad)
        raw[column] = values.astype(int)

    # Person attributes may carry the API-listed -9/-999 sentinels
    # (the same convention the slot validator tolerates); they map to
    # NA rather than refusing the file. Anything else out of range
    # still refuses.
    for column, low, high in (("TAGE", 0, 120), ("ESEX", 1, 2)):
        values = pd.to_numeric(raw[column], errors="coerce")
        sentinel = values.isin((_MISSING, _MISSING_ID))
        bad = raw[column][
            ~sentinel & (values.isna() | (values < low) | (values > high))
        ]
        if len(bad):
            raise _domain_error(year, column, bad)
        raw[column] = values.where(~sentinel).astype("Int64")
    weights = pd.to_numeric(raw["WPFINWGT"], errors="coerce")
    sentinel = weights == _MISSING_ID
    bad_weight = raw["WPFINWGT"][~sentinel & (weights.isna() | (weights < 0))]
    if len(bad_weight):
        raise _domain_error(year, "WPFINWGT", bad_weight)
    raw["WPFINWGT"] = weights.where(~sentinel).astype(float)

    month_codes = frozenset(range(1, 13))
    long_parts: list[pd.DataFrame] = []
    for n in slots:
        job_id = pd.to_numeric(raw[f"EJB{n}_JOBID"], errors="coerce")
        active = job_id.notna() & (job_id != _MISSING_ID)
        bad_id = raw[f"EJB{n}_JOBID"][active & (job_id <= 0)]
        if len(bad_id):
            raise _domain_error(year, f"EJB{n}_JOBID", bad_id)
        bmonth = _validate_codes(
            year, raw, f"EJB{n}_BMONTH", active, month_codes
        )
        emonth = _validate_codes(
            year, raw, f"EJB{n}_EMONTH", active, month_codes
        )
        clwrk = _validate_codes(
            year, raw, f"EJB{n}_CLWRK", active, frozenset(CLWRK_LABELS)
        )
        jborse = _validate_codes(
            year, raw, f"EJB{n}_JBORSE", active, frozenset(JBORSE_LABELS)
        )
        # EMPSIZE's universe is narrower than "active slot": the
        # question is asked about an employer's location, so
        # self-employment (JBORSE 2) and other work arrangements
        # (JBORSE 3) carry a structurally blank EMPSIZE. Verified on
        # the real 2023 file (188,268 active job-1 months): EMPSIZE
        # is populated for every JBORSE 1 job and blank for every
        # JBORSE 2 job (JBORSE 3 is mixed). A blank EMPSIZE on an
        # employer-type job still refuses.
        no_establishment = pd.to_numeric(
            raw[f"EJB{n}_JBORSE"], errors="coerce"
        ).isin((2, 3))
        empsize = _validate_codes(
            year,
            raw,
            f"EJB{n}_EMPSIZE",
            active & ~no_establishment,
            _EMPSIZE_CODES,
        )
        # A garbage *string* on a no-establishment slot would coerce
        # to NaN and masquerade as the structural blank, so the raw
        # cell is checked too: non-blank raw + NaN coercion refuses,
        # keeping the loud-refusal posture symmetric with employer
        # slots (review note on #206).
        raw_size = raw[f"EJB{n}_EMPSIZE"]
        raw_blank = raw_size.isna() | (raw_size.astype(str).str.strip() == "")
        bad_size = raw_size[
            active
            & no_establishment
            & (
                (
                    empsize.notna()
                    & ~empsize.isin((_MISSING, _MISSING_ID))
                    & ~empsize.isin(sorted(_EMPSIZE_CODES))
                )
                | (empsize.isna() & ~raw_blank)
            )
        ]
        if len(bad_size):
            raise _domain_error(year, f"EJB{n}_EMPSIZE", bad_size)
        # TJB{n}_MSUM has API range 0-9999999 and its ONLY sentinel is
        # -999 (2022/2023 SIPP variable metadata); -9 is not a valid
        # value here, unlike the coded slot columns. So the asymmetry is
        # deliberate: -999 is tolerated (maps to NaN below) and every
        # other negative — including -9 — refuses. Do not "harmonise"
        # this to also excuse -9.
        earnings = pd.to_numeric(raw[f"TJB{n}_MSUM"], errors="coerce")
        bad_earn = raw[f"TJB{n}_MSUM"][
            active
            & earnings.notna()
            & (earnings < 0)
            & (earnings != _MISSING_ID)
        ]
        if len(bad_earn):
            raise _domain_error(year, f"TJB{n}_MSUM", bad_earn)

        part = pd.DataFrame(
            {
                "person_id": (
                    raw["SSUID"].astype(str) + "-" + raw["PNUM"].astype(str)
                ),
                "year": year,
                "ref_year": year - 1,
                "month": raw["MONTHCODE"],
                "wave": raw["SWAVE"],
                "job_slot": n,
                "job_id": job_id,
                "bmonth": bmonth,
                "emonth": emonth,
                "clwrk": clwrk,
                "jborse": jborse,
                "empsize_code": empsize,
                "industry": raw[f"TJB{n}_IND"],
                "earnings": earnings.where(earnings != _MISSING_ID),
                "age": raw["TAGE"],
                "sex": raw["ESEX"],
                "weight": raw["WPFINWGT"],
            }
        )[active.to_numpy()]
        long_parts.append(part)

    out = pd.concat(long_parts, ignore_index=True)
    out["job_id"] = out["job_id"].astype(int)

    # One JOBID must occupy one slot per person-month; a duplicate
    # would double-count spell months and split earnings shares, so
    # it refuses like any other dictionary violation.
    duplicated = out.duplicated(["person_id", "month", "job_id"])
    if duplicated.any():
        example = out.loc[duplicated.idxmax()]
        raise ValueError(
            f"SIPP {year}: the same EJB job id appears in more than "
            "one slot for one person-month "
            f"({int(duplicated.sum())} row(s), e.g. person "
            f"{example['person_id']!r} month {example['month']} job "
            f"{example['job_id']}); refusing a file whose job slots "
            "are not distinct jobs."
        )

    out["class_of_worker"] = out["clwrk"].map(
        lambda code: CLWRK_LABELS.get(code, "missing")
    )
    out["work_arrangement"] = out["jborse"].map(
        lambda code: JBORSE_LABELS.get(code, "missing")
    )

    # Shares are within *known* earnings: a co-job with missing
    # earnings (-999) does not shrink the others' shares, and a
    # month whose known earnings total zero gets NaN shares (its
    # top_earner flag can still be True for a known-zero job).
    month_totals = out.groupby(["person_id", "month"])["earnings"].transform(
        "sum"
    )
    known = out["earnings"].notna() & (month_totals > 0)
    out["earnings_share"] = np.where(
        known, out["earnings"] / month_totals, np.nan
    )
    # Top earner per person-month: highest known earnings wins; ties
    # break to the lowest job_slot (deterministic and arbitrary, like
    # the mode() tie-break in job_spells). NaN earnings sort last so a
    # month with only unknown earnings yields no top_earner.
    ranked = out.sort_values(
        ["person_id", "month", "earnings", "job_slot"],
        ascending=[True, True, False, True],
        na_position="last",
    )
    top = ranked.drop_duplicates(["person_id", "month"]).index
    out["top_earner"] = out.index.isin(top) & out["earnings"].notna()

    return out.sort_values(
        ["person_id", "month", "job_slot"], ignore_index=True
    )


def job_spells(job_months: pd.DataFrame) -> pd.DataFrame:
    """Collapse job-months into IC1-preview spell rows.

    A spell is a maximal run of consecutive reference months for one
    (person, job id). The output mirrors the IC1 spell schema of ADR
    0003 (Proposed — this is a preview, not the frozen contract) and
    doubles as Workstream B's generator for IC1-conforming fixtures.

    Args:
        job_months: Output of :func:`read_sipp_job_months`.

    Returns:
        One row per spell: ``person_id``, ``spell_id`` (unique
        within person), ``start_year``/``start_month``,
        ``end_year``/``end_month``, ``n_months``, ``job_id``,
        ``industry``/``empsize_code``/``class_of_worker`` (modal),
        ``attributes_constant`` (False when any of the three varied
        within the spell — surfaced, never silently averaged),
        ``total_earnings``, ``earnings_share`` (spell earnings over
        the person's total earnings in the spell's months),
        ``primary_job`` (top earner in a strict majority of the
        spell's months).

    Raises:
        ValueError: If ``job_months`` lacks the required columns, or if
            it spans more than one ``ref_year`` (see below).
    """
    needed = {
        "person_id",
        "ref_year",
        "month",
        "job_id",
        "industry",
        "empsize_code",
        "class_of_worker",
        "earnings",
        "top_earner",
    }
    missing = sorted(needed - set(job_months.columns))
    if missing:
        raise ValueError(
            f"job_months is missing column(s) {missing}; pass the "
            "output of read_sipp_job_months."
        )

    if job_months.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "start_year",
                "start_month",
                "end_year",
                "end_month",
                "n_months",
                "job_id",
                "industry",
                "empsize_code",
                "class_of_worker",
                "attributes_constant",
                "total_earnings",
                "earnings_share",
                "primary_job",
                "spell_id",
            ]
        )

    # Cross-year spell linkage is undefined in this IC1 preview: the
    # break/run detection, the person-month earnings lookup, and the
    # spell edges all key on the calendar ``month`` (1-12) alone, so two
    # different reference years sharing a month would collapse into one
    # run and mis-count spells. read_sipp_job_months emits a single
    # ref_year per file, so this only trips when caller-concatenated
    # multi-year input is passed; refuse it loudly rather than silently
    # mis-collapse. (To support it, fold ref_year into the sort/break
    # keys via an absolute ``ref_year * 12 + month`` index and take the
    # edges from the spell's first/last row.)
    ref_years = job_months["ref_year"].dropna().unique()
    if len(ref_years) > 1:
        raise ValueError(
            "job_spells received job-months spanning multiple ref_years "
            f"({sorted(int(y) for y in ref_years)}); cross-year spell "
            "linkage is undefined in this IC1 preview. Collapse one SIPP "
            "file's months at a time."
        )

    frame = job_months.sort_values(["person_id", "job_id", "month"]).copy()
    frame["_break"] = (
        frame.groupby(["person_id", "job_id"])["month"].diff().ne(1)
    )
    frame["_run"] = frame.groupby(["person_id", "job_id"])["_break"].cumsum()

    person_month_totals = (
        frame.groupby(["person_id", "month"])["earnings"]
        .sum(min_count=1)
        .rename("month_total")
    )

    rows = []
    grouped = frame.groupby(["person_id", "job_id", "_run"], sort=True)
    for (person_id, job_id, _), spell in grouped:
        months = spell["month"]
        # NB: industry and empsize_code carry their raw codes here,
        # including the -9/-999 sentinels (unlike class_of_worker, which
        # read_sipp_job_months has already mapped to a "missing" label).
        # A downstream consumer (firms/banding.py) must expect a sentinel
        # mixed in with the 1-8 EMPSIZE codes / the industry codes.
        # mode() sorts, so a tie resolves to the smallest value —
        # arbitrary but deterministic, and always surfaced via
        # attributes_constant=False.
        modal = {
            column: spell[column].mode(dropna=False).iloc[0]
            for column in ("industry", "empsize_code", "class_of_worker")
        }
        constant = all(
            spell[column].nunique(dropna=False) == 1
            for column in ("industry", "empsize_code", "class_of_worker")
        )
        total = spell["earnings"].sum(min_count=1)
        person_total = person_month_totals.loc[
            [(person_id, m) for m in months]
        ].sum(min_count=1)
        share = (
            float(total / person_total)
            if pd.notna(total) and pd.notna(person_total) and person_total
            else np.nan
        )
        rows.append(
            {
                "person_id": person_id,
                "start_year": int(spell["ref_year"].iloc[0]),
                "start_month": int(months.min()),
                "end_year": int(spell["ref_year"].iloc[0]),
                "end_month": int(months.max()),
                "n_months": int(len(months)),
                "job_id": int(job_id),
                "industry": modal["industry"],
                "empsize_code": modal["empsize_code"],
                "class_of_worker": modal["class_of_worker"],
                "attributes_constant": bool(constant),
                "total_earnings": (
                    float(total) if pd.notna(total) else np.nan
                ),
                "earnings_share": share,
                "primary_job": bool(
                    spell["top_earner"].sum() * 2 > len(spell)
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["person_id", "start_month", "job_id"], ignore_index=True
    )
    out["spell_id"] = out.groupby("person_id").cumcount() + 1
    return out
