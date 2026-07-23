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
  2015Q1 on (the ``year >= 2015`` filter; matches the provenance
  note). Feeds E11's national margin.
* **J2J R2026Q1 (us, sex x age, no firm characteristics)** -- national
  job-to-job hire/separation flows by sex x age group, all-industry
  margin only (the committed-extract size cap rules out the full
  sector detail), 2015Q1 on. Feeds gate E2's age x sex
  separation/hire/J2J rate references. NOTE: LEHD calls the sex x
  age tabulation ``sa`` (``se`` is sex x *education*).
* **J2JOD R2026Q1 (us, origin x destination firm size)** -- national
  job-to-job flows by origin firm size x destination firm size,
  2015Q1 on. Feeds gate E11's origin/destination size-ladder
  reference. The LEHD flat J2JOD files publish only the one-sided
  firm-size margins, so the full cross comes from the LED Extraction
  Tool query API (``ledextract.ces.census.gov``), which serves
  whatever release is current and reports no release identifier of
  its own (its schema version V4.14.0 is the *software* version and
  is identical across R2026Q1 and R2026Q2, so it cannot pin the
  release -- see the provenance note). The full detail is released
  for 2015Q1-2016Q1 only; later quarters are suppressed (status
  flag 11) and only the margins remain published. Because the tool
  re-runs the query live, that only-ever detail window is archived
  under ``data/external/raw/`` and is the builder's default input.

Run from the repository root::

    .venv/bin/python scripts/fetch_employer_firm_targets.py

Re-running is idempotent: raw files are cached in a temp directory,
verified against the pinned sha256 digests below, and the extracts
are rewritten deterministically.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external"

RETRIEVED = "2026-07-14"
#: Fetch date of the second-wave extracts (sex x age J2J, J2JOD
#: origin x destination firm size).
RETRIEVED_WAVE2 = "2026-07-17"

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
    # Release-stamped directories, not `latest_release`: the sha256
    # pins below are for the R2026Q1 bytes, and `latest_release` is a
    # moving alias that will serve different bytes once LEHD rotates to
    # R2026Q2 (the digest check would then misreport a re-issue and the
    # committed extracts would be unrebuildable). LEHD keeps every
    # release under its stamped path indefinitely.
    "qwi_us_sa_fs_gn_ns_op_u.csv.gz": (
        "https://lehd.ces.census.gov/data/qwi/R2026Q1/us/"
        "qwi_us_sa_fs_gn_ns_op_u.csv.gz",
        "76f147615d796eb0ba93401107762e1c67667a25c468b2ee4ca07e07" "d9eabd85",
    ),
    "j2j_us_d_fs_gn_ns_oslp_u.csv.gz": (
        "https://lehd.ces.census.gov/data/j2j/R2026Q1/us/j2j/"
        "j2j_us_d_fs_gn_ns_oslp_u.csv.gz",
        "abdd573d414d66f864828952501cee0a6ef6c8db88cb789da38af1a3" "c9d55c6f",
    ),
    "j2j_us_sa_f_gn_ns_oslp_u.csv.gz": (
        "https://lehd.ces.census.gov/data/j2j/R2026Q1/us/j2j/"
        "j2j_us_sa_f_gn_ns_oslp_u.csv.gz",
        "0e043fc8796bd3e11231ff6d174fdfebed926c9d40da4f069a3ad31e" "ed55aba0",
    ),
    "label_agegrp.csv": (
        "https://lehd.ces.census.gov/data/schema/latest/" "label_agegrp.csv",
        "eb478c6eda6c12a57609afaf89bbb42dd4d9fb2ee883f6dd0399fb71" "7b27889b",
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

#: LEHD sex code -> label (pinned from the J2J schema).
SEX_LABELS = {0: "All Sexes", 1: "Male", 2: "Female"}

#: LEHD age-group code -> label (pinned from label_agegrp.csv).
AGEGRP_LABELS = {
    "A00": "All Ages (14-99)",
    "A01": "14-18",
    "A02": "19-21",
    "A03": "22-24",
    "A04": "25-34",
    "A05": "35-44",
    "A06": "45-54",
    "A07": "55-64",
    "A08": "65-99",
}

# ----------------------------------------------------------------
# LED Extraction Tool (the J2JOD origin x destination firm-size
# cross is not published in the LEHD flat files -- they carry only
# the one-sided firm-size margins -- so it is pulled through the
# LED Extraction Tool's query API instead).
# ----------------------------------------------------------------

LED_BASE = "https://ledextract.ces.census.gov"

#: Query submitted to the LED Extraction Tool (POST /j2j/download).
#: ``oq`` is the ordinal quarter, year * 4 + (quarter - 1):
#: 8060 = 2015Q1 .. 8100 = 2025Q1 (the last quarter in R2026Q1).
#: Origin-destination queries must carry the ``*_orig`` keys.
LED_J2JOD_REQUEST = {
    "version": "V4.14.0",
    "seasonadj": ["U"],
    "geography": ["00"],
    "geography_orig": ["00"],
    "industry": ["00"],
    "industry_orig": ["00"],
    "firmage": ["0"],
    "firmage_orig": ["0"],
    "firmsize": ["0", "1", "2", "3", "4", "5"],
    "firmsize_orig": ["0", "1", "2", "3", "4", "5"],
    "sex": ["0"],
    "agegrp": ["A00"],
    "education": ["E0"],
    "race": ["A0"],
    "ethnicity": ["A0"],
    "indicator": ["J2J", "EE", "AQHire", "J2JS", "EES", "AQHireS"],
    "oq": list(range(8060, 8101)),
    "export_labels": False,
}

#: The archived raw tool response. The LED Extraction Tool serves
#: only its *current* release and re-runs the query live, so the
#: 2015Q1-2016Q1 origin x destination detail window -- the only
#: window in which that cross is published at all (see the
#: detail-window note in the provenance file) -- would be
#: unrecoverable the day the tool stops serving it. The response is
#: therefore committed, and it is the default input: a fetch is a
#: *verification* path, not the only path to the data.
LED_J2JOD_ARCHIVE = OUT_DIR / "raw" / "led_j2jod_us_fsfs_2015on.csv.gz"

#: sha256 of the archived response bytes, as served on 2026-07-23.
LED_J2JOD_SHA256 = (
    "7afad9f408319c54e7e7d802b068723e346f49c840d4fe861e7e55d9" "528d3944"
)

#: sha256 of the response's *content*, canonicalised (columns sorted,
#: rows sorted by year/quarter/firmsize_orig/firmsize, fixed float
#: format) before hashing.
#:
#: This -- not the byte digest -- is the integrity pin, because the
#: byte digest is not a property of the data. The response first
#: pinned on this PR (``adbd16e2...``) stopped matching six days
#: later while every one of the 1,476 rows was unchanged, value for
#: value: the tool had merely reordered its measure columns
#: (``EE,AQHire,EES,AQHireS,J2J,J2JS`` where it previously emitted
#: ``EE,AQHire,J2J,EES,AQHireS,J2JS``). A pin that fires on cosmetic
#: reordering is worse than no pin: it trains the reader to re-pin
#: on sight, so the one failure that matters -- an actual revision --
#: arrives looking exactly like the six false alarms before it.
#: Byte drift is now reported and tolerated; content drift raises.
LED_J2JOD_CONTENT_SHA256 = (
    "c52ec512bc3f478d6426efb7b03cccbe1edc952309214030ed53eb42" "f7a83354"
)

#: Key columns defining canonical row order for the content digest.
LED_J2JOD_KEY = ["year", "quarter", "firmsize_orig", "firmsize"]

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


J2JOD_MEASURES = ["EE", "AQHire", "J2J", "EES", "AQHireS", "J2JS"]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface the LED tool's 303 instead of following it (the
    redirect target is the HTML results page; the CSV lives at
    ``download.csv`` with the same query string)."""

    def redirect_request(self, *args, **kwargs):
        return None


def led_j2jod_content_digest(path: Path) -> str:
    """Canonical content digest of a raw LED J2JOD response.

    Columns sorted, rows sorted by :data:`LED_J2JOD_KEY`, fixed float
    format — so the digest is a property of the *data*, invariant to
    the tool's column ordering (see :data:`LED_J2JOD_CONTENT_SHA256`
    for why that distinction is load-bearing).
    """
    frame = pd.read_csv(path, low_memory=False)
    canonical = (
        frame[sorted(frame.columns)]
        .sort_values(LED_J2JOD_KEY)
        .reset_index(drop=True)
    )
    buf = io.StringIO()
    canonical.to_csv(buf, index=False, float_format="%.10g")
    return hashlib.sha256(buf.getvalue().encode()).hexdigest()


def _verify_led_j2jod(path: Path) -> Path:
    """Verify a raw response by content, reporting byte drift."""
    content = led_j2jod_content_digest(path)
    if content != LED_J2JOD_CONTENT_SHA256:
        raise RuntimeError(
            f"LED J2JOD extract: content sha256 {content} != pinned "
            f"{LED_J2JOD_CONTENT_SHA256}. Values changed, not just "
            "the response layout — LEHD revises across releases, so "
            "re-pin only after diffing the cells and updating the "
            "provenance note with the new release."
        )
    # LED_J2JOD_SHA256 pins the *uncompressed* response bytes, so the
    # archive path must be decompressed before hashing. Hashing the
    # .gz container against it made this note fire on every default
    # run — a permanently-lit byte-drift channel is one nobody reads,
    # which is exactly how a real byte-only change goes unnoticed.
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != LED_J2JOD_SHA256:
        # Cause unknown by construction: the content pin above has
        # already passed, so the values are identical and the change
        # is in the layout (column order, quoting, line endings, ...).
        # Naming one cause here would be a guess.
        print(
            f"note: LED J2JOD response bytes changed ({digest[:12]} != "
            f"{LED_J2JOD_SHA256[:12]}) while every value is unchanged; "
            "a layout-only difference. Not an error."
        )
    return path


def fetch_led_j2jod(cache_dir: Path, *, live: bool = False) -> Path:
    """Return the raw J2JOD firm-size cross, verified by content.

    Reads the committed archive (:data:`LED_J2JOD_ARCHIVE`) by
    default: the tool serves only its current release, so the
    2015Q1-2016Q1 detail window it carries is not re-fetchable in
    perpetuity and the archive is the durable copy.

    With ``live=True``, re-queries the tool instead — POST the JSON
    query to ``/j2j/download``; it answers 303 with the encoded query
    string, and ``/j2j/download.csv?<query>`` serves the extract —
    and verifies the result against the same content digest. That is
    the path that detects a genuine LEHD revision.
    """
    if not live:
        if not LED_J2JOD_ARCHIVE.exists():
            raise FileNotFoundError(
                f"Archived LED J2JOD response missing at "
                f"{LED_J2JOD_ARCHIVE}; re-fetch with live=True only "
                "if the tool still serves the detail window."
            )
        return _verify_led_j2jod(LED_J2JOD_ARCHIVE)

    path = cache_dir / "led_j2jod_us_fsfs_2015on.csv"
    if not path.exists():
        print("querying the LED Extraction Tool (J2JOD firm-size cross)")
        req = urllib.request.Request(
            LED_BASE + "/j2j/download",
            data=json.dumps(LED_J2JOD_REQUEST).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "populace-dynamics fetch script",
            },
        )
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            resp = opener.open(req, timeout=300)
            raise RuntimeError(
                "LED Extraction Tool did not redirect (HTTP "
                f"{resp.status}); the query API may have changed."
            )
        except urllib.error.HTTPError as err:
            if err.code != 303:
                raise
            location = err.headers["Location"]
        query = urllib.parse.urlsplit(location).query
        csv_req = urllib.request.Request(
            LED_BASE + "/j2j/download.csv?" + query,
            headers={"User-Agent": "populace-dynamics fetch script"},
        )
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(csv_req, timeout=300) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(path)
    return _verify_led_j2jod(path)


def fetch(name: str, cache_dir: Path) -> Path:
    """Download (or reuse) a pinned raw file and verify its sha256."""
    url, expected = SOURCES[name]
    path = cache_dir / name
    if not path.exists():
        print(f"downloading {url}")
        req = urllib.request.Request(
            url, headers={"User-Agent": "populace-dynamics fetch script"}
        )
        # Download to a temp file and rename into place only on success,
        # so an interrupted download never leaves a partial file that
        # the next run would misdiagnose as a re-issued source.
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(req, timeout=300) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(path)
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


def build_j2j_sexage(cache_dir: Path) -> None:
    """National J2J flows by sex x age group (all-industry margin).

    The raw ``sa`` file is sex x age x NAICS sector; the committed
    extract keeps the all-industry margin (``industry == "00"``) only
    -- with the full sector detail the file would breach the 1 MB
    extract cap -- but keeps the complete sex (0-2) x age (A00-A08)
    grid, margins included, so aggregation identities stay testable.
    Pure row/column filter, no re-aggregation, from LEHD_START_YEAR
    on.
    """
    path = fetch("j2j_us_sa_f_gn_ns_oslp_u.csv.gz", cache_dir)
    with gzip.open(path, "rt") as fh:
        raw = pd.read_csv(fh, low_memory=False)
    keep = raw[
        (raw["year"] >= LEHD_START_YEAR)
        & (raw["industry"].astype(str) == "00")
    ].copy()
    id_cols = ["year", "quarter", "sex", "agegrp"]
    flag_cols = [f"s{m}" for m in J2J_MEASURES]
    out = keep[id_cols + J2J_MEASURES + flag_cols].copy()
    out.insert(3, "sex_label", out["sex"].map(SEX_LABELS))
    out.insert(5, "agegrp_label", out["agegrp"].map(AGEGRP_LABELS))
    out = out.sort_values(id_cols).reset_index(drop=True)
    out.to_csv(
        OUT_DIR / "j2j_us_sexage_2015on.csv",
        index=False,
        float_format="%.10g",
    )
    print(f"j2j_us_sexage_2015on.csv: {len(out)} rows")


def build_j2jod_firmsize(cache_dir: Path) -> None:
    """National J2J flows by origin x destination firm size.

    From the LED Extraction Tool (see :func:`fetch_led_j2jod`); the
    committed extract keeps the full 6 x 6 grid (codes 0-5 on both
    sides: the 25 detail cells plus the tool's aggregated margins,
    status flag 10/12). Column subset and sort only, no
    re-aggregation. Suppressed cells (status flag 11) load as NaN.
    NOTE: the tool's margins are aggregates of the firm-size-coded
    tabulation and do **not** sit systematically below the flat-file
    ``d_fs`` margins: checked across all 41 quarters the tool is
    *above* in 37, deviating −1.00% to +2.02% (mean +0.75%). The
    public-sector (firm size "N") explanation is refuted by that
    direction — excluding size N could only bias the tool downward.
    The gap is dominated by independent noise infusion applied to the
    two tabulations; see the provenance note and
    ``scripts/check_j2jod_margin_agreement.py``.
    """
    raw = pd.read_csv(fetch_led_j2jod(cache_dir), low_memory=False)
    keep = raw[raw["year"] >= LEHD_START_YEAR].copy()
    id_cols = ["year", "quarter", "firmsize_orig", "firmsize"]
    flag_cols = [f"s{m}" for m in J2JOD_MEASURES]
    out = keep[id_cols + J2JOD_MEASURES + flag_cols].copy()
    out.insert(4, "firmsize_orig_label", _firmsize_label(out["firmsize_orig"]))
    out.insert(5, "firmsize_label", _firmsize_label(out["firmsize"]))
    out = out.sort_values(id_cols).reset_index(drop=True)
    out.to_csv(
        OUT_DIR / "j2jod_us_firmsize_od_2015on.csv",
        index=False,
        float_format="%.10g",
    )
    print(f"j2jod_us_firmsize_od_2015on.csv: {len(out)} rows")


def main() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "employer_firm_raw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_susb(cache_dir)
    build_bds(cache_dir)
    build_qwi(cache_dir)
    build_j2j(cache_dir)
    build_j2j_sexage(cache_dir)
    build_j2jod_firmsize(cache_dir)
    for f in sorted(OUT_DIR.glob("*_us_*2015on.csv")) + [
        OUT_DIR / "susb_us_sector_size_2022.csv",
        OUT_DIR / "bds_us_firm_size_1978_2022.csv",
    ]:
        size = f.stat().st_size
        if size >= 1_000_000:
            raise RuntimeError(
                f"{f.name} is {size} bytes (>1 MB); extracts must stay "
                "small aggregate derivatives."
            )
        print(f"{f.name}: {size} bytes")


if __name__ == "__main__":
    main()
