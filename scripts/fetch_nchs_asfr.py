"""Fetch NCHS national age-specific fertility rates (ASFR) as a reference.

The external fertility anchor for the gate-2 family-transition floors. NCHS
publishes age-specific birth rates (live births per 1,000 women in a
5-year maternal age group) in the annual *Births: Final Data* National
Vital Statistics Report, and serves the same numbers machine-readably from
its Data Query System (DQS) on ``data.cdc.gov`` via the Socrata JSON API.

Source dataset ``daba-4vfq`` ("DQS Birth and fertility rates, by age
group, race, and Hispanic origin of mother: United States"; NVSS Birth
File). The successor to the retired static natality tables under
``cdc.gov/nchs/data/dvs/``. We pull the U.S., all-races, birth-rate rows
for the standard maternal age groups (``group_id='10'`` = maternal age
group; ``subtopic_id='1'`` = birth rate) for the latest final year (2024)
and the prior year (2023) as a cross-check.

Two labeling conventions NCHS documents and we carry through verbatim:

* the youngest reported bin is ``10-14`` and the oldest ``45-54`` -- the
  ``45-54`` row is the official "45-49" rate (all births at age 45+
  divided by the 45-49 female population), so it is stored under the band
  key ``45-49`` matching :data:`populace_dynamics.data.transitions.ASFR_AGE_BANDS`;
* the Total Fertility Rate is not a Socrata field; it is ``5 * sum`` of the
  eight single ASFR bins (10-14 ... 45-49) and reproduces the published
  headline exactly, so we compute and validate it here.

The parsed ASFRs are validated against the published NVSR headline values
(digit for digit) so a changed feed fails loudly. Written with full
provenance -- query URL, fetch timestamp, sha256 of the raw JSON response,
the NVSR citation, and the headline cross-check -- to
``data/external/nchs_asfr_<year>.json``.

Run from the repository root::

    .venv/bin/python scripts/fetch_nchs_asfr.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external"

#: Latest NCHS *final* natality vintage (published 2026-06-09) and the
#: prior year kept as a cross-check.
VINTAGE_YEAR = 2024
PRIOR_YEAR = 2023

#: Annual ASFR series for the gate-2 period-matched fertility anchor
#: (finding 6): the earliest calendar year the DQS resource serves through
#: the vintage. The Socrata feed for daba-4vfq begins at 2016; the annual
#: rates net the vintage confound out of the PSID/NCHS ASFR ratios by
#: matching each PSID woman-year to its own calendar year's national rate.
ANNUAL_START_YEAR = 2016

#: data.cdc.gov Socrata resource for the DQS birth/fertility-rate table.
RESOURCE = "daba-4vfq"
BASE_URL = f"https://data.cdc.gov/resource/{RESOURCE}.json"

#: Socrata subgroup_id -> the transitions.py band key. The oldest bin is
#: labeled "45-54 years" but is the official 45-49 rate (see module
#: docstring); "10-14" is kept for the TFR sum but is outside the gate-2
#: ASFR bands.
SUBGROUP_TO_BAND: dict[str, str] = {
    "20025": "10-14",
    "20100": "15-19",
    "20200": "20-24",
    "20300": "25-29",
    "20400": "30-34",
    "20500": "35-39",
    "20600": "40-44",
    "20700": "45-49",
}
#: The seven gate-2 ASFR bands (10-14 excluded; used only for TFR).
GATE2_BANDS: tuple[str, ...] = (
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
)

#: Published NVSR ASFR headline values (live births per 1,000 women), from
#: *Births: Final Data for 2024* (NVSR 75-2, Table 2) and *...2023*
#: (NVSR 74-1, Table 2). The parsed feed must match these exactly.
PUBLISHED_ASFR: dict[int, dict[str, float]] = {
    2024: {
        "10-14": 0.2,
        "15-19": 12.6,
        "20-24": 55.8,
        "25-29": 89.5,
        "30-34": 93.7,
        "35-39": 54.3,
        "40-44": 12.7,
        "45-49": 1.1,
    },
    2023: {
        "10-14": 0.2,
        "15-19": 13.1,
        "20-24": 57.7,
        "25-29": 91.0,
        "30-34": 94.3,
        "35-39": 54.3,
        "40-44": 12.5,
        "45-49": 1.1,
    },
}
#: Published Total Fertility Rate (births per 1,000 women), NVSR headline.
PUBLISHED_TFR: dict[int, float] = {2024: 1599.5, 2023: 1621.0}

NVSR_CITATION = (
    "Osterman MJK, Hamilton BE, Martin JA, Driscoll AK, Valenzuela CP. "
    "Births: Final Data for 2024. National Vital Statistics Reports, "
    "Vol. 75, No. 2. Hyattsville, MD: National Center for Health "
    "Statistics; June 9, 2026."
)
REPORT_PDF_URL = "https://www.cdc.gov/nchs/data/nvsr/nvsr75/nvsr75-02.pdf"
DQS_PAGE = f"https://data.cdc.gov/d/{RESOURCE}"


def _ssl_context() -> ssl.SSLContext:
    """A verifying SSL context, preferring certifi's CA bundle."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _query_url(years: tuple[int, ...]) -> str:
    """Socrata query for the U.S. all-races ASFR rows for ``years``."""
    year_list = ",".join(f"'{y}'" for y in years)
    subgroups = ",".join(f"'{s}'" for s in SUBGROUP_TO_BAND)
    where = (
        f"subtopic_id='1' AND group_id='10' "
        f"AND time_period in({year_list}) "
        f"AND subgroup_id in({subgroups})"
    )
    params = {
        "$select": (
            "time_period,subgroup,subgroup_id,estimate,estimate_type,"
            "subgroup_order"
        ),
        "$where": where,
        "$order": "time_period,subgroup_order",
        "$limit": "500",
    }
    return f"{BASE_URL}?{urllib.parse.urlencode(params)}"


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (populace-dynamics fetch)"}
    )
    with urllib.request.urlopen(
        request, timeout=120, context=_ssl_context()
    ) as response:
        return response.read()


def parse_asfr(rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """Group the Socrata rows into ``{year: {band: rate}}``.

    Socrata serializes numbers as strings and drops trailing zeros, so
    every estimate is ``float()``-cast; the ``estimate_type`` is verified
    to be the live-births-per-1,000-females birth rate so a wrong slice
    fails loudly.
    """
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        etype = str(row.get("estimate_type", ""))
        if "per 1,000" not in etype:
            raise ValueError(
                f"unexpected estimate_type {etype!r}; expected the "
                "per-1,000-females birth rate"
            )
        sid = str(row["subgroup_id"])
        band = SUBGROUP_TO_BAND.get(sid)
        if band is None:
            continue
        year = int(row["time_period"])
        out.setdefault(year, {})[band] = float(row["estimate"])
    return out


def validate_year(year: int, bands: dict[str, float]) -> dict[str, Any]:
    """Check the parsed ASFRs against the published NVSR headline + TFR."""
    published = PUBLISHED_ASFR[year]
    missing = set(published) - set(bands)
    if missing:
        raise ValueError(f"{year}: missing ASFR bands {sorted(missing)}.")
    for band, value in published.items():
        if round(bands[band], 1) != value:
            raise ValueError(
                f"{year} {band}: parsed ASFR {bands[band]} does not match "
                f"the published {value}."
            )
    # TFR = 5 * sum of the eight single ASFR bins; reproduces the headline.
    tfr = 5.0 * sum(bands[b] for b in published)
    if round(tfr, 1) != PUBLISHED_TFR[year]:
        raise ValueError(
            f"{year}: computed TFR {tfr:.1f} != published "
            f"{PUBLISHED_TFR[year]}."
        )
    # Unimodal hump: rises to a single peak across the gate-2 bands.
    series = [bands[b] for b in GATE2_BANDS]
    peak = series.index(max(series))
    nondecr_up = all(series[i] <= series[i + 1] for i in range(peak))
    nonincr_down = all(
        series[i] >= series[i + 1] for i in range(peak, len(series) - 1)
    )
    if not (nondecr_up and nonincr_down):
        raise ValueError(f"{year}: ASFR is not a unimodal hump: {series}.")
    return {
        "tfr_computed": round(tfr, 1),
        "tfr_published_headline": PUBLISHED_TFR[year],
        "peak_band": GATE2_BANDS[peak],
        "unimodal_hump": True,
    }


def build(verbose: bool = True) -> dict[str, Any]:
    fetched_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    url = _query_url((VINTAGE_YEAR, PRIOR_YEAR))
    if verbose:
        print(f"fetching ASFR: {url}")
    payload = _fetch_bytes(url)
    sha256 = hashlib.sha256(payload).hexdigest()
    rows = json.loads(payload)
    by_year = parse_asfr(rows)

    validation: dict[int, Any] = {}
    for year in (VINTAGE_YEAR, PRIOR_YEAR):
        if year not in by_year:
            raise ValueError(f"feed returned no rows for {year}.")
        validation[year] = validate_year(year, by_year[year])
        if verbose:
            v = validation[year]
            print(
                f"  {year}: peak {v['peak_band']}, TFR {v['tfr_computed']} "
                f"(headline {v['tfr_published_headline']})"
            )

    annual = _build_annual(verbose=verbose)

    return {
        "schema_version": "nchs_asfr.v1",
        "vintage_year": VINTAGE_YEAR,
        "prior_year": PRIOR_YEAR,
        "measure": "age_specific_fertility_rate_per_1000_women",
        "report": {
            "title": f"Births: Final Data for {VINTAGE_YEAR}",
            "nvsr_citation": NVSR_CITATION,
            "publisher": "National Center for Health Statistics (NCHS)",
            "report_pdf_url": REPORT_PDF_URL,
            "dqs_dataset_page": DQS_PAGE,
        },
        "fetch": {
            "fetched_utc": fetched_utc,
            "fetched_by": "scripts/fetch_nchs_asfr.py",
            "source_format": "json (data.cdc.gov Socrata DQS API)",
            "socrata_resource": RESOURCE,
            "query_url": url,
            "response_sha256": sha256,
            "n_bytes": len(payload),
        },
        "band_note": (
            "ASFR is live births per 1,000 women in the maternal age band. "
            "The '45-49' band is NCHS's official 45-49 rate: all births to "
            "women age 45+ over the 45-49 female population (the Socrata "
            "feed labels this row '45-54 years'). '10-14' is retained for "
            "the TFR sum only and is outside the gate-2 ASFR bands."
        ),
        "gate2_bands": list(GATE2_BANDS),
        "tables": {
            str(year): by_year[year] for year in (VINTAGE_YEAR, PRIOR_YEAR)
        },
        "total_fertility_rate": {
            str(year): validation[year]["tfr_computed"]
            for year in (VINTAGE_YEAR, PRIOR_YEAR)
        },
        "validation": {str(y): validation[y] for y in validation},
        "annual": annual,
    }


def _unimodal_hump(bands: dict[str, float]) -> tuple[bool, str]:
    """(is the gate-2 ASFR series a single-peaked age hump, peak band)."""
    series = [bands[b] for b in GATE2_BANDS]
    peak = series.index(max(series))
    up = all(series[i] <= series[i + 1] for i in range(peak))
    down = all(
        series[i] >= series[i + 1] for i in range(peak, len(series) - 1)
    )
    return (up and down), GATE2_BANDS[peak]


def _build_annual(verbose: bool = True) -> dict[str, Any]:
    """The annual ASFR series for the gate-2 period-matched anchor.

    The DQS resource serves the same maternal-age birth-rate rows for
    every ``time_period`` from :data:`ANNUAL_START_YEAR` through the
    vintage. Each year is checked for band completeness and the unimodal
    age hump; only 2023/2024 carry a published-headline cross-check (in
    ``tables``). Stored with its own query URL and sha256 so the series is
    reproducible and pinned.
    """
    years = tuple(range(ANNUAL_START_YEAR, VINTAGE_YEAR + 1))
    url = _query_url(years)
    if verbose:
        print(f"fetching annual ASFR series {years[0]}-{years[-1]}: {url}")
    payload = _fetch_bytes(url)
    sha256 = hashlib.sha256(payload).hexdigest()
    by_year = parse_asfr(json.loads(payload))
    present = sorted(by_year)
    if not present:
        raise ValueError("annual ASFR feed returned no rows.")

    tables: dict[str, dict[str, float]] = {}
    validation: dict[str, Any] = {}
    for year in present:
        bands = by_year[year]
        missing = set(GATE2_BANDS) - set(bands)
        if missing:
            raise ValueError(
                f"annual {year}: missing ASFR bands {sorted(missing)}."
            )
        hump, peak = _unimodal_hump(bands)
        if not hump:
            raise ValueError(
                f"annual {year}: ASFR is not a unimodal hump: "
                f"{[bands[b] for b in GATE2_BANDS]}."
            )
        tables[str(year)] = {b: bands[b] for b in GATE2_BANDS}
        validation[str(year)] = {"unimodal_hump": True, "peak_band": peak}
    if verbose:
        print(f"  annual years present: {present}")
    return {
        "band_note": (
            "Annual national ASFR (live births per 1,000 women in the "
            "maternal age band) for every calendar year the DQS resource "
            "serves; the gate-2 period-matched fertility anchor weights "
            "these by PSID woman-year exposure so the PSID/NCHS ratio is "
            "not confounded by the secular fertility decline. Only 2023 "
            "and 2024 carry a published-headline cross-check (see tables); "
            "intermediate years are band-complete and unimodal-checked."
        ),
        "years": [str(y) for y in present],
        "tables": tables,
        "validation": validation,
        "fetch": {
            "query_url": url,
            "response_sha256": sha256,
            "n_bytes": len(payload),
        },
    }


def main() -> None:
    artifact = build(verbose=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"nchs_asfr_{VINTAGE_YEAR}.json"
    out_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
