"""Extract the <=2014-vintage SSA trust-fund effective-rate series.

The committed source is the exact SSA HTTP response body captured on
December 20, 2014.  It ends with calendar year 2013, so it cannot leak the
2014-2025 observations now present on SSA's continuously updated page.  The
body was recovered from a Common Crawl WARC after direct programmatic access
to ssa.gov returned HTTP 403.  Its WARC payload digest was independently
verified before commit; the sha256 below is the build-time tamper gate.

Run from the repository root::

    .venv/bin/python scripts/extract_ssa_effective_interest_rates_2014.py
"""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
SOURCE_PATH = EXTERNAL / "ssa_effective_interest_rates_2014.source.html"
OUT_PATH = EXTERNAL / "ssa_effective_interest_rates_2014.json"

SCHEMA_VERSION = "ssa_effective_interest_rates.v1"
VINTAGE_YEAR = 2014
SOURCE_CAPTURE_DATE = "2014-12-20T03:49:23Z"
LATEST_OBSERVATION_YEAR = 2013
SOURCE_SHA256 = (
    "867704b8e479adcee9a7047e762848fe9f50df22028f6a80657ad8bf683550f7"
)
SOURCE_URL = "https://www.ssa.gov/oact/ProgData/effectiveRates.html"
RETRIEVAL_DATE = "2026-07-15T21:23:09Z"
FETCH_METHOD = (
    "Exact SSA HTTP 200 response body recovered from the Common Crawl "
    "CC-MAIN-2014-52 WARC capture dated 2014-12-20T03:49:23Z after direct "
    "browser-header curl to ssa.gov returned HTTP 403. The response-body "
    "SHA-1 was verified against WARC-Payload-Digest "
    "sha1:YJ2IJWUUVP6EPTXTSWJNZ4QI6P4ULVJL; this extractor verifies the "
    "committed body's sha256 before parsing."
)

TABLE_TITLE = (
    "Effective Interest Rates Earned By the Assets of the OASI and DI "
    "Trust Funds"
)
TABLE_CAPTION = f"{TABLE_TITLE} [Percent]"
TABLE_HEADERS = ("Calendar year", "OASI", "DI", "OASDI")
FIRST_OBSERVATION_YEAR = 1980


def _normalize(text: str) -> str:
    """Normalize display text without changing numeric or label content."""
    for dash in "\u2010\u2011\u2012\u2013\u2014\u2212":
        text = text.replace(dash, "-")
    return " ".join(text.replace("\xa0", " ").split())


class _TableParser(HTMLParser):
    """Collect table rows and captions, including rows in nested tables."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.captions: list[str] = []
        self._row_stack: list[dict[str, Any]] = []
        self._caption: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag == "tr":
            self._row_stack.append({"cells": [], "cell": None})
        elif tag in {"th", "td"} and self._row_stack:
            self._row_stack[-1]["cell"] = []
        elif tag == "caption":
            self._caption = []

    def handle_data(self, data: str) -> None:
        if self._row_stack and self._row_stack[-1]["cell"] is not None:
            self._row_stack[-1]["cell"].append(data)
        if self._caption is not None:
            self._caption.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._row_stack:
            context = self._row_stack[-1]
            if context["cell"] is not None:
                context["cells"].append(_normalize("".join(context["cell"])))
                context["cell"] = None
        elif tag == "tr" and self._row_stack:
            context = self._row_stack.pop()
            self.rows.append(context["cells"])
        elif tag == "caption" and self._caption is not None:
            self.captions.append(_normalize("".join(self._caption)))
            self._caption = None


def read_source() -> str:
    """Return the committed HTML after asserting its pinned sha256."""
    raw = SOURCE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"{SOURCE_PATH} sha256 {digest} != pinned {SOURCE_SHA256}; "
            "re-verify source provenance before rebuilding"
        )
    return raw.decode("utf-8")


def parse_rates(source: str) -> dict[int, dict[str, float]]:
    """Parse and label-check the two published rate-table panels."""
    parser = _TableParser()
    parser.feed(source)

    if TABLE_CAPTION not in parser.captions:
        raise ValueError(
            f"expected caption {TABLE_CAPTION!r} not found; got "
            f"{parser.captions!r}"
        )
    header_count = sum(tuple(row) == TABLE_HEADERS for row in parser.rows)
    if header_count != 2:
        raise ValueError(
            f"expected two {TABLE_HEADERS!r} headers, found {header_count}"
        )

    rates: dict[int, dict[str, float]] = {}
    for cells in parser.rows:
        if len(cells) != 4 or not re.fullmatch(r"\d{4}", cells[0]):
            continue
        year = int(cells[0])
        if year in rates:
            raise ValueError(f"duplicate calendar year {year}")
        try:
            rates[year] = {
                "oasi": float(cells[1]),
                "di": float(cells[2]),
                "oasdi": float(cells[3]),
            }
        except ValueError as error:
            raise ValueError(f"non-numeric rate row {cells!r}") from error

    expected_years = list(
        range(FIRST_OBSERVATION_YEAR, LATEST_OBSERVATION_YEAR + 1)
    )
    if sorted(rates) != expected_years:
        raise ValueError(
            f"calendar years {sorted(rates)} != expected "
            f"{FIRST_OBSERVATION_YEAR}-{LATEST_OBSERVATION_YEAR}"
        )
    if rates[2013] != {"oasi": 3.8, "di": 4.5, "oasdi": 3.8}:
        raise ValueError(f"unexpected boundary row: {rates[2013]!r}")
    return rates


def build() -> dict[str, Any]:
    """Build the deterministic external-reference artifact."""
    rates = parse_rates(read_source())
    return {
        "schema_version": SCHEMA_VERSION,
        "series": TABLE_TITLE,
        "table": TABLE_TITLE,
        "unit": "percent",
        "rate_basis": "estimated_effective_rate_earned_by_assets",
        "interpretation_note": (
            "SSA's estimated effective interest rate earned by trust-fund "
            "assets; distinct from the inflation-adjusted ultimate real "
            "interest assumption in 2014 Trustees Report Table II.C1."
        ),
        "vintage_year": VINTAGE_YEAR,
        "source_capture_date": SOURCE_CAPTURE_DATE,
        "latest_observation_year": LATEST_OBSERVATION_YEAR,
        "provenance": {
            "source": "Social Security Administration, Office of the Chief Actuary",
            "source_url": SOURCE_URL,
            "source_file": (
                "data/external/ssa_effective_interest_rates_2014.source.html"
            ),
            "source_sha256": SOURCE_SHA256,
            "source_capture_date": SOURCE_CAPTURE_DATE,
            "retrieval_date": RETRIEVAL_DATE,
            "fetch_method": FETCH_METHOD,
            "table_of_record": TABLE_TITLE,
            "archive_warc_payload_digest": (
                "sha1:YJ2IJWUUVP6EPTXTSWJNZ4QI6P4ULVJL"
            ),
            "vintage_note": (
                "The contemporaneous December 2014 source ends at calendar "
                "year 2013. The live page now contains post-boundary rows; "
                "those rows are intentionally absent from this source and "
                "artifact."
            ),
        },
        "validation": {
            "first_observation_year": FIRST_OBSERVATION_YEAR,
            "latest_observation_year": LATEST_OBSERVATION_YEAR,
            "n_observations": len(rates),
            "expected_headers": list(TABLE_HEADERS),
            "continuous_calendar_years": True,
        },
        "build": {
            "built_by": (
                "scripts/extract_ssa_effective_interest_rates_2014.py"
            ),
            "source_file": (
                "data/external/ssa_effective_interest_rates_2014.source.html"
            ),
            "source_sha256": SOURCE_SHA256,
            "reproducible": (
                "parsed deterministically from the committed, "
                "sha256-verified HTML source; no network or wall-clock "
                "timestamp is used"
            ),
        },
        "data": {str(year): rates[year] for year in sorted(rates)},
    }


def main() -> None:
    artifact = build()
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    digest = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(
        f"wrote {OUT_PATH} "
        f"({artifact['validation']['n_observations']} observations)"
    )
    print(f"json sha256: {digest}")


if __name__ == "__main__":
    main()
