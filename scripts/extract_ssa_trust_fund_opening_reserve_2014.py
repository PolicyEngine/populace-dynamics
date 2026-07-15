"""Extract the 2014-TR estimate of the end-of-2014 OASDI reserve.

The 2014 Trustees Report was published on July 28, 2014.  Its calendar-year
2014 row is therefore an intermediate-assumptions projection, not a realized
historical balance.  Actual end-of-2014 reserves first appear in the forbidden
2015 report.  This artifact keeps that timing distinction loud while binding
the only end-of-2014 reserve compatible with the campaign's vintage boundary.

Run from the repository root::

    .venv/bin/python scripts/extract_ssa_trust_fund_opening_reserve_2014.py
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
SOURCE_PATH = EXTERNAL / "ssa_tr_2014_vi_g8.source.html"
OUT_PATH = EXTERNAL / "ssa_trust_fund_opening_reserve_2014.json"

SCHEMA_VERSION = "ssa_trust_fund_opening_reserve.v1"
TRUSTEES_REPORT_YEAR = 2014
VINTAGE_YEAR = 2014
PUBLICATION_DATE = "2014-07-28"
AS_OF_DATE = "2014-12-31"
OPENING_YEAR = 2015
SCENARIO = "intermediate"
SOURCE_SHA256 = (
    "0cebeb4774112932f43432a8f9e60e1981a5506459340cb31ad2144121f6bc26"
)
SOURCE_URL = "https://www.ssa.gov/oact/TR/2014/lr6f8.html"
RETRIEVAL_DATE = "2026-07-15T21:25:41Z"
FETCH_METHOD = (
    "Browser network-response capture on 2026-07-15 after browser-header "
    "curl returned HTTP 403. Headless Google Chrome fetched the authoritative "
    "SSA URL; Fetch API arrayBuffer() exposed the content-decoded response "
    "body, which is committed here. The extractor verifies the committed "
    "body's sha256 before parsing."
)

TABLE = "VI.G8"
TABLE_CAPTION = (
    "Table VI.G8.- Operations of the Combined OASI and DI Trust Funds, in "
    "Current Dollars, Calendar Years 1970-2090 [In billions]"
)
TABLE_HEADERS = (
    "Calendar year",
    "Non-interest income",
    "Interest income",
    "Total income",
    "Cost",
    "Assets at end of year",
)
EXPECTED_INTERMEDIATE_2014_ROW = (
    "2014",
    "783.4",
    "99.0",
    "882.4",
    "863.1",
    "2,783.7",
)


def _normalize(text: str) -> str:
    """Normalize presentation whitespace and typographic dash variants."""
    for dash in "\u2010\u2011\u2012\u2013\u2014\u2212":
        text = text.replace(dash, "-")
    return " ".join(text.replace("\xa0", " ").split())


class _AllTablesParser(HTMLParser):
    """Collect captions and rows from every table, preserving table scope."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self._stack: list[dict[str, Any]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag == "table":
            self._stack.append(
                {
                    "rows": [],
                    "row": None,
                    "cell": None,
                    "caption": None,
                    "caption_chunks": None,
                }
            )
            return
        if not self._stack:
            return
        table = self._stack[-1]
        if tag == "tr":
            table["row"] = []
        elif tag in {"th", "td"} and table["row"] is not None:
            table["cell"] = []
        elif tag == "caption":
            table["caption_chunks"] = []
        elif tag == "br":
            if table["cell"] is not None:
                table["cell"].append(" ")
            if table["caption_chunks"] is not None:
                table["caption_chunks"].append(" ")

    def handle_data(self, data: str) -> None:
        if not self._stack:
            return
        table = self._stack[-1]
        if table["cell"] is not None:
            table["cell"].append(data)
        if table["caption_chunks"] is not None:
            table["caption_chunks"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        table = self._stack[-1]
        if tag in {"th", "td"} and table["row"] is not None:
            if table["cell"] is not None:
                table["row"].append(_normalize("".join(table["cell"])))
                table["cell"] = None
        elif tag == "tr" and table["row"] is not None:
            table["rows"].append(table["row"])
            table["row"] = None
        elif tag == "caption" and table["caption_chunks"] is not None:
            table["caption"] = _normalize("".join(table["caption_chunks"]))
            table["caption_chunks"] = None
        elif tag == "table":
            self.tables.append(self._stack.pop())


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


def parse_opening_reserve(source: str) -> dict[str, Any]:
    """Select the intermediate 2014 end-of-year asset-reserve cell."""
    parser = _AllTablesParser()
    parser.feed(source)
    matches = [
        table for table in parser.tables if table["caption"] == TABLE_CAPTION
    ]
    if len(matches) != 1:
        captions = [table["caption"] for table in parser.tables]
        raise ValueError(
            f"expected one {TABLE!r} table, found {len(matches)}; "
            f"captions={captions!r}"
        )
    rows = matches[0]["rows"]
    nonempty_rows = [tuple(cell for cell in row if cell) for row in rows]
    if TABLE_HEADERS not in nonempty_rows:
        raise ValueError(f"expected header {TABLE_HEADERS!r} not found")

    scenario: str | None = None
    selected: tuple[str, ...] | None = None
    for row in nonempty_rows:
        if len(row) == 1 and row[0] in {
            "Historical data:",
            "Intermediate:",
            "Low-cost:",
            "High-cost:",
        }:
            scenario = row[0]
            continue
        if scenario == "Intermediate:" and row and row[0] == "2014":
            selected = row
            break
    if selected != EXPECTED_INTERMEDIATE_2014_ROW:
        raise ValueError(
            f"intermediate 2014 row {selected!r} != expected "
            f"{EXPECTED_INTERMEDIATE_2014_ROW!r}"
        )

    amount_billions = Decimal(selected[-1].replace(",", ""))
    amount_millions = int(amount_billions * 1000)
    return {
        "as_of_date": AS_OF_DATE,
        "opening_year": OPENING_YEAR,
        "funds": ["OASI", "DI"],
        "fund_combination": "OASI + DI (combined OASDI)",
        "amount_billions_usd": float(amount_billions),
        "amount_millions_usd": amount_millions,
        "published_value": selected[-1],
        "estimate_status": "projected_intermediate_assumptions",
    }


def build() -> dict[str, Any]:
    """Build the deterministic opening-reserve reference artifact."""
    opening_reserve = parse_opening_reserve(read_source())
    return {
        "schema_version": SCHEMA_VERSION,
        "trustees_report_year": TRUSTEES_REPORT_YEAR,
        "vintage_year": VINTAGE_YEAR,
        "publication_date": PUBLICATION_DATE,
        "scenario": SCENARIO,
        "table": TABLE,
        "unit": "billions_current_us_dollars",
        "provenance": {
            "source": "2014 OASDI Trustees Report, Table VI.G8",
            "source_url": SOURCE_URL,
            "source_file": "data/external/ssa_tr_2014_vi_g8.source.html",
            "source_sha256": SOURCE_SHA256,
            "retrieval_date": RETRIEVAL_DATE,
            "fetch_method": FETCH_METHOD,
            "table_of_record": TABLE_CAPTION,
            "cell_binding": ("Intermediate: / 2014 / Assets at end of year"),
            "timing_note": (
                "The 2014 report predates December 31, 2014, so this is the "
                "report's intermediate-assumptions estimate. Actual "
                "end-of-2014 reserves first appear in the out-of-bound 2015 "
                "report and are intentionally not used."
            ),
        },
        "validation": {
            "expected_headers": list(TABLE_HEADERS),
            "published_intermediate_2014_row": list(
                EXPECTED_INTERMEDIATE_2014_ROW
            ),
            "value_status_label_checked": True,
            "opens_calendar_year": OPENING_YEAR,
        },
        "build": {
            "built_by": (
                "scripts/extract_ssa_trust_fund_opening_reserve_2014.py"
            ),
            "source_file": "data/external/ssa_tr_2014_vi_g8.source.html",
            "source_sha256": SOURCE_SHA256,
            "reproducible": (
                "parsed deterministically from the committed, "
                "sha256-verified HTML response body; no network or "
                "wall-clock timestamp is used"
            ),
        },
        "opening_reserve": opening_reserve,
    }


def main() -> None:
    artifact = build()
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    digest = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    amount = artifact["opening_reserve"]["amount_billions_usd"]
    print(f"wrote {OUT_PATH} (${amount:,.1f} billion)")
    print(f"json sha256: {digest}")


if __name__ == "__main__":
    main()
