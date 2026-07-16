"""Extract the 2014 Trustees Report intermediate ultimate assumptions.

The source is the committed browser response body for 2014 OASDI Trustees
Report table II.C1.  Direct curl access returned HTTP 403, so a local browser
fetched the authoritative page and exposed the response bytes without a
transcription step.  The extractor verifies those bytes before selecting the
five requested values from the exact ``Intermediate`` column.

Run from the repository root::

    .venv/bin/python scripts/extract_ssa_tr_ultimate_assumptions_2014.py
"""

from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
SOURCE_PATH = EXTERNAL / "ssa_tr_2014_ii_c1.source.html"
OUT_PATH = EXTERNAL / "ssa_tr_ultimate_assumptions_2014.json"

SCHEMA_VERSION = "ssa_tr_ultimate_assumptions.v1"
TRUSTEES_REPORT_YEAR = 2014
VINTAGE_YEAR = 2014
PUBLICATION_DATE = "2014-07-28"
SCENARIO = "intermediate"
SOURCE_SHA256 = (
    "737ece58d16e4290702c0c48832d1fa3995d052d4211d87713ad7910bc2e4e60"
)
SOURCE_URL = "https://www.ssa.gov/oact/tr/2014/II_C_assump.html"
RETRIEVAL_DATE = "2026-07-15T21:25:16Z"
FETCH_METHOD = (
    "Browser network-response capture on 2026-07-15 after browser-header "
    "curl returned HTTP 403. Headless Google Chrome fetched the authoritative "
    "SSA URL; Fetch API arrayBuffer() exposed the content-decoded response "
    "body, which is committed here. The extractor verifies the committed "
    "body's sha256 before parsing."
)

TABLE_CLASS = "UltimateKeyAssumps"
TABLE = "II.C1"
TABLE_CAPTION = (
    "Table II.C1.-Long-Range Valuesa of Key Assumptions for the 75-year "
    "Projection Period"
)
TABLE_HEADERS = (
    "Long-range assumptions",
    "Intermediate",
    "Low-cost",
    "High-cost",
)

_ROW_SPECS: dict[str, dict[str, Any]] = {
    "ultimate_total_fertility_rate": {
        "label": (
            "Total fertility rate (children per woman), for 2038 and later"
        ),
        "published_scenarios": ("2.0", "2.3", "1.7"),
        "unit": "children_per_woman",
        "horizon": "2038 and later",
    },
    "average_mortality_improvement_rate": {
        "label": (
            "Average annual percentage reduction in total age-sex-adjusted "
            "death rates from 2013 to 2088"
        ),
        "published_scenarios": (".79", ".41", "1.20"),
        "unit": "percent_per_year",
        "horizon": "2013-2088 average annual reduction",
        "interpretation_note": (
            "A 75-year average reduction, not a terminal-year mortality rate."
        ),
    },
    "ultimate_cpi_w_growth_rate": {
        "label": "Consumer Price Index (CPI-W), for 2020 and later",
        "published_scenarios": ("2.70", "3.40", "2.00"),
        "unit": "percent_per_year",
        "horizon": "2020 and later",
    },
    "ultimate_real_wage_differential": {
        "label": (
            "Average annual real-wage differential (percent) for 2025 to 2088"
        ),
        "published_scenarios": ("1.13", "1.76", ".52"),
        "unit": "percentage_points_per_year",
        "horizon": "2025-2088",
    },
    "ultimate_real_interest_rate": {
        "label": (
            "Annual trust fund real interest rate (percent), for 2025 and later"
        ),
        "published_scenarios": ("2.9", "3.4", "2.4"),
        "unit": "percent_per_year",
        "horizon": "2025 and later",
    },
}


def _normalize(text: str) -> str:
    """Normalize presentation whitespace and typographic dash variants."""
    for dash in "\u2010\u2011\u2012\u2013\u2014\u2212":
        text = text.replace(dash, "-")
    return " ".join(text.replace("\xa0", " ").split())


class _UltimateTableParser(HTMLParser):
    """Collect only rows from the uniquely named II.C1 table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.captions: list[str] = []
        self.target_count = 0
        self._table_depth = 0
        self._row_stack: list[dict[str, Any]] = []
        self._caption: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "table":
            if self._table_depth:
                self._table_depth += 1
            elif TABLE_CLASS in (attributes.get("class") or "").split():
                self._table_depth = 1
                self.target_count += 1
            return
        if not self._table_depth:
            return
        if tag == "tr":
            self._row_stack.append({"cells": [], "cell": None})
        elif tag in {"th", "td"} and self._row_stack:
            self._row_stack[-1]["cell"] = []
        elif tag == "caption":
            self._caption = []

    def handle_data(self, data: str) -> None:
        if not self._table_depth:
            return
        if self._row_stack and self._row_stack[-1]["cell"] is not None:
            self._row_stack[-1]["cell"].append(data)
        if self._caption is not None:
            self._caption.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._table_depth:
            return
        if tag in {"th", "td"} and self._row_stack:
            context = self._row_stack[-1]
            if context["cell"] is not None:
                context["cells"].append(_normalize("".join(context["cell"])))
                context["cell"] = None
        elif tag == "tr" and self._row_stack:
            self.rows.append(self._row_stack.pop()["cells"])
        elif tag == "caption" and self._caption is not None:
            self.captions.append(_normalize("".join(self._caption)))
            self._caption = None
        elif tag == "table":
            self._table_depth -= 1


def read_source() -> str:
    """Return the committed XHTML after asserting its pinned sha256."""
    raw = SOURCE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"{SOURCE_PATH} sha256 {digest} != pinned {SOURCE_SHA256}; "
            "re-verify source provenance before rebuilding"
        )
    return raw.decode("utf-8")


def parse_assumptions(source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse II.C1 and select the exact intermediate scenario column."""
    parser = _UltimateTableParser()
    parser.feed(source)
    if parser.target_count != 1:
        raise ValueError(
            f"expected one table class {TABLE_CLASS!r}, found "
            f"{parser.target_count}"
        )
    if parser.captions != [TABLE_CAPTION]:
        raise ValueError(
            f"caption {parser.captions!r} != expected {[TABLE_CAPTION]!r}"
        )
    if list(TABLE_HEADERS) not in parser.rows:
        raise ValueError(f"expected header {TABLE_HEADERS!r} not found")

    rows = {row[0]: row for row in parser.rows if len(row) == 4 and row[0]}
    assumptions: dict[str, Any] = {}
    published_scenarios: dict[str, Any] = {}
    for key, spec in _ROW_SPECS.items():
        label = spec["label"]
        row = rows.get(label)
        if row is None:
            raise ValueError(f"published row {label!r} not found")
        expected = spec["published_scenarios"]
        if tuple(row[1:]) != expected:
            raise ValueError(
                f"{label!r} scenario values {row[1:]!r} != {expected!r}"
            )
        record = {
            "source_label": label,
            "published_value": row[1],
            "value": float(row[1]),
            "unit": spec["unit"],
            "horizon": spec["horizon"],
        }
        if "interpretation_note" in spec:
            record["interpretation_note"] = spec["interpretation_note"]
        assumptions[key] = record
        published_scenarios[key] = {
            TABLE_HEADERS[index]: row[index]
            for index in range(1, len(TABLE_HEADERS))
        }
    return assumptions, published_scenarios


def build() -> dict[str, Any]:
    """Build the deterministic 2014 intermediate-assumption bundle."""
    assumptions, published_scenarios = parse_assumptions(read_source())
    return {
        "schema_version": SCHEMA_VERSION,
        "trustees_report_year": TRUSTEES_REPORT_YEAR,
        "vintage_year": VINTAGE_YEAR,
        "publication_date": PUBLICATION_DATE,
        "scenario": SCENARIO,
        "table": TABLE,
        "title": (
            "Long-Range Values of Key Assumptions for the 75-year "
            "Projection Period"
        ),
        "provenance": {
            "source": "2014 OASDI Trustees Report, Table II.C1",
            "source_url": SOURCE_URL,
            "source_file": "data/external/ssa_tr_2014_ii_c1.source.html",
            "source_sha256": SOURCE_SHA256,
            "retrieval_date": RETRIEVAL_DATE,
            "fetch_method": FETCH_METHOD,
            "table_of_record": TABLE_CAPTION,
            "scenario_binding": (
                "Exact column header 'Intermediate' in Table II.C1"
            ),
            "design_crosswalk": (
                "The M7 design cites Table V.B1 for its three carried TR "
                "constants. Table II.C1 is the smallest authoritative 2014 "
                "Trustees Report table that contains and label-binds the "
                "complete five-assumption bundle requested here, including "
                "the same 2.9 percent real-interest, 2.70 percent CPI-W, and "
                "1.13 percentage-point real-wage values."
            ),
        },
        "validation": {
            "expected_headers": list(TABLE_HEADERS),
            "n_selected_assumptions": len(assumptions),
            "all_scenario_values_label_checked": True,
            "published_scenario_values": published_scenarios,
        },
        "build": {
            "built_by": (
                "scripts/extract_ssa_tr_ultimate_assumptions_2014.py"
            ),
            "source_file": "data/external/ssa_tr_2014_ii_c1.source.html",
            "source_sha256": SOURCE_SHA256,
            "reproducible": (
                "parsed deterministically from the committed, "
                "sha256-verified XHTML response body; no network or "
                "wall-clock timestamp is used"
            ),
        },
        "assumptions": assumptions,
    }


def main() -> None:
    artifact = build()
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    digest = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(
        f"wrote {OUT_PATH} "
        f"({artifact['validation']['n_selected_assumptions']} assumptions)"
    )
    print(f"json sha256: {digest}")


if __name__ == "__main__":
    main()
