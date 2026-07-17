"""Structural validation for the append-only timeline forecast ledger.

The ledger (docs/forecasts/timeline_ledger.json) is process tooling, not
part of the gates.yaml evaluation contract. These tests keep it honest as
a data structure: parseable, monotonically identified, decidable, and
graded only through the resolution block.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "forecasts" / "timeline_ledger.json"

REQUIRED_FIELDS = {
    "id",
    "registered_at",
    "forecaster",
    "claim",
    "resolution_criterion",
    "p50",
    "p80",
    "status",
    "resolution",
}
STATUSES = {"open", "resolved", "superseded"}


def _ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def test_ledger_parses_with_expected_schema():
    doc = _ledger()
    assert doc["schema_version"] == "forecast_ledger.v1"
    assert isinstance(doc["entries"], list) and doc["entries"]


def test_entries_have_required_fields_and_monotonic_ids():
    entries = _ledger()["entries"]
    ids = [e["id"] for e in entries]
    assert ids == sorted(ids) and len(ids) == len(set(ids))
    for e in entries:
        assert REQUIRED_FIELDS <= set(e), e["id"]
        assert e["status"] in STATUSES, e["id"]
        # registered_at is a parseable UTC timestamp
        assert e["registered_at"].endswith("Z"), e["id"]
        dt.datetime.fromisoformat(e["registered_at"].replace("Z", "+00:00"))


def test_open_and_resolved_entries_have_decidable_dates():
    for e in _ledger()["entries"]:
        if e["status"] == "superseded" and e["p50"] is None:
            continue
        p50, p80 = _parse_date(e["p50"]), _parse_date(e["p80"])
        assert p50 <= p80, e["id"]


def test_resolution_blocks_are_complete_exactly_when_resolved():
    for e in _ledger()["entries"]:
        if e["status"] == "resolved":
            r = e["resolution"]
            assert r is not None, e["id"]
            assert {
                "resolved_at",
                "evidence",
                "error_days_p50",
                "within_p50",
                "within_p80",
            } <= set(r), e["id"]
        else:
            assert e["resolution"] is None, e["id"]


def test_supersession_links_are_internally_consistent():
    entries = {e["id"]: e for e in _ledger()["entries"]}
    for e in entries.values():
        if e["status"] == "superseded":
            successors = e.get("superseded_by")
            assert successors, e["id"]
            for s in successors:
                assert s in entries, (e["id"], s)
                assert entries[s]["registered_at"] >= e["registered_at"]
        for field in ("supersedes",):
            target = e.get(field)
            if target is not None:
                assert target in entries, (e["id"], target)
                assert entries[target]["status"] == "superseded"
