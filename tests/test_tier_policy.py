"""Regression test for the collection-time pytest tier policy."""

import json
from collections import Counter
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("tier_counts.json")


def test__given_collected_suite__then_tiers_match_policy_manifest(
    tier_policy_collection,
):
    """Every test has one tier and committed counts expose tier drift."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tier_names = tier_policy_collection["tier_names"]
    inventory = tier_policy_collection["items"]

    assert set(manifest) == {"schema_version", "counts"}
    assert manifest["schema_version"] == 1
    assert set(manifest["counts"]) == set(tier_names)

    invalid = {
        nodeid: markers for nodeid, markers in inventory if len(markers) != 1
    }
    assert not invalid, f"Tests without exactly one tier marker: {invalid}"

    actual_counts = Counter(markers[0] for _, markers in inventory)
    assert dict(actual_counts) == manifest["counts"]
