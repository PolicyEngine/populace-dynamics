"""Automatically assign one execution tier to every collected test."""

import sys
from pathlib import Path

import pytest

TIER_MARKERS = (
    "unit",
    "artifact",
    "integration_psid",
    "reproduction_legacy",
    "oracle_policyengine",
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PSID_DATA_INDICATORS = (
    "~/PolicyEngine/psid-data",
    "POPULACE_DYNAMICS_PSID_DIR",
)
_POLICYENGINE_ORACLE_INDICATORS = (
    "POPULACE_DYNAMICS_PE_US_DIR",
    "~/PolicyEngine/policyengine-us",
)
# Section 2.8.4 explicitly assigns this one truth-side-only, byte-identity
# check to the artifact tier even though it reads staged PSID when available.
# Keep the exception path-exact so no projected or general integration test
# can inherit the lower tier accidentally.
_SANCTIONED_ARTIFACT_MODULES = frozenset(
    {Path("tests/test_m6_truth_identity.py")}
)
_TIER_POLICY_COLLECTION = pytest.StashKey()


@pytest.fixture(autouse=True)
def _replay_frozen_qstar_preproduction_registry(request, monkeypatch):
    """Replay the frozen selector against its historical incumbent registry.

    The selector and its proof suite are frozen preproduction blobs.  Once the
    production engine appends streams 4/5, their historical runtime assertion
    still needs to see the pre-implementation 1/2/3 registry without changing
    either frozen blob.
    """
    source_path = Path(str(request.fspath)).resolve()
    if source_path.relative_to(_REPO_ROOT) != Path(
        "tests/test_m6_qstar_selection.py"
    ):
        return
    selector = sys.modules["select_m6_qstar_train_only"]
    monkeypatch.setattr(
        selector.fe,
        "SUBSTREAM_CODES",
        {"gate": 1, "donor-draw": 2, "re-entry-draw": 3},
    )


def _references_run_artifact(source: str) -> bool:
    """Return whether source refers to a JSON file under ``runs/``."""
    runs_path_indicators = ('"runs"', "'runs'", '"runs/', "'runs/")
    return ".json" in source and any(
        indicator in source for indicator in runs_path_indicators
    )


def _classify_test_module(relative_path: Path, source: str) -> str:
    """Classify a test module using only its path and static source."""
    if relative_path.match("tests/test_gate2_candidate*.py"):
        return "reproduction_legacy"
    if any(
        indicator in source for indicator in _POLICYENGINE_ORACLE_INDICATORS
    ):
        return "oracle_policyengine"
    if relative_path in _SANCTIONED_ARTIFACT_MODULES:
        return "artifact"
    if any(indicator in source for indicator in _PSID_DATA_INDICATORS):
        return "integration_psid"
    if _references_run_artifact(source):
        return "artifact"
    return "unit"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Mark items before pytest applies its built-in ``-m`` selection."""
    tier_by_path = {}

    for item in items:
        source_path = Path(str(item.fspath)).resolve()
        if source_path not in tier_by_path:
            source = source_path.read_text(encoding="utf-8")
            relative_path = source_path.relative_to(_REPO_ROOT)
            tier_by_path[source_path] = _classify_test_module(
                relative_path,
                source,
            )
        item.add_marker(tier_by_path[source_path])

    inventory = tuple(
        (
            item.nodeid,
            tuple(
                marker.name
                for marker in item.iter_markers()
                if marker.name in TIER_MARKERS
            ),
        )
        for item in items
    )
    config.stash[_TIER_POLICY_COLLECTION] = inventory


@pytest.fixture(scope="session")
def tier_policy_collection(pytestconfig):
    """Expose the complete pre-deselection tier inventory to policy tests."""
    return {
        "items": pytestconfig.stash[_TIER_POLICY_COLLECTION],
        "tier_names": TIER_MARKERS,
    }
