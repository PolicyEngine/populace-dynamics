"""Real-frame column-contract audit (staged PSID).

Loads the certified loaders' real output and asserts the M6 phase manifest is
satisfiable against the actual columns -- the definitive form of the check whose
absence let the second registered gate_m6 run crash at first real contact
(grading #42 comment 4972045579).  Skipped unless staged PSID is present at
``~/PolicyEngine/psid-data`` (or ``POPULACE_DYNAMICS_PSID_DIR``).  Reads only
schema; computes no scored statistic.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from populace_dynamics.harness.m6_schema_audit import (
    COMMITTED_FRAME_SCHEMAS,
    audit_columns,
)

_PSID_DIR = Path(
    os.environ.get("POPULACE_DYNAMICS_PSID_DIR", "~/PolicyEngine/psid-data")
).expanduser()
needs_staged_psid = pytest.mark.skipif(
    not _PSID_DIR.exists(),
    reason=(
        "staged PSID not present (~/PolicyEngine/psid-data or "
        "POPULACE_DYNAMICS_PSID_DIR); real-frame schema audit skipped"
    ),
)


def _real_frames() -> dict[str, object]:
    from populace_dynamics.data import deaths, disability, family
    from populace_dynamics.models.household_composition import data as hc_data
    from populace_dynamics.models.household_composition.components.cohabitation_overlay import (  # noqa: E501
        cohabitation_flag,
    )

    sources = hc_data.load_sources()
    death_records = deaths.read_death_records()
    disability_status = disability.read_disability_status()
    disability_panel = disability.build_disability_panel(
        disability_status, death_records
    )
    return {
        "demographic_panel": sources["demo"],
        "death_records": death_records,
        "earnings_panel": family.family_earnings_panel(),
        "disability_status": disability_status,
        "marital.person_years": sources["mpanel"].person_years,
        "marital.events": sources["mpanel"].events,
        "marital.attrs": sources["mpanel"].attrs,
        "household.person_waves": sources["hh"].person_waves,
        "cohabitation": cohabitation_flag(sources["rel_map"]),
        "disability_panel.person_years": disability_panel.person_years,
    }


@needs_staged_psid
def test_manifest_reads_are_satisfied_by_the_real_loaders():
    real = {name: set(frame.columns) for name, frame in _real_frames().items()}
    assert audit_columns(real) == []


@needs_staged_psid
def test_committed_schemas_match_the_real_loader_columns():
    real = {name: set(frame.columns) for name, frame in _real_frames().items()}
    drift = {
        name: {
            "committed_not_real": sorted(
                set(COMMITTED_FRAME_SCHEMAS[name]) - cols
            ),
            "real_not_committed": sorted(
                cols - set(COMMITTED_FRAME_SCHEMAS[name])
            ),
        }
        for name, cols in real.items()
        if set(COMMITTED_FRAME_SCHEMAS[name]) != cols
    }
    assert not drift, f"committed schema drift vs real loaders: {drift}"
