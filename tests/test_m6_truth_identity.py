"""PSID-backed truth-only byte identity for the M6 floor/harness seam.

This is deliberately not a projected-vs-realized test.  It computes the same
seed-0 realized side-A cell values through (1) the floor-script exports and (2)
the harness reducer, on exactly the same support, then compares canonical
IEEE-754 bytes.  The §2.8.3a earnings-domain support filter is tested separately
with synthetic frames in ``test_m6_scoring.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from populace_dynamics.data import deaths, disability, family, panels
from populace_dynamics.harness import m6_cells, m6_inputs
from populace_dynamics.harness.m6_scoring import (
    GATED_CELL_NAMES,
    cell_value_bytes,
    reduce_gated_cells,
    side_a_person_ids,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_psid = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2015").is_dir()
    or not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID family and marriage-history files not staged",
)


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_real_psid
def test_truth_side_floor_and_harness_cell_values_are_byte_identical():
    v1 = _load_script("build_m6_holdout_floors")
    v3 = _load_script("build_m6_holdout_floors_v3")

    # Extraction identity is distinct from support identity.
    assert v1.earnings_cells is m6_cells.earnings_cells
    assert v3.marital_cells is m6_cells.coarsened_marital_cells
    assert v3.disab_cells is m6_cells.coarsened_disability_cells
    assert (
        m6_inputs.marital_tables_from_panel
        is m6_cells.marital_tables_from_panel
    )

    demo = panels.demographic_panel()
    death_records = deaths.read_death_records()
    status = disability.read_disability_status()
    earnings_panel = family.family_earnings_panel()
    anchor = v1.build_anchor_frame(demo)
    present = v1.presence_by_wave(demo)
    events, person_years = v1.marital_tables(death_records, anchor, present)
    pairs = v1.disability_pairs(status, death_records, anchor)
    earnings = v1.earnings_frame(earnings_panel, anchor)

    events = events[events.window == "gated"]
    person_years = person_years[person_years.window == "gated"]
    pairs = pairs[pairs.window == "gated"]
    marital_ids = side_a_person_ids(anchor, split_unit="household", seed=0)
    person_ids = side_a_person_ids(anchor, split_unit="person", seed=0)
    side_events = events[events.person_id.isin(marital_ids)]
    side_person_years = person_years[person_years.person_id.isin(marital_ids)]
    side_pairs = pairs[pairs.person_id.isin(person_ids)]
    side_earnings = earnings[earnings.person_id.isin(person_ids)]

    floor_cells = {}
    floor_cells.update(
        v3.marital_cells(
            side_events,
            side_person_years,
            "first_marriage",
            v3.MARITAL_LADDER[0][1],
            v3.MARITAL_LADDER[0][2],
        )
    )
    floor_cells.update(
        v3.marital_cells(
            side_events,
            side_person_years,
            "divorce",
            v3.MARITAL_LADDER[2][1],
            v3.MARITAL_LADDER[2][2],
        )
    )
    floor_cells.update(
        v3.marital_cells(
            side_events,
            side_person_years,
            "remarriage",
            v3.MARITAL_LADDER[3][1],
            v3.MARITAL_LADDER[3][2],
        )
    )
    floor_cells.update(
        v3.disab_cells(
            side_pairs,
            "incidence",
            v3.DISABILITY_LADDER[-1][1],
            v3.DISABILITY_LADDER[-1][2],
        )
    )
    floor_cells.update(
        v3.disab_cells(
            side_pairs,
            "recovery",
            v3.DISABILITY_LADDER[-1][1],
            v3.DISABILITY_LADDER[-1][2],
        )
    )
    floor_cells.update(v1.earnings_cells(side_earnings))
    floor_cells = {cell: floor_cells[cell] for cell in GATED_CELL_NAMES}

    harness_cells = reduce_gated_cells(
        side_events,
        side_person_years,
        side_pairs,
        side_earnings,
    )
    assert cell_value_bytes(harness_cells) == cell_value_bytes(floor_cells)
