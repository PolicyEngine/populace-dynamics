"""PSID/frame reproduction pins for W1 forensics 3 (skips off-machine).

Marked ``integration_psid`` (references ``~/PolicyEngine/psid-data`` and the
certified frame export ``POPULACE_DYNAMICS_FRAME_PICKLE``); skipped when the
PSID Family Relationship Matrix or the frame pickle are not staged. Rebuilds
the bit-identity claims the artifact records: the Q10 re-run of the committed
#117 ledger on the transported frame reproduces the cube c2 exhaustion deltas,
and the Q11 re-run of ``regenerate_person_frame_v3`` reproduces the cube
hh_size cells. Single-draw checks (the artifact-build runs the full K-draw
decompositions).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate_w1_forensics3_v1.json"
CANDIDATE3 = ROOT / "runs" / "gate_w1_candidate3_v1.json"
SCRIPTS = ROOT / "scripts"
REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
FRAME_PICKLE = os.environ.get("POPULACE_DYNAMICS_FRAME_PICKLE", "")

needs_data = pytest.mark.skipif(
    not (REAL_DATA / "MX23REL").is_dir()
    or not (FRAME_PICKLE and Path(FRAME_PICKLE).exists()),
    reason="PSID MX23REL and/or the certified frame pickle "
    "(POPULACE_DYNAMICS_FRAME_PICKLE) not staged",
)
pytestmark = needs_data


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def env():
    sys.path.insert(0, str(SCRIPTS))
    import gate_w1_forensics3 as gf3

    gens = gf3.fit_generators()
    persons = gf3._load_frame()
    return {"gf3": gf3, "gens": gens, "persons": persons, "art": _artifact()}


def test_q10_ledger_rerun_reproduces_cube_exhaustion_deltas(env):
    gf3, gens, persons = env["gf3"], env["gens"], env["persons"]
    deployed = gf3._run_m2_on_transport(persons, gens)
    committed = json.loads(CANDIDATE3.read_text())["family_c"]["fingerprints"][
        "c2"
    ]["provision_deltas"]["our_exhaustion_deltas"]
    for prov, val in committed.items():
        assert abs(deployed["exhaustion_deltas"][prov] - float(val)) == 0.0


def test_q11_regen_reproduces_cube_hh_size_cells(env):
    gf3, gens, persons = env["gf3"], env["gens"], env["persons"]
    from populace_dynamics.data import deployment_frame as dfm
    from populace_dynamics.models import transport_deployment_v3 as td

    hold = gf3._seed0_holdout(persons)
    entry_anchor = td.build_cps_entry_anchor(
        dfm.reference_moments(persons, weighted=True)
    )
    regen0 = td.regenerate_person_frame_v3(
        hold, gens, entry_anchor, 0, td.FAMILY_A_STREAM_BASE
    )
    cells0 = dfm.reference_moments(regen0, weighted=True)
    cube = np.array(json.loads(CANDIDATE3.read_text())["family_a"]["cube"])
    gated = json.loads(CANDIDATE3.read_text())["family_a"]["gated_cells"]
    checked = 0
    for ci, cell in enumerate(gated):
        if cell.startswith("hh_size_share.") and cell in cells0:
            assert abs(cells0[cell]["rate"] - float(cube[0, ci, 0])) == 0.0
            checked += 1
    assert checked == 5  # mutation-check: all five hh_size cells found


def test_artifact_bit_identity_flags_match_rebuild(env):
    a = env["art"]
    assert (
        a["q10_cap150k_adjacency"]["instrumentation_fidelity"]["bit_identical"]
        is True
    )
    assert (
        a["q11_hhsize_residual"]["instrumentation_fidelity"]["bit_identical"]
        is True
    )
