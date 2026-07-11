"""PSID reproduction pins for gate-2b forensics 2 (skips off-machine).

Marked ``integration_psid`` (references ``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``); skipped when the PSID Family Relationship
Matrix is not staged. Each of the three frozen questions gets one reproduction
pin that rebuilds seed 0's deterministic train-side (side B) decomposition live
and matches the recorded per-seed values to float precision, plus the
instrumented-draw fidelity proof that the diagnostic's component decomposition
is bit-identical to the committed ``simulate_draw_v4``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_forensics2_v1.json"
SCRIPTS = ROOT / "scripts"
REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()

needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "MX23REL").is_dir(),
    reason="PSID MX23REL relationship matrix not staged "
    "(POPULACE_DYNAMICS_PSID_DIR)",
)
pytestmark = needs_psid

ATOL = 1e-9


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def seed0():
    """Load the panels, split seed 0, and fit candidate 4 on side B."""
    sys.path.insert(0, str(SCRIPTS))
    import gate2b_forensics2 as gf2

    from populace_dynamics.harness import panel as hpanel
    from populace_dynamics.models import household_composition_sim_v4 as hcs4

    data = gf2.c4.load_all()
    links = gf2._all_parent_links(data["bh"])
    hh, mpanel = data["hh"], data["mpanel"]
    _side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = {int(x) for x in side_b.person_id.unique()}
    model = hcs4.fit_household_model_v4(
        hh,
        mpanel,
        data["demo"],
        data["mh"],
        data["bh"],
        data["order_map"],
        data["rel_map"],
        ids_b,
        father_links_child=data["father_links_child"],
        parent_pairs=data["parent_pairs"],
        fu_sizes=data["fu_sizes"],
        legal_flag=data["legal_flag"],
    )
    recorded = next(s for s in _artifact()["per_seed"] if s["seed"] == 0)
    return {
        "gf2": gf2,
        "data": data,
        "links": links,
        "ids_b": ids_b,
        "model": model,
        "recorded": recorded,
    }


def test_pin_instrumented_draw_is_bit_identical(seed0):
    """The instrumented draw reproduces simulate_draw_v4 bit-for-bit."""
    gf2 = seed0["gf2"]
    fid = gf2.fidelity_check_v4(
        seed0["data"]["hh"],
        seed0["data"]["mpanel"],
        seed0["model"],
        seed0["ids_b"],
        gf2.DRAW_SEED_BASE,
    )
    assert fid["bit_identical"] is True
    assert fid["max_abs_rate_deviation_vs_committed_simulate_draw_v4"] == 0.0


def test_pin_q5_selection_basis_reproduces(seed0):
    """Q5: seed-0 observable vs child-record custody + maternal complement."""
    gf2 = seed0["gf2"]
    q5 = gf2.q5_custodial_selection(
        seed0["data"]["hh"],
        seed0["data"]["mpanel"],
        seed0["data"]["demo"],
        seed0["data"]["rel_map"],
        seed0["data"]["parent_pairs"],
        seed0["links"],
        seed0["ids_b"],
    )
    rec = seed0["recorded"]["q5"]
    # The maternal four-way partition reproduces and reconciles to 1.
    ov = q5["maternal_complement_overall"]
    rov = rec["maternal_complement_overall"]
    for key in (
        "with_both",
        "with_father_only",
        "with_mother_only",
        "with_neither",
        "p_with_father",
        "p_with_mother",
    ):
        assert ov[key] == pytest.approx(rov[key], abs=ATOL)
    assert abs(ov["partition_reconciliation_remainder"]) < 1e-12
    # The selection-gap table reproduces cell-by-cell.
    for key, cell in q5["selection_gap_table"].items():
        rcell = rec["selection_gap_table"][key]
        assert cell["observable_basis_p_coresident"] == pytest.approx(
            rcell["observable_basis_p_coresident"], abs=ATOL
        )
        assert cell["child_record_basis_p_coresident"] == pytest.approx(
            rcell["child_record_basis_p_coresident"], abs=ATOL
        )


def test_pin_q6_reference_channels_reproduce(seed0):
    """Q6: seed-0 grandchild channel partition + component coupling."""
    gf2 = seed0["gf2"]
    q6 = gf2.q6_reference_channels(
        seed0["data"]["hh"], seed0["data"]["rel_map"], seed0["ids_b"]
    )
    rec = seed0["recorded"]["q6_reference"]
    assert q6["reference_stock_55plus_female"] == pytest.approx(
        rec["reference_stock_55plus_female"], abs=ATOL
    )
    for name, val in q6["channels"].items():
        assert val == pytest.approx(rec["channels"][name], abs=ATOL)
    assert abs(q6["channel_reconciliation_remainder"]) < 1e-12
    rc = q6["reference_component_rates_55plus_female"]
    rrc = rec["reference_component_rates_55plus_female"]
    for key in rc:
        assert rc[key] == pytest.approx(rrc[key], abs=ATOL)
    # The joint IS the composed channel (a), coupled above independence.
    assert (
        rc["multigen_and_child_and_notparent"]
        > 2 * rc["independence_product_multigen_x_child_x_notparent"]
    )
    # In-law codes 67/69 and social-great 83 are excluded from the ref link.
    for code, crec in q6["grandparent_code_inventory"].items():
        assert crec["in_reference_link"] == (int(code) in {66, 68, 82, 87, 88})


def test_pin_q7_reference_routes_reproduce(seed0):
    """Q7: seed-0 reference actual + core size-3 route partitions."""
    gf2 = seed0["gf2"]
    q7 = gf2.q7_reference_routes(
        seed0["data"]["hh"], seed0["data"]["rel_map"], seed0["ids_b"]
    )
    rec = seed0["recorded"]["q7_reference"]
    assert q7["reference_actual_size3_total"] == pytest.approx(
        rec["reference_actual_size3_total"], abs=ATOL
    )
    assert q7["reference_core_size3_total"] == pytest.approx(
        rec["reference_core_size3_total"], abs=ATOL
    )
    for name, val in q7["reference_actual_size3_routes"].items():
        assert val == pytest.approx(
            rec["reference_actual_size3_routes"][name], abs=ATOL
        )
    for name, val in q7["reference_core_size3_routes"].items():
        assert val == pytest.approx(
            rec["reference_core_size3_routes"][name], abs=ATOL
        )
    assert abs(q7["reference_actual_reconciliation_remainder"]) < 1e-12
    assert abs(q7["reference_core_reconciliation_remainder"]) < 1e-12
