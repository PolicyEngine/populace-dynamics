"""PSID reproduction pins for gate-2b forensics 3 (skips off-machine).

Marked ``integration_psid`` (references ``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``); skipped when the PSID Family Relationship
Matrix is not staged. Each of the three frozen questions gets one reproduction
pin that rebuilds seed 0's deterministic train-side (side B) decomposition live
and matches the recorded per-seed values to float precision, plus the
instrumented-draw fidelity proof that the diagnostic's component decomposition
is bit-identical to the committed ``simulate_draw_v5`` (and that the four child
channels and the linked marital split reconcile exactly).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_forensics3_v1.json"
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
    """Load the panels, split seed 0, and fit candidate 5 on side B."""
    sys.path.insert(0, str(SCRIPTS))
    import gate2b_forensics3 as gf3

    from populace_dynamics.harness import panel as hpanel
    from populace_dynamics.models import household_composition_sim_v5 as hcs5

    data = gf3.c5.load_all()
    links = gf3.f2._all_parent_links(data["bh"])
    hh, mpanel = data["hh"], data["mpanel"]
    _side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = {int(x) for x in side_b.person_id.unique()}
    model = hcs5.fit_household_model_v5(
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
        child_record_expo=data["child_record_expo"],
        parent_counts=data["parent_counts"],
    )
    recorded = next(s for s in _artifact()["per_seed"] if s["seed"] == 0)
    return {
        "gf3": gf3,
        "data": data,
        "links": links,
        "ids_b": ids_b,
        "model": model,
        "recorded": recorded,
    }


def test_pin_instrumented_draw_is_bit_identical(seed0):
    """The instrumented draw reproduces simulate_draw_v5 bit-for-bit, and the
    four child channels + linked marital split reconcile exactly."""
    gf3 = seed0["gf3"]
    fid = gf3.fidelity_check_v5(
        seed0["data"]["hh"],
        seed0["data"]["mpanel"],
        seed0["model"],
        seed0["ids_b"],
        gf3.DRAW_SEED_BASE,
    )
    assert fid["bit_identical"] is True
    assert fid["max_abs_rate_deviation_vs_committed_simulate_draw_v5"] == 0.0
    assert fid["child_channel_additivity_residual"] == 0
    assert fid["linked_marital_split_additivity_residual"] == 0


def test_pin_q8_custodial_bases_reproduce(seed0):
    """Q8: seed-0 observable vs child-record custody + the 45-54|F anchors."""
    gf3 = seed0["gf3"]
    q8 = gf3.q8_reference(
        seed0["data"]["hh"],
        seed0["data"]["mpanel"],
        seed0["data"]["demo"],
        seed0["data"]["rel_map"],
        seed0["data"]["parent_pairs"],
        seed0["links"],
        seed0["ids_b"],
    )
    rec = seed0["recorded"]["q8_reference"]
    # The observable-vs-child-record selection gap reproduces cell-by-cell.
    gap = q8["custodial_bases_q5_reused"]["selection_gap_table"]
    rgap = rec["custodial_bases_q5_reused"]["selection_gap_table"]
    for key, cell in gap.items():
        assert cell["observable_basis_p_coresident"] == pytest.approx(
            rgap[key]["observable_basis_p_coresident"], abs=ATOL
        )
        assert cell["child_record_basis_p_coresident"] == pytest.approx(
            rgap[key]["child_record_basis_p_coresident"], abs=ATOL
        )
    # The 0-4 not-married sign inversion (child-record HIGHER at 0-4, LOWER at
    # school ages) is a live property of the data, not an artifact of caching.
    g04 = gap["0-4|not_married"]
    g512 = gap["5-12|not_married"]
    assert (
        g04["child_record_basis_p_coresident"]
        > g04["observable_basis_p_coresident"]
    )
    assert (
        g512["child_record_basis_p_coresident"]
        < g512["observable_basis_p_coresident"]
    )
    # The 45-54|female coupling signature + presence anchors reproduce.
    cs = q8["cell_45_54_female_reference"]["coupling_signature"]
    rcs = rec["cell_45_54_female_reference"]["coupling_signature"]
    for key in cs:
        assert cs[key] == pytest.approx(rcs[key], abs=ATOL)


def test_pin_q9_spouse_concept_reproduces(seed0):
    """Q9: seed-0 spouse code-20/code-22 shares (forensics-1 splitter)."""
    gf3 = seed0["gf3"]
    concept = gf3.q9_reference(
        seed0["data"]["rel_map"],
        seed0["data"]["hh"].person_waves,
        seed0["ids_b"],
    )
    rec = seed0["recorded"]["q9_concept"]
    assert concept["share_code20_legal_only"] == pytest.approx(
        rec["share_code20_legal_only"], abs=ATOL
    )
    assert concept["share_code22_cohab_only"] == pytest.approx(
        rec["share_code22_cohab_only"], abs=ATOL
    )
    assert abs(concept["reconciliation_remainder"]) < 1e-12
    for cell in (
        "coresident_spouse.25-34|female",
        "coresident_spouse.65-74|female",
    ):
        got = concept["by_cell_code_share"][cell]
        want = rec["by_cell_code_share"][cell]
        assert got["share_code20_legal"] == pytest.approx(
            want["share_code20_legal"], abs=ATOL
        )
        assert got["share_code22_cohab"] == pytest.approx(
            want["share_code22_cohab"], abs=ATOL
        )


def test_pin_q10_reference_joint_reproduces(seed0):
    """Q10: seed-0 (core size, non-core count) train joint + core dist."""
    gf3 = seed0["gf3"]
    q10 = gf3.q10_reference(
        seed0["data"]["hh"],
        seed0["data"]["fu_sizes"],
        seed0["data"]["parent_pairs"],
        seed0["ids_b"],
    )
    rec = seed0["recorded"]["q10_reference"]
    for size, val in q10["reference_hh_size_distribution"].items():
        assert val == pytest.approx(
            rec["reference_hh_size_distribution"][size], abs=ATOL
        )
    for core, val in q10["reference_noncore_incidence_by_capped_core"].items():
        assert val == pytest.approx(
            rec["reference_noncore_incidence_by_capped_core"][core], abs=ATOL
        )
    # The registered incidence shape: core-3 carries the most non-core members
    # (~0.35), cores 4/5 far fewer (~0.08).
    inc = q10["reference_noncore_incidence_by_capped_core"]
    assert inc["3"] > 0.25
    assert inc["4"] < 0.15
    assert inc["5"] < 0.15
