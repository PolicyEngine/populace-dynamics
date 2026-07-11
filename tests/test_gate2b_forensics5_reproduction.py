"""PSID reproduction pins for gate-2b forensics 5 (skips off-machine).

Marked ``integration_psid`` (references ``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``); skipped when the PSID Family Relationship
Matrix is not staged. The pins rebuild seed 0's train-side (side B)
decompositions live and match the recorded per-seed values to float precision:
the instrumented-draw fidelity proof (bit-identity to ``simulate_draw_v7`` and
the exact channel additivities), the Q14 linked-father reference anchors and
completed-family-size distributions, and the Q15 reference hh_size kernel
sanity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_forensics5_v1.json"
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
    """Load the panels, split seed 0, and fit candidate 7 on side B."""
    sys.path.insert(0, str(SCRIPTS))
    import gate2b_forensics5 as gf5

    from populace_dynamics.harness import panel as hpanel
    from populace_dynamics.models import household_composition_sim_v7 as hcs7

    data = gf5.c7.load_all()
    hh, mpanel = data["hh"], data["mpanel"]
    _side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = {int(x) for x in side_b.person_id.unique()}
    model = hcs7.fit_household_model_v7(
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
        marital_by_year=data["marital_by_year"],
        fu_sizes=data["fu_sizes"],
        legal_flag=data["legal_flag"],
        child_record_expo=data["child_record_expo"],
        parent_counts=data["parent_counts"],
    )
    recorded = next(s for s in _artifact()["per_seed"] if s["seed"] == 0)
    return {
        "gf5": gf5,
        "data": data,
        "ids_b": ids_b,
        "model": model,
        "recorded": recorded,
    }


def test_pin_instrumented_draw_is_bit_identical(seed0):
    """The instrumented draw reproduces simulate_draw_v7 bit-for-bit, and the
    child channels + linked marital split reconcile exactly."""
    gf5 = seed0["gf5"]
    fid = gf5.fidelity_check_v7(
        seed0["data"]["hh"],
        seed0["data"]["mpanel"],
        seed0["model"],
        seed0["ids_b"],
        gf5.DRAW_SEED_BASE,
    )
    assert fid["bit_identical"] is True
    assert fid["max_abs_rate_deviation_vs_committed_simulate_draw_v7"] == 0.0
    assert fid["child_channel_additivity_residual"] == 0
    assert fid["linked_marital_split_additivity_residual"] == 0
    assert fid["analytic_stock_out_of_range_or_nonzero_no_exposure"] == 0


def test_pin_q14_linked_reference_reproduces(seed0):
    """Q14: seed-0 linked-father anchors at the older male cells reproduce."""
    gf5 = seed0["gf5"]
    linked = gf5.q14_linked_reference(
        seed0["data"]["hh"],
        seed0["data"]["father_links_child"],
        seed0["data"]["parent_pairs"],
        seed0["data"]["marital_by_year"],
        seed0["model"],
        seed0["ids_b"],
        gf5.Q14_MALE_CELLS,
    )
    rec = seed0["recorded"]["q14_linked_reference"]
    for cell in gf5.Q14_MALE_CELLS:
        for key in (
            "reference_full_rate",
            "reference_linked_any_contribution",
            "reference_unlinked_any_contribution",
            "reference_a_refexp_joinable_contribution",
            "reference_s_joinable_restricted_contribution",
        ):
            assert linked[cell][key] == pytest.approx(
                rec[cell][key], abs=ATOL
            ), (cell, key)


def test_pin_q14_train_completed_size_reproduces(seed0):
    """Q14/Q15: the train completed-family-size cell distributions reproduce."""
    gf5 = seed0["gf5"]
    hh = seed0["data"]["hh"]
    ids_b = seed0["ids_b"]
    size_map = gf5.train_completed_size(seed0["data"]["parent_pairs"], ids_b)
    pw_b = hh.person_waves[hh.person_waves["person_id"].isin(ids_b)]
    pw_b = pw_b[pw_b["band"].notna()]
    rec = seed0["recorded"]["q14_train_dk"]
    for cell in gf5.Q14_CELLS:
        bl, sx = gf5._cell_of(cell)
        sub = pw_b[(pw_b["band"] == bl) & (pw_b["sex"] == sx)][
            ["person_id", "weight", "coresident_child"]
        ]
        d, k, full = gf5._train_cell_dk(sub, size_map)
        assert full == pytest.approx(rec[cell]["ref_full"], abs=ATOL)
        for b in gf5.SIZE_BUCKETS:
            assert d[b] == pytest.approx(rec[cell]["d_train"][b], abs=ATOL)
            assert k[b] == pytest.approx(rec[cell]["k_train"][b], abs=ATOL)


def test_pin_q15_reference_hh_kernel_reproduces(seed0):
    """Q15: the reference hh_size kernel reproduces the direct hh moments."""
    gf5 = seed0["gf5"]
    hh = seed0["data"]["hh"]
    ids_b = seed0["ids_b"]
    size_map = gf5.train_completed_size(seed0["data"]["parent_pairs"], ids_b)
    pw_all = hh.person_waves[hh.person_waves["person_id"].isin(ids_b)][
        ["person_id", "weight", "hh_size"]
    ]
    _d_all, _h, ref_hh = gf5._train_hh_kernel(pw_all, size_map)
    rec = seed0["recorded"]["q15_reference_hh"]
    rate_b = seed0["recorded"]["rate_b"]
    for j in ("3", "4", "5+"):
        assert ref_hh[j] == pytest.approx(rec[j], abs=ATOL)
        # The kernel convolution reproduces the direct reference moment.
        assert ref_hh[j] == pytest.approx(rate_b[f"hh_size.{j}"], abs=ATOL)
