"""PSID reproduction pins for gate-2b forensics 4 (skips off-machine).

Marked ``integration_psid`` (references ``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``); skipped when the PSID Family Relationship
Matrix is not staged. The pins rebuild seed 0's train-side (side B)
decompositions live and match the recorded per-seed values to float precision:
the instrumented-draw fidelity proof (bit-identity to ``simulate_draw_v6`` and
the exact channel additivities), the Q11 deterministic reference (existence
counts, analytic occupancy, episode lengths), the Q12 delta-3-off inertness at
the 25-34|female target via a single-draw replay, and the Q13 reference core /
large-family structure.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_forensics4_v1.json"
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
    """Load the panels, split seed 0, and fit candidate 6 on side B."""
    sys.path.insert(0, str(SCRIPTS))
    import gate2b_forensics4 as gf4

    from populace_dynamics.harness import panel as hpanel
    from populace_dynamics.models import household_composition_sim_v3 as hcs3
    from populace_dynamics.models import household_composition_sim_v6 as hcs6

    data = gf4.c6.load_all()
    hh, mpanel = data["hh"], data["mpanel"]
    _side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = {int(x) for x in side_b.person_id.unique()}
    model = hcs6.fit_household_model_v6(
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
        "gf4": gf4,
        "hcs3": hcs3,
        "hcs6": hcs6,
        "data": data,
        "ids_b": ids_b,
        "model": model,
        "recorded": recorded,
    }


def test_pin_instrumented_draw_is_bit_identical(seed0):
    """The instrumented draw reproduces simulate_draw_v6 bit-for-bit, and the
    child channels + linked marital split reconcile exactly."""
    gf4 = seed0["gf4"]
    fid = gf4.fidelity_check_v6(
        seed0["data"]["hh"],
        seed0["data"]["mpanel"],
        seed0["model"],
        seed0["ids_b"],
        gf4.DRAW_SEED_BASE,
    )
    assert fid["bit_identical"] is True
    assert fid["max_abs_rate_deviation_vs_committed_simulate_draw_v6"] == 0.0
    assert fid["child_channel_additivity_residual"] == 0
    assert fid["linked_marital_split_additivity_residual"] == 0
    assert fid["analytic_stock_out_of_range_or_nonzero_no_exposure"] == 0


def test_pin_q11_reference_reproduces(seed0):
    """Q11: seed-0 existence counts, analytic occupancy, and episode lengths."""
    gf4 = seed0["gf4"]
    hcs3 = seed0["hcs3"]
    marital_by_year = hcs3._father_marital_by_year(seed0["data"]["mpanel"])
    q11 = gf4.q11_reference(
        seed0["data"]["hh"],
        seed0["data"]["father_links_child"],
        seed0["data"]["parent_pairs"],
        marital_by_year,
        seed0["model"],
        seed0["ids_b"],
    )
    rec = seed0["recorded"]["q11_reference"]
    for cell in gf4.Q11_CELLS:
        got = q11["per_cell"][cell]
        want = rec["per_cell"][cell]
        for key in (
            "reference_full_male_rate",
            "reference_linked_any_contribution",
            "reference_linked_restricted_contribution",
            "reference_a_refexp_all_contribution",
            "reference_a_refexp_joinable_contribution",
            "reference_unlinked_any_contribution",
            "mean_exposed_linked_children_per_father_wave",
            "mean_coresident_linked_children_per_father_wave",
        ):
            assert got[key] == pytest.approx(want[key], abs=ATOL), (cell, key)
    # The episode-length distribution reproduces (reference contiguous spells).
    for b, v in q11["reference_episode_length_distribution"].items():
        assert v == pytest.approx(
            rec["reference_episode_length_distribution"][b], abs=ATOL
        )
    assert q11["reference_mean_episode_length"] == pytest.approx(
        rec["reference_mean_episode_length"], abs=ATOL
    )
    # The reference episodes are LONGER than the sim's (occupancy vs episode).
    assert (
        rec["reference_mean_episode_length"]
        > seed0["recorded"]["q11_sim_episodes"]["mean_episode_length"]
    )


def test_pin_q11_existence_is_identical_exposure(seed0):
    """Q11: the exposed-linked-child count per father-wave is a shared value
    (sim == reference) -- the existence channel is a structural zero."""
    gf4 = seed0["gf4"]
    hcs3 = seed0["hcs3"]
    marital_by_year = hcs3._father_marital_by_year(seed0["data"]["mpanel"])
    q11 = gf4.q11_reference(
        seed0["data"]["hh"],
        seed0["data"]["father_links_child"],
        seed0["data"]["parent_pairs"],
        marital_by_year,
        seed0["model"],
        seed0["ids_b"],
    )
    art = _artifact()["question_11_linked_father_child_supply"]
    recorded = seed0["recorded"]["q11_reference"]["per_cell"]
    for cell in gf4.Q11_CELLS:
        # The existence channel is an EXACT structural zero.
        assert (
            art["per_cell"][cell]["channels"]["existence_identical_exposure"]
            == 0.0
        )
        # The live seed-0 reference exposed distribution reproduces the recorded
        # seed-0 distribution to float precision.
        ref_exposed = q11["per_cell"][cell][
            "reference_exposed_count_distribution"
        ]
        rec_exposed = recorded[cell]["reference_exposed_count_distribution"]
        art_ref = art["per_cell"][cell]["existence_distributions"][
            "reference_exposed_linked_count"
        ]
        art_sim = art["per_cell"][cell]["existence_distributions"][
            "sim_exposed_linked_count"
        ]
        for b in ("0", "1", "2", "3+"):
            assert ref_exposed[b] == pytest.approx(rec_exposed[b], abs=ATOL)
            # Sim and reference exposed distributions essentially coincide.
            assert art_sim[b] == pytest.approx(art_ref[b], abs=0.03)


def test_pin_q12_delta3_inert_at_target_via_replay(seed0):
    """Q12: disabling delta 3 does not move the 25-34|female spouse rate."""
    gf4 = seed0["gf4"]
    model = seed0["model"]
    model_no_d3 = dataclasses.replace(
        model, cohab_entry_age_female={}, cohab_exit_age_female={}
    )
    hh, mpanel, ids_b = (
        seed0["data"]["hh"],
        seed0["data"]["mpanel"],
        seed0["ids_b"],
    )
    bands = ["25-34"]
    d_c6 = gf4.instrumented_draw_v6(hh, mpanel, model, ids_b, 5200)
    d_no = gf4.instrumented_draw_v6(hh, mpanel, model_no_d3, ids_b, 5200)
    r_c6 = gf4._spouse_female_rates(d_c6, bands)["25-34"]["full"]
    r_no = gf4._spouse_female_rates(d_no, bands)["25-34"]["full"]
    # Delta 3 coincides with candidate 4's estimator at 25-34: inert.
    assert abs(r_c6 - r_no) < 1e-3
    # And the recorded 20-draw effect is inert too.
    q12 = _artifact()["question_12_spouse_25_34_female_movement"]
    assert abs(q12["delta3_target_effect_full"]) < 1e-6


def test_pin_q13_reference_core_structure_reproduces(seed0):
    """Q13: seed-0 reference core-5+, hh-5+, and 3+-own-child share."""
    hh = seed0["data"]["hh"]
    fu = seed0["data"]["fu_sizes"]
    pp = seed0["data"]["parent_pairs"]
    ids_b = seed0["ids_b"]
    ppw = hh.person_waves[hh.person_waves["person_id"].isin(ids_b)][
        ["person_id", "year", "weight", "hh_size"]
    ].merge(fu, on=["person_id", "year"], how="left")
    ppw["family_unit_size"] = ppw["family_unit_size"].fillna(1).astype("int64")
    rw = ppw["weight"].to_numpy(np.float64)
    rtot = float(rw.sum())
    core_ref = ppw["family_unit_size"].to_numpy()
    ref_core5plus = float(rw[core_ref >= 5].sum() / rtot)
    ref_hh5plus = float(rw[ppw["hh_size"].to_numpy() >= 5].sum() / rtot)
    # 3+-own-child family share (the forensics-3 Q10 upstream fertility signal).
    pp_b = pp[pp["parent_person_id"].isin(ids_b)]
    childcnt = pp_b.groupby(["parent_person_id", "year"]).size().rename("_n")
    owncnt = ppw[["person_id", "year", "weight"]].merge(
        childcnt.reset_index().rename(
            columns={"parent_person_id": "person_id"}
        ),
        on=["person_id", "year"],
        how="left",
    )
    owncnt["_n"] = owncnt["_n"].fillna(0).astype("int64")
    ow = owncnt["weight"].to_numpy(np.float64)
    ref_child3plus = float(ow[owncnt["_n"].to_numpy() >= 3].sum() / ow.sum())
    rec = seed0["recorded"]
    assert ref_core5plus == pytest.approx(
        rec["q13_reference_core5plus"], abs=ATOL
    )
    assert ref_hh5plus == pytest.approx(rec["q13_reference_hh5plus"], abs=ATOL)
    assert ref_child3plus == pytest.approx(
        rec["q13_reference_child3plus_share"], abs=ATOL
    )
    # The sim core-5+ under-produces the reference (the upstream deficit).
    assert rec["q13_sim_core5plus"] < rec["q13_reference_core5plus"]
