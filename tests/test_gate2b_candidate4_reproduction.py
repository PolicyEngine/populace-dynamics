"""PSID reproduction pin for gate-2b candidate 4 (skips off-machine).

Candidate 4's scoring path shares the candidate hard-stop precheck: it must
reproduce the committed gate-2b floor (``runs/gate2b_floors_v1.json``)
bit-for-bit before any simulated cell is scored. This module pins that against
the staged PSID Family Relationship Matrix (``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``) and proves the candidate-4 carry + delta
properties on real data:

* **Byte-identical carry.** ``simulate_draw_v4`` reproduces candidate 3's
  carried families exactly, so on the same draw seed ``coresident_parent`` and
  ``multigen`` are EQUAL to ``simulate_draw_v3``'s, cell for cell -- the
  byte-identical carried-family proof the artifact reports (coresident_spouse
  is NOT carried; deltas 1 and 2 target it).
* **Maternal / women's child untouched.** Women's ``coresident_child`` is
  byte-identical to candidate 3 (delta 3a gates only the paternal LINKED
  children of men).
* **The four deltas fire.** The legal-spouse residual overlay, the age-refined
  cohabitation overlay, the non-family 2+ tail spread, and the 5-year skip-gen
  rebuild each realize their states as specified.

Marked ``integration_psid``; skipped when the PSID products are not staged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.data import deaths
from populace_dynamics.data import household_composition as hc
from populace_dynamics.harness import panel as hpanel
from populace_dynamics.models import household_composition_sim_v3 as hcs3
from populace_dynamics.models import household_composition_sim_v4 as hcs4

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()

needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "MX23REL").is_dir(),
    reason="PSID MX23REL relationship matrix not staged",
)
pytestmark = needs_psid

GATE_SEEDS = (0, 1, 2, 3, 4)
EXACT_ATOL = 1e-12


@pytest.fixture(scope="module")
def household_panel():
    return hc.build_household_panel()


@pytest.fixture(scope="module")
def floor():
    return json.loads(FLOOR.read_text())


@pytest.fixture(scope="module")
def loaders():
    from populace_dynamics.data import (
        births,
        marriage,
        panels,
        relmap,
        transitions,
    )

    rel_map = relmap.relationship_map()
    demo = panels.demographic_panel()
    sex = deaths.read_death_records()
    mh = marriage.marriage_history()
    bh = births.birth_history()
    demo_pos = demo[demo.weight > 0]
    person_weight = (
        demo_pos.sort_values("period")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["weight"]
    )
    mpanel = transitions.build_marital_panel(mh, sex, person_weight)
    ep = marriage.marriage_episodes(mh)
    ep = ep[ep["start_year"].notna()].copy()
    ep["start_year"] = ep["start_year"].astype("int64")
    ep = ep.sort_values(["person_id", "start_year"])
    ep["order"] = ep.groupby("person_id").cumcount() + 1
    order_map = ep[["person_id", "start_year", "order"]].drop_duplicates(
        ["person_id", "start_year"]
    )
    return {
        "rel_map": rel_map,
        "demo": demo,
        "mh": mh,
        "bh": bh,
        "mpanel": mpanel,
        "order_map": order_map,
        "father_links_child": hcs3.father_link_births_with_child(bh),
        "parent_pairs": hcs3.parent_child_coresidence_pairs(rel_map),
        "fu_sizes": hcs3.family_unit_sizes(rel_map),
        "legal_flag": hcs4.legal_spouse_flag(rel_map),
    }


def _fit_v3_v4(hh, loaders, ids_b):
    m3 = hcs3.fit_household_model_v3(
        hh,
        loaders["mpanel"],
        loaders["demo"],
        loaders["mh"],
        loaders["bh"],
        loaders["order_map"],
        loaders["rel_map"],
        ids_b,
        father_links_child=loaders["father_links_child"],
        parent_pairs=loaders["parent_pairs"],
        fu_sizes=loaders["fu_sizes"],
    )
    m4 = hcs4.fit_household_model_v4(
        hh,
        loaders["mpanel"],
        loaders["demo"],
        loaders["mh"],
        loaders["bh"],
        loaders["order_map"],
        loaders["rel_map"],
        ids_b,
        father_links_child=loaders["father_links_child"],
        parent_pairs=loaders["parent_pairs"],
        fu_sizes=loaders["fu_sizes"],
        legal_flag=loaders["legal_flag"],
    )
    return m3, m4


def test_full_panel_reference_moments_reproduce_floor(household_panel, floor):
    ref = hc.reference_moments(household_panel, weighted=True)
    committed = floor["reference_moments"]
    max_dev = max(
        abs(ref[k]["rate"] - committed[k]["rate"]) for k in committed
    )
    assert max_dev <= EXACT_ATOL


def test_per_seed_rate_a_and_holdout_ids_reproduce(household_panel, floor):
    committed_ho = {p["seed"]: p for p in floor["holdout_ids"]["per_seed"]}
    committed_ns = {p["seed"]: p for p in floor["noise_floor_per_seed"]}
    for seed in GATE_SEEDS:
        side_a, _ = hpanel.split_panel_by_person(
            household_panel.attrs, "person_id", fraction=0.5, seed=seed
        )
        ids = sorted(int(x) for x in side_a.person_id.unique())
        digest = hashlib.sha256(
            ",".join(str(i) for i in ids).encode()
        ).hexdigest()
        assert digest == committed_ho[seed]["holdout_person_id_sha256"]
        cells = hc.reference_moments(household_panel, set(ids), weighted=True)
        committed = committed_ns[seed]["cells"]
        max_dev = max(
            abs(cells[k]["rate"] - committed[k]["rate_a"]) for k in committed
        )
        assert max_dev <= EXACT_ATOL, seed


def test_carried_families_are_byte_identical_to_candidate_3(
    household_panel, loaders
):
    """coresident_parent and multigen equal candidate 3's simulate_draw_v3
    output cell for cell (candidate 4 reproduces candidate 3's 0xB2B / 0xC2 /
    0xC3 streams before overlaying the shape-preserving refits and the
    isolated-0xC4 additions)."""
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    m3, m4 = _fit_v3_v4(hh, loaders, ids_b)

    p3, _ = hcs3.simulate_draw_v3(hh, loaders["mpanel"], m3, ids_a, 5200)
    p4, _ = hcs4.simulate_draw_v4(hh, loaders["mpanel"], m4, ids_a, 5200)

    key = ["person_id", "year"]
    a3 = p3.person_waves.set_index(key).sort_index()
    a4 = p4.person_waves.set_index(key).sort_index()
    assert a3.index.equals(a4.index)
    for col in ("coresident_parent", "multigen"):
        assert (a3[col].to_numpy() == a4[col].to_numpy()).all(), col
    # coresident_spouse IS changed (deltas 1 and 2 target it).
    assert not (
        a3["coresident_spouse"].to_numpy()
        == a4["coresident_spouse"].to_numpy()
    ).all()


def test_women_child_byte_identical_delta3a_touches_men_only(
    household_panel, loaders
):
    """Delta 3a re-keys only the paternal LINKED children, so women's
    coresident_child is byte-identical to candidate 3."""
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    m3, m4 = _fit_v3_v4(hh, loaders, ids_b)
    p3, _ = hcs3.simulate_draw_v3(hh, loaders["mpanel"], m3, ids_a, 5200)
    p4, _ = hcs4.simulate_draw_v4(hh, loaders["mpanel"], m4, ids_a, 5200)

    key = ["person_id", "year"]
    a3 = p3.person_waves.set_index(key).sort_index()
    a4 = p4.person_waves.set_index(key).sort_index()
    sexmap = (
        deaths.read_death_records()
        .drop_duplicates("person_id")
        .set_index("person_id")["sex"]
    )
    sx = pd.Index(a4.index.get_level_values(0)).map(sexmap)
    is_w = np.asarray(sx == "female")
    assert (
        a3["coresident_child"].to_numpy()[is_w]
        == a4["coresident_child"].to_numpy()[is_w]
    ).all()


def test_simulate_draw_v4_realizes_the_four_deltas(
    household_panel, floor, loaders
):
    """One candidate-4 draw of the seed-0 holdout is schema-valid,
    undefined-free, and realizes the four deltas as specified."""
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    _, m4 = _fit_v3_v4(hh, loaders, ids_b)
    sim, diag = hcs4.simulate_draw_v4(hh, loaders["mpanel"], m4, ids_a, 5200)
    sim_pw = sim.person_waves

    real = hh.person_waves[hh.person_waves["person_id"].isin(ids_a)]
    assert len(sim_pw) == len(real)
    for col in (
        "coresident_spouse",
        "coresident_parent",
        "coresident_child",
        "coresident_grandchild",
        "multigen",
        "hh_size",
    ):
        assert col in sim_pw.columns
    assert (sim_pw["hh_size"] >= 1).all()

    # Delta 2: the legal-spouse residual overlay fires.
    assert diag["n_legal_residual_person_waves_simulated"] > 0
    # Delta 1: cohabitation single-year overlay is active.
    assert diag["n_cohab_person_waves_simulated"] > 0
    # Delta 3b: the 2+ tail spread lifts the mean count above the truncated 2.
    assert diag["mean_nonfamily_count_within_2plus_simulated"] > 2.0
    # Delta 4: the skip-gen occupancy is active and grandchild is a superset of
    # the composed grandchild.
    assert diag["n_skipgen_person_waves_simulated"] > 0
    gc = sim_pw["coresident_grandchild"].to_numpy()
    composed = (
        sim_pw["multigen"].to_numpy()
        & sim_pw["coresident_child"].to_numpy()
        & ~sim_pw["coresident_parent"].to_numpy()
    )
    assert ((gc | composed) == gc).all()

    # Delta 3a: the custodial gate is active.
    assert diag["n_linked_child_coresident_wave_units"] > 0

    # Every gated cell has a positive simulated denominator on this draw.
    tol_cells = set(floor["gate_partition"]["gate_eligible"])
    cand = hc.reference_moments(sim, ids_a, weighted=True)
    for cell in tol_cells:
        assert cand[cell]["den_wt"] > 0, cell


def test_legal_residual_fires_where_core_underproduces(
    household_panel, loaders
):
    """Delta 2's residual overlay is active exactly where the target
    (ref_code20 - core_legal) is positive -- the older-male bands the
    forensics found the core under-produces."""
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = set(int(x) for x in side_b.person_id.unique())
    _, m4 = _fit_v3_v4(hh, loaders, ids_b)
    # The 65-74 and 75+ male bands under-produce -> positive target.
    assert m4.legal_residual_target[("65-74", "male")] > 0
    assert m4.legal_residual_target[("75+", "male")] > 0
    # Young bands where the core meets the reference -> zero (overlay off).
    assert m4.legal_residual_target[("35-44", "female")] == 0.0
