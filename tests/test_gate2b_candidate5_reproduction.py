"""PSID reproduction pin for gate-2b candidate 5 (skips off-machine).

Candidate 5's scoring path shares the candidate hard-stop precheck: it must
reproduce the committed gate-2b floor (``runs/gate2b_floors_v1.json``)
bit-for-bit before any simulated cell is scored. This module pins that against
the staged PSID Family Relationship Matrix (``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``) and proves the candidate-5 carry + delta
properties on real data:

* **Byte-identical carry (incl. spouse).** ``simulate_draw_v5`` reproduces
  candidate 4's carried families exactly, so on the same draw seed
  ``coresident_spouse``, ``coresident_parent`` and ``multigen`` are EQUAL to
  ``simulate_draw_v4``'s, cell for cell.
* **Multigen marginal unchanged.** The delta-1 coupling reads the multigen
  state but never changes it -- ``multigen`` is byte-identical to candidate 4
  (the load-bearing spec constraint).
* **Women's / married-father child untouched.** Women's ``coresident_child`` is
  byte-identical to candidate 4 (delta 2 corrects only NOT-married linked
  fathers, who are men).
* **The three deltas fire.** The coupled adult-child grandchild, the not-married
  child-record custodial swap, the core-size bridge reach and the per-ego
  parent-count draw each realize their states as specified.

Marked via the PSID skip; skipped when the PSID products are not staged.
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
from populace_dynamics.models import household_composition_sim_v4 as hcs4
from populace_dynamics.models import household_composition_sim_v5 as hcs5

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
    from populace_dynamics.models import household_composition_sim_v3 as hcs3

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
    father_links_child = hcs3.father_link_births_with_child(bh)
    parent_pairs = hcs3.parent_child_coresidence_pairs(rel_map)
    marital_by_year = hcs3._father_marital_by_year(mpanel)
    return {
        "rel_map": rel_map,
        "demo": demo,
        "mh": mh,
        "bh": bh,
        "mpanel": mpanel,
        "order_map": order_map,
        "father_links_child": father_links_child,
        "parent_pairs": parent_pairs,
        "fu_sizes": hcs3.family_unit_sizes(rel_map),
        "legal_flag": hcs4.legal_spouse_flag(rel_map),
        "parent_counts": hcs5.parent_link_counts(rel_map),
        "child_record_expo": hcs5.build_child_record_exposure(
            father_links_child, parent_pairs, marital_by_year, demo, rel_map
        ),
    }


def _fit_v4_v5(hh, loaders, ids_b):
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
    m5 = hcs5.fit_household_model_v5(
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
        child_record_expo=loaders["child_record_expo"],
        parent_counts=loaders["parent_counts"],
    )
    return m4, m5


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


def test_carried_families_byte_identical_to_candidate_4(
    household_panel, loaders
):
    """coresident_spouse, coresident_parent and multigen equal candidate 4's
    simulate_draw_v4 output cell for cell -- candidate 5 reproduces the
    candidate-4 0xB2B / 0xC2 / 0xC3 / 0xC4 streams before the three deltas."""
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    m4, m5 = _fit_v4_v5(hh, loaders, ids_b)

    p4, _ = hcs4.simulate_draw_v4(hh, loaders["mpanel"], m4, ids_a, 5200)
    p5, _ = hcs5.simulate_draw_v5(hh, loaders["mpanel"], m5, ids_a, 5200)

    key = ["person_id", "year"]
    a4 = p4.person_waves.set_index(key).sort_index()
    a5 = p5.person_waves.set_index(key).sort_index()
    assert a4.index.equals(a5.index)
    # coresident_spouse IS carried in candidate 5 (no c5 delta targets it).
    for col in ("coresident_spouse", "coresident_parent", "multigen"):
        assert (a4[col].to_numpy() == a5[col].to_numpy()).all(), col


def test_multigen_marginal_unchanged_gated_cells(household_panel, loaders):
    """The delta-1 load-bearing constraint: every gated multigen stock cell
    rate is byte-identical to candidate 4 (the coupling reads multigen but
    never changes the marginal)."""
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    m4, m5 = _fit_v4_v5(hh, loaders, ids_b)
    p4, _ = hcs4.simulate_draw_v4(hh, loaders["mpanel"], m4, ids_a, 5200)
    p5, _ = hcs5.simulate_draw_v5(hh, loaders["mpanel"], m5, ids_a, 5200)
    r4 = hc.reference_moments(p4, ids_a, weighted=True)
    r5 = hc.reference_moments(p5, ids_a, weighted=True)
    for cell in (
        "multigen.15-24|female",
        "multigen.25-34|female",
        "multigen.45-54|female",
        "multigen_entry",
        "multigen_exit",
    ):
        assert abs(r4[cell]["rate"] - r5[cell]["rate"]) <= EXACT_ATOL, cell


def test_women_and_married_child_byte_identical(household_panel, loaders):
    """Delta 2 corrects only NOT-married linked fathers (men), so women's
    coresident_child is byte-identical to candidate 4."""
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    m4, m5 = _fit_v4_v5(hh, loaders, ids_b)
    p4, _ = hcs4.simulate_draw_v4(hh, loaders["mpanel"], m4, ids_a, 5200)
    p5, _ = hcs5.simulate_draw_v5(hh, loaders["mpanel"], m5, ids_a, 5200)
    key = ["person_id", "year"]
    a4 = p4.person_waves.set_index(key).sort_index()
    a5 = p5.person_waves.set_index(key).sort_index()
    sexmap = (
        deaths.read_death_records()
        .drop_duplicates("person_id")
        .set_index("person_id")["sex"]
    )
    sx = pd.Index(a5.index.get_level_values(0)).map(sexmap)
    is_w = np.asarray(sx == "female")
    assert (
        a4["coresident_child"].to_numpy()[is_w]
        == a5["coresident_child"].to_numpy()[is_w]
    ).all()
    # The gated 45-54|female grandchild cell is byte-identical (coupling 55+).
    r4 = hc.reference_moments(p4, ids_a, weighted=True)
    r5 = hc.reference_moments(p5, ids_a, weighted=True)
    c = "coresident_grandchild.45-54|female"
    assert abs(r4[c]["rate"] - r5[c]["rate"]) <= EXACT_ATOL


def test_simulate_draw_v5_realizes_the_three_deltas(
    household_panel, floor, loaders
):
    """One candidate-5 draw of the seed-0 holdout is schema-valid,
    undefined-free, and realizes the three deltas as specified."""
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    _, m5 = _fit_v4_v5(hh, loaders, ids_b)
    sim, diag = hcs5.simulate_draw_v5(hh, loaders["mpanel"], m5, ids_a, 5200)
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

    # Delta 1: the coupled adult-child grandchild fires and lifts the joint
    # above the independence product (the coupling).
    assert diag["n_coupled_grandchild_waves_simulated"] > 0
    assert (
        diag["coupling_gc55f_joint_mg_child_notparent"]
        > diag["coupling_gc55f_independence_product"]
    )
    # Delta 2: the custodial gate is active and the not-married child-record
    # rates were fit.
    assert diag["n_linked_child_coresident_wave_units"] > 0
    nm = m5.meta["custodial_child_record"]["not_married_child_record_by_band"]
    # forensics-2: observable over-states at school ages -> child-record < obs.
    assert 0.0 < nm["5-12"] < 0.6
    # Delta 3: the parent-count draw yields a mix (not the fixed 2), and the
    # core-size bridge conditions on dense cells.
    assert 1.0 < diag["mean_n_parents_among_coresident_parent"] < 2.0
    assert m5.meta["nonfamily_by_core"]["n_dense_core_band_sex_cells"] > 0

    # grandchild is a superset of the composed grandchild (skipgen union).
    gc = sim_pw["coresident_grandchild"].to_numpy()
    composed = (
        sim_pw["multigen"].to_numpy()
        & sim_pw["coresident_child"].to_numpy()
        & ~sim_pw["coresident_parent"].to_numpy()
    )
    below55 = sim_pw["age"].to_numpy() < hcs5.GRANDCHILD_LO
    # Below 55 the composed grandchild is a subset of the union (unchanged
    # composition | skipgen); at 55+ the coupling replaces the composed input.
    assert ((gc | composed)[below55] == gc[below55]).all()

    # Every gated cell has a positive simulated denominator on this draw.
    tol_cells = set(floor["gate_partition"]["gate_eligible"])
    cand = hc.reference_moments(sim, ids_a, weighted=True)
    for cell in tol_cells:
        assert cand[cell]["den_wt"] > 0, cell


def test_coupling_lifts_grandchild_and_bridge_lowers_hh_size3(
    household_panel, loaders
):
    """Delta 1 lifts grandchild 55+|female far above candidate 4; delta 3
    lowers hh_size.3 toward the reference (single-draw direction check)."""
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    m4, m5 = _fit_v4_v5(hh, loaders, ids_b)
    p4, _ = hcs4.simulate_draw_v4(hh, loaders["mpanel"], m4, ids_a, 5200)
    p5, _ = hcs5.simulate_draw_v5(hh, loaders["mpanel"], m5, ids_a, 5200)
    r4 = hc.reference_moments(p4, ids_a, weighted=True)
    r5 = hc.reference_moments(p5, ids_a, weighted=True)
    # Coupling roughly doubles the grandchild 55+|female stock.
    assert (
        r5["coresident_grandchild.55+|female"]["rate"]
        > 1.8 * r4["coresident_grandchild.55+|female"]["rate"]
    )
    # Bridge reach + parent-count composition lowers hh_size.3.
    assert r5["hh_size.3"]["rate"] < r4["hh_size.3"]["rate"]
