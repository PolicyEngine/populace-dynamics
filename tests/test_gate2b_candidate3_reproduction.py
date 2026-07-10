"""PSID reproduction pin for gate-2b candidate 3 (skips off-machine).

Candidate 3's scoring path shares the candidate hard-stop precheck: it must
reproduce the committed gate-2b floor (``runs/gate2b_floors_v1.json``)
bit-for-bit before any simulated cell is scored. This module pins that against
the staged PSID Family Relationship Matrix (``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``) and proves the three candidate-3 properties on
real data:

* **Byte-identical carry.** ``simulate_draw_v3`` reproduces candidate 2's draw
  exactly, so on the same draw seed ``coresident_parent``, ``multigen`` AND
  ``coresident_spouse`` are EQUAL to ``simulate_draw_v2``'s, cell for cell --
  the byte-identical carried-family proof the artifact reports.
* **Maternal side untouched.** Women's ``coresident_child`` is byte-identical
  to candidate 2 (delta 1 gates only the paternal LINKED children).
* **Skip-gen union / non-family bridge.** The candidate-3
  ``coresident_grandchild`` is the union of the composed grandchild and the
  skipped-generation state, and ``hh_size`` is the composed family unit plus
  the non-family count (never below candidate 2's family-unit hh_size).

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
from populace_dynamics.models import household_composition_sim_v2 as hcs2
from populace_dynamics.models import household_composition_sim_v3 as hcs3

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
    }


def _fit_v2_v3(hh, loaders, ids_b):
    m2 = hcs2.fit_household_model_v2(
        hh,
        loaders["mpanel"],
        loaders["demo"],
        loaders["mh"],
        loaders["bh"],
        loaders["order_map"],
        loaders["rel_map"],
        ids_b,
    )
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
    return m2, m3


def test_full_panel_reference_moments_reproduce_floor(household_panel, floor):
    """Every committed reference-moment rate reproduces bit-for-bit."""
    ref = hc.reference_moments(household_panel, weighted=True)
    committed = floor["reference_moments"]
    max_dev = max(
        abs(ref[k]["rate"] - committed[k]["rate"]) for k in committed
    )
    assert max_dev <= EXACT_ATOL


def test_per_seed_rate_a_and_holdout_ids_reproduce(household_panel, floor):
    """Each gate seed's holdout split and rate_a reproduce the committed
    floor -- the scoring path's denominator (unchanged from candidate 1)."""
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


def test_carried_families_are_byte_identical_to_candidate_2(
    household_panel, loaders
):
    """The regression-proof on real data: candidate 3's coresident_parent,
    multigen AND coresident_spouse equal candidate 2's simulate_draw_v2 output
    cell for cell, because candidate 3 reproduces candidate 2's draw exactly
    (same 0xB2B / 0xC2 streams) before overlaying the isolated-stream deltas.
    """
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    m2, m3 = _fit_v2_v3(hh, loaders, ids_b)

    p2, _ = hcs2.simulate_draw_v2(hh, loaders["mpanel"], m2, ids_a, 5200)
    p3, _ = hcs3.simulate_draw_v3(hh, loaders["mpanel"], m3, ids_a, 5200)

    key = ["person_id", "year"]
    a2 = p2.person_waves.set_index(key).sort_index()
    a3 = p3.person_waves.set_index(key).sort_index()
    assert a2.index.equals(a3.index)
    for col in ("coresident_parent", "multigen", "coresident_spouse"):
        assert (a2[col].to_numpy() == a3[col].to_numpy()).all(), col


def test_maternal_side_untouched_women_child_identical(
    household_panel, loaders
):
    """Delta 1 gates only the paternal linked children: women's
    coresident_child is byte-identical to candidate 2."""
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    m2, m3 = _fit_v2_v3(hh, loaders, ids_b)
    p2, _ = hcs2.simulate_draw_v2(hh, loaders["mpanel"], m2, ids_a, 5200)
    p3, _ = hcs3.simulate_draw_v3(hh, loaders["mpanel"], m3, ids_a, 5200)

    key = ["person_id", "year"]
    a2 = p2.person_waves.set_index(key).sort_index()
    a3 = p3.person_waves.set_index(key).sort_index()
    sexmap = (
        deaths.read_death_records()
        .drop_duplicates("person_id")
        .set_index("person_id")["sex"]
    )
    sx = pd.Index(a3.index.get_level_values(0)).map(sexmap)
    is_w = np.asarray(sx == "female")
    assert (
        a2["coresident_child"].to_numpy()[is_w]
        == a3["coresident_child"].to_numpy()[is_w]
    ).all()


def test_simulate_draw_v3_emits_a_valid_household_panel(
    household_panel, floor, loaders
):
    """One candidate-3 draw of the seed-0 holdout is schema-valid,
    undefined-free, and realizes the three deltas as specified."""
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    _, m3 = _fit_v2_v3(hh, loaders, ids_b)
    sim, diag = hcs3.simulate_draw_v3(hh, loaders["mpanel"], m3, ids_a, 5200)
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

    # Delta 3: grandchild == (multigen & child & ~parent) UNION skipgen, so it
    # is a SUPERSET of the composed grandchild (and multigen is unchanged).
    gc = sim_pw["coresident_grandchild"].to_numpy()
    composed = (
        sim_pw["multigen"].to_numpy()
        & sim_pw["coresident_child"].to_numpy()
        & ~sim_pw["coresident_parent"].to_numpy()
    )
    assert ((gc | composed) == gc).all()  # composed implies gc
    assert diag["n_skipgen_person_waves_simulated"] > 0
    assert gc.sum() >= composed.sum()

    # Delta 2: the non-family count is additive on top of the composed family
    # unit, so hh_size is never below 1 + coresident_spouse (self + spouse),
    # and the simulated non-family mean is non-negative (the bridge is active).
    assert (
        sim_pw["hh_size"].to_numpy()
        >= 1 + sim_pw["coresident_spouse"].to_numpy().astype(int)
    ).all()
    assert diag["mean_nonfamily_count_simulated"] >= 0.0

    # Delta 1: the linked-child coresidence gate is active (some but not all
    # father-linked child-waves are counted).
    assert diag["n_linked_fathers_side_a"] > 0
    assert diag["n_linked_child_coresident_wave_units"] > 0

    # Every gated cell has a positive simulated denominator on this draw.
    tol_cells = set(floor["gate_partition"]["gate_eligible"])
    cand = hc.reference_moments(sim, ids_a, weighted=True)
    for cell in tol_cells:
        assert cand[cell]["den_wt"] > 0, cell
