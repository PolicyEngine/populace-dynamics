"""PSID reproduction pin for gate-2b candidate 1 (skips off-machine).

The candidate's scoring path must reproduce the committed gate-2b floor
(``runs/gate2b_floors_v1.json``) bit-for-bit -- the same person-disjoint
half-split and the same weighted household-composition moments the tolerances
are derived from -- before any simulated cell is scored. This is the hard-stop
precheck the one-shot run runs first, pinned here against the staged PSID
Family Relationship Matrix (``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``). It also exercises one simulated draw of the
holdout to prove the six-component generator emits a schema-valid,
undefined-free household panel on real data.

Marked ``integration_psid``; skipped when the PSID products are not staged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from populace_dynamics.data import household_composition as hc
from populace_dynamics.harness import panel as hpanel

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


def test_full_panel_reference_moments_reproduce_floor(household_panel, floor):
    """Every committed reference-moment rate reproduces bit-for-bit."""
    ref = hc.reference_moments(household_panel, weighted=True)
    committed = floor["reference_moments"]
    max_dev = max(
        abs(ref[k]["rate"] - committed[k]["rate"]) for k in committed
    )
    assert max_dev <= EXACT_ATOL


def test_per_seed_rate_a_and_holdout_ids_reproduce(household_panel, floor):
    """Each gate seed's holdout split and rate_a reproduce the committed floor.

    This is the scoring path's denominator: the candidate is scored against
    exactly these side-A rates, so reproducing them is the precondition of a
    faithful one-shot.
    """
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


def test_simulate_draw_emits_a_valid_household_panel(household_panel, floor):
    """One draw of the seed-0 holdout is schema-valid and undefined-free.

    Same rows as the real side-A exposure, hh_size >= 1, and every gated cell
    has a positive simulated denominator (the undefined-draw rule would
    otherwise invalidate the run).
    """
    from populace_dynamics.data import (
        births,
        deaths,
        marriage,
        panels,
        relmap,
        transitions,
    )
    from populace_dynamics.models import household_composition_sim as hcs

    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

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

    model = hcs.fit_household_model(
        hh, mpanel, demo, mh, bh, order_map, rel_map, ids_b
    )
    sim = hcs.simulate_draw(hh, mpanel, model, ids_a, 5200)
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
    # Composition identity holds on the simulated panel (component 6).
    gc = sim_pw["coresident_grandchild"].to_numpy()
    implied = (
        sim_pw["multigen"].to_numpy()
        & sim_pw["coresident_child"].to_numpy()
        & ~sim_pw["coresident_parent"].to_numpy()
    )
    assert (gc == implied).all()

    # Every gated cell has a positive simulated denominator on this draw.
    tol_cells = set(floor["gate_partition"]["gate_eligible"])
    cand = hc.reference_moments(sim, ids_a, weighted=True)
    for cell in tol_cells:
        assert cand[cell]["den_wt"] > 0, cell
