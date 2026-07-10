"""PSID reproduction pin for gate-2b candidate 2 (skips off-machine).

Candidate 2's scoring path shares candidate 1's hard-stop precheck: it must
reproduce the committed gate-2b floor (``runs/gate2b_floors_v1.json``)
bit-for-bit before any simulated cell is scored. This module pins that against
the staged PSID Family Relationship Matrix (``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``) and, crucially, proves the two candidate-2
properties on real data:

* **Byte-identical carry.** ``simulate_draw_v2`` reads ``coresident_parent``
  and ``multigen`` off candidate 1's ``simulate_draw`` unchanged, so on the
  same draw seed those states are EQUAL, cell for cell -- the regression-proof
  the artifact's cleared-family check reports.
* **Additive spouse union.** The candidate-2 ``coresident_spouse`` is a
  SUPERSET of candidate 1's legal-marriage state (the cohabitation overlay
  only adds partner mass; it never removes a legal spouse).

Marked ``integration_psid``; skipped when the PSID products are not staged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from populace_dynamics.data import household_composition as hc
from populace_dynamics.harness import panel as hpanel
from populace_dynamics.models import household_composition_sim as hcs
from populace_dynamics.models import household_composition_sim_v2 as hcs2

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
        deaths,
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
    }


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

    This is the scoring path's denominator: candidate 2 is scored against
    exactly these side-A rates (unchanged from candidate 1), so reproducing
    them is the precondition of a faithful one-shot.
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


def test_carried_families_are_byte_identical_to_candidate_1(
    household_panel, loaders
):
    """The regression-proof on real data: candidate 2's coresident_parent and
    multigen equal candidate 1's simulate_draw output cell for cell.

    Because candidate 2 reads those states off candidate 1's simulate_draw
    unchanged (same seed, same occupancy tag), they cannot drift -- so the
    parental-home and multigen families that cleared in candidate 1 cannot
    regress.
    """
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

    model = hcs2.fit_household_model_v2(
        hh,
        loaders["mpanel"],
        loaders["demo"],
        loaders["mh"],
        loaders["bh"],
        loaders["order_map"],
        loaders["rel_map"],
        ids_b,
    )
    # candidate 1's simulate on the same base model + seed.
    c1_panel = hcs.simulate_draw(
        hh, loaders["mpanel"], model.base, ids_a, 5200
    )
    v2_panel, _ = hcs2.simulate_draw_v2(
        hh, loaders["mpanel"], model, ids_a, 5200
    )

    key = ["person_id", "year"]
    c1 = c1_panel.person_waves.set_index(key).sort_index()
    v2 = v2_panel.person_waves.set_index(key).sort_index()
    assert c1.index.equals(v2.index)
    # The two carried families are byte-identical.
    assert (c1["coresident_parent"] == v2["coresident_parent"]).all()
    assert (c1["multigen"] == v2["multigen"]).all()
    # The spouse union is a superset of candidate 1's legal-marriage state.
    c1_spouse = c1["coresident_spouse"].to_numpy(dtype=bool)
    v2_spouse = v2["coresident_spouse"].to_numpy(dtype=bool)
    assert (v2_spouse | c1_spouse == v2_spouse).all()  # c1 implies v2
    assert v2_spouse.sum() >= c1_spouse.sum()


def test_simulate_draw_v2_emits_a_valid_household_panel(
    household_panel, floor, loaders
):
    """One candidate-2 draw of the seed-0 holdout is schema-valid and
    undefined-free (the undefined-draw rule would otherwise invalidate)."""
    hh = household_panel
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

    model = hcs2.fit_household_model_v2(
        hh,
        loaders["mpanel"],
        loaders["demo"],
        loaders["mh"],
        loaders["bh"],
        loaders["order_map"],
        loaders["rel_map"],
        ids_b,
    )
    sim, diag = hcs2.simulate_draw_v2(
        hh, loaders["mpanel"], model, ids_a, 5200
    )
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
    # Composition identity holds on the simulated panel (grandchild).
    gc = sim_pw["coresident_grandchild"].to_numpy()
    implied = (
        sim_pw["multigen"].to_numpy()
        & sim_pw["coresident_child"].to_numpy()
        & ~sim_pw["coresident_parent"].to_numpy()
    )
    assert (gc == implied).all()

    # Delta-2 diagnostics: linked links present, a shadow residual remains.
    assert diag["n_paternal_linked_births"] > 0
    assert diag["n_linked_fathers_side_a"] > 0
    assert diag["n_linked_fathers_side_a"] < diag["n_side_a_men"]

    # Every gated cell has a positive simulated denominator on this draw.
    tol_cells = set(floor["gate_partition"]["gate_eligible"])
    cand = hc.reference_moments(sim, ids_a, weighted=True)
    for cell in tol_cells:
        assert cand[cell]["den_wt"] > 0, cell
