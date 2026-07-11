"""PSID reproduction pin for gate-2b candidate 9 (skips off-machine).

Proves the candidate-9 scope properties on the staged PSID Family Relationship
Matrix (``~/PolicyEngine/psid-data`` / ``POPULACE_DYNAMICS_PSID_DIR``):

* **Deficit cohorts byte-identical to candidate 8.** ``simulate_draw_v9``
  reproduces ``simulate_draw_v8`` exactly on the same draw seed for the four
  deficit scope child cells and every carried family (the scoped write gate
  reuses candidate 8's per-cohort draws bit-for-bit).
* **Non-deficit cohorts revert to candidate 7.** Every non-deficit
  coresident_child cell equals ``simulate_draw_v7``'s exactly -- the four
  candidate-8 collateral cells return to their cleared candidate-7 state.
* **The pre-run analytic check** prices ``hh_size.5+`` (the middle cohorts carry
  the large majority of candidate 8's lift) and predicts the four collateral
  cells revert to cleared.

Marked via the PSID skip; skipped when the PSID products are not staged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from populace_dynamics.data import household_composition as hc
from populace_dynamics.harness import panel as hpanel
from populace_dynamics.models import household_composition_sim_v3 as hcs3
from populace_dynamics.models import household_composition_sim_v4 as hcs4
from populace_dynamics.models import household_composition_sim_v5 as hcs5
from populace_dynamics.models import household_composition_sim_v7 as hcs7
from populace_dynamics.models import household_composition_sim_v8 as hcs8
from populace_dynamics.models import household_composition_sim_v9 as hcs9

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
GATES = ROOT / "gates.yaml"
REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()

needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "MX23REL").is_dir(),
    reason="PSID MX23REL relationship matrix not staged",
)
pytestmark = needs_psid

GATE_SEEDS = (0, 1, 2, 3, 4)
EXACT_ATOL = 1e-12
LIFTED_SPOUSE_CELL = "coresident_spouse.25-34|female"
DRAW_SEED = 5200


@pytest.fixture(scope="module")
def household_panel():
    return hc.build_household_panel()


@pytest.fixture(scope="module")
def floor():
    return json.loads(FLOOR.read_text())


@pytest.fixture(scope="module")
def tol():
    gates = yaml.safe_load(GATES.read_text())
    th = gates["gates"]["gate_2"]["gate_2b"]["thresholds"]
    out = {}
    for view in th["views"].values():
        for cell, value in view["tolerances"].items():
            out[cell] = float(value)
    return out


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
        "marital_by_year": marital_by_year,
        "fu_sizes": hcs3.family_unit_sizes(rel_map),
        "legal_flag": hcs4.legal_spouse_flag(rel_map),
        "parent_counts": hcs5.parent_link_counts(rel_map),
        "child_record_expo": hcs5.build_child_record_exposure(
            father_links_child, parent_pairs, marital_by_year, demo, rel_map
        ),
    }


def _fit(fn, hh, loaders, ids_b, **kw):
    return fn(
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
        marital_by_year=loaders["marital_by_year"],
        fu_sizes=loaders["fu_sizes"],
        legal_flag=loaders["legal_flag"],
        child_record_expo=loaders["child_record_expo"],
        parent_counts=loaders["parent_counts"],
        **kw,
    )


def _seed0(hh):
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    return (
        set(int(x) for x in side_a.person_id.unique()),
        set(int(x) for x in side_b.person_id.unique()),
    )


@pytest.fixture(scope="module")
def draws(household_panel, loaders):
    """v7 / v8 / v9 reference moments on seed 0, draw 5200 (fit once, reuse)."""
    hh = household_panel
    ids_a, ids_b = _seed0(hh)
    m7 = _fit(hcs7.fit_household_model_v7, hh, loaders, ids_b)
    m8 = _fit(hcs8.fit_household_model_v8, hh, loaders, ids_b)
    p7, _ = hcs7.simulate_draw_v7(hh, loaders["mpanel"], m7, ids_a, DRAW_SEED)
    p8, _ = hcs8.simulate_draw_v8(hh, loaders["mpanel"], m8, ids_a, DRAW_SEED)
    p9, d9 = hcs9.simulate_draw_v9(hh, loaders["mpanel"], m8, ids_a, DRAW_SEED)
    return {
        "ids_a": ids_a,
        "ids_b": ids_b,
        "m8": m8,
        "r7": hc.reference_moments(p7, ids_a, weighted=True),
        "r8": hc.reference_moments(p8, ids_a, weighted=True),
        "r9": hc.reference_moments(p9, ids_a, weighted=True),
        "diag9": d9,
    }


# --------------------------------------------------------------------------
# Hard-stop precheck reproduction
# --------------------------------------------------------------------------
def test_full_panel_reference_moments_reproduce_floor(household_panel, floor):
    ref = hc.reference_moments(household_panel, weighted=True)
    committed = floor["reference_moments"]
    max_dev = max(
        abs(ref[k]["rate"] - committed[k]["rate"]) for k in committed
    )
    assert max_dev <= EXACT_ATOL


def test_per_seed_holdout_ids_reproduce(household_panel, floor):
    committed_ho = {p["seed"]: p for p in floor["holdout_ids"]["per_seed"]}
    for seed in GATE_SEEDS:
        side_a, _ = hpanel.split_panel_by_person(
            household_panel.attrs, "person_id", fraction=0.5, seed=seed
        )
        ids = sorted(int(x) for x in side_a.person_id.unique())
        digest = hashlib.sha256(
            ",".join(str(i) for i in ids).encode()
        ).hexdigest()
        assert digest == committed_ho[seed]["holdout_person_id_sha256"]


# --------------------------------------------------------------------------
# Deficit cohorts + carries byte-identical to candidate 8
# --------------------------------------------------------------------------
def test_carried_families_byte_identical_to_candidate8_and_7(draws):
    r7, r8, r9 = draws["r7"], draws["r8"], draws["r9"]
    for cell in (
        "coresident_spouse.35-44|male",
        "coresident_spouse.55-64|female",
        "coresident_parent.15-24|male",
        "multigen.25-34|female",
        "multigen_entry",
        "parental_home_exit.25-34|female",
        "coresident_grandchild.45-54|female",
    ):
        assert abs(r9[cell]["rate"] - r8[cell]["rate"]) <= EXACT_ATOL, cell
        assert abs(r9[cell]["rate"] - r7[cell]["rate"]) <= EXACT_ATOL, cell


def test_deficit_scope_cells_byte_identical_to_candidate8(draws):
    r8, r9 = draws["r8"], draws["r9"]
    for cell in hcs9.FERTILITY_LIFT_CELLS:
        assert abs(r9[cell]["rate"] - r8[cell]["rate"]) <= EXACT_ATOL, cell


def test_lifted_spouse_cell_byte_identical_to_candidate8(draws):
    # delta 2 is carried byte-faithfully; the 25-34|female lift equals c8's.
    r8, r9 = draws["r8"], draws["r9"]
    assert (
        abs(r9[LIFTED_SPOUSE_CELL]["rate"] - r8[LIFTED_SPOUSE_CELL]["rate"])
        <= EXACT_ATOL
    )


# --------------------------------------------------------------------------
# Non-deficit cohorts revert to candidate 7 byte-identically
# --------------------------------------------------------------------------
def test_reverted_child_cells_byte_identical_to_candidate7(draws):
    r7, r9 = draws["r7"], draws["r9"]
    for cell in hcs9.REVERTED_CHILD_CELLS:
        assert abs(r9[cell]["rate"] - r7[cell]["rate"]) <= EXACT_ATOL, cell


def test_collateral_cells_revert_and_move_off_candidate8(draws):
    """The four candidate-8 collateral cells return to candidate 7 (revert) and
    move off candidate 8 (the global-lift overshoot removed)."""
    r7, r8, r9 = draws["r7"], draws["r8"], draws["r9"]
    for cell in hcs9.COLLATERAL_CELLS:
        assert abs(r9[cell]["rate"] - r7[cell]["rate"]) <= EXACT_ATOL, cell
        assert abs(r9[cell]["rate"] - r8[cell]["rate"]) > 1e-6, cell


def test_only_deficit_cohorts_move_off_candidate7(draws):
    """Among the gated coresident_child cells, exactly the four deficit cohorts
    move off candidate 7; every other reverts."""
    r7, r9 = draws["r7"], draws["r9"]
    child_cells = [c for c in r9 if c.startswith("coresident_child.")]
    for cell in child_cells:
        if cell not in r7:
            continue
        moved = abs(r9[cell]["rate"] - r7[cell]["rate"]) > 1e-6
        expected = cell in hcs9.FERTILITY_LIFT_CELLS
        # report-only 75+ cells are also composition bands but not gated; the
        # gated deficit cohorts are the four scope cells.
        if cell in hcs9.FERTILITY_LIFT_CELLS or cell in (
            hcs9.REVERTED_CHILD_CELLS
        ):
            assert moved == expected, cell


# --------------------------------------------------------------------------
# hh_size.5+ sits between candidate 7 and candidate 8 (the priced cell)
# --------------------------------------------------------------------------
def test_hh_size_5plus_between_candidate7_and_candidate8(draws):
    r7, r8, r9 = draws["r7"], draws["r8"], draws["r9"]
    v7 = r7["hh_size.5+"]["rate"]
    v8 = r8["hh_size.5+"]["rate"]
    v9 = r9["hh_size.5+"]["rate"]
    # the scoped lift delivers only the deficit cohorts' share -> between the
    # unlifted candidate 7 and the fully-lifted candidate 8.
    assert v7 <= v9 <= v8
    assert v9 < v8  # strictly less lift than the global candidate 8


# --------------------------------------------------------------------------
# The pre-run analytic check reproduces (priced hh_size.5+ + collateral revert)
# --------------------------------------------------------------------------
def test_analytic_check_prices_hh_size_and_predicts_collateral(
    household_panel, loaders, draws, tol
):
    hh = household_panel
    ids_b = draws["ids_b"]
    reference_b = hc.reference_moments(hh, ids_b, weighted=True)
    ac = hcs9.scoped_lift_analytic_check(
        draws["m8"], hh, loaders["mpanel"], ids_b, reference_b, tol
    )
    hp = ac["hh_size_5plus_priced"]
    # the middle cohorts carry the large majority of candidate 8's lift.
    assert hp["global_lift_candidate8"] > hp["scoped_lift_candidate9"] > 0.0
    assert 0.7 < hp["middle_cohort_share_of_lift"] < 0.95
    # the collateral cells' scoped counterfactual is their sim rate (reverted).
    for cell in hcs9.COLLATERAL_CELLS:
        c = ac["cells"][cell]
        assert c["in_scope"] is False
        assert abs(c["scoped_counterfactual"] - c["sim_train"]) <= 1e-9
        assert c["predicted_within_tolerance"] is True
    # the deficit scope cells keep the global (fertility) counterfactual.
    for cell in ("coresident_child.55-64|male", "coresident_child.65-74|male"):
        c = ac["cells"][cell]
        assert c["in_scope"] is True
        assert (
            abs(c["scoped_counterfactual"] - c["global_counterfactual"])
            <= 1e-9
        )
