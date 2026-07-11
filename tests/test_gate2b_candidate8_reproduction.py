"""PSID reproduction pin for gate-2b candidate 8 (skips off-machine).

Proves the candidate-8 carry + delta properties on the staged PSID Family
Relationship Matrix (``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``):

* **Byte-identical carry.** ``simulate_draw_v8`` reproduces candidate 7's
  carried families exactly on the same draw seed -- ``coresident_parent``,
  ``coresident_spouse`` (every band EXCEPT the delta-2 lifted 25-34|female),
  ``multigen`` and ``coresident_grandchild`` are EQUAL to ``simulate_draw_v7``'s,
  cell for cell (candidate 8 changes ONLY coresident_child / hh_size /
  coresident_spouse.25-34|female on the isolated 0xC8 stream).
* **Delta 1 (fertility-core lift).** hh_size.5+ and coresident_child.55-64|male
  lift toward the reference; the fit reproduces the Q15 headline.
* **Delta 2 (cohab-overlay lift).** coresident_spouse.25-34|female lifts by the
  Bernoulli superposition of the -0.045 overlay shortfall.
* **Delta 3 (band-signed retention + link-coverage).** The exit-origin closure is
  band-signed (reduce 45-54|female over-retention, lift 65-74) and the
  link-coverage channel is closed at the older-male bands.

Marked via the PSID skip; skipped when the PSID products are not staged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from populace_dynamics.data import household_composition as hc
from populace_dynamics.harness import panel as hpanel
from populace_dynamics.models import household_composition_sim_v3 as hcs3
from populace_dynamics.models import household_composition_sim_v4 as hcs4
from populace_dynamics.models import household_composition_sim_v5 as hcs5
from populace_dynamics.models import household_composition_sim_v7 as hcs7
from populace_dynamics.models import household_composition_sim_v8 as hcs8

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
FORENSICS = ROOT / "runs" / "gate2b_forensics5_v1.json"
REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()

needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "MX23REL").is_dir(),
    reason="PSID MX23REL relationship matrix not staged",
)
pytestmark = needs_psid

GATE_SEEDS = (0, 1, 2, 3, 4)
EXACT_ATOL = 1e-12
LIFTED_SPOUSE_CELL = "coresident_spouse.25-34|female"


@pytest.fixture(scope="module")
def household_panel():
    return hc.build_household_panel()


@pytest.fixture(scope="module")
def floor():
    return json.loads(FLOOR.read_text())


@pytest.fixture(scope="module")
def forensics():
    return json.loads(FORENSICS.read_text())


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


def _fit_v7(hh, loaders, ids_b):
    return hcs7.fit_household_model_v7(
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
    )


def _fit_v8(hh, loaders, ids_b, n_fit_draws=3):
    return hcs8.fit_household_model_v8(
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
        n_fit_draws=n_fit_draws,
    )


def _seed0(hh):
    side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    return (
        set(int(x) for x in side_a.person_id.unique()),
        set(int(x) for x in side_b.person_id.unique()),
    )


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


def test_per_seed_rate_a_and_holdout_ids_reproduce(household_panel, floor):
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
# Byte-identical carry vs candidate 7
# --------------------------------------------------------------------------
def test_carried_families_byte_identical_to_candidate_7(
    household_panel, loaders
):
    hh = household_panel
    ids_a, ids_b = _seed0(hh)
    m7 = _fit_v7(hh, loaders, ids_b)
    m8 = _fit_v8(hh, loaders, ids_b)
    p7, _ = hcs7.simulate_draw_v7(hh, loaders["mpanel"], m7, ids_a, 5200)
    p8, _ = hcs8.simulate_draw_v8(hh, loaders["mpanel"], m8, ids_a, 5200)
    key = ["person_id", "year"]
    a7 = p7.person_waves.set_index(key).sort_index()
    a8 = p8.person_waves.set_index(key).sort_index()
    assert a7.index.equals(a8.index)
    # coresident_parent / multigen / coresident_grandchild carried UNCHANGED.
    for col in (
        "coresident_parent",
        "multigen",
        "coresident_grandchild",
    ):
        assert (a7[col].to_numpy() == a8[col].to_numpy()).all(), col
    # gated carried cells byte-identical (spouse EXCEPT the lifted 25-34|female).
    r7 = hc.reference_moments(p7, ids_a, weighted=True)
    r8 = hc.reference_moments(p8, ids_a, weighted=True)
    for cell in (
        "coresident_spouse.35-44|male",
        "coresident_spouse.55-64|female",
        "coresident_parent.15-24|male",
        "multigen.25-34|female",
        "multigen_entry",
        "multigen_exit",
        "parental_home_exit.25-34|female",
        "coresident_grandchild.55+|female",
        "coresident_grandchild.45-54|female",
    ):
        assert abs(r7[cell]["rate"] - r8[cell]["rate"]) <= EXACT_ATOL, cell


# --------------------------------------------------------------------------
# Delta 2: the cohabitation-overlay lift at 25-34|female
# --------------------------------------------------------------------------
def test_delta2_lifts_spouse_25_34_female(household_panel, loaders):
    hh = household_panel
    ids_a, ids_b = _seed0(hh)
    m7 = _fit_v7(hh, loaders, ids_b)
    m8 = _fit_v8(hh, loaders, ids_b)
    p7, _ = hcs7.simulate_draw_v7(hh, loaders["mpanel"], m7, ids_a, 5200)
    p8, diag = hcs8.simulate_draw_v8(hh, loaders["mpanel"], m8, ids_a, 5200)
    r7 = hc.reference_moments(p7, ids_a, weighted=True)
    r8 = hc.reference_moments(p8, ids_a, weighted=True)
    # the overlay lift raises the fragile cell (Bernoulli superposition
    # new = old + 0.045 * (1 - old); the realized rate change is
    # 0.045 * (1 - old) ~ 0.019 at old ~ 0.58, NOT the 0.045 parameter).
    assert r8[LIFTED_SPOUSE_CELL]["rate"] > r7[LIFTED_SPOUSE_CELL]["rate"]
    assert 0.012 < diag["cohab_overlay_lift"]["realized_lift"] < 0.03


# --------------------------------------------------------------------------
# Delta 1: fertility-core lift raises the large-family / older-male cells
# --------------------------------------------------------------------------
def test_delta1_lifts_hh_size_5plus_and_55_64_male(household_panel, loaders):
    hh = household_panel
    ids_a, ids_b = _seed0(hh)
    m7 = _fit_v7(hh, loaders, ids_b)
    m8 = _fit_v8(hh, loaders, ids_b)
    p7, _ = hcs7.simulate_draw_v7(hh, loaders["mpanel"], m7, ids_a, 5200)
    p8, _ = hcs8.simulate_draw_v8(hh, loaders["mpanel"], m8, ids_a, 5200)
    r7 = hc.reference_moments(p7, ids_a, weighted=True)
    r8 = hc.reference_moments(p8, ids_a, weighted=True)
    assert r8["hh_size.5+"]["rate"] > r7["hh_size.5+"]["rate"]
    assert (
        r8["coresident_child.55-64|male"]["rate"]
        > r7["coresident_child.55-64|male"]["rate"]
    )
    # the hard 65-74|male cell lifts under the multi-channel closure.
    assert (
        r8["coresident_child.65-74|male"]["rate"]
        > r7["coresident_child.65-74|male"]["rate"]
    )


# --------------------------------------------------------------------------
# Delta 3: band-signed exit-origin closure (both signs) + link coverage
# --------------------------------------------------------------------------
def test_delta3_band_signed_shift_signs(household_panel, loaders):
    hh = household_panel
    _ids_a, ids_b = _seed0(hh)
    m8 = _fit_v8(hh, loaders, ids_b)
    shift = m8.retention_link_shift
    # 45-54|female over-retains -> the shift REDUCES it (negative).
    assert shift["coresident_child.45-54|female"] < 0.0
    # 65-74|male / 65-74|female under-retain -> the shift LIFTS (positive).
    assert shift["coresident_child.65-74|male"] > 0.0
    assert shift["coresident_child.65-74|female"] > 0.0
    # 55-64|male gets the link-coverage closure (positive).
    assert shift["coresident_child.55-64|male"] > 0.0


def test_delta3_reduces_45_54_female_over_retention(household_panel, loaders):
    hh = household_panel
    ids_a, ids_b = _seed0(hh)
    m7 = _fit_v7(hh, loaders, ids_b)
    m8 = _fit_v8(hh, loaders, ids_b)
    p7, _ = hcs7.simulate_draw_v7(hh, loaders["mpanel"], m7, ids_a, 5200)
    p8, _ = hcs8.simulate_draw_v8(hh, loaders["mpanel"], m8, ids_a, 5200)
    r7 = hc.reference_moments(p7, ids_a, weighted=True)
    r8 = hc.reference_moments(p8, ids_a, weighted=True)
    # candidate 7 over-produced 45-54|female; delta 3 reduces it (band-signed).
    assert (
        r8["coresident_child.45-54|female"]["rate"]
        < r7["coresident_child.45-54|female"]["rate"]
    )


# --------------------------------------------------------------------------
# The three deltas' fit-vs-raw checks reproduce forensics-5
# --------------------------------------------------------------------------
def test_delta_checks_reproduce_forensics5(
    household_panel, loaders, forensics
):
    hh = household_panel
    _ids_a, ids_b = _seed0(hh)
    m8 = _fit_v8(hh, loaders, ids_b)
    ck = hcs8.c8_delta_checks(m8, forensics)
    hh5 = ck["delta_1_fertility_core_lift"]["reproduction_hh_size_5plus"]
    assert hh5["headline"] == "0.128 -> 0.144 (vs reference 0.139)"
    q16 = ck["delta_2_cohab_overlay_lift"]["reproduction_spouse_25_34_female"]
    assert q16["headline"] == "0.588 -> 0.606 (vs reference 0.621)"
    # the fitted delta-3 exit-origin signs match the Q14 measured channels.
    bs = ck["delta_3_retention_link_refit"]["band_sign_fit_vs_raw"]
    assert bs["coresident_child.45-54|female"]["exit_sign"] == (
        "reduce_over_retention"
    )
    assert bs["coresident_child.65-74|male"]["exit_sign"] == (
        "lift_under_retention"
    )
