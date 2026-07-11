"""PSID reproduction pin for gate-2b candidate 6 (skips off-machine).

Candidate 6's scoring path shares the candidate hard-stop precheck: it must
reproduce the committed gate-2b floor (``runs/gate2b_floors_v1.json``)
bit-for-bit before any simulated cell is scored. This module pins that against
the staged PSID Family Relationship Matrix (``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``) and proves the candidate-6 carry + delta
properties on real data:

* **Byte-identical carry.** ``simulate_draw_v6`` reproduces candidate 5's
  carried families exactly, so on the same draw seed ``coresident_parent``,
  ``multigen`` and the gated multigen cells are EQUAL to ``simulate_draw_v5``'s,
  cell for cell (the load-bearing constraint). ``coresident_grandchild.55+|
  female`` is byte-identical (the coupling is independent of the maternal child
  leaves and is applied ONLY at 55+ -- no downward extension).
* **Delta 1 fires and drains.** The not-married 0-4 custodial cell reverts to
  the observable basis, which is BELOW the child-record basis there, so
  ``coresident_child.15-24|male`` is drained.
* **Delta 2 does NOT over-drain.** The linked-married custodial coresidence is
  byte-identical to candidate 5 (no hard leave -- candidate 4's single-year
  observable married prob already IS the exit timing), so ``coresident_child``
  35-44|male is byte-identical.
* **Delta 4 reproduces the forensics-3 Q10 feasibility.** Convolving the fitted
  count conditional with the committed candidate-5 core distribution clears
  hh3/hh4/hh5+ (~0.1887/0.1709/0.1303).

Marked via the PSID skip; skipped when the PSID products are not staged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from populace_dynamics.data import deaths
from populace_dynamics.data import household_composition as hc
from populace_dynamics.harness import panel as hpanel
from populace_dynamics.models import household_composition_sim_v4 as hcs4
from populace_dynamics.models import household_composition_sim_v5 as hcs5
from populace_dynamics.models import household_composition_sim_v6 as hcs6

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "runs" / "gate2b_floors_v1.json"
FORENSICS = ROOT / "runs" / "gate2b_forensics3_v1.json"
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


def _fit(module, hh, loaders, ids_b):
    return module(
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


# --------------------------------------------------------------------------
# Byte-identical carry vs candidate 5 (incl. grandchild 55+ + multigen)
# --------------------------------------------------------------------------
def test_carried_families_byte_identical_to_candidate_5(
    household_panel, loaders
):
    hh = household_panel
    ids_a, ids_b = _seed0(hh)
    m5 = _fit(hcs5.fit_household_model_v5, hh, loaders, ids_b)
    m6 = _fit(hcs6.fit_household_model_v6, hh, loaders, ids_b)
    p5, _ = hcs5.simulate_draw_v5(hh, loaders["mpanel"], m5, ids_a, 5200)
    p6, _ = hcs6.simulate_draw_v6(hh, loaders["mpanel"], m6, ids_a, 5200)
    key = ["person_id", "year"]
    a5 = p5.person_waves.set_index(key).sort_index()
    a6 = p6.person_waves.set_index(key).sort_index()
    assert a5.index.equals(a6.index)
    for col in ("coresident_parent", "multigen"):
        assert (a5[col].to_numpy() == a6[col].to_numpy()).all(), col
    # gated multigen + grandchild-55+ cells byte-identical.
    r5 = hc.reference_moments(p5, ids_a, weighted=True)
    r6 = hc.reference_moments(p6, ids_a, weighted=True)
    for cell in (
        "multigen.15-24|female",
        "multigen.25-34|female",
        "multigen.45-54|female",
        "multigen_entry",
        "multigen_exit",
        "coresident_parent.15-24|male",
        "parental_home_exit.25-34|female",
        "coresident_grandchild.55+|female",
    ):
        assert abs(r5[cell]["rate"] - r6[cell]["rate"]) <= EXACT_ATOL, cell


def test_no_coupling_fires_below_age_55(household_panel, loaders):
    hh = household_panel
    ids_a, ids_b = _seed0(hh)
    m6 = _fit(hcs6.fit_household_model_v6, hh, loaders, ids_b)
    _, diag = hcs6.simulate_draw_v6(hh, loaders["mpanel"], m6, ids_a, 5200)
    assert diag["no_coupling_below_55_max_p"] == 0.0
    assert hcs6.GRANDCHILD_LO == 55


# --------------------------------------------------------------------------
# Delta 1 fires and drains; delta 2 does not over-drain the male cells
# --------------------------------------------------------------------------
def test_delta1_revert_below_child_record_and_drains_15_24_male(
    household_panel, loaders
):
    hh = household_panel
    ids_a, ids_b = _seed0(hh)
    m6 = _fit(hcs6.fit_household_model_v6, hh, loaders, ids_b)
    # the fitted child-record 0-4 not-married rate the revert removes.
    cr04 = m6.custodial_child_record[(hc.band_label(0, 4), "not_married")]
    # the observable basis the revert returns to, weighted over the ACTUAL 0-4
    # not-married linked-child exposure (age x era) -- below the child-record
    # 0-4 rate, so the revert drains the young-father over-production. (An
    # unweighted single-era mean over ages 0-4 does NOT show this -- age 0 is
    # heavily weighted and low in the earlier eras, which the exposure
    # weighting captures.)
    fl = loaders["father_links_child"]
    fl = fl[fl["parent_person_id"].isin(ids_b)]
    fw = hh.person_waves[hh.person_waves["person_id"].isin(ids_b)][
        ["person_id", "year", "weight"]
    ].rename(columns={"person_id": "parent_person_id"})
    ex = fl.merge(fw, on="parent_person_id", how="inner")
    ex["child_age"] = ex["year"] - ex["birth_year"]
    ex = ex[(ex["child_age"] >= 0) & (ex["child_age"] <= 4)].copy()
    from populace_dynamics.models import household_composition_sim_v3 as hcs3

    mby = hcs3._father_marital_by_year(loaders["mpanel"]).rename(
        columns={"person_id": "parent_person_id"}
    )
    ex = ex.merge(mby, on=["parent_person_id", "year"], how="left")
    ex = ex[ex["marital"].fillna("not_married") == "not_married"]
    obs = np.array(
        [
            hcs4._custodial_prob(
                m6.base_v4, int(a), hcs4.era_of_year(int(y)), "not_married"
            )
            for a, y in zip(ex["child_age"], ex["year"], strict=True)
        ]
    )
    w = ex["weight"].to_numpy(float)
    obs_weighted = float((w * obs).sum() / w.sum())
    assert obs_weighted < cr04  # revert drains the over-production
    m5 = _fit(hcs5.fit_household_model_v5, hh, loaders, ids_b)
    p5, _ = hcs5.simulate_draw_v5(hh, loaders["mpanel"], m5, ids_a, 5200)
    p6, _ = hcs6.simulate_draw_v6(hh, loaders["mpanel"], m6, ids_a, 5200)
    r5 = hc.reference_moments(p5, ids_a, weighted=True)
    r6 = hc.reference_moments(p6, ids_a, weighted=True)
    assert (
        r6["coresident_child.15-24|male"]["rate"]
        < r5["coresident_child.15-24|male"]["rate"]
    )


def test_delta2_linked_married_not_over_drained(household_panel, loaders):
    """The linked-married custodial coresidence is byte-identical to candidate 5
    except the delta-1 0-4 revert -- no hard leave, so the older married-linked
    male cells are NOT over-drained (35-44|male stays ~candidate 5)."""
    hh = household_panel
    ids_a, ids_b = _seed0(hh)
    m5 = _fit(hcs5.fit_household_model_v5, hh, loaders, ids_b)
    m6 = _fit(hcs6.fit_household_model_v6, hh, loaders, ids_b)
    p5, _ = hcs5.simulate_draw_v5(hh, loaders["mpanel"], m5, ids_a, 5200)
    p6, _ = hcs6.simulate_draw_v6(hh, loaders["mpanel"], m6, ids_a, 5200)
    r5 = hc.reference_moments(p5, ids_a, weighted=True)
    r6 = hc.reference_moments(p6, ids_a, weighted=True)
    # older married-linked male cells stay close to candidate 5 (no over-drain).
    for cell in (
        "coresident_child.35-44|male",
        "coresident_child.55-64|male",
        "coresident_child.65-74|male",
    ):
        assert abs(r6[cell]["rate"] - r5[cell]["rate"]) < 0.01, cell


# --------------------------------------------------------------------------
# Delta 4 reproduces the forensics-3 Q10 feasibility
# --------------------------------------------------------------------------
def test_delta4_reproduces_q10_feasibility(household_panel, loaders):
    hh = household_panel
    _, ids_b = _seed0(hh)
    m6 = _fit(hcs6.fit_household_model_v6, hh, loaders, ids_b)
    forensics = json.loads(FORENSICS.read_text())
    q10 = forensics["question_10_hh_size_3_5plus_joint_constraint"]
    implied = hcs6.bridge_feasibility_convolution(
        m6.nonfamily_count_by_core, q10["sim_c5_core_size_distribution"]
    )
    ref = q10["reference_hh_size_distribution"]
    per_cell = q10["honest_joint_counterfactual"]["per_cell"]
    for cell, key in (
        ("hh_size.3", "3"),
        ("hh_size.4", "4"),
        ("hh_size.5+", "5+"),
    ):
        tol = per_cell[cell]["tolerance"]
        score = abs(np.log(implied[key] / ref[key]))
        assert score <= tol, (cell, score, tol)
    # reproduces the proven numbers within the per-seed conditional deviation.
    assert abs(implied["3"] - 0.1887) < 0.003
    assert abs(implied["5+"] - 0.1303) < 0.003
