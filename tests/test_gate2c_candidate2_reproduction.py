"""PSID / pe-us reproduction pin for gate-2c candidate 2 (skips off-machine).

Proves the candidate-2 one-shot on the staged PSID products
(``~/PolicyEngine/psid-data``) and the certified pe-us AIME oracle
(``POPULACE_DYNAMICS_PE_US_DIR``):

* a fresh side-B v2 fit + K=20 side-A simulate reproduces the committed seed-0
  gated-cell ``rbar`` and per-draw cube to bit precision;
* two independent v2 simulations at the same draw seed are byte-identical;
* THE BYTE-CARRY: a live candidate-1 simulate and the candidate-2 simulate at
  the same seed / draw produce IDENTICAL carried-family rates (assort /
  remarriage / event window / shared), and the first_marriage cells equal
  ``m(tercile | band, sex)`` times candidate 1's -- the delta touches only the
  first-marriage event weights;
* the marginal-preservation constraint ``sum_t m * phi_cert = 1`` holds.

Needs both the PSID Marriage History / family panel and the pe-us NAWI/AIME
chain; skipped when either is unavailable (an oracle-tier reproduction pin).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault(
    "POPULACE_DYNAMICS_PE_US_DIR",
    str(Path("~/PolicyEngine/policyengine-us-main").expanduser()),
)

from populace_dynamics.data import (  # noqa: E402
    births,
    deaths,
    family,
    marriage,
    panels,
    transitions,
)
from populace_dynamics.data import (
    couple_earnings as ce,
)
from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.models import (  # noqa: E402
    couple_formation_sim_v1 as cfs1,
)
from populace_dynamics.models import (  # noqa: E402
    couple_formation_sim_v2 as cfs,
)
from populace_dynamics.models.family_transitions.common import (  # noqa: E402
    marriage_order_map,
)
from populace_dynamics.ss.params import load_ssa_parameters  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2c_hazard_v2.json"
PSID_DATA = Path("~/PolicyEngine/psid-data").expanduser()
PE_US_DIR = Path(os.environ["POPULACE_DYNAMICS_PE_US_DIR"])

needs_data = pytest.mark.skipif(
    not (PSID_DATA / "MX23REL").is_dir() or not PE_US_DIR.is_dir(),
    reason="PSID products or the pe-us AIME oracle checkout not staged",
)
pytestmark = needs_data

GATE_SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 20
DRAW_SEED_BASE = 5200
SPLIT_COLUMN = "component_id"
CARRIED_FAMILIES = {
    "assort_mating",
    "remarriage_by_earnings",
    "earnings_around_marriage",
    "earnings_around_divorce",
    "shared_earnings_ratio",
}


@pytest.fixture(scope="module")
def sources():
    params = load_ssa_parameters()
    mh = marriage.marriage_history()
    dr = deaths.read_death_records()
    bh = births.birth_history()
    demo = panels.demographic_panel()
    demo_pos = demo[demo["weight"] > 0]
    person_weight = (
        demo_pos.sort_values("period")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["weight"]
    )
    earnings_panel = family.family_earnings_panel()
    ce_panel = ce.build_couple_panel(
        params=params,
        marriage_records=mh,
        earnings_panel=earnings_panel,
        death_records=dr,
        person_weight=person_weight,
    )
    mpanel = transitions.build_marital_panel(mh, dr, person_weight)
    axis = cfs.build_committed_axis(
        ce_panel,
        earnings_panel=earnings_panel,
        marriage_records=mh,
        params=params,
        person_weight=person_weight,
    )
    return {
        "ce_panel": ce_panel,
        "mpanel": mpanel,
        "demo": demo,
        "mh": mh,
        "bh": bh,
        "order_map": marriage_order_map(mh),
        "axis": axis,
    }


@pytest.fixture(scope="module")
def art():
    return json.loads(ARTIFACT.read_text())


def _fit_v2(sources, seed):
    ce_panel = sources["ce_panel"]
    side_a, side_b = hpanel.split_panel_by_person(
        ce_panel.attrs, SPLIT_COLUMN, fraction=0.5, seed=seed
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    model = cfs.fit_couple_model_v2(
        ce_panel,
        sources["mpanel"],
        demographic_panel=sources["demo"],
        marriage_records=sources["mh"],
        birth_records=sources["bh"],
        marriage_order_map=sources["order_map"],
        axis=sources["axis"],
        train_ids=ids_b,
    )
    return model, ids_a, ids_b


@pytest.fixture(scope="module")
def seed0(sources):
    model, ids_a, ids_b = _fit_v2(sources, 0)
    per_draw = {}
    for k in range(N_DRAWS):
        sim_panel, _ = cfs.simulate_draw_v2(
            sources["ce_panel"],
            sources["mpanel"],
            model,
            sources["axis"],
            ids_a,
            DRAW_SEED_BASE + k,
        )
        cells = ce.reference_moments(sim_panel, weighted=True)
        for c, rec in cells.items():
            per_draw.setdefault(c, []).append(float(rec["rate"]))
    return {"model": model, "ids_a": ids_a, "per_draw": per_draw}


# --------------------------------------------------------------------------
# Reproduction: the committed rbar + cube
# --------------------------------------------------------------------------
def test_seed0_rbar_reproduces_committed_artifact(seed0, art):
    committed = {s["seed"]: s for s in art["per_seed"]}[0]["gated_cells"]
    for cell, rec in committed.items():
        rbar = float(np.mean(seed0["per_draw"][cell]))
        assert abs(rbar - rec["rbar"]) <= 1e-9, cell


def test_seed0_per_draw_cube_reproduces(seed0, art):
    cube = art["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    si = cube["seed_index"].index(0)
    for ci, cell in enumerate(cube["cell_index"]):
        for k in range(N_DRAWS):
            committed = cube["rates"][k][ci][si]
            assert abs(seed0["per_draw"][cell][k] - committed) <= 1e-9, (
                cell,
                k,
            )


def test_simulate_draw_v2_is_deterministic(sources, seed0):
    model = seed0["model"]
    ids_a = seed0["ids_a"]
    a, _ = cfs.simulate_draw_v2(
        sources["ce_panel"],
        sources["mpanel"],
        model,
        sources["axis"],
        ids_a,
        DRAW_SEED_BASE,
    )
    b, _ = cfs.simulate_draw_v2(
        sources["ce_panel"],
        sources["mpanel"],
        model,
        sources["axis"],
        ids_a,
        DRAW_SEED_BASE,
    )
    ca = ce.reference_moments(a, weighted=True)
    cb = ce.reference_moments(b, weighted=True)
    max_dev = max(abs(ca[c]["rate"] - cb[c]["rate"]) for c in ca)
    assert max_dev == 0.0


# --------------------------------------------------------------------------
# THE BYTE-CARRY: candidate 1 vs candidate 2 at the same seed / draw
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def paired_seed0(sources):
    ce_panel = sources["ce_panel"]
    side_a, side_b = hpanel.split_panel_by_person(
        ce_panel.attrs, SPLIT_COLUMN, fraction=0.5, seed=0
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    m1 = cfs1.fit_couple_model_v1(
        ce_panel,
        sources["mpanel"],
        demographic_panel=sources["demo"],
        marriage_records=sources["mh"],
        birth_records=sources["bh"],
        marriage_order_map=sources["order_map"],
        axis=sources["axis"],
        train_ids=ids_b,
    )
    m2 = cfs.fit_couple_model_v2(
        ce_panel,
        sources["mpanel"],
        demographic_panel=sources["demo"],
        marriage_records=sources["mh"],
        birth_records=sources["bh"],
        marriage_order_map=sources["order_map"],
        axis=sources["axis"],
        train_ids=ids_b,
    )
    p1, _ = cfs1.simulate_draw_v1(
        ce_panel, sources["mpanel"], m1, sources["axis"], ids_a, DRAW_SEED_BASE
    )
    p2, d2 = cfs.simulate_draw_v2(
        ce_panel, sources["mpanel"], m2, sources["axis"], ids_a, DRAW_SEED_BASE
    )
    return {
        "m1": m1,
        "m2": m2,
        "c1": ce.reference_moments(p1, weighted=True),
        "c2": ce.reference_moments(p2, weighted=True),
        "d2": d2,
    }


def test_certified_core_fit_byte_identical(paired_seed0):
    # the candidate-2 base fit is candidate 1 verbatim.
    assert np.array_equal(
        paired_seed0["m1"].assort_kernel, paired_seed0["m2"].assort_kernel
    )


def test_carried_families_byte_identical_v1_vs_v2(paired_seed0):
    c1 = paired_seed0["c1"]
    c2 = paired_seed0["c2"]
    max_dev = 0.0
    n = 0
    for cell in c1:
        if cell.split(".")[0] in CARRIED_FAMILIES:
            max_dev = max(max_dev, abs(c1[cell]["rate"] - c2[cell]["rate"]))
            n += 1
    assert max_dev == 0.0
    assert n >= 20


def test_first_marriage_equals_modifier_times_candidate1(paired_seed0):
    c1 = paired_seed0["c1"]
    c2 = paired_seed0["c2"]
    modifier = paired_seed0["m2"].fm_modifier
    max_dev = 0.0
    n = 0
    for cell in c1:
        if not cell.startswith("first_marriage_by_earnings"):
            continue
        body = cell.split(".", 1)[1]  # t{terc}.{band}|{sex}
        terc = int(body[1])
        band, sex = body[3:].split("|")
        m = modifier.lookup(
            np.array([sex]), np.array([band]), np.array([terc])
        )[0]
        pred = m * c1[cell]["rate"]
        max_dev = max(max_dev, abs(pred - c2[cell]["rate"]))
        n += 1
    assert max_dev <= 1e-12
    assert n == 24


def test_marginal_preservation_constraint_holds(paired_seed0):
    chk = paired_seed0["d2"]["marginal_preservation_check"]
    assert chk["constraint_holds"] is True
    assert chk["constraint_max_abs_dev_from_one"] <= 1e-9
    con = paired_seed0["m2"].fm_modifier.constraint_per_band()
    phi_sum = paired_seed0["m2"].fm_modifier.phi_cert.sum(axis=2)
    assert np.allclose(con[phi_sum > 0], 1.0, atol=1e-9)


def test_modifier_shape_and_bands(seed0):
    mod = seed0["model"].fm_modifier
    assert mod.m_norm.shape == (2, 4, 3)
    assert mod.alpha == cfs.MODIFIER_SHRINKAGE_ALPHA
    # every applied modifier is finite and positive.
    assert np.all(mod.m_norm > 0)
    assert np.all(np.isfinite(mod.m_norm))
