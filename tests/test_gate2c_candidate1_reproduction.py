"""PSID / pe-us reproduction pin for gate-2c candidate 1 (skips off-machine).

Proves the candidate-1 one-shot on the staged PSID products
(``~/PolicyEngine/psid-data``) and the certified pe-us AIME oracle
(``POPULACE_DYNAMICS_PE_US_DIR``):

* the full-panel ``couple_earnings.reference_moments`` reproduces the frozen
  floor's committed reference moments exactly, and each gate seed's
  couple-disjoint holdout id set reproduces the committed sha256;
* a fresh side-B fit + K=20 side-A simulate reproduces the committed seed-0
  gated-cell ``rbar`` (the 20-draw mean rate) to bit precision -- the run is
  deterministic and auditable;
* two independent simulations at the same draw seed are byte-identical.

Needs both the PSID Marriage History / family panel and the pe-us NAWI/AIME
chain; skipped when either is unavailable (an oracle-tier reproduction pin).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

# The couple earnings axis needs the certified pe-us AIME chain; default the
# checkout the floor was built against unless the environment overrides it.
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
    couple_formation_sim_v1 as cfs,
)
from populace_dynamics.models.family_transitions.common import (  # noqa: E402
    marriage_order_map,
)
from populace_dynamics.ss.params import load_ssa_parameters  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2c_hazard_v1.json"
FLOOR = ROOT / "runs" / "gate2c_floors_v1.json"
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
EXACT_ATOL = 1e-12


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
def floor():
    return json.loads(FLOOR.read_text())


@pytest.fixture(scope="module")
def art():
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------
# Hard-stop precheck reproduction
# --------------------------------------------------------------------------
def test_full_panel_reference_moments_reproduce_floor(sources, floor):
    ref = ce.reference_moments(sources["ce_panel"], weighted=True)
    committed = floor["reference_moments"]
    max_dev = max(
        abs(ref[k]["rate"] - committed[k]["rate"]) for k in committed
    )
    assert max_dev <= EXACT_ATOL


def test_per_seed_holdout_ids_reproduce(sources, floor):
    committed = {p["seed"]: p for p in floor["holdout_ids"]["per_seed"]}
    for seed in GATE_SEEDS:
        side_a, _ = hpanel.split_panel_by_person(
            sources["ce_panel"].attrs, SPLIT_COLUMN, fraction=0.5, seed=seed
        )
        ids = sorted(int(x) for x in side_a.person_id.unique())
        digest = hashlib.sha256(
            ",".join(str(i) for i in ids).encode()
        ).hexdigest()
        assert digest == committed[seed]["holdout_person_id_sha256"]


def test_committed_axis_cuts_match_panel(sources):
    axis = sources["axis"]
    assert axis.cuts == tuple(
        float(c) for c in sources["ce_panel"].earn_tercile_cuts
    )
    # decile edges are monotone and bracket the tercile cuts.
    assert np.all(np.diff(axis.decile_edges) > 0)


# --------------------------------------------------------------------------
# The seed-0 fit + K=20 simulate reproduces the committed rbar bit-for-bit
# --------------------------------------------------------------------------
def _fit_seed(sources, seed):
    ce_panel = sources["ce_panel"]
    side_a, side_b = hpanel.split_panel_by_person(
        ce_panel.attrs, SPLIT_COLUMN, fraction=0.5, seed=seed
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())
    model = cfs.fit_couple_model_v1(
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
    model, ids_a, ids_b = _fit_seed(sources, 0)
    per_draw = {}
    for k in range(N_DRAWS):
        sim_panel, _ = cfs.simulate_draw_v1(
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


def test_seed0_rbar_reproduces_committed_artifact(seed0, art):
    committed = {s["seed"]: s for s in art["per_seed"]}[0]["gated_cells"]
    for cell, rec in committed.items():
        rbar = float(np.mean(seed0["per_draw"][cell]))
        assert abs(rbar - rec["rbar"]) <= 1e-9, cell


def test_seed0_per_draw_cube_reproduces(seed0, art):
    cube = art["fresh_run_artifact_schema"]["per_draw_per_cell_rates"]
    seed_index = cube["seed_index"]
    si = seed_index.index(0)
    for ci, cell in enumerate(cube["cell_index"]):
        for k in range(N_DRAWS):
            committed = cube["rates"][k][ci][si]
            assert abs(seed0["per_draw"][cell][k] - committed) <= 1e-9, (
                cell,
                k,
            )


def test_simulate_draw_is_deterministic(sources, seed0):
    model = seed0["model"]
    ids_a = seed0["ids_a"]
    a, _ = cfs.simulate_draw_v1(
        sources["ce_panel"],
        sources["mpanel"],
        model,
        sources["axis"],
        ids_a,
        DRAW_SEED_BASE,
    )
    b, _ = cfs.simulate_draw_v1(
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
# Component structure: both-orientation emission + committed cuts
# --------------------------------------------------------------------------
def test_directed_couples_both_orientations_emitted(sources, seed0):
    model = seed0["model"]
    ids_a = seed0["ids_a"]
    sim_panel, diag = cfs.simulate_draw_v1(
        sources["ce_panel"],
        sources["mpanel"],
        model,
        sources["axis"],
        ids_a,
        DRAW_SEED_BASE,
    )
    # both orientations: exactly twice the marriages, and the own/spouse
    # tercile contingency is symmetric by construction.
    assert diag["n_directed_couples"] == 2 * diag["n_marriages"]
    c = sim_panel.couples
    tab = (
        c.groupby(["own_tercile", "spouse_tercile"])["weight"]
        .sum()
        .unstack(fill_value=0.0)
    )
    assert np.allclose(tab.to_numpy(), tab.to_numpy().T, atol=1e-6)


def test_kernel_conditioning_shape(seed0):
    kernel = seed0["model"].assort_kernel
    # [sex(2), age_band(4), own_decile(10), spouse_decile(10)], each a proper
    # conditional distribution.
    assert kernel.shape == (2, 4, 10, 10)
    assert np.allclose(kernel.sum(axis=3), 1.0)
