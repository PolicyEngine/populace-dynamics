"""Reproduction (oracle) pins for the M4 candidate-1 scored run.

Skipped off-machine (needs the staged PSID individual file). On a staged
machine it is the independent verification-by-rerun the standing addendum
requires: it refits the hazard machinery on each side B, re-simulates the
K=20 holdout draws, and reproduces the committed cube, per-cell scores and
the gate verdict to 1e-9 -- plus the fit-fidelity (train copy == module
reference rates) and simulate_draw determinism the candidate rests on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_gate_m4_candidate1 as runner  # noqa: E402

from populace_dynamics.data import disability  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402
from populace_dynamics.models import (  # noqa: E402
    disability_hazard_sim as dhs,
)

PSID_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_data = pytest.mark.skipif(
    not (PSID_DATA / "MX23REL").is_dir(),
    reason="PSID individual file not staged at ~/PolicyEngine/psid-data",
)
pytestmark = needs_data

ARTIFACT = ROOT / "runs" / "gate_m4_hazard_v1.json"


@pytest.fixture(scope="module")
def panel() -> disability.DisabilityPanel:
    return runner.load_panel()


@pytest.fixture(scope="module")
def contract() -> dict:
    return runner.resolve_contract(json.loads(runner.FLOOR_RUN.read_text()))


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(ARTIFACT.read_text())


def _split(panel: disability.DisabilityPanel, seed: int):
    ids_frame = panel.person_years[["person_id"]].drop_duplicates()
    left, right = hpanel.split_panel_by_person(
        ids_frame, "person_id", fraction=0.5, seed=seed
    )
    return set(left["person_id"]), set(right["person_id"])


def test_fit_is_the_module_reference_rates(
    panel: disability.DisabilityPanel,
) -> None:
    """Every fitted hazard equals the unmodified module rate on side B --
    the no-free-hyperparameter / train-copy property the gate rests on."""
    _, ids_b = _split(panel, 0)
    model = dhs.fit(panel, ids_b)
    inc = disability.incidence_cells(panel, ids_b)
    rec = disability.recovery_cells(panel, ids_b)
    prev = disability.prevalence_cells(panel, ids_b)
    for bi, (lo, hi) in enumerate(disability.DI_AGE_BANDS):
        band = disability.band_label(lo, hi)
        for si, sex in enumerate(disability.SEXES):
            assert model.incidence[bi + 1, si] == pytest.approx(
                inc[f"incidence.{band}|{sex}"]["rate"], abs=1e-15
            )
            assert model.recovery[bi + 1, si] == pytest.approx(
                rec[f"recovery.{band}|{sex}"]["rate"], abs=1e-15
            )
            assert model.prevalence0[bi + 1, si] == pytest.approx(
                prev[f"prevalence.{band}|{sex}"]["rate"], abs=1e-15
            )


def test_simulate_draw_is_deterministic(
    panel: disability.DisabilityPanel,
) -> None:
    ids_a, ids_b = _split(panel, 0)
    model = dhs.fit(panel, ids_b)
    a = disability.reference_moments(
        dhs.simulate_draw(panel, model, ids_a, 5200)
    )
    b = disability.reference_moments(
        dhs.simulate_draw(panel, model, ids_a, 5200)
    )
    for cell in runner.LOCKED_TOLERANCES:
        assert a[cell]["rate"] == b[cell]["rate"]


def test_reproduces_committed_cube_and_verdict(
    panel: disability.DisabilityPanel, contract: dict, committed: dict
) -> None:
    """Refit + re-simulate every seed and reproduce the committed run."""
    committed_seed = {s["seed"]: s for s in committed["per_seed"]}
    cube = committed["per_draw_per_cell_rates"]["cube"]
    n_pass = 0
    for seed in runner.GATE_SEEDS:
        fresh = runner.score_seed(seed, panel, contract, verbose=False)
        ref = committed_seed[seed]
        for cell in contract["internal_cells"]:
            f = fresh["internal_cells"][cell]
            r = ref["internal_cells"][cell]
            assert f["rbar"] == pytest.approx(r["rbar"], abs=1e-9)
            assert f["rate_a"] == pytest.approx(r["rate_a"], abs=1e-12)
            assert f["per_draw_rate"] == pytest.approx(
                r["per_draw_rate"], abs=1e-12
            )
            assert (f["score"] == pytest.approx(r["score"], abs=1e-9)) or (
                not np.isfinite(f["score"]) and not np.isfinite(r["score"])
            )
            assert f["pass"] == r["pass"]
        for cell in contract["anchor_cells"]:
            f = fresh["anchor_cells"][cell]
            r = ref["anchor_cells"][cell]
            assert f["invariant_value"] == pytest.approx(
                r["invariant_value"], abs=1e-9
            )
            assert f["margin_sigma_units"] == pytest.approx(
                r["margin_sigma_units"], abs=1e-9
            )
            assert f["pass"] == r["pass"]
        # the fresh per-draw rates reproduce the committed [20, 8, 5] cube
        for ci, cell in enumerate(contract["internal_cells"]):
            plane = [cube[k][ci][seed] for k in range(runner.N_DRAWS)]
            assert fresh["internal_cells"][cell]["per_draw_rate"] == (
                pytest.approx(plane, abs=1e-12)
            )
        assert fresh["seed_pass"] == ref["seed_pass"]
        n_pass += int(fresh["seed_pass"])

    assert n_pass == committed["verdict"]["n_seeds_pass"]
    assert (n_pass >= 4) == committed["verdict"]["gate_m4_pass"]
