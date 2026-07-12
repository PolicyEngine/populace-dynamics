"""PSID/frame reproduction pins for W1 forensics 1 (skips off-machine).

Marked ``integration_psid`` (references ``~/PolicyEngine/psid-data`` and the
certified frame export ``POPULACE_DYNAMICS_FRAME_PICKLE``); skipped when the
PSID Family Relationship Matrix or the frame pickle are not staged. Rebuilds
the bit-identity claims the artifact records: the instrumentation reproduces
the committed ``transport_deployment_v1`` machinery bit-for-bit.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate_w1_forensics1_v1.json"
CANDIDATE1 = ROOT / "runs" / "gate_w1_candidate1_v1.json"
SCRIPTS = ROOT / "scripts"
REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
FRAME_PICKLE = os.environ.get("POPULACE_DYNAMICS_FRAME_PICKLE", "")

ATOL = 1e-9

needs_data = pytest.mark.skipif(
    not (REAL_DATA / "MX23REL").is_dir()
    or not (FRAME_PICKLE and Path(FRAME_PICKLE).exists()),
    reason="PSID MX23REL and/or the certified frame pickle "
    "(POPULACE_DYNAMICS_FRAME_PICKLE) not staged",
)
pytestmark = needs_data


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def env():
    sys.path.insert(0, str(SCRIPTS))
    import gate_w1_forensics1 as gf1

    gens = gf1.fit_generators()
    persons = gf1._load_frame()
    return {
        "gf1": gf1,
        "gens": gens,
        "persons": persons,
        "art": _artifact(),
        "c1": json.loads(CANDIDATE1.read_text()),
    }


def test_q1_marital_reproduces_committed_cube_bit_identically(env):
    """seed-0 draw-0 holdout marital + coresident cells == the committed
    candidate-1 cube, exactly (0.0)."""
    from populace_dynamics.data import deployment_frame as dfm
    from populace_dynamics.models import transport_deployment_v1 as td

    gf1, gens, persons = env["gf1"], env["gens"], env["persons"]
    c1 = env["c1"]
    gated = c1["family_a"]["gated_cells"]
    cube = np.array(c1["family_a"]["cube"])
    side_a = set(
        td.holdout_side_a_households(
            persons["household_id"].to_numpy(), 0
        ).tolist()
    )
    hold = persons[persons["household_id"].isin(side_a)].reset_index(drop=True)
    ah = hold[hold["age"] >= td.ADULT_MIN_AGE].reset_index(drop=True)
    marital = td.regenerate_marital(
        ah["person_id"].to_numpy(),
        ah["age"].to_numpy(dtype=np.float64),
        ah["is_female"].to_numpy(dtype=bool),
        ah["weight"].to_numpy(dtype=np.float64),
        gens.fitted_ft,
        td.FAMILY_A_STREAM_BASE + 0,
    )
    repro = dfm.reference_moments(
        gf1._marital_frame(ah, marital), weighted=True
    )
    max_dev = 0.0
    for ci, cell in enumerate(gated):
        if cell.startswith(("marital_share.", "coresident_spouse.")):
            if cell in repro:
                max_dev = max(
                    max_dev, abs(repro[cell]["rate"] - float(cube[0, ci, 0]))
                )
    assert max_dev == 0.0
    assert (
        env["art"]["q1_marital_equilibration"]["instrumentation_fidelity"][
            "max_abs_rate_deviation_vs_committed_cube"
        ]
        == 0.0
    )


def test_q1_observed_init_reproduces_frame_a_maritl(env):
    """observed-init O == the frame's own A_MARITL married share (the
    identity that reproduces rate_a)."""
    from populace_dynamics.data import deployment_frame as dfm

    o = dfm.reference_moments(env["persons"], weighted=True)
    rec = env["art"]["q1_marital_equilibration"]["per_band_sex"]
    for band_sex, cell in rec.items():
        key = f"marital_share.married.{band_sex}"
        assert o[key]["rate"] == pytest.approx(
            cell["observed_init_O"], abs=ATOL
        )


def test_q5_upper_read_career_panel_is_bit_identical(env):
    """the apply_p0=False career panel reproduces td.transport_career_panel
    bit-for-bit (earnings + frac-above-cap)."""
    from populace_dynamics.models import transport_deployment_v1 as td

    gf1, gens, persons = env["gf1"], env["gens"], env["persons"]
    mine = gf1._career_panel(persons, gens, apply_p0=False)
    committed = td.transport_career_panel(persons, gens)
    a = mine["panel"]["earnings"].to_numpy(dtype=np.float64)
    b = committed["panel"]["earnings"].to_numpy(dtype=np.float64)
    assert len(a) == len(b)
    assert np.array_equal(a, b)
    assert (
        mine["frac_payroll_above_wage_base"]
        == committed["frac_payroll_above_wage_base"]
        == env["art"]["q5_tail_upper_read"]["upper_read"][
            "frac_payroll_above_wage_base"
        ]
    )


def test_q5_corrected_tail_is_lighter(env):
    """applying the certified p0 (zero/low-earning years) lightens the tail
    -- the conservative-direction correction the C1 answer relies on."""
    gf1, gens, persons = env["gf1"], env["gens"], env["persons"]
    up = gf1._career_panel(persons, gens, apply_p0=False)
    cor = gf1._career_panel(persons, gens, apply_p0=True)
    assert (
        cor["frac_payroll_above_wage_base"]
        < up["frac_payroll_above_wage_base"]
    )
    assert cor["frac_payroll_above_wage_base"] == pytest.approx(
        env["art"]["q5_tail_upper_read"]["corrected_tail"][
            "frac_payroll_above_wage_base"
        ],
        abs=ATOL,
    )


def test_q2_psid_boundary_support_reproduces(env):
    """the train-fitted boundary participation / profile reproduce."""
    gf1 = env["gf1"]
    import run_gate1_baseline as g1base

    raw = g1base.family_earnings_panel()
    raw = raw[
        (raw.period >= g1base.PERIOD_MIN)
        & (raw.period <= g1base.PERIOD_MAX)
        & (raw.weight > 0)
    ]
    prime = raw[(raw.age >= 35) & (raw.age <= 44)]
    prime_med = gf1._weighted_median_pos(prime)
    rec = env["art"]["q2_participation_boundary"]["psid_boundary_support"]
    for lo, hi, label in ((18, 24, "18-24"), (62, 69, "62-69")):
        sub = raw[(raw.age >= lo) & (raw.age <= hi)]
        assert gf1._weighted_participation(sub) == pytest.approx(
            rec[label]["participation"], abs=ATOL
        )
        assert (gf1._weighted_median_pos(sub) / prime_med) == pytest.approx(
            rec[label]["profile_ratio"], abs=ATOL
        )
