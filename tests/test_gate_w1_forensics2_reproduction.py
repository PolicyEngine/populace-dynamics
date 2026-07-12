"""PSID/frame reproduction pins for W1 forensics 2 (skips off-machine).

Marked ``integration_psid`` (references ``~/PolicyEngine/psid-data`` and the
certified frame export ``POPULACE_DYNAMICS_FRAME_PICKLE``); skipped when the
PSID Family Relationship Matrix or the frame pickle are not staged. Rebuilds
the bit-identity claims the artifact records: the instrumentation reproduces
the committed ``transport_deployment_v2`` machinery bit-for-bit. These are
single-draw checks (the artifact-build runs the full K-draw decompositions).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate_w1_forensics2_v1.json"
CANDIDATE2 = ROOT / "runs" / "gate_w1_candidate2_v1.json"
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
    import gate_w1_forensics2 as gf2

    gens = gf2.fit_generators()
    persons = gf2._load_frame()
    return {
        "gf2": gf2,
        "gens": gens,
        "persons": persons,
        "art": _artifact(),
        "c2": json.loads(CANDIDATE2.read_text()),
    }


def _seed0_holdout(gf2, persons):
    from populace_dynamics.models import transport_deployment_v1 as td1

    side_a = set(
        td1.holdout_side_a_households(
            persons["household_id"].to_numpy(), 0
        ).tolist()
    )
    return persons[persons["household_id"].isin(side_a)].reset_index(drop=True)


def test_q6_marital_reproduces_committed_cube_bit_identically(env):
    """seed-0 draw-0 holdout marital + coresident cells == the committed
    candidate-2 cube, exactly (0.0)."""
    from populace_dynamics.data import deployment_frame as dfm
    from populace_dynamics.models import transport_deployment_v2 as td

    gf2, gens, persons = env["gf2"], env["gens"], env["persons"]
    c2 = env["c2"]
    gated = c2["family_a"]["gated_cells"]
    cube = np.array(c2["family_a"]["cube"])
    hold = _seed0_holdout(gf2, persons)
    ah = hold[hold["age"] >= td.ADULT_MIN_AGE].reset_index(drop=True)
    hpid = ah["person_id"].to_numpy()
    marital = td.regenerate_marital_v2(
        hpid,
        ah["age"].to_numpy(dtype=np.float64),
        ah["is_female"].to_numpy(dtype=bool),
        ah["weight"].to_numpy(dtype=np.float64),
        gens.fitted_ft,
        gens.initial_state_model,
        td.FAMILY_A_STREAM_BASE + 0,
    )
    repro = dfm.reference_moments(
        gf2._marital_frame(
            hpid,
            ah["age"].to_numpy(dtype=np.float64),
            ah["is_female"].to_numpy(dtype=bool),
            ah["weight"].to_numpy(dtype=np.float64),
            marital.reindex(hpid).to_numpy(dtype=object),
        ),
        weighted=True,
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
        env["art"]["q6_marital_calibration_frame"]["instrumentation_fidelity"][
            "max_abs_rate_deviation_vs_committed_cube"
        ]
        == 0.0
    )


def test_q6_frame_A_is_the_a_maritl_cross_section(env):
    """the decomposition's frame anchor A == the frame's own A_MARITL married
    share, exactly."""
    from populace_dynamics.data import deployment_frame as dfm

    A = dfm.reference_moments(env["persons"], weighted=True)
    rec = env["art"]["q6_marital_calibration_frame"]["per_band_sex"]
    for band_sex, cell in rec.items():
        key = f"marital_share.married.{band_sex}"
        assert A[key]["rate"] == pytest.approx(
            cell["frame_A_MARITL_A"], abs=ATOL
        )


def test_q7_hh_size_reproduces_committed_cube_bit_identically(env):
    """seed-0 draw-0 holdout hh_size cells == the committed candidate-2 cube,
    exactly (0.0), via regenerate_person_frame_v2."""
    from populace_dynamics.data import deployment_frame as dfm
    from populace_dynamics.models import transport_deployment_v2 as td

    gf2, gens, persons = env["gf2"], env["gens"], env["persons"]
    c2 = env["c2"]
    gated = c2["family_a"]["gated_cells"]
    cube = np.array(c2["family_a"]["cube"])
    hold = _seed0_holdout(gf2, persons)
    regen = td.regenerate_person_frame_v2(
        hold, gens, 0, td.FAMILY_A_STREAM_BASE
    )
    cells = dfm.reference_moments(regen, weighted=True)
    max_dev = 0.0
    for ci, cell in enumerate(gated):
        if cell.startswith("hh_size_share.") and cell in cells:
            max_dev = max(
                max_dev, abs(cells[cell]["rate"] - float(cube[0, ci, 0]))
            )
    assert max_dev == 0.0


def test_q7_coresident_parent_reduces_size1(env):
    """the train-fitted coresident_parent roster lever cuts the size-1 share
    below the committed baseline (one draw)."""
    from populace_dynamics.models import transport_deployment_v2 as td

    gf2, gens, persons = env["gf2"], env["gens"], env["persons"]
    hold = _seed0_holdout(gf2, persons)
    ad = hold[hold["age"] >= td.ADULT_MIN_AGE].reset_index(drop=True)
    pid = ad["person_id"].to_numpy()
    age = ad["age"].to_numpy(dtype=np.float64)
    fem = ad["is_female"].to_numpy(dtype=bool)
    wt = ad["weight"].to_numpy(dtype=np.float64)
    cp = gf2._train_coresident_parent_rate()
    base, _ = gf2._household_config(
        pid, age, fem, wt, gens, td.FAMILY_A_STREAM_BASE, None, False
    )
    lever_a, _ = gf2._household_config(
        pid, age, fem, wt, gens, td.FAMILY_A_STREAM_BASE, cp, False
    )
    assert lever_a["1"] < base["1"]


def test_q8_base_earnings_reproduces_committed_interior(env):
    """the base (no-sex-covariate) re-draw reproduces the committed interior
    earnings cells bit-for-bit -- the byte-carry baseline."""
    from populace_dynamics.data import deployment_frame as dfm
    from populace_dynamics.models import transport_deployment_v2 as td

    gf2, gens, persons = env["gf2"], env["gens"], env["persons"]
    c2 = env["c2"]
    gated = c2["family_a"]["gated_cells"]
    cube = np.array(c2["family_a"]["cube"])
    hold = _seed0_holdout(gf2, persons)
    ah = hold[hold["age"] >= td.ADULT_MIN_AGE].reset_index(drop=True)
    age = ah["age"].to_numpy(dtype=np.float64)
    fem = ah["is_female"].to_numpy(dtype=bool)
    e = gf2._regen_earnings_base(age, fem, gens, td.FAMILY_A_STREAM_BASE + 0)
    import pandas as pd

    fr = pd.DataFrame(
        {
            "person_id": ah["person_id"].to_numpy(),
            "weight": ah["weight"].to_numpy(dtype=np.float64),
            "age": age,
            "is_female": fem,
            "earnings": e,
            "marital_status": np.array(["never_married"] * len(ah)),
            "hh_size": np.ones(len(ah)),
            "coresident_spouse": np.zeros(len(ah), dtype=bool),
        }
    )
    cells = dfm.reference_moments(fr, weighted=True)
    interior_tokens = ("25-34", "35-44", "45-54", "55-61")
    max_dev = 0.0
    for ci, cell in enumerate(gated):
        if (
            cell.startswith("earnings_")
            and any(t in cell for t in interior_tokens)
            and cell in cells
        ):
            max_dev = max(
                max_dev, abs(cells[cell]["rate"] - float(cube[0, ci, 0]))
            )
    assert max_dev == 0.0


def test_q8_interior_sex_covariate_clears_targets(env):
    """the interior sex-covariate rescore clears >=3 of the four target cells
    (K_EARN draws)."""
    from populace_dynamics.data import deployment_frame as dfm
    from populace_dynamics.models import transport_deployment_v2 as td

    gf2, gens, persons = env["gf2"], env["gens"], env["persons"]
    c2 = env["c2"]
    pc0 = c2["family_a"]["per_seed"][0]["per_cell"]
    hold = _seed0_holdout(gf2, persons)
    ah = hold[hold["age"] >= td.ADULT_MIN_AGE].reset_index(drop=True)
    age = ah["age"].to_numpy(dtype=np.float64)
    fem = ah["is_female"].to_numpy(dtype=bool)
    wt = ah["weight"].to_numpy(dtype=np.float64)
    interior, _ = gf2._fit_interior_marginals()
    import pandas as pd

    acc: dict[str, list[float]] = {}
    for k in range(gf2.K_EARN):
        e = gf2._regen_earnings_sexcov(
            age, fem, gens, interior, td.FAMILY_A_STREAM_BASE + k
        )
        fr = pd.DataFrame(
            {
                "person_id": ah["person_id"].to_numpy(),
                "weight": wt,
                "age": age,
                "is_female": fem,
                "earnings": e,
                "marital_status": np.array(["never_married"] * len(ah)),
                "hh_size": np.ones(len(ah)),
                "coresident_spouse": np.zeros(len(ah), dtype=bool),
            }
        )
        cc = dfm.reference_moments(fr, weighted=True)
        for c in gf2.Q8_TARGET_CELLS:
            if c in cc:
                acc.setdefault(c, []).append(float(cc[c]["rate"]))
    n_clear = 0
    for c in gf2.Q8_TARGET_CELLS:
        rb = float(np.mean(acc[c]))
        ra = float(pc0[c]["rate_a"])
        n_clear += int(abs(math.log(rb / ra)) <= pc0[c]["tolerance"])
    assert n_clear >= 3
    assert (
        n_clear
        == env["art"]["q8_interior_sex_covariate"]["n_target_cells_clear"]
    )


def test_q9_concept_gap_reproduces(env):
    """the 18-24 head/spouse-vs-all-person concept gap reproduces the
    recorded train measurement."""
    gf2, gens, persons = env["gf2"], env["gens"], env["persons"]
    f1 = json.loads((ROOT / "runs" / "gate_w1_forensics1_v1.json").read_text())
    q9 = gf2.q9_concept_cells(persons, gens, env["c2"], f1)
    rec = env["art"]["q9_concept_cells"]["concept_gap_18_24_participation"]
    assert q9["concept_gap_18_24_participation"][
        "pooled_gap_pp"
    ] == pytest.approx(rec["pooled_gap_pp"], abs=ATOL)
    assert q9["concept_gap_18_24_participation"]["psid_head_spouse_universe"][
        "pooled"
    ] == pytest.approx(rec["psid_head_spouse_universe"]["pooled"], abs=ATOL)
