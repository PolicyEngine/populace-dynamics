"""PSID reproduction pins for gate-2b forensics 1 (skips off-machine).

Marked ``integration_psid`` (references ``~/PolicyEngine/psid-data`` /
``POPULACE_DYNAMICS_PSID_DIR``); skipped when the PSID Family Relationship
Matrix is not staged. Each of the four frozen questions gets one reproduction
pin that rebuilds seed 0's deterministic train-side (side B) decomposition live
and matches the recorded per-seed values to float precision, plus the
instrumented-draw fidelity proof that the diagnostic's component
decomposition is bit-identical to the committed ``simulate_draw_v3``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2b_forensics1_v1.json"
SCRIPTS = ROOT / "scripts"
REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()

needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "MX23REL").is_dir(),
    reason="PSID MX23REL relationship matrix not staged "
    "(POPULACE_DYNAMICS_PSID_DIR)",
)
pytestmark = needs_psid

ATOL = 1e-9


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def seed0():
    """Load the panels, split seed 0, and fit the candidate-3 model on side B."""
    sys.path.insert(0, str(SCRIPTS))
    import gate2b_forensics1 as gf

    from populace_dynamics.harness import panel as hpanel
    from populace_dynamics.models import household_composition_sim_v3 as hcs3

    data = gf.c3.load_all()
    hh, mpanel = data["hh"], data["mpanel"]
    _side_a, side_b = hpanel.split_panel_by_person(
        hh.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = {int(x) for x in side_b.person_id.unique()}
    model = hcs3.fit_household_model_v3(
        hh,
        mpanel,
        data["demo"],
        data["mh"],
        data["bh"],
        data["order_map"],
        data["rel_map"],
        ids_b,
        father_links_child=data["father_links_child"],
        parent_pairs=data["parent_pairs"],
        fu_sizes=data["fu_sizes"],
    )
    linked_ids = (
        {int(x) for x in model.father_links["parent_person_id"]}
        & {
            int(x)
            for x in mpanel.attrs[mpanel.attrs["sex"] == "male"]["person_id"]
        }
        & ids_b
    )
    recorded = next(s for s in _artifact()["per_seed"] if s["seed"] == 0)
    return {
        "gf": gf,
        "data": data,
        "ids_b": ids_b,
        "model": model,
        "linked_ids": linked_ids,
        "recorded": recorded,
    }


def test_pin_q1_concept_and_reference_reproduce(seed0):
    """Q1: seed-0 spouse concept-code enumeration + reference rate_b."""
    gf = seed0["gf"]
    from populace_dynamics.data import household_composition as hc

    concept = gf.spouse_concept_codes(
        seed0["data"]["rel_map"],
        seed0["data"]["hh"].person_waves,
        seed0["ids_b"],
    )
    rec = seed0["recorded"]["q1_concept"]
    assert concept["distinct_spouse_link_codes_present"] == [20, 22]
    assert concept["share_beyond_20_22"] == pytest.approx(0.0, abs=ATOL)
    for key in (
        "share_code20_legal_only",
        "share_code22_cohab_only",
        "share_both_codes",
    ):
        assert concept[key] == pytest.approx(rec[key], abs=ATOL)
    # The train reference rate_b of a failing spouse cell reproduces.
    rate_b = hc.reference_moments(
        seed0["data"]["hh"], seed0["ids_b"], weighted=True
    )
    assert float(
        rate_b["coresident_spouse.15-24|male"]["rate"]
    ) == pytest.approx(
        seed0["recorded"]["rate_b"]["coresident_spouse.15-24|male"], abs=ATOL
    )


def test_pin_q2_unlinked_profile_reproduces(seed0):
    """Q2: seed-0 linked/unlinked men profile (age, married share, era)."""
    gf = seed0["gf"]
    prof = gf.unlinked_men_profile(
        seed0["data"]["mpanel"],
        seed0["data"]["hh"].person_waves,
        seed0["linked_ids"],
        seed0["ids_b"],
    )
    rec = seed0["recorded"]["q2_unlinked_profile"]
    assert prof["n_men"] == rec["n_men"]
    assert prof["unlinked_fraction"] == pytest.approx(
        rec["unlinked_fraction"], abs=ATOL
    )
    assert prof["unlinked_profile"]["married_share"] == pytest.approx(
        rec["unlinked_profile"]["married_share"], abs=ATOL
    )
    assert prof["linked_profile"]["married_share"] == pytest.approx(
        rec["linked_profile"]["married_share"], abs=ATOL
    )


def test_pin_q3_reference_hh_size_reproduces(seed0):
    """Q3: seed-0 train reference hh_size distribution (the size-3 target)."""
    from populace_dynamics.data import household_composition as hc

    rate_b = hc.reference_moments(
        seed0["data"]["hh"], seed0["ids_b"], weighted=True
    )
    for size in (
        "hh_size.1",
        "hh_size.2",
        "hh_size.3",
        "hh_size.4",
        "hh_size.5+",
    ):
        assert float(rate_b[size]["rate"]) == pytest.approx(
            seed0["recorded"]["rate_b"][size], abs=ATOL
        )


def test_pin_q4_skipgen_age_structure_reproduces(seed0):
    """Q4: seed-0 raw skipped-generation stock by single-year age (55+ female)."""
    gf = seed0["gf"]
    age = gf.skipgen_single_year_age(
        seed0["data"]["hh"].person_waves,
        seed0["model"].skipgen_entry,
        seed0["ids_b"],
    )
    rec = seed0["recorded"]["q4_skipgen_age"]
    for band in ("55-64", "65-74", "75+"):
        assert age["band_detail_female"][band][
            "raw_skipgen_stock_share"
        ] == pytest.approx(
            rec["band_detail_female"][band]["raw_skipgen_stock_share"],
            abs=ATOL,
        )


def test_pin_instrumented_draw_is_bit_identical(seed0):
    """The instrumented draw reproduces the committed simulate_draw_v3 exactly.

    The fidelity proof underlying every component decomposition: at seed 0's
    first draw (5200) the instrumented panel's every scored cell equals the
    committed ``simulate_draw_v3`` panel's to bit precision.
    """
    gf = seed0["gf"]
    fid = gf.fidelity_check(
        seed0["data"]["hh"],
        seed0["data"]["mpanel"],
        seed0["model"],
        seed0["ids_b"],
        gf.DRAW_SEED_BASE,
    )
    assert fid["bit_identical"] is True
    assert fid["max_abs_rate_deviation_vs_committed_simulate_draw_v3"] == 0.0
    # The recorded protocol block carries the same proof.
    recorded_fid = _artifact()["protocol"]["instrumentation_fidelity"]
    assert recorded_fid["bit_identical"] is True
