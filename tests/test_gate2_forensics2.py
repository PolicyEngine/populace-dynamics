"""Tests for gate-2 forensics 2 (reported, not gated).

REPORTED-NOT-GATED. The consistency tests read only the forensics-2 artifact
(``runs/gate2_forensics2_v1.json``), the committed candidate-10 gate artifact
(``runs/gate2_hazard_v10.json``) and ``gates.yaml``; they never rerun the
diagnostic and need no PSID, so they run in CI. They audit that every headline
recomputes from the stored per-seed values: the Q4 widowed-stock gap and its
reconciling onset-bucket / inflow / outflow / initial-states / spousal-gap
decomposition, and the Q5 pathway table and its age-at-dissolution
misallocation probe, plus the published-outer context match.

One reproduction pin (``test_pin_*``) rebuilds the train-side inputs live and
matches seed 0's reference widowed-stock share and its first simulation draw
to float precision (skipped when the PSID marriage-history files are absent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2_forensics2_v1.json"
CANDIDATE10 = ROOT / "runs" / "gate2_hazard_v10.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_psid = pytest.mark.skipif(
    not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID marriage-history files not staged",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------
# Top-level provenance
# --------------------------------------------------------------------------
def test_artifact_schema_and_reported_not_gated():
    a = _artifact()
    assert a["schema_version"] == "gate2_forensics2.v1"
    assert a["reported_not_gated"] is True
    assert a["registration"].endswith("#issuecomment-4917987214")
    assert a["registration_pointer"] == "4917987214"
    assert "candidate 10" in a["candidate_under_diagnosis"]
    for block in ("question_4_widow_stock", "question_5_female_count"):
        assert block in a


def test_protocol_is_train_side_only():
    p = _artifact()["protocol"]
    assert p["train_side_only"] is True
    assert "never" in p["outer_holdout_contact"].lower()
    assert p["gate_seeds"] == [0, 1, 2, 3, 4]
    assert p["n_draws"] == 20
    assert "5200" in p["draw_rng_rule"]


def test_no_gate_verdict_written_and_lock_untouched():
    a = _artifact()
    # A diagnostic, not a gate run: no gate pass/fail verdict is emitted.
    assert "verdict" not in a
    assert "gate_2_pass" not in a
    assert a["revision_pins"]["gates_yaml_locked"] is True


def test_candidate_11_registers_only_on_this_evidence():
    a = _artifact()
    assert "candidate 11" in a["candidate_11_implications"]
    assert "age at dissolution" in a["candidate_11_implications"]


# --------------------------------------------------------------------------
# Question 4 -- widowed-stock gap decomposition
# --------------------------------------------------------------------------
def test_q4_gap_recomputes():
    q = _artifact()["question_4_widow_stock"]["widowed_stock_75plus"]
    assert q["gap_reference_minus_simulated"] == pytest.approx(
        q["reference"] - q["simulated"], abs=1e-9
    )
    # The registration's headline: the simulation under-produces the stock.
    assert q["gap_reference_minus_simulated"] > 0
    assert q["log_ratio_sim_over_ref"] < 0


def test_q4_onset_buckets_reconcile_to_share():
    d = _artifact()["question_4_widow_stock"]["onset_bucket_decomposition"]
    q = _artifact()["question_4_widow_stock"]["widowed_stock_75plus"]
    # The buckets partition the 75+ widowed stock, so they sum to the share.
    assert sum(d["reference"].values()) == pytest.approx(
        q["reference"], abs=1e-6
    )
    assert sum(d["simulated"].values()) == pytest.approx(
        q["simulated"], abs=1e-6
    )
    # Bucket gaps recompute and sum to the total gap.
    for k in d["reference"]:
        assert d["gap_reference_minus_simulated"][k] == pytest.approx(
            d["reference"][k] - d["simulated"][k], abs=1e-9
        )
    assert sum(d["gap_reference_minus_simulated"].values()) == pytest.approx(
        q["gap_reference_minus_simulated"], abs=1e-6
    )
    # aging_in_gap == the two aged-in onset buckets' gaps.
    assert d["aging_in_gap"] == pytest.approx(
        d["gap_reference_minus_simulated"]["onset_lt65"]
        + d["gap_reference_minus_simulated"]["onset_65_74"],
        abs=1e-9,
    )


def test_q4_aging_in_carries_the_gap():
    d = _artifact()["question_4_widow_stock"]["onset_bucket_decomposition"]
    # The named verdict: aging-in carries the gap, dwarfing the fresh-75+
    # onset and the (structurally absent) initial-states margins.
    assert abs(d["aging_in_gap"]) > abs(d["fresh_75plus_onset_gap"])
    assert abs(d["aging_in_gap"]) > abs(d["initial_states_gap"])


def test_q4_initial_states_absent():
    init = _artifact()["question_4_widow_stock"]["initial_states"]
    # Retrospective-to-age-15 exposure => no left-censored widowhood.
    assert init["n_entry_widowed_persons_total"] == 0
    assert init["entry_widowed_75plus_stock_share"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_q4_outflow_is_elderly_widow_over_remarriage():
    outflow = _artifact()["question_4_widow_stock"][
        "outflow_elderly_widow_remarriage"
    ]
    for _band, rec in outflow.items():
        assert rec["sim_over_ref_ratio"] == pytest.approx(
            rec["simulated"] / rec["reference"], rel=1e-9
        )
    # The 75+ band is over-remarried (the pooled 50+ band's misallocation);
    # its ratio exceeds every younger band's.
    r75 = outflow["75+"]["sim_over_ref_ratio"]
    assert r75 > 1.0
    assert r75 > outflow["50-64"]["sim_over_ref_ratio"]
    assert r75 > outflow["65-74"]["sim_over_ref_ratio"]


def test_q4_inflow_ratios_recompute():
    inflow = _artifact()["question_4_widow_stock"]["inflow_incidence"]
    for rec in inflow.values():
        assert rec["sim_over_ref_ratio"] == pytest.approx(
            rec["simulated"] / rec["reference"], rel=1e-9
        )


def test_q4_spousal_gap_draw_does_not_underage_husbands():
    t = _artifact()["question_4_widow_stock"]["spousal_age_gap_test"]
    # The registration asks whether the draw under-ages husbands; it does not.
    assert t["husband_underaged_by_draw"] is False
    assert t["imputed_minus_observed_mean_husband_age"] >= 0
    assert "REJECTED" in t["finding"]
    # Consistency of the mean-husband-age delta with the two means.
    assert t["imputed_minus_observed_mean_husband_age"] == pytest.approx(
        t["imputed_mean_husband_age"] - t["observed_mean_husband_age"],
        abs=1e-9,
    )


def test_q4_per_seed_gap_recomputes_from_per_seed_block():
    a = _artifact()
    per_seed = {s["seed"]: s for s in a["per_seed"]}
    for rec in a["question_4_widow_stock"]["per_seed"]:
        s = per_seed[rec["seed"]]
        ref = s["ref_stock"]["share_75plus"]
        sim = s["sim_stock_mean"]["share_75plus"]
        assert rec["reference_share_75plus"] == pytest.approx(ref, abs=1e-12)
        assert rec["simulated_share_75plus"] == pytest.approx(sim, abs=1e-12)
        assert rec["gap"] == pytest.approx(ref - sim, abs=1e-12)


def test_q4_sim_share_is_mean_over_20_draws():
    for s in _artifact()["per_seed"]:
        draws = s["per_draw_share_widowed_75plus"]
        assert len(draws) == 20
        assert s["sim_stock_mean"]["share_75plus"] == pytest.approx(
            float(np.mean(draws)), abs=1e-9
        )


# --------------------------------------------------------------------------
# Question 5 -- female count residual under candidate 10
# --------------------------------------------------------------------------
def test_q5_pathway_deficit_is_reference_minus_simulated():
    by_sex = _artifact()["question_5_female_count"]["by_sex"]
    for sex in ("female", "male"):
        for rec in by_sex[sex]["pathway_cells"].values():
            assert rec["deficit"] == pytest.approx(
                rec["reference"] - rec["simulated"], abs=1e-9
            )


def test_q5_first_marriage_is_intensive_margin_near_one():
    by_sex = _artifact()["question_5_female_count"]["by_sex"]
    for sex in ("female", "male"):
        first = by_sex[sex]["pathway_cells"]["first"]
        assert first["reference"] == pytest.approx(1.0, abs=0.05)
        assert first["simulated"] == pytest.approx(1.0, abs=0.05)


def test_q5_female_over_production_recomputes_and_is_positive():
    fem = _artifact()["question_5_female_count"]["by_sex"]["female"]
    assert fem["in_exposure_over_production"] == pytest.approx(
        fem["in_exposure_simulated"] - fem["in_exposure_reference"], abs=1e-9
    )
    # The female in-exposure count is over-produced (sim exceeds reference).
    assert fem["in_exposure_over_production"] > 0
    # The largest over-produced cell has a negative deficit (sim > ref).
    assert fem["largest_over_produced_cells"][0]["deficit"] < 0


def test_q5_over_production_sits_in_after_widowhood():
    split = _artifact()["question_5_female_count"][
        "female_over_production_origin_split"
    ]
    for _k, rec in split.items():
        assert rec["over_production"] == pytest.approx(
            rec["simulated"] - rec["reference"], abs=1e-9
        )
    # After-widowhood carries positive over-production, larger than the
    # after-divorce pathway the male age-conditioning fix addressed.
    assert split["after_widowhood"]["over_production"] > 0
    assert (
        split["after_widowhood"]["over_production"]
        > split["after_divorce"]["over_production"]
    )


def test_q5_conditioning_margin_is_age_at_dissolution():
    probe = _artifact()["question_5_female_count"]["conditioning_margin_probe"]
    assert probe["margin"] == "age_at_dissolution"
    tbl = probe["female_remarriage_by_age_at_dissolution"]
    for origin in ("divorced", "widowed"):
        for rec in tbl[origin].values():
            if rec["reference"] > 0:
                assert rec["sim_over_ref_ratio"] == pytest.approx(
                    rec["simulated"] / rec["reference"], rel=1e-9
                )
    # The misallocation signature: elderly (age-at-dissolution 65+) widows are
    # over-remarried relative to the reference.
    assert tbl["widowed"]["65+"]["sim_over_ref_ratio"] > 1.0


# --------------------------------------------------------------------------
# Published-outer context (candidate 10, never re-simulated)
# --------------------------------------------------------------------------
def test_published_outer_matches_committed_candidate10():
    committed = json.loads(CANDIDATE10.read_text())
    by_seed = {s["seed"]: s for s in committed["per_seed"]}
    cells = _artifact()["published_outer_context"]["cells"]
    cell = "share_widowed.75+|female"
    for seed_str, rec in cells[cell].items():
        gc = by_seed[int(seed_str)]["gated_cells"][cell]
        assert rec["rate_a"] == pytest.approx(gc["rate_a"], abs=1e-9)
        assert rec["score"] == pytest.approx(gc["score"], abs=1e-9)
        assert rec["pass"] == gc["pass"]
    # The decider failed the outer gate on 4 of 5 seeds.
    n_fail = sum(1 for r in cells[cell].values() if r["pass"] is False)
    assert n_fail == 4


# --------------------------------------------------------------------------
# Reproduction pin (needs the staged PSID marriage-history files)
# --------------------------------------------------------------------------
@needs_psid
def test_pin_seed0_reference_and_first_draw_reproduce():
    """Seed 0's train-side reference share and first draw reproduce.

    Rebuilds the panel, fits candidate 10 on the seed-0 train half, and matches
    the recorded reference widowed-stock share and the first simulation draw
    (5200) to float precision -- the train-side pipeline pinned bit-for-bit.
    """
    sys.path.insert(0, str(SCRIPTS))
    import gate2_forensics as gf1
    import gate2_forensics2 as gf2
    import run_gate2_candidate10 as c10

    from populace_dynamics.harness import panel as hpanel

    data = gf1._load_inputs()
    panel = data["panel"]
    _side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = {int(x) for x in side_b.person_id.unique()}

    recorded = next(s for s in _artifact()["per_seed"] if s["seed"] == 0)
    ref_stock = gf2.widowed_stock_shares(panel, ids_b)
    assert ref_stock["share_75plus"] == pytest.approx(
        recorded["ref_stock"]["share_75plus"], abs=1e-9
    )

    components = c10.fit_components(
        panel,
        data["demo"],
        data["death_records"],
        data["mh_records"],
        data["birth_records"],
        data["order_map"],
        ids_b,
    )
    sim_panel, _ = c10.simulate_holdout(
        panel, ids_b, components, gf2.DRAW_SEED_BASE
    )
    sim_stock = gf2.widowed_stock_shares(sim_panel, ids_b)
    assert sim_stock["share_75plus"] == pytest.approx(
        recorded["per_draw_share_widowed_75plus"][0], abs=1e-9
    )
