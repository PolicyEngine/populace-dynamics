"""Tests for gate-2 forensics 3 (reported, not gated).

REPORTED-NOT-GATED. The consistency tests read only the forensics-3 artifact
(``runs/gate2_forensics3_v1.json``), the committed candidate-11 gate artifact
(``runs/gate2_hazard_v11.json``) and ``gates.yaml``; they never rerun the
diagnostic and need no PSID, so they run in CI. They audit that every headline
recomputes from the stored per-seed values: the Q6 support-reachability
taxonomy (its partition of the 75+ widowed stock, the structurally-unreachable
share, and the observed-initial-state-fix applicability) and the Q7 widowed
exposure-by-age blocks (the 50-64 pool-inflation ratio, the incidence inflow
shortfall by band, and the survival-in-widowhood curves), plus the
published-outer context match.

One reproduction pin (``test_pin_*``) rebuilds the train-side inputs live and
matches seed 0's reference support taxonomy and its first candidate-11
simulation draw to float precision (skipped when the PSID marriage-history
files are absent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate2_forensics3_v1.json"
CANDIDATE11 = ROOT / "runs" / "gate2_hazard_v11.json"
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
    assert a["schema_version"] == "gate2_forensics3.v1"
    assert a["reported_not_gated"] is True
    assert a["registration"].endswith("#issuecomment-4920355640")
    assert a["registration_pointer"] == "4920355640"
    assert "candidate 11" in a["candidate_under_diagnosis"]
    for block in (
        "question_6_reference_reachability",
        "question_7_widowed_exposure_by_age",
    ):
        assert block in a


def test_protocol_is_train_side_only():
    p = _artifact()["protocol"]
    assert p["train_side_only"] is True
    assert "never" in p["outer_holdout_contact"].lower()
    assert p["gate_seeds"] == [0, 1, 2, 3, 4]
    assert p["n_draws"] == 20
    assert "5200" in p["draw_rng_rule"]
    # The Q6 boundary is the observed PSID support, not the age-15 exposure.
    assert "demographic_panel" in p["q6_support_boundary"]


def test_no_gate_verdict_written_and_lock_untouched():
    a = _artifact()
    assert "verdict" not in a
    assert "gate_2_pass" not in a
    assert a["revision_pins"]["gates_yaml_locked"] is True


def test_candidate_12_registers_only_on_this_evidence():
    a = _artifact()
    assert "candidate 12" in a["candidate_12_implications"]
    assert "initial-state" in a["candidate_12_implications"].lower()


# --------------------------------------------------------------------------
# Question 6 -- support-reachability taxonomy
# --------------------------------------------------------------------------
def test_q6_buckets_partition_the_widowed_stock():
    q = _artifact()["question_6_reference_reachability"]
    buckets = q["taxonomy_share_of_widowed_stock"]
    # The four buckets partition the 75+ widowed stock, so they sum to 1.
    assert sum(buckets.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(buckets) == {
        "reachable_within_support",
        "carried_predates_support",
        "onset_after_support_end",
        "non_derivable",
    }


def test_q6_structurally_unreachable_recomputes():
    q = _artifact()["question_6_reference_reachability"]
    b = q["taxonomy_share_of_widowed_stock"]
    assert q["structurally_unreachable_share"] == pytest.approx(
        b["carried_predates_support"]
        + b["onset_after_support_end"]
        + b["non_derivable"],
        abs=1e-9,
    )
    # The initial-state-fixable share is exactly the carried-status mass.
    assert q["initial_state_fixable_share"] == pytest.approx(
        b["carried_predates_support"], abs=1e-9
    )
    assert q["non_derivable_residual_share"] == pytest.approx(
        b["non_derivable"], abs=1e-12
    )


def test_q6_most_of_stock_is_reachable_but_a_carried_share_exists():
    q = _artifact()["question_6_reference_reachability"]
    b = q["taxonomy_share_of_widowed_stock"]
    # The registration's headline: most of the stock is simulation-reachable,
    # but a non-trivial carried-status share is structurally unreachable.
    assert b["reachable_within_support"] > 0.75
    assert b["carried_predates_support"] > 0.01
    assert q["structurally_unreachable_share"] > 0.01


def test_q6_non_derivable_residual_is_zero():
    q = _artifact()["question_6_reference_reachability"]
    # Unlike the marriage-count residual, every widowed person-year is derived
    # from a datable widowhood episode (years_since_dissolution never NA), so
    # there is no "not derivable from any datable episode" mass.
    assert q["non_derivable_residual_share"] == pytest.approx(0.0, abs=1e-4)


def test_q6_observed_initial_state_fix_applies():
    q = _artifact()["question_6_reference_reachability"]
    # The unreachable mass is carried status with a zero non-derivable
    # residual, so the observed-initial-state fix (marriage-count precedent)
    # applies; a rate fix cannot recover it.
    assert q["observed_initial_state_fix_applies"] is True
    assert "APPLIES" in q["fix_verdict"]


def test_q6_reconstruction_window_hides_the_carried_share():
    q = _artifact()["question_6_reference_reachability"]
    b = q["taxonomy_share_of_widowed_stock"]
    # Under the retrospective-to-age-15 exposure window (forensics 2's
    # boundary) there is no carried mass; the observed-support boundary is what
    # exposes it. This pins WHY forensics 3 differs from forensics 2.
    assert q["reconstruction_window_carried_share"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert (
        b["carried_predates_support"]
        > q["reconstruction_window_carried_share"]
    )


def test_q6_per_seed_recomputes_from_per_seed_block():
    a = _artifact()
    per_seed = {s["seed"]: s for s in a["per_seed"]}
    for rec in a["question_6_reference_reachability"]["per_seed"]:
        tax = per_seed[rec["seed"]]["ref_support_taxonomy"]
        b = tax["buckets_share_of_widowed_stock"]
        assert rec["reachable_within_support"] == pytest.approx(
            b["reachable_within_support"], abs=1e-12
        )
        assert rec["structurally_unreachable_share"] == pytest.approx(
            rec["carried_predates_support"]
            + rec["onset_after_support_end"]
            + rec["non_derivable"],
            abs=1e-9,
        )


# --------------------------------------------------------------------------
# Question 7 -- widowed exposure by age
# --------------------------------------------------------------------------
def test_q7_widowed_by_age_ratios_recompute():
    tbl = _artifact()["question_7_widowed_exposure_by_age"][
        "widowed_person_years_by_age_sex"
    ]
    for rec in tbl.values():
        if rec["reference_widowed_py_weight"] > 0:
            assert rec["sim_over_ref_weight_ratio"] == pytest.approx(
                rec["simulated_widowed_py_weight"]
                / rec["reference_widowed_py_weight"],
                rel=1e-9,
            )


def test_q7_5064_pool_is_inflated():
    q = _artifact()["question_7_widowed_exposure_by_age"]
    pool = q["pool_5064_female"]
    # The registration's hypothesis: the simulated widowed 50-64 female pool is
    # inflated (which explains the female count over-production under correct
    # rates).
    assert pool["sim_over_ref_weight_ratio"] > 1.0
    assert "inflated" in q["pool_inflation_verdict"].lower()
    assert "not inflated" not in q["pool_inflation_verdict"].lower()


def test_q7_inflow_ratios_and_shortfall_recompute():
    inflow = _artifact()["question_7_widowed_exposure_by_age"][
        "inflow_incidence_by_age_sex"
    ]
    for _sex, bands in inflow.items():
        for rec in bands.values():
            if rec["reference"] > 0:
                assert rec["sim_over_ref_ratio"] == pytest.approx(
                    rec["simulated"] / rec["reference"], rel=1e-9
                )
                assert rec["inflow_shortfall"] == pytest.approx(
                    1.0 - rec["sim_over_ref_ratio"], abs=1e-9
                )


def test_q7_inflow_shortfall_concentrates_at_old_bands():
    q = _artifact()["question_7_widowed_exposure_by_age"]
    worst = q["inflow_shortfall_worst_female_band"]
    # The ~10% inflow shortfall concentrates at the old bands (the untouched
    # widowhood-incidence side that keeps the 75+ stock down).
    assert worst in ("65-74", "75+")
    fem = q["inflow_incidence_by_age_sex"]["female"]
    # The 75+ band under-produces incidence vs the reference.
    assert fem["75+"]["sim_over_ref_ratio"] < 1.0
    # The worst band's shortfall is the largest over the female bands.
    assert fem[worst]["inflow_shortfall"] == max(
        rec["inflow_shortfall"] for rec in fem.values()
    )


def test_q7_survival_curves_are_valid_and_reported_matches():
    surv = _artifact()["question_7_widowed_exposure_by_age"][
        "survival_in_widowhood_female"
    ]
    for label in ("50-64", "65+"):
        block = surv[label]
        for side in ("reference_survival_curve", "simulated_survival_curve"):
            curve = block[side]
            # Survival starts at 1 and is non-increasing (KM).
            assert curve[0] == pytest.approx(1.0, abs=1e-12)
            for i in range(1, len(curve)):
                assert curve[i] <= curve[i - 1] + 1e-12
        # Reported snapshots index the stored curve.
        for k, v in block["reference_reported"].items():
            assert block["reference_survival_curve"][int(k)] == pytest.approx(
                v, abs=1e-12
            )
        for k, v in block["simulated_reported"].items():
            assert block["simulated_survival_curve"][int(k)] == pytest.approx(
                v, abs=1e-12
            )
        assert block["sim_minus_ref_rmst"] == pytest.approx(
            block["simulated_restricted_mean_survival_years"]
            - block["reference_restricted_mean_survival_years"],
            abs=1e-9,
        )


def test_q7_elderly_onset_survival_tracks_reference():
    surv = _artifact()["question_7_widowed_exposure_by_age"][
        "survival_in_widowhood_female"
    ]
    # Candidate 11's 50+ remarriage-band split fixed the elderly-widow
    # over-remarriage, so 65+ onset survival-in-widowhood tracks the reference
    # (the residual 75+ gap is inflow, not outflow).
    assert abs(surv["65+"]["sim_minus_ref_rmst"]) < 2.5


def test_q7_sim_share_is_mean_over_20_draws():
    for s in _artifact()["per_seed"]:
        draws = s["per_draw_share_widowed_75plus"]
        assert len(draws) == 20
        assert s["sim_stock_mean"]["share_75plus"] == pytest.approx(
            float(np.mean(draws)), abs=1e-9
        )


# --------------------------------------------------------------------------
# Published-outer context (candidate 11, never re-simulated)
# --------------------------------------------------------------------------
def test_published_outer_matches_committed_candidate11():
    committed = json.loads(CANDIDATE11.read_text())
    by_seed = {s["seed"]: s for s in committed["per_seed"]}
    cells = _artifact()["published_outer_context"]["cells"]
    cell = "share_widowed.75+|female"
    n_fail_committed = 0
    for seed_str, rec in cells[cell].items():
        gc = by_seed[int(seed_str)]["gated_cells"][cell]
        assert rec["rate_a"] == pytest.approx(gc["rate_a"], abs=1e-9)
        assert rec["score"] == pytest.approx(gc["score"], abs=1e-9)
        assert rec["pass"] == gc["pass"]
        if gc["pass"] is False:
            n_fail_committed += 1
    # The registered modal cell persisted as a failure on the outer holdout.
    n_fail_ctx = sum(1 for r in cells[cell].values() if r["pass"] is False)
    assert n_fail_ctx == n_fail_committed
    assert n_fail_ctx >= 1


# --------------------------------------------------------------------------
# Reproduction pin (needs the staged PSID marriage-history files)
# --------------------------------------------------------------------------
@needs_psid
def test_pin_seed0_reference_taxonomy_and_first_draw_reproduce():
    """Seed 0's reference support taxonomy and first draw reproduce.

    Rebuilds the panel, computes the seed-0 train-half reference support
    taxonomy, fits candidate 11 and matches the recorded structurally-
    unreachable share and the first simulation draw (5200) to float precision
    -- the train-side pipeline pinned bit-for-bit.
    """
    sys.path.insert(0, str(SCRIPTS))
    import gate2_forensics as gf1
    import gate2_forensics3 as gf3
    import run_gate2_candidate11 as c11

    from populace_dynamics.harness import panel as hpanel

    data = gf1._load_inputs()
    panel = data["panel"]
    support = gf3.observed_support(data["demo"])
    _side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=0
    )
    ids_b = {int(x) for x in side_b.person_id.unique()}

    recorded = next(s for s in _artifact()["per_seed"] if s["seed"] == 0)
    tax = gf3.widowed_75plus_support_taxonomy(panel, support, ids_b)
    assert tax["structurally_unreachable_share"] == pytest.approx(
        recorded["ref_support_taxonomy"]["structurally_unreachable_share"],
        abs=1e-9,
    )
    assert tax["buckets_share_of_widowed_stock"][
        "carried_predates_support"
    ] == pytest.approx(
        recorded["ref_support_taxonomy"]["buckets_share_of_widowed_stock"][
            "carried_predates_support"
        ],
        abs=1e-9,
    )

    components = c11.fit_components(
        panel,
        data["demo"],
        data["death_records"],
        data["mh_records"],
        data["birth_records"],
        data["order_map"],
        ids_b,
    )
    sim_panel, _ = c11.simulate_holdout(
        panel, ids_b, components, gf3.DRAW_SEED_BASE
    )
    sim_stock = gf3._stock_shares(sim_panel, ids_b)
    assert sim_stock["share_75plus"] == pytest.approx(
        recorded["per_draw_share_widowed_75plus"][0], abs=1e-9
    )
