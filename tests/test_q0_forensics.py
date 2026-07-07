"""Tests for the Q0-forensics artifact (why zero-anchor careers overstate).

Two tiers, mirroring the gate runs' and downstream-relevance test structure:

* Always-runnable internal-consistency tests (no PSID, no populace-fit) that
  touch only the committed artifact plus the pure helpers in
  :mod:`scripts.q0_forensics`: the schema is sane and marked
  reported-not-gated with the no-holdout-real-contact framing; the Q1
  participation and level channels ADD to the total Q0 gap (an exact
  additive decomposition); the participation subsplit sums to the net
  participation channel; the pooled scalars recompute as the per-seed means;
  and the candidate-8 cross-check records the byte-identical-gate finding on
  every seed.
* A seed-0 reproduction pin (skipped when PSID family files are absent, and
  ``importorskip('populace.fit')`` because the candidate-7/8 participation
  gate needs it) that reruns the seed-0 measurement through the build
  machinery and pins the committed seed-0 numbers to float precision. It is
  run live in the dedicated gate venv before the artifact is committed.

The build script is REPORTED-NOT-GATED: it reads no gate and changes no gate,
and reads no holdout-real career beyond the pooled Q0 mean the ratified
benefit-space gate already scores. These tests assert that framing is
recorded and that the artifact's arithmetic is internally reproducible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "q0_forensics_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import q0_forensics as builder

    return builder


# --------------------------------------------------------------------------
# Schema and reported-not-gated framing (always runnable)
# --------------------------------------------------------------------------
def test_schema_and_reported_not_gated():
    art = _artifact()
    assert art["schema_version"] == "q0_forensics.v1"
    assert art["run"] == "q0_forensics_v1"
    assert art["reported_not_gated"] is True
    # The load-bearing framing: reported, no holdout-real contact, purpose.
    assert "changes no gate" in art["no_holdout_real_contact"] or (
        "gate already scores" in art["no_holdout_real_contact"]
    )
    assert isinstance(art["purpose"], str) and art["purpose"].strip()
    # The comparison population is TRAIN zero-anchor persons.
    assert "TRAIN" in art["definitions"]["comparison_population"]
    # The proxy-is-an-average disclaimer is present (the mechanical premise).
    assert "average" in art["definitions"]["pia_proxy"].lower()
    assert art["revision_pins"]["pe_us_revision"]
    assert art["seeds"] == [0, 1, 2, 3, 4]
    assert [s["seed"] for s in art["per_seed"]] == [0, 1, 2, 3, 4]


def test_candidate7_reference_recorded():
    art = _artifact()
    ref = art["candidate7_reference"]
    assert ref["path"] == "runs/gate1_rank_knn_v1.json"
    # The committed pooled Q0 mean the block gates on is recorded.
    assert ref["committed_pooled_q0_mean_pct"] == pytest.approx(
        9.295292319680367, abs=1e-9
    )


# --------------------------------------------------------------------------
# Q1: the exact additive decomposition (always runnable)
# --------------------------------------------------------------------------
def test_participation_plus_level_equals_total_gap():
    """participation channel + level channel == total Q0 gap, every seed."""
    art = _artifact()
    for s in art["per_seed"]:
        q1 = s["q1_participation_vs_level"]
        total = q1["total_gap_pct"]
        partic = q1["participation_channel_pct_of_real_mean"]
        level = q1["level_channel_pct_of_real_mean"]
        assert partic + level == pytest.approx(
            total, rel=1e-9, abs=1e-9
        ), f"seed {s['seed']}: {partic} + {level} != {total}"


def test_participation_subsplit_sums_to_net():
    """zero->positive + positive->zero == net participation, every seed."""
    art = _artifact()
    for s in art["per_seed"]:
        q1 = s["q1_participation_vs_level"]
        bc = q1["by_category"]
        z2p = bc["zero_to_positive"]["gap_contribution_pct_of_real_mean"]
        p2z = bc["positive_to_zero"]["gap_contribution_pct_of_real_mean"]
        net = q1["participation_channel_pct_of_real_mean"]
        assert z2p + p2z == pytest.approx(net, rel=1e-9, abs=1e-9), s["seed"]


def test_all_four_categories_sum_to_total_gap():
    """The four transition categories partition the total gap exactly."""
    art = _artifact()
    for s in art["per_seed"]:
        q1 = s["q1_participation_vs_level"]
        total = q1["total_gap_pct"]
        parts = sum(
            q1["by_category"][name]["gap_contribution_pct_of_real_mean"]
            for name in (
                "zero_to_zero",
                "zero_to_positive",
                "positive_to_zero",
                "positive_to_positive",
            )
        )
        assert parts == pytest.approx(total, rel=1e-9, abs=1e-9), s["seed"]


def test_category_weight_shares_sum_to_one():
    art = _artifact()
    for s in art["per_seed"]:
        bc = s["q1_participation_vs_level"]["by_category"]
        shares = sum(bc[name]["weight_share"] for name in bc)
        assert shares == pytest.approx(1.0, rel=1e-9, abs=1e-9), s["seed"]


# --------------------------------------------------------------------------
# The ranked-mechanisms conclusion the PR leads with (always runnable)
# --------------------------------------------------------------------------
def test_participation_dominates_level_pooled():
    """The pooled conclusion: participation is the driver, level is not."""
    art = _artifact()
    ranked = art["ranked_mechanisms"]["ranked_by_abs_pooled_gap_pct"]
    # First-ranked mechanism is participation and materially exceeds level.
    assert "participation" in ranked[0]["mechanism"]
    assert abs(ranked[0]["pooled_gap_pct"]) > abs(ranked[1]["pooled_gap_pct"])
    # The participation channel alone is >= the whole gate-breaching gap.
    sub = art["ranked_mechanisms"]["participation_subsplit"]
    assert sub["net_participation_pct"] > 5.0  # breaches the 5% Q0 band alone


def test_covariate_ceiling_exceeds_production_set():
    """Q4: realized n_pos R^2 exceeds the age+span production set (pooled)."""
    art = _artifact()
    p = art["pooled"]
    assert (
        p["q4_r2_n_pos_NOT_production"]["mean"]
        > p["q4_r2_age_plus_span"]["mean"]
    )
    # Adding anchor position on top of age+span barely helps (near-useless).
    assert p["q4_r2_age_span_position"]["mean"] == pytest.approx(
        p["q4_r2_age_plus_span"]["mean"], abs=0.02
    )


def test_candidate8_gate_byte_identical_all_seeds():
    """The c8 cross-check: same participation gate, so same all-zero share."""
    art = _artifact()
    for s in art["per_seed"]:
        c8 = s["candidate8_cross_check"]
        assert c8["gen_share_all_zero_matches_c7"] is True, s["seed"]
    assert art["pooled"]["c8_gate_match_all_seeds"] is True
    # And the consistency verdict is recorded on the ranked block.
    cc = art["ranked_mechanisms"]["candidate8_consistency"]
    assert cc["participation_gate_byte_identical_all_seeds"] is True
    assert "consistent" in cc["verdict"].lower()


# --------------------------------------------------------------------------
# Pooled scalars recompute as the per-seed means (always runnable)
# --------------------------------------------------------------------------
def test_pooled_scalars_recompute_from_per_seed():
    art = _artifact()
    p = art["pooled"]

    def per_seed(path: list) -> list[float]:
        out = []
        for s in art["per_seed"]:
            node = s
            for k in path:
                node = node[k]
            out.append(node)
        return out

    checks = {
        "q1_total_gap_pct": ["q1_participation_vs_level", "total_gap_pct"],
        "q1_participation_channel_pct": [
            "q1_participation_vs_level",
            "participation_channel_pct_of_real_mean",
        ],
        "q1_level_channel_pct": [
            "q1_participation_vs_level",
            "level_channel_pct_of_real_mean",
        ],
        "c8_total_gap_pct": ["candidate8_cross_check", "total_gap_pct"],
    }
    for pooled_key, path in checks.items():
        assert p[pooled_key]["mean"] == pytest.approx(
            float(np.mean(per_seed(path))), rel=1e-9
        ), pooled_key


# --------------------------------------------------------------------------
# Seed-0 reproduction pin (needs PSID + populace-fit)
# --------------------------------------------------------------------------
@needs_real_family
def test_seed0_reproduces_committed_artifact():
    """Rerun the seed-0 forensics; pin the committed seed-0 numbers."""
    pytest.importorskip(
        "populace.fit",
        reason="populace-fit not installed (gate runs use a dedicated venv)",
    )
    builder = _import_builder()

    art = _artifact()
    ref0 = next(s for s in art["per_seed"] if s["seed"] == 0)

    params = builder.load_ssa_parameters()
    assert params.pe_us_revision == art["revision_pins"]["pe_us_revision"], (
        "policyengine-us checkout revision differs from the pinned build; "
        "set POPULACE_DYNAMICS_PE_US_DIR to the pinned checkout"
    )
    panel = builder.load_filtered_panel()
    all_anchor = builder.anchor_rows(panel)
    cutpoints = builder.anchor_quintile_cutpoints(all_anchor)

    got = builder.measure_seed(0, panel, all_anchor, params, cutpoints, False)

    # Q1 channel split reproduces to float precision.
    g1 = got["q1_participation_vs_level"]
    r1 = ref0["q1_participation_vs_level"]
    for key in (
        "total_gap_pct",
        "participation_channel_pct_of_real_mean",
        "level_channel_pct_of_real_mean",
    ):
        assert g1[key] == pytest.approx(r1[key], abs=1e-9), key

    # Q1 counterfactual swap reproduces.
    for side in (
        "gen_participation_real_levels",
        "real_participation_gen_levels",
    ):
        assert got["q1_counterfactual_swap"][side][
            "pct_diff_vs_real"
        ] == pytest.approx(
            ref0["q1_counterfactual_swap"][side]["pct_diff_vs_real"], abs=1e-9
        ), side

    # Q2 re-entry ranks reproduce.
    g2 = got["q2_reentry_channel"]
    r2 = ref0["q2_reentry_channel"]
    assert g2["drawn_reentry_rank"]["weighted_mean"] == pytest.approx(
        r2["drawn_reentry_rank"]["weighted_mean"], abs=1e-9
    )
    assert g2["drawn_minus_real_mean_rank_gap"] == pytest.approx(
        r2["drawn_minus_real_mean_rank_gap"], abs=1e-9
    )

    # Q3 participation shares reproduce.
    g3 = got["q3_participation_channel"]
    r3 = ref0["q3_participation_channel"]
    assert g3["generated"]["weighted_share_all_zero"] == pytest.approx(
        r3["generated"]["weighted_share_all_zero"], abs=1e-9
    )
    assert g3["train_real_zero_anchor"][
        "weighted_share_all_zero"
    ] == pytest.approx(
        r3["train_real_zero_anchor"]["weighted_share_all_zero"], abs=1e-9
    )

    # Q4 R^2 reproduces.
    for key in ("age_plus_span", "n_pos_only_NOT_production"):
        assert got["q4_covariate_discrimination"]["weighted_r2"][
            key
        ] == pytest.approx(
            ref0["q4_covariate_discrimination"]["weighted_r2"][key], abs=1e-9
        ), key

    # Candidate-8 cross-check reproduces and the gate stays byte-identical.
    assert got["candidate8_cross_check"]["total_gap_pct"] == pytest.approx(
        ref0["candidate8_cross_check"]["total_gap_pct"], abs=1e-9
    )
    assert got["candidate8_cross_check"]["gen_share_all_zero_matches_c7"] is (
        True
    )

    # Subgroup sizes reproduce exactly (deterministic given the split).
    assert got["n_q0_holdout"] == ref0["n_q0_holdout"]
    assert got["n_q0_train"] == ref0["n_q0_train"]
