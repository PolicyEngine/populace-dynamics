"""Internal-consistency tests for the C2ST forensics artifact.

These are reported-not-gated: the artifact
(``runs/c2st_forensics_v1.json``) is a forensic analysis of two already-run
gate-1 candidates, never a gate run, and it makes NO holdout contact for
any reported forensic value. The tests here touch only that artifact and
never ``gates.yaml`` or the committed gate ``runs/`` artifacts, so they
run anywhere (no PSID, no populace-fit).

They check that the stored numbers are internally consistent with the
claims the artifact makes -- the mirrored classifier equals the harness
C2ST, marginal/pair AUCs sit in the band the full-feature AUC and the
noise floor bound, the round-number break is present, and the
per-candidate attribution (splice = marginal-carried, kernel =
joint-carried) holds -- so a reader can audit the write-up against the
artifact without rerunning the (PSID-gated) analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "c2st_forensics_v1.json"

PAIRS = "psid_family_earnings_pairs"
RUNS = "psid_family_earnings_runs"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_artifact_present_and_reported_not_gated():
    """Schema, provenance, and the no-holdout-contact flags are recorded."""
    a = _artifact()
    assert a["schema_version"] == "c2st_forensics.v1"
    assert a["reported_not_gated"] is True
    assert a["holdout_contact"] is False
    assert a["seed"] == 0
    assert set(a["by_view"]) == {PAIRS, RUNS}


@pytest.mark.parametrize("view", [PAIRS, RUNS])
@pytest.mark.parametrize("cand", ["splice", "kernel"])
def test_mirror_equals_harness(view: str, cand: str):
    """The mirrored classifier reproduces the harness C2ST bit-for-bit.

    This is what makes every other number in the artifact a statement
    about the gate's own classifier and not a look-alike.
    """
    sa = _artifact()["by_view"][view]["sanity_anchor"][cand]
    assert sa["mirror_equals_harness"] is True
    assert sa["mirror_c2st_vs_holdout"] == pytest.approx(
        sa["harness_c2st_vs_holdout"], abs=1e-12
    )


@pytest.mark.parametrize("view", [PAIRS, RUNS])
@pytest.mark.parametrize("cand", ["splice", "kernel"])
def test_attribution_band(view: str, cand: str):
    """Marginal and pair AUCs lie between the noise floor and full AUC.

    No single feature or pair should separate more than the full feature
    set (up to CV slack), and none should fall meaningfully below the
    real-vs-real floor -- the attribution decomposes the full AUC rather
    than exceeding it.
    """
    block = _artifact()["by_view"][view]
    fa = block["candidates"][cand]["feature_attribution"]
    floor = block["shared_signal"]["noise_floor"]
    full = fa["full_auc"]
    slack = 0.02
    for name, auc in {**fa["marginal_auc"], **fa["pairwise_auc"]}.items():
        assert auc <= full + slack, f"{name} exceeds full AUC"
        assert auc >= floor - slack, f"{name} below the noise floor"


@pytest.mark.parametrize("view", [PAIRS, RUNS])
def test_distinct_not_shared_signal(view: str):
    """Splice vs kernel is at least as separable as either vs real.

    The two candidates land at similar candidate-vs-real AUC but are told
    apart from each other at least as well -- distinct defects, not one
    shared residual signal.
    """
    ss = _artifact()["by_view"][view]["shared_signal"]
    assert ss["noise_floor"] < ss["splice_vs_kernel"]
    assert (
        ss["splice_vs_kernel"]
        >= min(ss["splice_vs_train"], ss["kernel_vs_train"]) - 1e-9
    )


@pytest.mark.parametrize("view", [PAIRS, RUNS])
def test_noise_floor_below_candidates(view: str):
    """Real-vs-real (train halves) is easier to confuse than any candidate."""
    ss = _artifact()["by_view"][view]["shared_signal"]
    assert ss["noise_floor"] < ss["splice_vs_train"]
    assert ss["noise_floor"] < ss["kernel_vs_train"]


@pytest.mark.parametrize("view", [PAIRS, RUNS])
@pytest.mark.parametrize("cand", ["splice", "kernel"])
def test_round_number_break_present(view: str, cand: str):
    """Both candidates inject non-integer earnings; real PSID has none.

    A cosmetic footprint the report explicitly separates from what drives
    the AUC (the rounding-repair ablation), so the artifact must carry both
    the break and the ablation.
    """
    dist = _artifact()["by_view"][view]["candidates"][cand]["distributional"]
    rn = dist["round_number"]
    assert rn["train"]["non_integer_share"] == pytest.approx(0.0, abs=1e-6)
    assert rn["candidate"]["non_integer_share"] > 0.05
    repair = _artifact()["by_view"][view]["candidates"][cand][
        "rounding_repair"
    ]
    assert "raw" in repair and "snap_100" in repair


def test_kernel_is_joint_carried_pairs():
    """Kernel: marginals ~ chance, but the earnings joint ~ the full AUC.

    The rank kernel matches the one-period marginals and leaves its signal
    in the (t, t+1) joint, so the earnings-only joint AUC is well above the
    strongest single earnings marginal.
    """
    fa = _artifact()["by_view"][PAIRS]["candidates"]["kernel"][
        "feature_attribution"
    ]
    best_earn_marginal = max(
        fa["marginal_auc"]["earnings_t0"], fa["marginal_auc"]["earnings_t1"]
    )
    assert fa["earnings_only_auc"] > best_earn_marginal + 0.02
    # Marginals are near chance (matched distributions).
    assert best_earn_marginal < 0.53


def test_splice_is_marginal_carried_pairs():
    """Splice: a single earnings marginal already separates near the full.

    Segment rescaling distorts the marginal earnings distribution, so the
    strongest earnings marginal recovers most of the full-feature AUC.
    """
    fa = _artifact()["by_view"][PAIRS]["candidates"]["splice"][
        "feature_attribution"
    ]
    best_earn_marginal = max(
        fa["marginal_auc"]["earnings_t0"], fa["marginal_auc"]["earnings_t1"]
    )
    assert best_earn_marginal > 0.53
    assert best_earn_marginal >= fa["full_auc"] - 0.02


def test_kernel_runs_view_hardest():
    """The kernel's runs-view (window=3) AUC exceeds its pairs-view AUC.

    Two chained kernel steps accumulate the persistence deficit, so the
    three-observation window separates more than the two-observation one --
    the artifact's recorded published scores and the recomputed vs-train
    values agree on the direction.
    """
    a = _artifact()
    pairs = a["by_view"][PAIRS]["sanity_anchor"]["kernel"][
        "mirror_c2st_vs_holdout"
    ]
    runs = a["by_view"][RUNS]["sanity_anchor"]["kernel"][
        "mirror_c2st_vs_holdout"
    ]
    assert runs > pairs
