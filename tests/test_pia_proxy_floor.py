"""Tests for the PIA-proxy anchor floor (runs/pia_proxy_floor_9822.json).

This artifact is the real-vs-real PIA-proxy floor at deployment scale that
the gate-1 amendment proposal cites for its proposed benefit_space KS
threshold. It is a committed evidence anchor, pinned like the other
``runs/`` floors.

Two tiers, mirroring the other run/floor tests:

* Always-runnable internal-consistency tests (no PSID, no populace-fit)
  that touch only the committed artifact: the schema is sane and marked a
  reported anchor; every pooled statistic recomputes from the stored
  per-seed values; and the headline KS block is exposed in the
  committed-floor convention so the amendment's KS derivation binds.
* A seed-0 reproduction pin (skipped when the PSID family files are
  absent) that reruns the seed-0 real-vs-real measurement through the
  build machinery and pins the committed artifact's seed-0 numbers to
  float precision.

Crucially, the floor is REAL-VS-REAL: no candidate is generated, so the
reproduction path imports and runs the oracle WITHOUT ``populace.fit``.
The test asserts that: importing the builder and running seed 0 must not
pull ``populace.fit`` into ``sys.modules``. This is the property that lets
the anchor be reproduced off the dedicated gate venv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "pia_proxy_floor_9822.json"
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
    import build_pia_proxy_floor as builder

    return builder


# --------------------------------------------------------------------------
# Schema and reported-anchor framing (always runnable)
# --------------------------------------------------------------------------
def test_schema_and_reported_anchor():
    art = _artifact()
    assert art["schema_version"] == "pia_proxy_floor.v1"
    assert art["run"] == "pia_proxy_floor_9822"
    assert art["reported_anchor_not_gated"] is True
    assert isinstance(art["purpose"], str) and art["purpose"].strip()
    assert "changes no gate" in art["purpose"]
    # Proxy-not-full-415(b) disclaimer present.
    assert "STATUTE-SHAPED PROXY" in art["not_full_415b"]
    assert "415(b)" in art["not_full_415b"]
    # Oracle revision pinned, and the functional is imported (single
    # source of truth), not re-implemented.
    assert art["oracle"]["pe_us_revision"]
    assert (
        "build_downstream_relevance.py:person_pia_proxy"
        in art["functional"]["source"]
    )
    # Headline KS block in the committed-floor convention.
    assert set(art["noise_floor_seeds_0_4"]) == {"ks_distance"}
    ks = art["noise_floor_seeds_0_4"]["ks_distance"]
    assert {"mean", "sd", "values"} <= set(ks)
    assert len(ks["values"]) == len(art["protocol"]["seeds"])


def test_floor_full_panel_not_train_split():
    """The anchor is built on the FULL locked filtered panel (deployment
    scale), not a train split -- that is what distinguishes it from the
    downstream artifact's noise anchor."""
    art = _artifact()
    assert "FULL locked filtered panel" in art["data"]
    assert art["n_persons"] > 20000  # full panel, not a 0.8 train split
    # Each seed's two disjoint halves are ~20% of persons each.
    for row in art["per_seed"]:
        assert (
            0.15 * art["n_persons"]
            < row["n_persons_side_a"]
            < (0.25 * art["n_persons"])
        )


def test_pooled_stats_recompute_from_per_seed():
    """Every pooled floor statistic recomputes from the per-seed values."""
    art = _artifact()
    per = art["per_seed"]
    floor = art["floor_seeds_0_4"]

    def dist(row, path):
        node = row["distribution"]
        for key in path:
            node = node[key]
        return node

    # KS mean/sd/values.
    ks_vals = [dist(r, ["ks_distance"]) for r in per]
    assert floor["ks_distance"]["values"] == pytest.approx(ks_vals)
    assert floor["ks_distance"]["mean"] == pytest.approx(np.mean(ks_vals))
    assert floor["ks_distance"]["sd"] == pytest.approx(np.std(ks_vals, ddof=1))

    # Absolute percent-gap magnitudes from the signed per-seed gaps.
    abs_mean = [abs(dist(r, ["mean", "pct_diff"])) for r in per]
    abs_median = [abs(dist(r, ["median", "pct_diff"])) for r in per]
    assert floor["abs_mean_pct_diff"]["values"] == pytest.approx(abs_mean)
    assert floor["abs_mean_pct_diff"]["mean"] == pytest.approx(
        np.mean(abs_mean)
    )
    assert floor["abs_median_pct_diff"]["values"] == pytest.approx(abs_median)

    # Per-decile absolute magnitudes.
    for dkey, block in floor["abs_decile_pct_diff"].items():
        signed = [dist(r, ["deciles", dkey, "pct_diff"]) for r in per]
        assert block["values"] == pytest.approx([abs(v) for v in signed])

    # Pooled Q0 magnitude = |across-seed mean of signed Q0 gaps|.
    q0_signed = [r["q0_zero_anchor"]["mean_pct_diff"] for r in per]
    assert floor["pooled_abs_q0_mean_pct_diff"] == pytest.approx(
        abs(np.mean(q0_signed))
    )
    assert floor["abs_q0_mean_pct_diff"]["values"] == pytest.approx(
        [abs(v) for v in q0_signed]
    )

    # Per-decile 5%-clip counts.
    for dkey, count in floor["decile_seeds_clipping_5pct"].items():
        vals = floor["signed_decile_pct_diff"][dkey]["values"]
        assert count == sum(abs(v) > 5.0 for v in vals), dkey


def test_headline_ks_block_mirrors_floor():
    """The exposed noise_floor_seeds_0_4 KS equals the floor KS block."""
    art = _artifact()
    assert (
        art["noise_floor_seeds_0_4"]["ks_distance"]
        == art["floor_seeds_0_4"]["ks_distance"]
    )


def test_q0_gap_is_within_five_percent_real_vs_real():
    """The real-vs-real Q0 floor (pooled) sits inside the +/-5% band.

    This is the load-bearing coherence property: the proposed Q0 gate
    (abs pooled Q0 % <= 5) must PASS real-vs-real, or it would reject
    reality. The per-seed floor is noisy; the pooled magnitude is clean.
    """
    art = _artifact()
    assert art["floor_seeds_0_4"]["pooled_abs_q0_mean_pct_diff"] <= 5.0


# --------------------------------------------------------------------------
# Seed-0 reproduction pin (needs PSID; NO populace-fit)
# --------------------------------------------------------------------------
@needs_real_family
def test_seed0_reproduces_committed_artifact_without_populace_fit():
    """Rerun the seed-0 real-vs-real block; match to float precision.

    The floor is real-vs-real, so no candidate is generated and the
    oracle import path must NOT pull populace.fit. This test asserts that
    property: after importing the builder and running seed 0, populace.fit
    is absent from sys.modules.
    """
    assert "populace.fit" not in sys.modules, (
        "populace.fit already imported before the anchor path ran; this "
        "test must observe a clean oracle path"
    )
    builder = _import_builder()
    assert (
        "populace.fit" not in sys.modules
    ), "importing the anchor builder pulled populace.fit"

    art = _artifact()
    ref = next(s for s in art["per_seed"] if s["seed"] == 0)

    params = builder.load_ssa_parameters()
    assert params.pe_us_revision == art["oracle"]["pe_us_revision"], (
        "policyengine-us checkout revision differs from the pinned build; "
        "set POPULACE_DYNAMICS_PE_US_DIR to the pinned checkout"
    )
    panel = builder.load_filtered_panel()
    all_anchor = builder.anchor_rows(panel)
    cutpoints = builder.anchor_quintile_cutpoints(all_anchor)

    got = builder.measure_seed(0, panel, all_anchor, cutpoints, params, False)

    # No candidate generation ran, so populace.fit stayed out.
    assert (
        "populace.fit" not in sys.modules
    ), "the anchor seed-0 measurement pulled populace.fit"

    # Distribution block: mean/median/decile values and KS.
    for key in ("mean", "median"):
        assert got["distribution"][key]["candidate"] == pytest.approx(
            ref["distribution"][key]["candidate"], abs=1e-9
        ), key
        assert got["distribution"][key]["real"] == pytest.approx(
            ref["distribution"][key]["real"], abs=1e-9
        ), key
    for dkey in ref["distribution"]["deciles"]:
        got_d = got["distribution"]["deciles"][dkey]
        ref_d = ref["distribution"]["deciles"][dkey]
        assert got_d["candidate"] == pytest.approx(
            ref_d["candidate"], abs=1e-9
        ), dkey
        assert got_d["real"] == pytest.approx(ref_d["real"], abs=1e-9), dkey
    assert got["distribution"]["ks_distance"] == pytest.approx(
        ref["distribution"]["ks_distance"], abs=1e-12
    )

    # Q0 subgroup gap.
    assert got["q0_zero_anchor"]["mean_pct_diff"] == pytest.approx(
        ref["q0_zero_anchor"]["mean_pct_diff"], abs=1e-9
    )

    # Person counts.
    assert got["n_persons_side_a"] == ref["n_persons_side_a"]
    assert got["n_persons_side_b"] == ref["n_persons_side_b"]
    assert got["n_persons_q0_side_a"] == ref["n_persons_q0_side_a"]
    assert got["n_persons_q0_side_b"] == ref["n_persons_q0_side_b"]
