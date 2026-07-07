"""Tests for the reform-delta diagnostic (runs/reform_delta_diagnostic_v1.json).

The artifact asks whether the gate-passing generator (candidate 11, run 13)
reproduces reform INCIDENCE -- per-person and distributional benefit deltas --
that real histories imply, comparing the real-vs-generated gap against a
real-vs-real half-split floor. It is REPORTED, NOT GATED: it reads no gate and
changes no gate.

Two tiers, mirroring the other run/floor tests:

* Always-runnable internal-consistency tests (no PSID, no populace-fit) that
  touch only the committed artifact plus the pure helpers in
  :mod:`scripts.reform_delta_diagnostic`: the schema is sane and marked
  reported-not-gated; the registration pointer and the proxy-not-full-415(b)
  disclaimer (bound verbatim to the committed downstream artifact) are
  present; the reform definitions match the pinned oracle's baseline factors
  (ReformA = (0.95, baseline f2, baseline f3)); every pooled statistic, the
  decile/winners/Q0 blocks, the floor blocks, and the within-floor verdicts
  RECOMPUTE from the stored per-seed values; and the paired stats are labelled
  descriptive-only. Pure helpers reproduce closed-form answers.
* A seed-0 reproduction pin (skipped without the PSID family files, and
  ``importorskip('populace.fit')`` because the candidate-11 generation needs
  it) that reruns the seed-0 fit+generate+measure through the build machinery
  and pins the committed seed-0 numbers to float precision. Run live in the
  dedicated gate venv before the artifact is committed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "reform_delta_diagnostic_v1.json"
DOWNSTREAM_ARTIFACT = ROOT / "runs" / "downstream_relevance_c7_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)

REFORMS = ("reform_a", "reform_b")
SCALAR_METRICS = ("mean_delta", "median_delta", "winners_share")
COMPARED_DECILES = ("d3", "d4", "d5", "d6", "d7", "d8", "d9")
REPORTED_ONLY_DECILES = ("d1", "d2", "d10")
SEEDS = [0, 1, 2, 3, 4]


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import reform_delta_diagnostic as builder

    return builder


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


# --------------------------------------------------------------------------
# Schema and reported-not-gated framing (always runnable)
# --------------------------------------------------------------------------
def test_schema_and_reported_not_gated():
    art = _artifact()
    assert art["schema_version"] == "reform_delta_diagnostic.v1"
    assert art["run"] == "reform_delta_diagnostic_v1"
    assert art["reported_not_gated"] is True
    assert isinstance(art["purpose"], str) and art["purpose"].strip()
    assert "changes no gate" in art["purpose"]
    # Registration pointer to the frozen spec (#42 comment 4906177609).
    assert art["registration_pointer"] == "#42 comment 4906177609"
    assert art["registration"].endswith("4906177609")
    # Proxy-not-full-415(b) disclaimer present.
    assert "STATUTE-SHAPED PROXY" in art["not_full_415b"]
    assert "415(b)" in art["not_full_415b"]


def test_within_floor_tol_is_small_positive_guard():
    """within_floor is a descriptive flag with a tiny 0-vs-0 fp guard, not a
    verdict; the tolerance is far below any real gap."""
    art = _artifact()
    tol = art["within_floor_abs_tol"]
    assert 0.0 < tol <= 1e-6
    assert "not a verdict" in art["within_floor_note"]


def test_disclaimer_carried_verbatim_from_downstream():
    """The 415(b) disclaimer is byte-identical to the committed downstream one."""
    art = _artifact()
    downstream = json.loads(DOWNSTREAM_ARTIFACT.read_text())
    assert art["not_full_415b"] == downstream["not_full_415b"]


def test_seeds_present_realgen_and_floor():
    art = _artifact()
    for name in REFORMS:
        block = art["reform_results"][name]
        assert [r["seed"] for r in block["per_seed"]] == SEEDS
        assert [r["seed"] for r in block["floor_per_seed"]] == SEEDS
        # Real-vs-gen rows are person-aligned (paired present); floor rows are
        # disjoint halves (no paired block).
        for r in block["per_seed"]:
            assert {"real", "generated", "paired"} <= set(r)
        for f in block["floor_per_seed"]:
            assert {"side_a", "side_b"} <= set(f)
            assert "paired" not in f


def test_revision_pins():
    art = _artifact()
    pins = art["revision_pins"]
    assert pins["pe_us_revision"]
    assert pins["sklearn_version"]
    assert pins["artifact_schema_version"] == "reform_delta_diagnostic.v1"
    gates = pins["gates_yaml"]
    assert gates["gate_1_locked"] is True
    # Amendment 2 (mean-based classifier gating) is ratified in the pinned gate.
    assert len(gates["amendments_ratified"]) >= 1
    assert "no_self_rescue" in gates["amendment_rules"]


# --------------------------------------------------------------------------
# Reform definitions match the pinned oracle's baseline params
# --------------------------------------------------------------------------
def test_reform_definitions_match_baseline_factors():
    """ReformA = (0.95, baseline f2, baseline f3); ReformB removes the cap."""
    art = _artifact()
    reforms = art["reforms"]
    base = reforms["baseline"]["pia_factors"]
    # Baseline first factor is the statutory 0.90 (the reform is 0.90 -> 0.95).
    assert base[0] == pytest.approx(0.90)
    ra = reforms["reform_a"]
    assert ra["baseline_pia_factors"] == base
    assert ra["pia_factors"][0] == pytest.approx(0.95)
    assert ra["pia_factors"][1] == pytest.approx(base[1])
    assert ra["pia_factors"][2] == pytest.approx(base[2])
    rb = reforms["reform_b"]
    assert "wage_base_for(year) -> inf" in rb["change"]
    assert rb["baseline_wage_base_2022"] > 0
    # The per-reform result block carries the same definition.
    assert art["reform_results"]["reform_a"]["definition"] == ra
    assert art["reform_results"]["reform_b"]["definition"] == rb


# --------------------------------------------------------------------------
# Pooled scalar metrics recompute from per-seed values
# --------------------------------------------------------------------------
def _signed(rows, side_hi, side_lo, key):
    out = []
    for r in rows:
        hi, lo = r[side_hi][key], r[side_lo][key]
        if hi is not None and lo is not None:
            out.append(hi - lo)
    return out


def test_scalar_pooled_recompute_from_per_seed():
    art = _artifact()
    tol = art["within_floor_abs_tol"]
    for name in REFORMS:
        block = art["reform_results"][name]
        rows, floor_rows = block["per_seed"], block["floor_per_seed"]
        pooled = block["pooled"]
        for key in SCALAR_METRICS:
            blk = pooled[key]
            # Side pools.
            assert blk["real_pooled"]["mean"] == pytest.approx(
                _mean([r["real"][key] for r in rows]), rel=1e-9, abs=1e-12
            )
            assert blk["generated_pooled"]["mean"] == pytest.approx(
                _mean([r["generated"][key] for r in rows]),
                rel=1e-9,
                abs=1e-12,
            )
            # Real-vs-gen gap: signed per-seed, abs summary, scale (= abs.mean).
            rg_signed = _signed(rows, "generated", "real", key)
            assert blk["realgen_gap"]["per_seed_signed"] == pytest.approx(
                rg_signed, rel=1e-9, abs=1e-12
            )
            assert blk["realgen_gap"]["abs"]["values"] == pytest.approx(
                [abs(v) for v in rg_signed], rel=1e-9, abs=1e-12
            )
            assert blk["realgen_scale"] == pytest.approx(
                _mean([abs(v) for v in rg_signed]), rel=1e-9, abs=1e-12
            )
            # Floor gap: side_a - side_b.
            fl_signed = _signed(floor_rows, "side_a", "side_b", key)
            assert blk["floor_gap"]["per_seed_signed"] == pytest.approx(
                fl_signed, rel=1e-9, abs=1e-12
            )
            assert blk["floor_scale"] == pytest.approx(
                _mean([abs(v) for v in fl_signed]), rel=1e-9, abs=1e-12
            )
            assert blk["within_floor"] == (
                blk["realgen_scale"] <= blk["floor_scale"] + tol
            )


def test_q0_pooled_recompute_pooled_signed():
    """Q0 metric uses the pooled-signed scale |across-seed mean of the gap|."""
    art = _artifact()
    tol = art["within_floor_abs_tol"]
    for name in REFORMS:
        block = art["reform_results"][name]
        rows, floor_rows = block["per_seed"], block["floor_per_seed"]
        blk = block["pooled"]["q0_mean_delta"]
        # Per-seed signed real/gen values for both sides recorded.
        rg_signed = _signed(rows, "generated", "real", "q0_mean_delta")
        assert blk["realgen_gap"]["per_seed_signed"] == pytest.approx(
            rg_signed, rel=1e-9, abs=1e-12
        )
        # Pooled scale is |mean_s(signed)|, NOT mean_s(|signed|).
        assert blk["realgen_scale"] == pytest.approx(
            abs(_mean(rg_signed)), rel=1e-9, abs=1e-12
        )
        fl_signed = _signed(floor_rows, "side_a", "side_b", "q0_mean_delta")
        assert blk["floor_scale"] == pytest.approx(
            abs(_mean(fl_signed)), rel=1e-9, abs=1e-12
        )
        assert blk["within_floor"] == (
            blk["realgen_scale"] <= blk["floor_scale"] + tol
        )
        # Both sides' pooled across-seed means are recorded (per the spec).
        assert blk["real_pooled"]["mean"] == pytest.approx(
            _mean([r["real"]["q0_mean_delta"] for r in rows]),
            rel=1e-9,
            abs=1e-12,
        )
        assert blk["generated_pooled"]["mean"] == pytest.approx(
            _mean([r["generated"]["q0_mean_delta"] for r in rows]),
            rel=1e-9,
            abs=1e-12,
        )


def test_decile_incidence_recompute_and_convention():
    """Per-decile incidence gaps recompute; d3-d9 compared, d1/d2/d10 reported."""
    art = _artifact()
    tol = art["within_floor_abs_tol"]
    for name in REFORMS:
        block = art["reform_results"][name]
        rows, floor_rows = block["per_seed"], block["floor_per_seed"]
        dec = block["pooled"]["decile_incidence"]
        assert dec["compared_deciles"] == list(COMPARED_DECILES)
        assert dec["reported_only_deciles"] == list(REPORTED_ONLY_DECILES)
        per_decile = dec["per_decile"]
        assert set(per_decile) == {f"d{k}" for k in range(1, 11)}
        for dkey, cell in per_decile.items():
            rg = [
                r["generated"]["decile_mean_delta"][dkey]
                - r["real"]["decile_mean_delta"][dkey]
                for r in rows
                if r["generated"]["decile_mean_delta"][dkey] is not None
                and r["real"]["decile_mean_delta"][dkey] is not None
            ]
            assert cell["realgen_scale"] == pytest.approx(
                _mean([abs(v) for v in rg]), rel=1e-9, abs=1e-12
            )
            fl = [
                f["side_a"]["decile_mean_delta"][dkey]
                - f["side_b"]["decile_mean_delta"][dkey]
                for f in floor_rows
                if f["side_a"]["decile_mean_delta"][dkey] is not None
                and f["side_b"]["decile_mean_delta"][dkey] is not None
            ]
            assert cell["floor_scale"] == pytest.approx(
                _mean([abs(v) for v in fl]), rel=1e-9, abs=1e-12
            )
            assert cell["within_floor"] == (
                cell["realgen_scale"] <= cell["floor_scale"] + tol
            )
            assert cell["compared"] == (dkey in COMPARED_DECILES)
        # Headline: max over d3-d9.
        assert dec["max_realgen_scale_d3_d9"] == pytest.approx(
            max(per_decile[d]["realgen_scale"] for d in COMPARED_DECILES),
            rel=1e-9,
            abs=1e-12,
        )
        assert dec["max_floor_scale_d3_d9"] == pytest.approx(
            max(per_decile[d]["floor_scale"] for d in COMPARED_DECILES),
            rel=1e-9,
            abs=1e-12,
        )
        assert dec["within_floor_d3_d9"] == (
            dec["max_realgen_scale_d3_d9"]
            <= dec["max_floor_scale_d3_d9"] + tol
        )


def test_floor_is_disjoint_half_split_scale():
    """Floor halves are ~20%-of-persons disjoint real samples (ctx20 scale)."""
    art = _artifact()
    n_persons = art["n_persons"]
    assert n_persons > 20000  # the FULL filtered panel, not a train split
    for name in REFORMS:
        for f in art["reform_results"][name]["floor_per_seed"]:
            for side in ("side_a", "side_b"):
                assert (
                    0.15 * n_persons < f[side]["n_persons"] < 0.25 * n_persons
                )


def test_paired_stats_labeled_descriptive():
    """Metric 5 (paired) is present, pooled, and labelled descriptive-only."""
    art = _artifact()
    for name in REFORMS:
        block = art["reform_results"][name]
        pd_block = block["pooled"]["paired_descriptive"]
        assert "DESCRIPTIVE ONLY" in pd_block["note"]
        # Per-seed paired stats carry the descriptive note too.
        for r in block["per_seed"]:
            assert "DESCRIPTIVE ONLY" in r["paired"]["note"]
        # Pooled corr recomputes from the per-seed corrs (dropping None).
        corrs = [
            r["paired"]["corr"]
            for r in block["per_seed"]
            if r["paired"]["corr"] is not None
        ]
        assert pd_block["corr"]["mean"] == pytest.approx(
            _mean(corrs), rel=1e-9, abs=1e-12
        )
        mads = [
            r["paired"]["weighted_mean_abs_diff"] for r in block["per_seed"]
        ]
        assert pd_block["weighted_mean_abs_diff"]["mean"] == pytest.approx(
            _mean(mads), rel=1e-9, abs=1e-12
        )
        # There is no floor key on the paired block (no real-vs-real analogue).
        assert "floor_gap" not in pd_block


def test_deltas_nonnegative_and_shares_valid():
    """Both reforms weakly raise the PIA-proxy, so mean/median Delta >= 0;
    winners shares are valid probabilities."""
    art = _artifact()
    for name in REFORMS:
        block = art["reform_results"][name]
        for r in block["per_seed"]:
            for side in ("real", "generated"):
                assert r[side]["mean_delta"] >= -1e-9
                assert r[side]["median_delta"] >= -1e-9
                assert 0.0 <= r[side]["winners_share"] <= 1.0
        for f in block["floor_per_seed"]:
            for side in ("side_a", "side_b"):
                assert f[side]["mean_delta"] >= -1e-9
                assert 0.0 <= f[side]["winners_share"] <= 1.0


def test_reform_incidence_shapes():
    """Reform A is bottom-loaded, Reform B top-loaded, in the real deciles.

    A raises the first PIA bracket, so its incidence is broad and does NOT
    grow toward the top; B removes the cap, so its incidence concentrates at
    the top decile. Checked on the pooled real-side decile means.
    """
    art = _artifact()

    def real_decile_mean(name, dkey):
        rows = art["reform_results"][name]["per_seed"]
        vals = [
            r["real"]["decile_mean_delta"][dkey]
            for r in rows
            if r["real"]["decile_mean_delta"][dkey] is not None
        ]
        return _mean(vals)

    # Reform B: top decile mean Delta strictly exceeds the middle decile.
    assert real_decile_mean("reform_b", "d10") > real_decile_mean(
        "reform_b", "d5"
    )
    # Reform B concentrates at the top far more than Reform A does.
    b_top_ratio = real_decile_mean("reform_b", "d10") / max(
        real_decile_mean("reform_b", "d5"), 1e-9
    )
    a_top_ratio = real_decile_mean("reform_a", "d10") / max(
        real_decile_mean("reform_a", "d5"), 1e-9
    )
    assert b_top_ratio > a_top_ratio


# --------------------------------------------------------------------------
# Pure-helper unit tests (import the builder; no PSID, no populace-fit)
# --------------------------------------------------------------------------
def test_reform_wrappers_delegate_and_override():
    builder = _import_builder()

    class FakeBase:
        nawi = {y: 1.0 for y in range(1998, 2023, 2)}
        pia_factors = (0.9, 0.32, 0.15)
        pe_us_revision = "fake"

        def wage_base_for(self, year):
            return 50_000.0

        def bend_points(self, year):
            return (1024.0, 6172.0)

    base = FakeBase()
    a = builder.ReformA(base)
    assert a.pia_factors == (0.95, 0.32, 0.15)  # f2/f3 delegated
    assert a.wage_base_for(2000) == 50_000.0  # delegated
    assert a.bend_points(2000) == (1024.0, 6172.0)  # delegated
    b = builder.ReformB(base)
    assert b.wage_base_for(2000) == float("inf")  # overridden
    assert b.pia_factors == (0.9, 0.32, 0.15)  # delegated
    assert b.bend_points(2000) == (1024.0, 6172.0)  # delegated
    assert b.pe_us_revision == "fake"  # arbitrary attribute delegated


def test_weighted_decile_groups_partition():
    builder = _import_builder()
    vals = np.arange(100, dtype=float)
    w = np.ones(100)
    g = builder._weighted_decile_groups(vals, w)
    assert set(g.tolist()) == set(range(10))
    assert np.bincount(g, minlength=10).tolist() == [10] * 10
    # A 30% mass at zero fills the bottom three deciles.
    vals2 = np.concatenate([np.zeros(30), np.arange(1, 71, dtype=float)])
    g2 = builder._weighted_decile_groups(vals2, np.ones(100))
    assert set(g2[:30].tolist()) <= {0, 1, 2}


def test_weighted_corr_closed_form():
    builder = _import_builder()
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert builder._weighted_corr(x, 2 * x + 1, np.ones(5)) == pytest.approx(
        1.0, abs=1e-12
    )
    assert builder._weighted_corr(x, -x, np.ones(5)) == pytest.approx(
        -1.0, abs=1e-12
    )
    assert builder._weighted_corr(x, np.ones(5), np.ones(5)) is None


def test_summary_matches_numpy():
    builder = _import_builder()
    s = builder._summary([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == pytest.approx(2.5)
    assert s["sd"] == pytest.approx(float(np.std([1, 2, 3, 4], ddof=1)))
    assert s["n_seeds"] == 4
    assert builder._summary([])["n_seeds"] == 0


def test_panel_deltas_and_metrics_closed_form():
    """panel_deltas + delta_metrics on a hand-built panel with fake params.

    Reform A (0.90->0.95) bumps every positive-baseline person by 5% of their
    first-bracket AIME; Reform B (cap removed) bumps only above-cap earners.
    Q0 (zero-anchor) persons are exactly the zero-last-period persons.
    """
    import pandas as pd

    builder = _import_builder()

    class FakeBase:
        nawi = {y: 1.0 for y in range(1998, 2023, 2)}
        pia_factors = (0.9, 0.32, 0.15)
        pe_us_revision = "fake"

        def wage_base_for(self, year):
            return 50_000.0

        def bend_points(self, year):
            return (1024.0, 6172.0)

    base = FakeBase()
    rows = []
    # p1: all zero (baseline 0 -> delta 0 under both reforms; Q0 anchor).
    # p2: below cap (Reform B no effect).
    # p3: above cap (Reform B raises).
    for pid, earns in [
        (1, [0, 0, 0]),
        (2, [30_000, 30_000, 30_000]),
        (3, [80_000, 80_000, 80_000]),
    ]:
        for per, e in zip([2018, 2020, 2022], earns, strict=True):
            rows.append(
                dict(
                    person_id=pid,
                    period=per,
                    earnings=float(e),
                    age=40,
                    weight=1.0,
                )
            )
    panel = pd.DataFrame(rows)
    anchor = builder.anchor_rows(panel)
    cuts = builder.anchor_quintile_cutpoints(anchor)

    df_a = builder.panel_deltas(panel, anchor, base, builder.ReformA(base))
    df_b = builder.panel_deltas(panel, anchor, base, builder.ReformB(base))
    da = dict(zip(df_a.person_id, df_a.delta, strict=True))
    db = dict(zip(df_b.person_id, df_b.delta, strict=True))
    assert da[1] == pytest.approx(0.0)  # zero baseline -> zero delta
    assert da[2] > 0.0 and da[3] > 0.0  # A bumps positive earners
    assert db[1] == pytest.approx(0.0)
    assert db[2] == pytest.approx(0.0)  # below cap -> B no effect
    assert db[3] > 0.0  # above cap -> B raises

    m_b = builder.delta_metrics(df_b, anchor, cuts)
    # Only p3 is a winner under Reform B (weighted share 1/3).
    assert m_b["winners_share"] == pytest.approx(1.0 / 3.0)
    # Q0 subgroup is p1 (zero anchor) -> mean delta 0.
    assert m_b["q0_mean_delta"] == pytest.approx(0.0)
    assert m_b["n_q0_persons"] == 1


# --------------------------------------------------------------------------
# Seed-0 reproduction pin (needs PSID + populace-fit; run in the gate venv)
# --------------------------------------------------------------------------
@needs_real_family
def test_seed0_reproduces_committed_artifact():
    """Rerun the seed-0 fit+generate+measure; match to float precision."""
    pytest.importorskip(
        "populace.fit",
        reason="populace-fit not installed (gate runs use a dedicated venv)",
    )
    builder = _import_builder()
    art = _artifact()

    params = builder.load_ssa_parameters()
    if params.pe_us_revision != art["revision_pins"]["pe_us_revision"]:
        pytest.skip(
            f"policyengine-us checkout at {params.pe_us_revision} differs "
            f"from the artifact's pinned "
            f"{art['revision_pins']['pe_us_revision']}; point "
            "POPULACE_DYNAMICS_PE_US_DIR at the pinned revision to run the "
            "reproduction"
        )
    # Reform definitions really do match the loaded oracle's baseline factors.
    base_factors = list(params.pia_factors)
    assert art["reforms"]["baseline"]["pia_factors"] == pytest.approx(
        base_factors
    )
    assert art["reforms"]["reform_a"]["pia_factors"] == pytest.approx(
        [0.95, base_factors[1], base_factors[2]]
    )

    panel = builder.load_filtered_panel()
    all_anchor = builder.anchor_rows(panel)
    cutpoints = builder.anchor_quintile_cutpoints(all_anchor)
    holdout, candidate = builder.fit_and_generate_candidate11(
        0, panel, all_anchor
    )

    for name, factory in builder.REFORMS.items():
        reform_params = factory(params)
        got_rg = builder.measure_realgen_seed(
            reform_params,
            params,
            all_anchor,
            cutpoints,
            holdout,
            candidate,
        )
        got_fl = builder.measure_floor_seed(
            0, reform_params, params, panel, all_anchor, cutpoints
        )
        block = art["reform_results"][name]
        ref_rg = next(r for r in block["per_seed"] if r["seed"] == 0)
        ref_fl = next(f for f in block["floor_per_seed"] if f["seed"] == 0)

        for side in ("real", "generated"):
            _assert_side(got_rg[side], ref_rg[side])
        for side in ("side_a", "side_b"):
            _assert_side(got_fl[side], ref_fl[side])

        assert got_rg["paired"]["corr"] == pytest.approx(
            ref_rg["paired"]["corr"], abs=1e-9
        )
        assert got_rg["paired"]["weighted_mean_abs_diff"] == pytest.approx(
            ref_rg["paired"]["weighted_mean_abs_diff"], abs=1e-9
        )
        assert got_rg["paired"]["n_persons"] == ref_rg["paired"]["n_persons"]


def _assert_side(got: dict, ref: dict) -> None:
    for key in (
        "mean_delta",
        "median_delta",
        "mean_baseline",
        "winners_share",
    ):
        assert got[key] == pytest.approx(ref[key], abs=1e-9), key
    if ref["q0_mean_delta"] is None:
        assert got["q0_mean_delta"] is None
    else:
        assert got["q0_mean_delta"] == pytest.approx(
            ref["q0_mean_delta"], abs=1e-9
        )
    assert got["n_persons"] == ref["n_persons"]
    assert got["n_q0_persons"] == ref["n_q0_persons"]
    for dkey, val in ref["decile_mean_delta"].items():
        if val is None:
            assert got["decile_mean_delta"][dkey] is None, dkey
        else:
            assert got["decile_mean_delta"][dkey] == pytest.approx(
                val, abs=1e-9
            ), dkey
