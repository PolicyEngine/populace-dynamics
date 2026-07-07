"""Tests for the Phase-A PI/PPI replication vs Mermin (2005).

Artifact: runs/replication_ppi_mermin_v1.json. Frozen spec: issue #42
comment 4907444903 (with the coordinator-directed generator-domain
deviation the artifact carries in ``deviation_from_registration``).
REPORTED, NOT GATED.

Two tiers, mirroring the other run/floor tests:

* Always-runnable internal-consistency tests (no PSID, no populace-fit)
  that touch only the committed artifact plus the pure helpers in
  :mod:`scripts.replication_ppi_mermin`: the schema is sane and marked
  reported-not-gated; the registration pointer, the deviation block, and
  the anchor provenance (Mermin's transcribed target rows with page
  citations) are present; the PI incidence ratio is quintile-invariant to
  1e-6 within every side (the encoding check -- PI scales all factors by
  the same wedge, so the ratio is the wedge for every career); the PPI
  gradient is monotone; and the pooled three-way table, the PI scalars,
  the floors, and the directional verdict all RECOMPUTE from the stored
  per-seed values. Pure helpers reproduce closed-form scheduled/PI/PPI
  amounts, the wedge, and the coverage selection.
* A seed-0 reproduction pin (skipped without the PSID family files, and
  ``importorskip('populace.fit')`` because candidate-11 generation needs
  it) that reruns the seed-0 fit+generate+measure through the build
  machinery and pins the committed seed-0 quintile ratios to float
  precision. Run live in the dedicated gate venv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "replication_ppi_mermin_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)

SEEDS = [0, 1, 2, 3, 4]
N_QUINTILES = 5
DYNASIM_PPI = (98.7, 90.4, 81.3, 75.7, 71.7)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import replication_ppi_mermin as builder

    return builder


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _ppi(side: dict) -> list[float]:
    """The five quintile PPI ratios of one side (in quintile order)."""
    return [q["ppi_ratio_pct"] for q in side["quintiles"]]


def _pi(side: dict) -> list[float]:
    return [q["pi_ratio_pct"] for q in side["quintiles"]]


# =====================================================================
# Schema and reported-not-gated framing (always runnable)
# =====================================================================
def test_schema_and_reported_not_gated():
    art = _artifact()
    assert art["schema_version"] == "replication_ppi_mermin.v1"
    assert art["run"] == "replication_ppi_mermin_v1"
    assert art["reported_not_gated"] is True
    assert "changes no gate" in art["purpose"]
    assert art["registration_pointer"] == "#42 comment 4907444903"
    assert art["registration"].endswith("4907444903")
    assert "issues/74" in art["program_design_issue"]


def test_deviation_block_present_and_explicit():
    """The coordinator-directed support restriction is carried prominently."""
    art = _artifact()
    dev = art["deviation_from_registration"]
    assert isinstance(dev, str) and dev.strip()
    # Names the restriction, its symmetry, and the DYNASIM-side delta.
    assert "common gate" in dev.lower() or "gate-filtered" in dev.lower()
    assert "symmetric" in dev.lower()
    assert "truncated" in dev.lower()


def test_anchor_provenance_rows_and_citations():
    """Mermin's transcribed target rows and page citations are present."""
    art = _artifact()
    prov = art["anchor_provenance"]
    t2 = prov["table2_retired_workers_62_67_in_2050"]
    assert tuple(t2["progressive_price_indexing_pct"]["by_quintile"]) == (
        DYNASIM_PPI
    )
    assert t2["price_indexing_pct"]["all"] == 67.8
    assert t2["scheduled_mean_2005usd"]["by_quintile"] == [
        9200,
        13900,
        18100,
        21900,
        25600,
    ]
    # Page citations present on each mechanic and on the tables.
    assert "p.4" in prov["pi_mechanics"]["citation"]
    assert "30th percentile" in prov["ppi_mechanics"]["quote"]
    assert "p.16" in t2["citation"]
    assert (
        prov["table1_75yr_payroll_effect_pct"]["values"][
            "progressive_price_indexing"
        ]
        == -0.14
    )
    # Wedge source: paper states no number; 2005 TR 1.1pp fallback, not tuned.
    ws = prov["wedge_source"]
    assert ws["paper_states_number"] is False
    assert "1.1" in ws["fallback"]
    assert "not tuned" in ws["fallback"].lower()
    # The population/quintile delta (spouse-shared vs own-record) is named.
    assert (
        "SHARED" in prov["population_and_quintile"]["quintile_variable_paper"]
    )
    assert any(
        "spouse-shared" in d or "own-record" in d
        for d in prov["named_population_deltas"]
    )


def test_transport_and_conventions():
    art = _artifact()
    tc = art["transport_and_conventions"]
    assert tc["eligibility_year"] == 2050
    assert tc["index_year"] == 2048
    assert 0.60 < tc["wedge"] < 0.72
    assert tc["wedge_formula"] == "(1.028/1.039)**(2050-2012)"
    b1, b2 = tc["bend_points_2050"]
    assert 0 < b1 < b2
    assert "415(g)" in tc["ratio_415g_note"]
    assert "30th percentile" in tc["ppi_bend"]


def test_revision_pins():
    art = _artifact()
    pins = art["revision_pins"]
    assert pins["pe_us_revision"]
    assert pins["sklearn_version"]
    assert pins["artifact_schema_version"] == "replication_ppi_mermin.v1"
    assert pins["gates_yaml"]["gate_1_locked"] is True


def test_seeds_present_realgen_and_floor():
    art = _artifact()
    assert [r["seed"] for r in art["per_seed"]] == SEEDS
    for r in art["per_seed"]:
        assert {"real", "generated", "floor"} <= set(r)
        assert {"side_a", "side_b"} <= set(r["floor"])
        for side in (r["real"], r["generated"]):
            assert len(side["quintiles"]) == N_QUINTILES


# =====================================================================
# PI encoding check: quintile-invariant ratio == the wedge, by construction
# =====================================================================
def test_pi_ratio_quintile_invariant_equals_wedge():
    """Within every side (real, generated, floor halves) and seed, the five
    PI quintile ratios are equal to 1e-6 and equal 100*wedge -- the check
    that validates the PI encoding (all factors scaled by one wedge)."""
    art = _artifact()
    wedge_pct = 100.0 * art["transport_and_conventions"]["wedge"]
    for r in art["per_seed"]:
        sides = [
            r["real"],
            r["generated"],
            r["floor"]["side_a"],
            r["floor"]["side_b"],
        ]
        for side in sides:
            pis = [
                q["pi_ratio_pct"]
                for q in side["quintiles"]
                if q.get("n_positive", 0) > 0
            ]
            assert max(pis) - min(pis) < 1e-6, pis
            for v in pis:
                assert v == pytest.approx(wedge_pct, abs=1e-6)
            assert side["overall_pi_ratio_pct"] == pytest.approx(
                wedge_pct, abs=1e-6
            )
    # The pooled PI scalar equals the wedge too.
    pi = art["three_way_comparison"]["pi_scalars"]
    assert pi["wedge_implied_scalar_pct"] == pytest.approx(wedge_pct, abs=1e-9)
    assert pi["real_pooled_mean_pct"]["mean"] == pytest.approx(
        wedge_pct, abs=1e-6
    )
    assert pi["generated_pooled_mean_pct"]["mean"] == pytest.approx(
        wedge_pct, abs=1e-6
    )
    assert pi["dynasim_pct"] == 67.8


def test_ppi_gradient_monotone_and_bounded():
    """PPI is progressive: the ratio is non-increasing across quintiles
    (Q1 highest, protected near 100%; Q5 lowest, toward the wedge)."""
    art = _artifact()
    wedge_pct = 100.0 * art["transport_and_conventions"]["wedge"]
    for r in art["per_seed"]:
        for side in (r["real"], r["generated"]):
            ppis = _ppi(side)
            assert all(
                ppis[k] >= ppis[k + 1] - 1e-6 for k in range(N_QUINTILES - 1)
            ), ppis
            assert ppis[0] == pytest.approx(100.0, abs=0.5)  # Q1 protected
            # Every PPI ratio lies between the wedge and 100% (a PPI career
            # is cut at most as much as full price indexing).
            for v in ppis:
                assert wedge_pct - 1e-6 <= v <= 100.0 + 1e-6


# =====================================================================
# Pooled three-way table + floors recompute from per-seed values
# =====================================================================
def test_ppi_three_way_pooled_recompute():
    art = _artifact()
    rows = art["per_seed"]
    table = art["three_way_comparison"]["ppi_by_quintile"]
    for k in range(N_QUINTILES):
        cell = table[k]
        assert cell["quintile"] == k + 1
        assert cell["dynasim_pct"] == DYNASIM_PPI[k]
        real_vals = [_ppi(r["real"])[k] for r in rows]
        gen_vals = [_ppi(r["generated"])[k] for r in rows]
        assert cell["real_pooled"]["mean"] == pytest.approx(
            _mean(real_vals), rel=1e-9, abs=1e-12
        )
        assert cell["real_pooled"]["values"] == pytest.approx(
            real_vals, rel=1e-9, abs=1e-12
        )
        assert cell["generated_pooled"]["mean"] == pytest.approx(
            _mean(gen_vals), rel=1e-9, abs=1e-12
        )
        # Real-vs-generated gap scale = mean of per-seed |gen - real|.
        rg = [g - r for g, r in zip(gen_vals, real_vals, strict=True)]
        assert cell["realgen_gap"]["per_seed_signed"] == pytest.approx(
            rg, rel=1e-9, abs=1e-12
        )
        assert cell["realgen_scale"] == pytest.approx(
            _mean([abs(v) for v in rg]), rel=1e-9, abs=1e-12
        )
        # Floor scale = mean of per-seed |side_a - side_b|.
        fa = [_ppi(r["floor"]["side_a"])[k] for r in rows]
        fb = [_ppi(r["floor"]["side_b"])[k] for r in rows]
        fg = [a - b for a, b in zip(fa, fb, strict=True)]
        assert cell["floor_scale"] == pytest.approx(
            _mean([abs(v) for v in fg]), rel=1e-9, abs=1e-12
        )
        assert cell["gap_exceeds_floor"] == (
            cell["realgen_scale"] > cell["floor_scale"]
        )
        assert cell["generated_cut_smaller_than_real"] == (_mean(rg) > 0.0)


def test_directional_prediction_recompute():
    """The directional verdict recomputes from the table (registration:
    generated Q5 cut smaller than real, gap > floor at Q5 only)."""
    art = _artifact()
    table = art["three_way_comparison"]["ppi_by_quintile"]
    d = art["three_way_comparison"]["directional_prediction"]
    q5 = table[4]
    assert (
        d["q5_generated_cut_smaller"] == q5["generated_cut_smaller_than_real"]
    )
    assert d["q5_gap_exceeds_floor"] == q5["gap_exceeds_floor"]
    assert d["q1_q3_gaps_within_floor"] == all(
        not table[k]["gap_exceeds_floor"] for k in (0, 1, 2)
    )
    assert d["prediction_held"] == (
        d["q5_generated_cut_smaller"]
        and d["q5_gap_exceeds_floor"]
        and d["q1_q3_gaps_within_floor"]
    )


def test_floor_is_disjoint_half_split_scale():
    """Floor halves are disjoint ~20%-of-selected real samples (ctx20)."""
    art = _artifact()
    n_selected = art["person_selection"]["n_selected"]
    assert n_selected > 1000
    for r in art["per_seed"]:
        for side in ("side_a", "side_b"):
            n = r["floor"][f"n_persons_{side}"]
            assert 0.12 * n_selected < n < 0.28 * n_selected


def test_quintile_cells_internally_consistent():
    """Every quintile cell: n_positive <= n_persons, AIME increases across
    quintiles (own-distribution ordering), scheduled amount positive."""
    art = _artifact()
    for r in art["per_seed"]:
        for side in (r["real"], r["generated"]):
            means = []
            for q in side["quintiles"]:
                if q.get("n_positive", 0) == 0:
                    continue
                assert q["n_positive"] <= q["n_persons"]
                assert q["mean_scheduled_amount"] > 0
                means.append(q["mean_aime"])
            # AIME strictly increases from Q1 to Q5 (quintiles of AIME).
            assert all(
                means[i] < means[i + 1] for i in range(len(means) - 1)
            ), means


# =====================================================================
# Pure-helper unit tests (import the builder; no PSID, no populace-fit)
# =====================================================================
def test_build_transport_wedge_and_bends():
    builder = _import_builder()

    class FakeParams:
        pia_factors = (0.9, 0.32, 0.15)
        pe_us_revision = "fake"
        nawi = {1977: 10_000.0, **{y: 50_000.0 for y in range(2020, 2036)}}

        def wage_base_for(self, year):
            return 100_000.0

    t = builder.build_transport(FakeParams())
    # Wedge = (1.028/1.039)**(2050-2012), not tuned.
    expected_w = (1.028 / 1.039) ** 38
    assert t["wedge"] == pytest.approx(expected_w, rel=1e-12)
    assert t["index_year"] == 2048
    assert t["eligibility_year"] == 2050
    # NAWI projected from 2035 at 3.9% to 2048; bends derived per 415(a)(1)(B).
    nawi_2048 = 50_000.0 * 1.039 ** (2048 - 2035)
    assert t["index_nawi"] == pytest.approx(nawi_2048, rel=1e-9)
    assert t["bend_points"][0] == pytest.approx(
        round(180.0 * nawi_2048 / 10_000.0)
    )
    assert t["bend_points"][1] == pytest.approx(
        round(1085.0 * nawi_2048 / 10_000.0)
    )


def test_scheduled_pi_ppi_closed_form():
    builder = _import_builder()
    transport = {
        "bend_points": (1000.0, 6000.0),
        "pia_factors": (0.9, 0.32, 0.15),
        "wedge": 0.67,
    }
    aime = np.array([500.0, 2000.0, 7000.0])
    sched = builder.scheduled_amount(aime, transport)
    assert sched[0] == pytest.approx(0.9 * 500)
    assert sched[1] == pytest.approx(0.9 * 1000 + 0.32 * 1000)
    assert sched[2] == pytest.approx(0.9 * 1000 + 0.32 * 5000 + 0.15 * 1000)
    # PI = wedge * scheduled for EVERY aime (=> ratio is the wedge, flat).
    pi = builder.price_indexed_amount(aime, transport)
    assert np.allclose(pi, 0.67 * sched)
    assert np.allclose(pi / sched, 0.67)
    # PPI with a bend at 2000: below unchanged, above scaled, continuous.
    ppi = builder.progressive_price_indexed_amount(aime, 2000.0, transport)
    assert ppi[0] == pytest.approx(sched[0])  # 500 < 2000 -> unchanged
    sched_bend = 0.9 * 1000 + 0.32 * 1000  # scheduled at aime=2000
    assert ppi[2] == pytest.approx(sched_bend + 0.67 * (sched[2] - sched_bend))
    # Continuity at the bend: PPI(2000) == scheduled(2000).
    at_bend = builder.progressive_price_indexed_amount(
        np.array([2000.0]), 2000.0, transport
    )
    assert at_bend[0] == pytest.approx(sched[1])


def test_coverage_selection_closed_form():
    import pandas as pd

    builder = _import_builder()
    rows = []
    # p1: 8 consecutive positive biennial obs -> selected (cov 1.0, n_pos 8).
    for per in range(1998, 2014, 2):
        rows.append(dict(person_id=1, period=per, earnings=40_000.0))
    # p2: 2 positive obs -> too few (n_pos 2 < 8).
    for per in (1998, 2000):
        rows.append(dict(person_id=2, period=per, earnings=40_000.0))
    # p3: 9-slot span with 8 positives (one zero) -> cov 8/9 >= 0.8, selected.
    for i, per in enumerate(range(1998, 2016, 2)):
        rows.append(
            dict(person_id=3, period=per, earnings=0.0 if i == 4 else 40_000.0)
        )
    # p4: 8 positives but spread over a 12-slot span (gaps) -> cov 8/12 < 0.8.
    for per in list(range(1998, 2012, 2)) + [2020]:
        rows.append(dict(person_id=4, period=per, earnings=40_000.0))
    panel = pd.DataFrame(rows)
    selected = builder.coverage_selected_persons(panel)
    assert selected == {1, 3}


def test_summary_empty_safe():
    builder = _import_builder()
    s = builder._summary([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == pytest.approx(2.5)
    assert s["n_seeds"] == 4
    assert builder._summary([])["n_seeds"] == 0  # empty-safe


# =====================================================================
# Seed-0 reproduction pin (needs PSID + populace-fit; run in the gate venv)
# =====================================================================
@needs_real_family
def test_seed0_reproduces_committed_artifact():
    """Rerun the seed-0 fit+generate+measure; match the committed seed-0
    quintile ratios and floors to float precision."""
    pytest.importorskip(
        "populace.fit",
        reason="populace-fit not installed (gate runs use a dedicated venv)",
    )
    builder = _import_builder()
    art = _artifact()

    params = builder.load_ssa_parameters()
    if params.pe_us_revision != art["revision_pins"]["pe_us_revision"]:
        pytest.skip(
            f"policyengine-us at {params.pe_us_revision} differs from the "
            f"artifact's pinned {art['revision_pins']['pe_us_revision']}"
        )
    transport = builder.build_transport(params)
    panel = builder.load_filtered_panel()
    all_anchor = builder.anchor_rows(panel)
    weight_of = dict(
        zip(all_anchor["person_id"], all_anchor["weight"], strict=True)
    )
    selected = builder.coverage_selected_persons(panel)
    selected_panel = panel[panel["person_id"].isin(selected)].reset_index(
        drop=True
    )

    holdout, candidate = builder.fit_and_generate_candidate11(
        0, panel, all_anchor
    )
    got_rg = builder.measure_realgen_seed(
        holdout, candidate, selected, weight_of, params, transport
    )
    got_fl = builder.measure_floor_seed(
        0, selected_panel, weight_of, params, transport
    )
    ref = next(r for r in art["per_seed"] if r["seed"] == 0)

    for side in ("real", "generated"):
        _assert_side(got_rg[side], ref[side])
    for side in ("side_a", "side_b"):
        _assert_side(got_fl[side], ref["floor"][side])


def _assert_side(got: dict, ref: dict) -> None:
    assert got["n_persons"] == ref["n_persons"]
    assert got["bend30_aime"] == pytest.approx(ref["bend30_aime"], rel=1e-9)
    for gq, rq in zip(got["quintiles"], ref["quintiles"], strict=True):
        assert gq["n_positive"] == rq["n_positive"]
        if rq.get("n_positive", 0) == 0:
            continue
        assert gq["pi_ratio_pct"] == pytest.approx(
            rq["pi_ratio_pct"], abs=1e-9
        )
        assert gq["ppi_ratio_pct"] == pytest.approx(
            rq["ppi_ratio_pct"], abs=1e-9
        )
        assert gq["mean_aime"] == pytest.approx(rq["mean_aime"], rel=1e-9)
