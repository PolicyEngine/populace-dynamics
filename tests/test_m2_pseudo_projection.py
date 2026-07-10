"""Tests for the M2 same-frame pseudo-projection.

Artifact: runs/m2_pseudo_projection_v1.json. Frozen spec: issue #42
comment 4931333382. REPORTED, NOT GATED.

Two tiers, mirroring the other run/floor tests:

* Always-runnable artifact-consistency tests (no PSID) that touch only the
  committed artifact plus the pure module constants: the schema is marked
  reported-not-gated with the registration pointer; the Smith anchor
  deltas and 2034 baseline equal the committed transcription AND an
  independent re-transcription; the reserve is calibrated so the baseline
  exhausts in 2034; F1 (signs), F2 (revenue-side exhaustion ordering + the
  Smith Kendall tau), F3 (FRA->72 ranks above cap-$150k and +1pp), and F4
  (outlay-side PI > NRA > PPI > COLA persistence) all RECOMPUTE from the
  stored per-provision rows and equal the stored results-vs-forecasts
  block; the floors are present with the quartet's PI floor exactly zero.
* A skipif-PSID reproduction pin that rebuilds the frame + ledger and pins
  the elimination exhaustion delta (the F2-ordering driver) and the F4
  quartet order to the committed artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "m2_pseudo_projection_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_psid = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir()
    or not (REAL_DATA / "mh85_23").is_dir()
    or not (REAL_DATA / "cah85_23").is_dir(),
    reason="PSID family/marriage/birth files not staged",
)

# Independently re-transcribed Smith (2015) solvency-year deltas + the
# 2034 baseline (verified against pdftotext of 72196-can-ss-be-solvent.pdf
# on 2026-07-09 per #74 protocol note 3), hardcoded so the test fails if
# the committed constants or the artifact drift from the source.
PDF_SMITH_YEAR_DELTAS = {
    "cap_150k": 1,  # Smith p.3 ("to 2035, an additional year")
    "elimination": 21,  # Smith p.3 ("through 2055, an additional 21 years")
    "payroll_plus_1pp": 5,  # Smith p.3 ("an additional 5 years")
    "payroll_plus_2pp": 18,  # Smith p.3 ("an additional 18 years")
}
PDF_SMITH_BASELINE_YEAR = 2034  # Smith p.1, p.2

REVENUE = (
    "cap_150k",
    "elimination",
    "payroll_plus_1pp",
    "payroll_plus_2pp",
)
SMITH = REVENUE + ("fra_to_72",)
QUARTET = (
    "price_indexing",
    "progressive_price_indexing",
    "nra_raised_to_70",
    "reduced_cola",
)
SMITH_REVENUE_ORDER = (
    "elimination",
    "payroll_plus_2pp",
    "payroll_plus_1pp",
    "cap_150k",
)
F4_TARGET_ORDER = (
    "price_indexing",
    "nra_raised_to_70",
    "progressive_price_indexing",
    "reduced_cola",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import m2_pseudo_projection as builder

    return builder


def _rows_by_name(art: dict) -> dict[str, dict]:
    return {r["provision"]: r for r in art["provisions"]}


# =====================================================================
# Schema + reported-not-gated framing (always runnable)
# =====================================================================
def test_schema_and_reported_not_gated():
    art = _artifact()
    assert art["schema_version"] == "m2_pseudo_projection.v1"
    assert art["run"] == "m2_pseudo_projection_v1"
    assert art["reported_not_gated"] is True
    assert "changes no gate" in art["purpose"]
    assert art["registration_pointer"] == "#42 comment 4931333382"
    assert art["registration"].endswith("4931333382")
    assert "issues/74" in art["program_design_issue"]
    assert "issues/113" in art["roadmap_issue"]
    assert "does not grade" in art["grading_note"]


def test_common_frame_and_named_deltas():
    art = _artifact()
    cf = art["common_frame"]
    assert cf["n_common_frame"] == 1549
    assert cf["n_common_frame"] == cf["n_careers_mermin_sex_resolvable"]
    deltas = art["named_deltas"]
    assert any("frame-relative levels" in d for d in deltas)
    assert any("calibrated (not derived) initial reserve" in d for d in deltas)
    assert any("compressed top tail" in d for d in deltas)
    assert any("no DI" in d for d in deltas)
    assert any("no phase-in" in d for d in deltas)


# =====================================================================
# Calibration: reserve set so the baseline exhausts in Smith's 2034
# =====================================================================
def test_calibration_to_smith_2034():
    art = _artifact()
    cal = art["calibration_disclosure"]
    assert cal["target_year"] == PDF_SMITH_BASELINE_YEAR
    assert cal["baseline_exhaustion_year"] == pytest.approx(2034.0, abs=1e-6)
    assert "2034" in cal["target_source"]
    assert cal["calibrated_reserve"] > 0.0
    assert "calibration anchors the level" in cal["note"]


# =====================================================================
# Smith anchor: committed constants + independent PDF transcription
# =====================================================================
def test_smith_anchor_values_match_committed_and_pdf():
    builder = _import_builder()
    rows = _rows_by_name(_artifact())
    for prov, pdf_delta in PDF_SMITH_YEAR_DELTAS.items():
        assert builder.SMITH_SOLVENCY_YEAR_DELTAS[prov] == pdf_delta
        assert rows[prov]["anchor_smith_year_delta"] == pdf_delta
    # FRA->72 is "<1 year" (recorded as a representative sub-one value).
    assert builder.SMITH_SOLVENCY_YEAR_DELTAS["fra_to_72"] < 1.0
    assert rows["fra_to_72"]["anchor_smith_year_delta"] < 1.0
    assert builder.SMITH_BASELINE_EXHAUSTION_YEAR == PDF_SMITH_BASELINE_YEAR
    prov = _artifact()["anchor_provenance"]
    assert "72196" in prov["smith_cite"]
    assert "2034" in prov["smith_cite"]
    assert "pdftotext" in prov["reverified_pdftotext"]


# =====================================================================
# Revenue side: combined OASDI rate sourced from the pe-us statute series
# =====================================================================
def test_oasdi_rate_from_statute_series():
    art = _artifact()
    rate = art["revenue_side"]["oasdi_combined_rate"]
    assert rate["combined"] == pytest.approx(0.124)
    assert rate["employee"] == pytest.approx(0.062)
    assert rate["employer"] == pytest.approx(0.062)
    assert "3101" in rate["reference"] and "3111" in rate["reference"]


# =====================================================================
# F1: sign agreement recomputes from the per-provision rows
# =====================================================================
def test_f1_sign_agreement_recomputes():
    art = _artifact()
    rows = _rows_by_name(art)
    n_ok = 0
    n_total = 0
    for prov in SMITH:
        row = rows[prov]
        n_ok += int(row["balance_analogue_delta"] > 0.0)
        n_ok += int(row["exhaustion_delta_years"] > 0.0)
        n_total += 2
    for name in QUARTET:
        row = rows[name]
        assert row["expected_sign"] == "negative"
        ok = row["outlay_delta"] < 0.0
        assert row["sign_ok"] is bool(ok)
        n_ok += int(ok)
        n_total += 1
    pct = round(100.0 * n_ok / n_total, 1)
    assert pct == pytest.approx(
        art["results_vs_forecasts"]["F1"]["result_pct"]
    )
    assert art["results_vs_forecasts"]["F1"]["met"] is (n_ok == n_total)


# =====================================================================
# F2: revenue-side exhaustion ordering + Smith Kendall tau, from the rows
# =====================================================================
def test_f2_revenue_ordering_recomputes():
    art = _artifact()
    rows = _rows_by_name(art)
    exh = {p: rows[p]["exhaustion_delta_years"] for p in REVENUE}
    our_order = sorted(REVENUE, key=lambda p: -exh[p])
    stored = art["results_vs_forecasts"]["F2"]["result_order"]
    assert our_order == stored
    # Kendall tau of our ordering vs Smith's, recomputed.
    our_rank = [our_order.index(p) for p in REVENUE]
    smith_rank = [list(SMITH_REVENUE_ORDER).index(p) for p in REVENUE]
    tau, _ = kendalltau(our_rank, smith_rank)
    assert tau == pytest.approx(
        art["results_vs_forecasts"]["F2"]["kendall_tau_vs_smith"]
    )
    # F2 is met iff our order equals Smith's exact order.
    assert art["results_vs_forecasts"]["F2"]["met"] is (
        our_order == list(SMITH_REVENUE_ORDER)
    )


# =====================================================================
# F3: FRA->72 ranks above cap-$150k and +1pp, from the rows
# =====================================================================
def test_f3_fra72_ranks_above_recomputes():
    art = _artifact()
    rows = _rows_by_name(art)
    fra = rows["fra_to_72"]["exhaustion_delta_years"]
    above_cap = fra > rows["cap_150k"]["exhaustion_delta_years"]
    above_p1 = fra > rows["payroll_plus_1pp"]["exhaustion_delta_years"]
    f3 = art["results_vs_forecasts"]["F3"]
    assert f3["result_ranks_above_cap_150k"] is bool(above_cap)
    assert f3["result_ranks_above_plus_1pp"] is bool(above_p1)
    assert f3["met"] is bool(above_cap and above_p1)
    # The FRA->72 mechanism: it does not exhaust within the horizon here.
    assert rows["fra_to_72"]["exhausts_within_horizon"] is False


# =====================================================================
# F4: outlay-side ordering persistence (the #115 T2 lesson), from the rows
# =====================================================================
def test_f4_outlay_ordering_recomputes():
    art = _artifact()
    rows = _rows_by_name(art)
    reduction = {n: -rows[n]["outlay_delta"] for n in QUARTET}
    our_order = sorted(QUARTET, key=lambda n: -reduction[n])
    stored = art["results_vs_forecasts"]["F4"]["result_order"]
    assert our_order == stored
    assert art["results_vs_forecasts"]["F4"]["met"] is (
        our_order == list(F4_TARGET_ORDER)
    )
    # The PPI<->NRA swap persists: NRA ranks above PPI by reduction.
    assert our_order.index("nra_raised_to_70") < our_order.index(
        "progressive_price_indexing"
    )


# =====================================================================
# F4 quartet deltas equal the #115 committed benefit-side deltas
# =====================================================================
def test_f4_quartet_matches_cost_ordering_committed():
    art = _artifact()
    rows = _rows_by_name(art)
    # The #115 cost-ordering artifact's benefit-side deltas on the same
    # frame (reused verbatim); pin to that committed run to 4 dp.
    committed = json.loads(
        (ROOT / "runs" / "replication_cost_ordering_v1.json").read_text()
    )
    co_rows = {r["provision"]: r for r in committed["provisions"]}
    for name in QUARTET:
        assert rows[name]["outlay_delta"] == pytest.approx(
            co_rows[name]["our_delta"], abs=1e-6
        )


# =====================================================================
# Floors: present, 5 seeds, the quartet's PI floor exactly zero
# =====================================================================
def test_floors_present_pi_zero():
    art = _artifact()
    floors = art["floors"]["per_provision"]
    for prov in SMITH:
        for key in ("balance_delta", "exhaustion_delta_years"):
            fl = floors["smith"][prov][key]
            assert len(fl["per_seed_signed_gap"]) == 5
            assert fl["abs"]["n_seeds"] == 5
    for name in QUARTET:
        assert len(floors["quartet"][name]["per_seed_signed_gap"]) == 5
    # Price indexing is a uniform scalar (W - 1), identical on every split.
    assert floors["quartet"]["price_indexing"]["abs"]["mean"] == pytest.approx(
        0.0, abs=1e-12
    )
    # Every reported quartet floor mean matches the provision row.
    rows = _rows_by_name(art)
    for name in QUARTET:
        assert rows[name]["outlay_delta_floor_abs_mean"] == pytest.approx(
            floors["quartet"][name]["abs"]["mean"]
        )


def test_provisions_count_and_families():
    art = _artifact()
    rows = _rows_by_name(art)
    assert len(rows) == 9
    assert all(rows[p]["family"] == "smith_2015_solvency" for p in SMITH)
    assert all(rows[n]["family"] == "mermin_quartet_outlay" for n in QUARTET)


# =====================================================================
# skipif-PSID reproduction pin: rebuild the frame + ledger, pin the
# elimination exhaustion delta (the F2 driver) and the F4 quartet order
# =====================================================================
@needs_real_psid
def test_elimination_delta_and_f4_reproduce_committed():
    builder = _import_builder()
    art = _artifact()
    rows = _rows_by_name(art)

    params = builder.load_ssa_parameters()
    transport = builder.build_transport(params)
    survival = builder._mr.Survival()
    rate = builder.load_oasdi_combined_rate()["combined"]

    common, mr_study, meta = builder.build_frame(params, transport)
    assert meta["n_common_frame"] == art["common_frame"]["n_common_frame"]

    factor_cache = builder._factor_cache(common, params)
    contribs, _ = builder.build_person_contribs(
        common, mr_study, params, transport, survival, factor_cache, rate
    )
    deltas = builder.provision_deltas(
        contribs.sum(axis=0), builder.TR2014_REAL_INTEREST
    )
    # Pin the elimination exhaustion delta (the elimination<->+2pp swap
    # driver) to the committed artifact; the full-frame ledger is
    # deterministic.
    assert deltas["elimination"]["exhaustion_delta_years"] == pytest.approx(
        rows["elimination"]["exhaustion_delta_years"], abs=1e-6
    )
    # And the revenue-side order reproduces (+2pp above elimination).
    exh = {p: deltas[p]["exhaustion_delta_years"] for p in REVENUE}
    our_order = sorted(REVENUE, key=lambda p: -exh[p])
    assert our_order == art["results_vs_forecasts"]["F2"]["result_order"]
    assert our_order[0] == "payroll_plus_2pp"
    assert our_order[1] == "elimination"

    # F4 quartet order reproduces on the rebuilt frame.
    quartet = builder.quartet_outlay_deltas(common, transport, survival)
    q_order = sorted(QUARTET, key=lambda n: quartet[n])  # most negative first
    assert q_order[0] == "price_indexing"
    assert q_order.index("nra_raised_to_70") < q_order.index(
        "progressive_price_indexing"
    )
