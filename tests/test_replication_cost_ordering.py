"""Tests for the cost-ordering synthesis vs the anchor cost columns.

Artifact: runs/replication_cost_ordering_v1.json. Frozen spec: issue #42
comment 4931034068. REPORTED, NOT GATED.

Two tiers, mirroring the other run/floor tests:

* Always-runnable artifact-consistency tests (no PSID) that touch only the
  committed artifact plus the pure module constants: the schema is sane and
  marked reported-not-gated with the registration pointer; the anchor cost
  columns in the per-provision rows equal the committed transcription AND
  the independently re-transcribed PDF values; T1 sign agreement and the T2
  and T3 Kendall taus RECOMPUTE from the stored per-provision rows and equal
  the stored test results and the results-vs-forecasts block; earnings
  sharing is recorded excluded-with-reason and descriptive; the floors are
  present with PI's exactly zero.
* A skipif-PSID reproduction pin that rebuilds the common frame and pins one
  provision's aggregate cost delta (progressive price indexing -- the delta
  that drives the T2 ordering) to the committed artifact to float precision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "replication_cost_ordering_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_psid = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir()
    or not (REAL_DATA / "mh85_23").is_dir()
    or not (REAL_DATA / "cah85_23").is_dir(),
    reason="PSID family/marriage/birth files not staged",
)

# Independently re-transcribed anchor values (verified against pdftotext of
# the archived PDFs on 2026-07-09 per #74 protocol note 3), hardcoded here so
# the test fails if the committed constants or the artifact drift from the
# source tables.
PDF_MERMIN_PAYROLL_PCT = {
    "price_indexing": 0.68,
    "progressive_price_indexing": -0.14,
    "reduced_cola": -1.12,
    "nra_raised_to_70": -0.5,
}  # Mermin (2005) 411260 Table 1, 75-yr deficit/surplus row, PDF p.15
PDF_CAREGIVER_COST_PCT = {
    "Biden": -0.12,
    "Buttigieg": -0.51,
    "Klobuchar": -0.12,
    "Warren": -0.30,
}  # Smith/Johnson/Favreault (2020) 103050 Table 3, printed p.19

MERMIN_QUARTET = (
    "price_indexing",
    "progressive_price_indexing",
    "nra_raised_to_70",
    "reduced_cola",
)
CAREGIVER_PLANS = ("Biden", "Buttigieg", "Klobuchar", "Warren")


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import replication_cost_ordering as builder

    return builder


def _rows_by_name(art: dict) -> dict[str, dict]:
    return {r["provision"]: r for r in art["provisions"]}


# =====================================================================
# Schema + reported-not-gated framing (always runnable)
# =====================================================================
def test_schema_and_reported_not_gated():
    art = _artifact()
    assert art["schema_version"] == "replication_cost_ordering.v1"
    assert art["run"] == "replication_cost_ordering_v1"
    assert art["reported_not_gated"] is True
    assert "changes no gate" in art["purpose"]
    assert art["registration_pointer"] == "#42 comment 4931034068"
    assert art["registration"].endswith("4931034068")
    assert "issues/74" in art["program_design_issue"]
    assert "does not grade" in art["grading_note"]


def test_common_frame_and_named_deltas():
    art = _artifact()
    cf = art["common_frame"]
    assert cf["n_common_frame"] > 0
    assert cf["n_common_frame"] == cf["n_careers_mermin_sex_resolvable"]
    assert "SCHEDULED" in cf["baseline"]
    assert "415(b)" in cf["aime_convention"]
    deltas = art["named_deltas"]
    assert any("no long-run projection" in d for d in deltas)
    assert any("common-support restriction" in d for d in deltas)
    assert any("PPI/NRA" in d for d in deltas)
    assert any("DI" in d for d in deltas)


# =====================================================================
# Anchor cost columns: committed constants + independent PDF transcription
# =====================================================================
def test_anchor_values_match_committed_and_pdf():
    builder = _import_builder()
    rows = _rows_by_name(_artifact())
    for prov in MERMIN_QUARTET:
        row_anchor = rows[prov]["anchor_value_payroll_pct"]
        assert row_anchor == pytest.approx(builder.MERMIN_PAYROLL_PCT[prov])
        assert row_anchor == pytest.approx(PDF_MERMIN_PAYROLL_PCT[prov])
    for plan in CAREGIVER_PLANS:
        row_anchor = rows[f"caregiver_{plan}"]["anchor_value_payroll_pct"]
        assert row_anchor == pytest.approx(
            builder.CAREGIVER_ANCHOR_A_TABLE[plan]
        )
        assert row_anchor == pytest.approx(PDF_CAREGIVER_COST_PCT[plan])


def test_mermin_scheduled_deficit_and_cite():
    builder = _import_builder()
    assert builder.MERMIN_SCHEDULED_DEFICIT_PCT == pytest.approx(-1.69)
    prov = _artifact()["anchor_provenance"]
    assert "411260" in prov["mermin_cite"]
    assert "Table 1" in prov["mermin_cite"]
    assert "103050" in prov["caregiver_cite"]
    assert "Table 3" in prov["caregiver_cite"]
    assert "pdftotext" in prov["reverified_pdftotext"]


# =====================================================================
# T1: sign agreement recomputes from the per-provision rows
# =====================================================================
def test_t1_sign_agreement_recomputes():
    art = _artifact()
    rows = _rows_by_name(art)
    n_ok = 0
    n_total = 0
    for prov in MERMIN_QUARTET:
        row = rows[prov]
        assert row["expected_sign"] == "negative"
        ok = row["our_delta"] < 0.0
        assert row["sign_ok"] is bool(ok)
        n_ok += int(ok)
        n_total += 1
    for plan in CAREGIVER_PLANS:
        row = rows[f"caregiver_{plan}"]
        assert row["expected_sign"] == "positive"
        ok = row["our_delta"] > 0.0
        assert row["sign_ok"] is bool(ok)
        n_ok += int(ok)
        n_total += 1
    pct = 100.0 * n_ok / n_total
    assert pct == pytest.approx(
        art["tests"]["T1_sign_agreement"]["pct_agreement"]
    )
    assert pct == pytest.approx(
        art["results_vs_forecasts"]["T1"]["result_pct"]
    )


# =====================================================================
# T2: Mermin-quartet Kendall tau recomputes from the rows
# =====================================================================
def test_t2_mermin_tau_recomputes():
    art = _artifact()
    rows = _rows_by_name(art)
    # Aligned so a bigger benefit reduction is a bigger number (our -delta;
    # anchor payroll saving). Kendall tau is shift-invariant, so the anchor
    # scheduled-deficit offset does not change the recomputed value.
    our_reduction = [-rows[p]["our_delta"] for p in MERMIN_QUARTET]
    anchor_saving = [
        rows[p]["anchor_value_payroll_pct"] for p in MERMIN_QUARTET
    ]
    tau, _ = kendalltau(our_reduction, anchor_saving)
    stored = art["tests"]["T2_mermin_kendall_tau"]["kendall_tau"]
    assert tau == pytest.approx(stored)
    assert tau == pytest.approx(
        art["results_vs_forecasts"]["T2"]["result_tau"]
    )
    # The percent-of-scheduled cross-check column agrees with the payroll
    # column on the ordering (same recomputed tau).
    our_pct = [100.0 * (1.0 + rows[p]["our_delta"]) for p in MERMIN_QUARTET]
    anchor_pct = [
        rows[p]["anchor_pct_of_scheduled_2050"] for p in MERMIN_QUARTET
    ]
    tau_x, _ = kendalltau([-v for v in our_pct], [-v for v in anchor_pct])
    assert tau_x == pytest.approx(
        art["tests"]["T2_mermin_kendall_tau"][
            "kendall_tau_pct_scheduled_xcheck"
        ]
    )


# =====================================================================
# T3: caregiver-quartet Kendall tau recomputes from the rows
# =====================================================================
def test_t3_caregiver_tau_recomputes():
    art = _artifact()
    rows = _rows_by_name(art)
    our_gain = [rows[f"caregiver_{p}"]["our_delta"] for p in CAREGIVER_PLANS]
    anchor_cost_mag = [
        -rows[f"caregiver_{p}"]["anchor_value_payroll_pct"]
        for p in CAREGIVER_PLANS
    ]
    tau, _ = kendalltau(our_gain, anchor_cost_mag)
    stored = art["tests"]["T3_caregiver_kendall_tau"]["kendall_tau"]
    assert tau == pytest.approx(stored)
    assert tau == pytest.approx(
        art["results_vs_forecasts"]["T3"]["result_tau"]
    )


def test_results_vs_forecasts_flags():
    art = _artifact()
    rvf = art["results_vs_forecasts"]
    # The forecasts are recorded verbatim; met-flags are internally
    # consistent with the stored numbers (the orchestrator grades on #42).
    assert rvf["T1"]["met"] is (rvf["T1"]["result_pct"] == 100.0)
    assert rvf["T2"]["met"] is (rvf["T2"]["result_tau"] >= 1.0 - 1e-9)
    assert rvf["T3"]["met"] is (rvf["T3"]["result_tau"] >= 0.8)


# =====================================================================
# Earnings sharing: excluded-with-reason, descriptive for T1
# =====================================================================
def test_earnings_sharing_excluded_descriptive():
    art = _artifact()
    r7 = art["earnings_sharing"]
    assert r7["excluded_from_common_frame"] is True
    assert r7["common_frame"] is False
    assert "couples provision" in r7["exclusion_reason"]
    assert r7["t1_role"] == "descriptive"
    # It is NOT one of the eight scored common-frame provisions.
    names = {r["provision"] for r in art["provisions"]}
    assert "earnings_sharing_1b" not in names
    assert len(names) == 8


# =====================================================================
# Floors: present, PI exactly zero (a uniform scalar)
# =====================================================================
def test_floors_present_pi_zero():
    art = _artifact()
    floors = art["floors"]["per_provision"]
    for prov in MERMIN_QUARTET:
        assert prov in floors
        assert len(floors[prov]["per_seed_signed_gap"]) == 5
    for plan in CAREGIVER_PLANS:
        assert f"caregiver_{plan}" in floors
    # Price indexing is a uniform scalar (W - 1), identical on every split.
    assert floors["price_indexing"]["abs"]["mean"] == pytest.approx(
        0.0, abs=1e-12
    )
    # Every reported per-provision floor mean matches the row.
    rows = _rows_by_name(art)
    for name, fl in floors.items():
        assert rows[name]["floor_abs_mean"] == pytest.approx(fl["abs"]["mean"])


# =====================================================================
# skipif-PSID reproduction pin: rebuild the frame, pin the PPI delta
# =====================================================================
@needs_real_psid
def test_ppi_delta_reproduces_committed():
    builder = _import_builder()
    art = _artifact()
    params = builder.load_ssa_parameters()
    transport = builder.build_transport(params)
    survival = builder.Survival()
    common, meta = builder.build_common_frame(params, transport)

    assert meta["n_common_frame"] == art["common_frame"]["n_common_frame"]

    deltas = builder._deltas_from_pairs(
        builder._all_pairs(common, transport, survival)
    )
    rows = _rows_by_name(art)
    # Pin the progressive-price-indexing delta (the T2-ordering driver) to
    # the committed artifact; the full-frame delta is deterministic.
    assert deltas["progressive_price_indexing"] == pytest.approx(
        rows["progressive_price_indexing"]["our_delta"], abs=1e-9
    )
    # And the recomputed T2 tau reproduces the committed value.
    t2 = builder.t2_mermin_tau(deltas)
    assert t2["kendall_tau"] == pytest.approx(
        art["tests"]["T2_mermin_kendall_tau"]["kendall_tau"]
    )
    assert deltas["progressive_price_indexing"] < 0.0
    assert not np.isnan(deltas["progressive_price_indexing"])
