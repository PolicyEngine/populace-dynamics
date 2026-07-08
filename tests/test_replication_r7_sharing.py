"""Tests for the R7 earnings-sharing replication vs Favreault-Steuerle (2007).

Artifact: runs/replication_r7_sharing_v1.json. Frozen spec: issue #42
comment 4911171806. REPORTED, NOT GATED.

Two tiers, mirroring the other run/floor tests:

* Always-runnable internal-consistency tests (no PSID) that touch only the
  committed artifact plus the pure helpers in
  :mod:`scripts.replication_r7_sharing`: the schema is sane and marked
  reported-not-gated; the registration pointer, the anchor provenance
  (Favreault-Steuerle package definitions with page citations and the full
  transcribed Table 3), the global-increase decision, and the 1c
  non-implementation are present; every cell's nine buckets sum to ~100 and
  its four thresholds derive from them; the per-cell floors, the directional
  verdict, and the magnitude check all RECOMPUTE from the stored per-seed
  half-split shares; and the never-married rows are near-zero by
  construction. Pure helpers reproduce the bucket boundaries, the threshold
  aggregation, the DYNASIM transcription, and the earnings-sharing history.
* A seed-0 reproduction pin (skipped without the PSID family + marriage
  files) that reruns the real-data load, scoring, and seed-0 half-split
  through the build machinery and pins the committed shares, coverage, and
  counts to float precision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "replication_r7_sharing_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_psid = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir()
    or not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID family/marriage files not staged",
)

SEEDS = [0, 1, 2, 3, 4]
MARITAL = ("married", "divorced", "widowed", "never_married")
SEXES = ("male", "female")
THRESHOLDS = ("lose_ge_20", "lose_ge_5", "gain_ge_5", "gain_ge_20")
BUCKETS = (
    "lose_ge_20",
    "lose_10_20",
    "lose_5_10",
    "lose_lt_5",
    "no_change",
    "gain_lt_5",
    "gain_5_10",
    "gain_10_20",
    "gain_ge_20",
)
# DYNASIM Table 3, package 1b, married female nine buckets (the flagship
# gainer cell) -- an external anchor pin transcribed from the PDF.
DYNASIM_1B_MARRIED_FEMALE = (8.4, 7.6, 6.1, 6.6, 0.0, 11.0, 6.5, 9.7, 44.1)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import replication_r7_sharing as builder

    return builder


def _cell(table: list[dict], status: str, sex: str) -> dict:
    return next(r for r in table if r["status"] == status and r["sex"] == sex)


# =====================================================================
# Schema and reported-not-gated framing (always runnable)
# =====================================================================
def test_schema_and_reported_not_gated():
    art = _artifact()
    assert art["schema_version"] == "replication_r7_sharing.v1"
    assert art["run"] == "replication_r7_sharing_v1"
    assert art["reported_not_gated"] is True
    assert art["real_couples_only"] is True
    assert "changes no gate" in art["purpose"]
    assert art["registration_pointer"] == "#42 comment 4911171806"
    assert art["registration"].endswith("4911171806")
    assert "issues/74" in art["program_design_issue"]
    assert art["primary_package"] == "1b"


def test_global_increase_decision_documented():
    """The +4.5% scalar decision is carried prominently with its evidence."""
    art = _artifact()
    dec = art["global_increase_decision"]
    assert "4.5" in dec
    assert "never-married" in dec.lower()
    assert art["conventions"]["global_increase"]["1b"] == 0.045
    assert art["conventions"]["global_increase"]["1a"] == 0.0271
    # The no-scalar sensitivity is present and collapses never-married to
    # no change (the scalar's fingerprint).
    ns = art["sensitivity_no_scalar"]["shares_by_cell"]["never_married:male"]
    assert ns["buckets"]["no_change"] == pytest.approx(100.0, abs=0.5)


def test_package_1c_documented_not_scored():
    art = _artifact()
    txt = art["package_1c_not_implemented"]
    assert "self-financed" in txt or "self-finance" in txt.lower()
    assert "actuarial" in txt.lower()
    assert "4.6 percent" in txt or "4.6%" in txt
    # 1c is not among the scored packages.
    assert "package_1c" not in art
    assert set(art["conventions"]["global_increase"]) == {"1a", "1b"}


def test_anchor_provenance_package_defs_and_citations():
    """Favreault-Steuerle package definitions and Table 3 transcription."""
    art = _artifact()
    prov = art["anchor_provenance"]
    assert "311436" in prov["paper"]
    assert "440v2" in prov["paper"]
    defs = prov["package_definitions"]
    assert "p.15" in defs["citation"]
    assert "2.71" in defs["1a"]
    assert "4.5" in defs["1b"]
    assert "PRIMARY" in defs["1b"]
    assert "taxable maximum" in defs["earnings_sharing_core"]
    # Full Table 3 transcription, all three packages, both sexes. Rows are
    # verbatim; most sum to 100.0-100.2 (the paper's 0.1-rounding). The one
    # exception is package-1c married women, whose printed cells sum to
    # 109.1 -- a documented printed typo in the source (the "no change"
    # cell reads 9.1 where every other married sharing cell is 0.0). 1c is
    # descriptive provenance only and feeds no result.
    t3 = prov["table3_winners_losers_2049"]
    assert "p.19" in t3["citation"]
    assert tuple(t3["bucket_order"]) == BUCKETS
    assert "source_arithmetic_note" in t3
    assert "109.1" in t3["source_arithmetic_note"]
    for pkg in ("1a", "1b", "1c"):
        for status in MARITAL:
            for sex in SEXES:
                row = t3["packages"][pkg][status][sex]
                assert len(row) == 9
                if (pkg, status, sex) == ("1c", "married", "female"):
                    # The documented printed-source typo (kept verbatim).
                    assert sum(row) == pytest.approx(109.1, abs=0.05)
                else:
                    assert sum(row) == pytest.approx(100.0, abs=0.3)
    # The flagship DYNASIM cell is transcribed exactly.
    assert (
        tuple(t3["packages"]["1b"]["married"]["female"])
        == DYNASIM_1B_MARRIED_FEMALE
    )
    # Table 5 poverty transcribed for provenance.
    t5 = prov["table5_poverty_2049"]
    assert "p.24" in t5["citation"]
    assert t5["rows"]["all_people"] == [5.17, 4.96, 5.48, 4.79]
    # Named deltas present.
    assert any(
        "1943-1957" in d or "1960-1980" in d
        for d in prov["named_population_deltas"]
    )


def test_study_population_and_coverage():
    art = _artifact()
    sp = art["study_population"]
    assert sp["n_career_frame"] > 1000
    assert sp["n_scored"] > 500
    assert sp["n_scored"] <= sp["n_career_frame"]
    cov = sp["both_spouse_coverage"]
    assert 0.0 < cov["both_spouse_coverage_share"] <= 1.0
    # per-status coverage present.
    for status in MARITAL:
        assert status in cov["by_marital_status"]
    # never-married is trivially fully covered.
    assert cov["by_marital_status"]["never_married"]["share"] == pytest.approx(
        1.0
    )
    # cell counts sum to n_scored.
    assert sum(sp["n_by_cell"].values()) == sp["n_scored"]


def test_revision_pins():
    art = _artifact()
    pins = art["revision_pins"]
    assert pins["pe_us_revision"]
    assert pins["artifact_schema_version"] == "replication_r7_sharing.v1"


def test_per_seed_present():
    art = _artifact()
    assert [r["seed"] for r in art["per_seed"]] == SEEDS
    for r in art["per_seed"]:
        assert {"package_1b", "package_1a"} <= set(r)
        for status in MARITAL:
            for sex in SEXES:
                key = f"{status}:{sex}"
                cell = r["package_1b"][key]
                assert {"side_a", "side_b", "n_a", "n_b"} <= set(cell)
                for thr in THRESHOLDS:
                    assert thr in cell["side_a"]


# =====================================================================
# Internal consistency of the three-way table (always runnable)
# =====================================================================
def test_buckets_sum_to_100_and_thresholds_derive():
    art = _artifact()
    for pkg in ("package_1b_primary", "package_1a_descriptive"):
        for row in art[pkg]["three_way_by_cell"]:
            if row["n_persons"] == 0:
                continue
            b = row["our_buckets"]
            assert sum(b.values()) == pytest.approx(100.0, abs=0.05)
            thr = row["thresholds"]
            assert thr["lose_ge_5"]["our_share_pct"] == pytest.approx(
                b["lose_ge_20"] + b["lose_10_20"] + b["lose_5_10"], abs=0.02
            )
            assert thr["gain_ge_5"]["our_share_pct"] == pytest.approx(
                b["gain_5_10"] + b["gain_10_20"] + b["gain_ge_20"], abs=0.02
            )
            assert thr["lose_ge_20"]["our_share_pct"] == pytest.approx(
                b["lose_ge_20"], abs=0.02
            )
            assert thr["gain_ge_20"]["our_share_pct"] == pytest.approx(
                b["gain_ge_20"], abs=0.02
            )


def test_dynasim_rows_match_transcription():
    """Each three-way cell's DYNASIM buckets equal the anchor transcription."""
    art = _artifact()
    t3 = art["anchor_provenance"]["table3_winners_losers_2049"]["packages"]
    for pkg_key, pkg in (
        ("package_1b_primary", "1b"),
        ("package_1a_descriptive", "1a"),
    ):
        for row in art[pkg_key]["three_way_by_cell"]:
            dyn = row["dynasim_buckets"]
            ref = dict(
                zip(BUCKETS, t3[pkg][row["status"]][row["sex"]], strict=False)
            )
            for bucket in BUCKETS:
                assert dyn[bucket] == pytest.approx(ref[bucket], abs=1e-9)


def test_floors_recompute_from_per_seed():
    """Each cell/threshold floor = mean|side_a - side_b| over the 5 seeds."""
    art = _artifact()
    per_seed = art["per_seed"]
    for pkg_key, seed_key in (
        ("package_1b_primary", "package_1b"),
        ("package_1a_descriptive", "package_1a"),
    ):
        for row in art[pkg_key]["three_way_by_cell"]:
            key = f"{row['status']}:{row['sex']}"
            for thr in THRESHOLDS:
                gaps = [
                    abs(
                        s[seed_key][key]["side_a"][thr]
                        - s[seed_key][key]["side_b"][thr]
                    )
                    for s in per_seed
                ]
                stored = row["thresholds"][thr]["floor_mean"]
                assert stored == pytest.approx(float(np.mean(gaps)), abs=0.02)


def test_abs_gap_and_exceeds_floor_recompute():
    art = _artifact()
    for row in art["package_1b_primary"]["three_way_by_cell"]:
        for thr in THRESHOLDS:
            cell = row["thresholds"][thr]
            assert cell["abs_gap_vs_dynasim"] == pytest.approx(
                abs(cell["our_share_pct"] - cell["dynasim_pct"]), abs=0.02
            )
            assert cell["our_exceeds_floor"] == (
                cell["our_share_pct"] > cell["floor_mean"]
            )


def test_directional_verdict_recomputes_and_holds():
    """The pre-registered directional expectations recompute and hold."""
    art = _artifact()
    v = art["package_1b_primary"]["directional_verdict"]
    tw_1b = art["package_1b_primary"]["three_way_by_cell"]
    tw_1a = art["package_1a_descriptive"]["three_way_by_cell"]

    mm = _cell(tw_1b, "married", "male")["thresholds"]
    assert v["married_men_are_losers"]["held"] == (
        mm["lose_ge_5"]["our_share_pct"] > mm["gain_ge_5"]["our_share_pct"]
    )
    df_ = _cell(tw_1b, "divorced", "female")["thresholds"]
    assert v["divorced_women_are_gainers"]["held"] == (
        df_["gain_ge_5"]["our_share_pct"] > df_["lose_ge_5"]["our_share_pct"]
    )
    wf_1b = _cell(tw_1b, "widowed", "female")["thresholds"]
    wf_1a = _cell(tw_1a, "widowed", "female")["thresholds"]
    assert v["survivor_variant_helps_widowed_women"]["held"] == (
        wf_1a["lose_ge_20"]["our_share_pct"]
        < wf_1b["lose_ge_20"]["our_share_pct"]
    )
    # The registration's pre-registered expectations all hold on real data.
    assert v["all_held"] is True


def test_never_married_near_zero_by_construction():
    """Never-married winners/losers at >=5% are near zero under 1b (the
    uniform +4.5% is below the 5% threshold)."""
    art = _artifact()
    for sex in SEXES:
        thr = _cell(
            art["package_1b_primary"]["three_way_by_cell"],
            "never_married",
            sex,
        )["thresholds"]
        assert thr["gain_ge_5"]["our_share_pct"] < 5.0
        assert thr["lose_ge_5"]["our_share_pct"] < 5.0


def test_magnitude_check_recomputes():
    art = _artifact()
    mc = art["package_1b_primary"]["magnitude_check"]
    tw = art["package_1b_primary"]["three_way_by_cell"]
    large = []
    for row in tw:
        for cell in row["thresholds"].values():
            if cell["dynasim_pct"] >= 20.0:
                large.append(cell["abs_gap_vs_dynasim"] <= 10.0)
    assert mc["n_large_cells"] == len(large)
    assert mc["n_within_10pp"] == sum(large)


def test_aggregate_cost_change_present():
    art = _artifact()
    acc = art["package_1b_primary"]["aggregate_cost_change"]
    # The scalar-off change is more negative than the scalar-on change
    # (the +4.5% partially offsets the removal of auxiliary benefits).
    assert (
        acc["weighted_aggregate_pct_change_1b_noscalar"]
        < acc["weighted_aggregate_pct_change_1b"]
    )


# =====================================================================
# Pure-helper unit tests (import the builder; no PSID)
# =====================================================================
def test_bucket_boundaries():
    b = _import_builder()
    assert b._bucket_of(-0.25) == "lose_ge_20"
    assert b._bucket_of(-0.20) == "lose_ge_20"
    assert b._bucket_of(-0.15) == "lose_10_20"
    assert b._bucket_of(-0.10) == "lose_10_20"
    assert b._bucket_of(-0.07) == "lose_5_10"
    assert b._bucket_of(-0.05) == "lose_5_10"
    assert b._bucket_of(-0.02) == "lose_lt_5"
    assert b._bucket_of(0.0) == "no_change"
    assert b._bucket_of(0.03) == "gain_lt_5"
    assert b._bucket_of(0.05) == "gain_5_10"
    assert b._bucket_of(0.12) == "gain_10_20"
    assert b._bucket_of(0.20) == "gain_ge_20"
    assert b._bucket_of(0.5) == "gain_ge_20"


def test_thresholds_from_buckets():
    b = _import_builder()
    buckets = dict(
        zip(
            BUCKETS,
            (10.0, 5.0, 3.0, 2.0, 0.0, 20.0, 6.0, 4.0, 50.0),
            strict=False,
        )
    )
    thr = b._thresholds_from_buckets(buckets)
    assert thr["lose_ge_20"] == 10.0
    assert thr["lose_ge_5"] == 18.0  # 10 + 5 + 3
    assert thr["gain_ge_5"] == 60.0  # 6 + 4 + 50
    assert thr["gain_ge_20"] == 50.0


def test_dynasim_cell_transcription():
    b = _import_builder()
    cell = b.dynasim_cell("1b", "married", "female")
    assert cell["buckets"]["gain_ge_20"] == 44.1
    assert cell["thresholds"]["gain_ge_20"] == 44.1
    assert cell["thresholds"]["gain_ge_5"] == pytest.approx(6.5 + 9.7 + 44.1)


def test_person_history_interpolates_single_year_gaps():
    import pandas as pd

    b = _import_builder()
    # Born 1950: ages 22-61 = 1972-2011. Biennial 2000,2002 with a gap at
    # 2001 -> interpolated to the mean; a two-year gap is NOT filled.
    rows = [
        {"person_id": 1, "period": 2000, "earnings": 40_000.0},
        {"person_id": 1, "period": 2002, "earnings": 44_000.0},
        {"person_id": 1, "period": 2005, "earnings": 50_000.0},
    ]
    sub = pd.DataFrame(rows)
    hist = b._person_history(sub, 1950)
    assert hist[2000] == 40_000.0
    assert hist[2001] == pytest.approx(42_000.0)  # single-year gap filled
    assert hist[2002] == 44_000.0
    assert 2003 not in hist  # two-year gap (2003, 2004) not filled
    assert hist[2005] == 50_000.0


def test_person_history_respects_age_window():
    import pandas as pd

    b = _import_builder()
    # Born 1950: age 22 = 1972, age 61 = 2011. A 1970 (age 20) row drops.
    rows = [
        {"person_id": 1, "period": 1970, "earnings": 9_999.0},
        {"person_id": 1, "period": 1972, "earnings": 10_000.0},
        {"person_id": 1, "period": 2011, "earnings": 60_000.0},
    ]
    hist = b._person_history(pd.DataFrame(rows), 1950)
    assert 1970 not in hist
    assert 1972 in hist and 2011 in hist


def test_weighted_bucket_shares_and_thresholds():
    b = _import_builder()
    # Four persons: -0.30 (lose>=20), -0.06 (lose5-10), +0.03 (gain<5),
    # +0.25 (gain>=20); equal weights -> 25% each.
    deltas = np.array([-0.30, -0.06, 0.03, 0.25])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    shares = b._weighted_bucket_shares(deltas, weights)
    assert shares["lose_ge_20"] == pytest.approx(25.0)
    assert shares["lose_5_10"] == pytest.approx(25.0)
    assert shares["gain_lt_5"] == pytest.approx(25.0)
    assert shares["gain_ge_20"] == pytest.approx(25.0)
    thr = b._thresholds_from_buckets(shares)
    assert thr["lose_ge_5"] == pytest.approx(50.0)  # lose>=20 + lose5-10
    assert thr["gain_ge_5"] == pytest.approx(25.0)  # only gain>=20


def test_weighted_bucket_shares_respects_weights():
    b = _import_builder()
    # A heavy loser and a light gainer -> loss share dominates.
    deltas = np.array([-0.30, 0.25])
    weights = np.array([3.0, 1.0])
    shares = b._weighted_bucket_shares(deltas, weights)
    assert shares["lose_ge_20"] == pytest.approx(75.0)
    assert shares["gain_ge_20"] == pytest.approx(25.0)


def test_summary_empty_safe():
    b = _import_builder()
    s = b._summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean"] == pytest.approx(3.0)
    assert s["n_seeds"] == 5
    assert b._summary([])["n_seeds"] == 0


# =====================================================================
# Seed-0 reproduction pin (needs PSID; run live)
# =====================================================================
@needs_real_psid
def test_seed0_reproduces_committed_artifact():
    """Rerun the real-data load, scoring, and the seed-0 half-split; match
    the committed shares, coverage, and counts to float precision."""
    builder = _import_builder()
    art = _artifact()

    params = builder.load_ssa_parameters()
    if params.pe_us_revision != art["revision_pins"]["pe_us_revision"]:
        pytest.skip(
            f"policyengine-us at {params.pe_us_revision} differs from the "
            f"artifact's pinned {art['revision_pins']['pe_us_revision']}"
        )

    study = builder.StudyData(params)
    df = builder.score_population(study)

    # Career-frame size, scored count, and coverage pin.
    sp = art["study_population"]
    assert len(study.careers) == sp["n_career_frame"]
    assert len(df) == sp["n_scored"]
    cov = builder._coverage_report(study)
    assert cov["both_spouse_coverage_share"] == pytest.approx(
        sp["both_spouse_coverage"]["both_spouse_coverage_share"], abs=1e-9
    )

    # Full-sample 1b shares pin (every cell, every threshold).
    shares_1b = builder.cell_shares(df, "delta_1b")
    for row in art["package_1b_primary"]["three_way_by_cell"]:
        got = shares_1b[(row["status"], row["sex"])]
        assert got["n_persons"] == row["n_persons"]
        for thr in THRESHOLDS:
            assert got["thresholds"][thr] == pytest.approx(
                row["thresholds"][thr]["our_share_pct"], abs=1e-6
            )

    # Seed-0 half-split pin (the floor inputs).
    seed0 = builder.seed_half_shares(df, "delta_1b", 0)
    ref0 = next(s for s in art["per_seed"] if s["seed"] == 0)["package_1b"]
    for key, cell in seed0.items():
        for thr in THRESHOLDS:
            assert cell["side_a"][thr] == pytest.approx(
                ref0[key]["side_a"][thr], abs=1e-6
            )
            assert cell["side_b"][thr] == pytest.approx(
                ref0[key]["side_b"][thr], abs=1e-6
            )
