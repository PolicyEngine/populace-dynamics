"""Tests for the caregiver-credit replication vs Smith/Johnson/Favreault
(2020).

Artifact: runs/replication_caregiver_v1.json. Frozen spec: issue #42
comment 4911453454. REPORTED, NOT GATED.

Two tiers, mirroring the other run/floor tests:

* Always-runnable internal-consistency tests (no PSID) that touch only the
  committed artifact plus the pure helpers in
  :mod:`scripts.replication_caregiver`: the schema is sane and marked
  reported-not-gated; the registration pointer and the anchor provenance
  (the four plan definitions with page citations and the transcribed
  Table 15 row) are present; the four-plan table's anchor cells equal the
  transcription and its per-plan floors, the per-quintile incidence, and
  the pre-registered expectation all RECOMPUTE from the stored per-seed
  and full-sample values. Pure helpers reproduce the transported AIME, the
  qualifying-year selection (including the benefit-maximising vs
  chronological cap), the plan metrics, and the Spearman ordering.
* A seed-0 reproduction pin (skipped without the PSID family + birth
  files) that reruns the real-data load, scoring, and the seed-0
  half-split through the build machinery and pins the committed shares and
  counts to float precision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "replication_caregiver_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_psid = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir()
    or not (REAL_DATA / "cah85_23").is_dir(),
    reason="PSID family/birth files not staged",
)

PLAN_ORDER = ("Biden", "Buttigieg", "Klobuchar", "Warren")
ANCHOR_TABLE15 = {
    "Biden": 54.0,
    "Buttigieg": 52.0,
    "Klobuchar": 62.0,
    "Warren": 55.0,
}
ANCHOR_SHARE_GAINING = {
    "Biden": 36.0,
    "Buttigieg": 45.0,
    "Klobuchar": 27.0,
    "Warren": 33.0,
}


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import replication_caregiver as builder

    return builder


def _plan_row(table: list[dict], plan: str) -> dict:
    return next(r for r in table if r["plan"] == plan)


# =====================================================================
# Schema and reported-not-gated framing (always runnable)
# =====================================================================
def test_schema_and_reported_not_gated():
    art = _artifact()
    assert art["schema_version"] == "replication_caregiver.v1"
    assert art["run"] == "replication_caregiver_v1"
    assert art["reported_not_gated"] is True
    assert art["real_data_only"] is True
    assert "changes no gate" in art["purpose"]
    assert art["registration_pointer"] == "#42 comment 4911453454"
    assert art["registration"].endswith("4911453454")
    assert "issues/74" in art["program_design_issue"]


def test_revision_pins():
    art = _artifact()
    pins = art["revision_pins"]
    assert pins["pe_us_revision"]
    assert pins["artifact_schema_version"] == "replication_caregiver.v1"


def test_per_seed_present():
    art = _artifact()
    assert [r["seed"] for r in art["per_seed"]] == [0, 1, 2, 3, 4]
    for r in art["per_seed"]:
        for plan in PLAN_ORDER:
            cell = r["half_metrics"][plan]
            assert {"side_a", "side_b"} <= set(cell)
            for side in ("side_a", "side_b"):
                assert "bottom_quintile_share_pct" in cell[side]
                assert "share_gaining_pct" in cell[side]


# =====================================================================
# Anchor provenance: plan definitions + Table 15 transcription + cites
# =====================================================================
def test_anchor_provenance_plan_defs_and_citations():
    art = _artifact()
    prov = art["anchor_provenance"]
    assert "103050" in prov["paper"]
    assert "ID980" in prov["paper"]
    defs = prov["plan_definitions"]
    assert "p. 8" in defs["citation"]
    assert "p. 11" in defs["citation"]
    # Maximum qualifying-child ages (Table 2): 11 / 17 / 5 / 5.
    ages = defs["maximum_age_for_qualifying_child"]
    assert ages == {"Biden": 11, "Buttigieg": 17, "Klobuchar": 5, "Warren": 5}
    caps = defs["year_cap"]
    assert caps["Biden"] == 5
    assert caps["Buttigieg"] is None and caps["Klobuchar"] is None
    assert caps["Warren"] is None
    # Biden's phase-out simplification is documented.
    assert "phase" in defs["biden_variant_note"].lower()


def test_anchor_table15_transcription():
    art = _artifact()
    t15 = art["anchor_provenance"]["table15_bottom_fifth_2065"]
    assert "p. 66" in t15["citation"]
    assert "Create caregiver credit" in t15["citation"]
    assert t15["values_pct"] == ANCHOR_TABLE15
    assert t15["band_pct"] == [52.0, 62.0]
    # Share-gaining narrative + cost table transcriptions with cites.
    sg = art["anchor_provenance"]["share_gaining_2065"]
    assert "p. 27-28" in sg["citation"]
    assert sg["values_pct"] == ANCHOR_SHARE_GAINING
    cost = art["anchor_provenance"]["cost_75yr_pct_payroll"]
    assert "p. 19" in cost["citation"]
    assert cost["values"]["Buttigieg"] == -0.51
    # Named deltas present (individual vs couples-shared; truncated window).
    deltas = art["anchor_provenance"]["named_population_deltas"]
    assert any("own-record" in d for d in deltas)
    assert any("truncated" in d for d in deltas)


# =====================================================================
# Four-plan table + internal consistency (always runnable)
# =====================================================================
def test_four_plan_table_structure_and_anchor_cells():
    art = _artifact()
    table = art["four_plan_table"]
    assert [r["plan"] for r in table] == list(PLAN_ORDER)
    for row in table:
        plan = row["plan"]
        assert row["anchor_bottom_fifth_pct"] == ANCHOR_TABLE15[plan]
        assert row["anchor_share_gaining_pct"] == ANCHOR_SHARE_GAINING[plan]
        # in_or_above_band recomputes (band lower bound 52).
        assert row["in_or_above_band"] == (
            row["our_bottom_quintile_share_pct"] >= 52.0
        )
        # abs_gap recomputes.
        assert row["abs_gap_vs_anchor"] == pytest.approx(
            abs(
                row["our_bottom_quintile_share_pct"]
                - row["anchor_bottom_fifth_pct"]
            ),
            abs=0.02,
        )
        # Credit design fields match the plan definitions.
        assert row["credit_fraction"] in (0.5, 1.0)
        assert row["child_age_limit_max"] in (11, 17, 5)


def test_incidence_sums_to_100_and_aime_monotone():
    art = _artifact()
    for _plan, m in art["full_sample_metrics"].items():
        inc = m["incidence_by_quintile"]
        assert len(inc) == 5
        total = sum(q["share_of_aggregate_gain_pct"] for q in inc)
        assert total == pytest.approx(100.0, abs=0.02)
        # Bottom-quintile share equals Q1's share-of-aggregate-gain.
        assert m["bottom_quintile_share_pct"] == pytest.approx(
            inc[0]["share_of_aggregate_gain_pct"], abs=1e-9
        )
        # Own-distribution ordering: baseline AIME strictly increases.
        aimes = [q["mean_baseline_aime"] for q in inc]
        assert all(aimes[i] < aimes[i + 1] for i in range(4)), aimes


def test_floors_recompute_from_per_seed():
    art = _artifact()
    per_seed = art["per_seed"]
    for row in art["four_plan_table"]:
        plan = row["plan"]
        gaps = [
            abs(
                s["half_metrics"][plan]["side_a"]["bottom_quintile_share_pct"]
                - s["half_metrics"][plan]["side_b"][
                    "bottom_quintile_share_pct"
                ]
            )
            for s in per_seed
        ]
        assert row["bottom_quintile_share_floor_mean"] == pytest.approx(
            float(np.mean(gaps)), abs=1e-6
        )


def test_floor_below_plan_spread():
    """The bottom-quintile-share floors are far smaller than the plan-to-
    plan spread, so the ordering is signal, not sampling noise."""
    art = _artifact()
    shares = [
        r["our_bottom_quintile_share_pct"] for r in art["four_plan_table"]
    ]
    spread = max(shares) - min(shares)
    max_floor = max(
        r["bottom_quintile_share_floor_mean"] for r in art["four_plan_table"]
    )
    assert max_floor < spread


def test_registered_expectation_recomputes_and_holds():
    art = _artifact()
    e = art["registered_expectation"]
    full = art["full_sample_metrics"]
    # Half-credit designs in or above the band.
    hc = e["half_credit_in_or_above_band"]
    assert hc["held"] == all(
        full[p]["bottom_quintile_share_pct"] >= 52.0
        for p in ("Biden", "Klobuchar")
    )
    # Ordering endpoints match the anchor.
    po = e["plan_ordering_matches_anchor"]
    our_conc = {p: full[p]["bottom_quintile_share_pct"] for p in PLAN_ORDER}
    assert po["our_concentration_high"] == max(our_conc, key=our_conc.get)
    assert po["our_concentration_low"] == min(our_conc, key=our_conc.get)
    assert po["anchor_concentration_high"] == "Klobuchar"
    assert po["anchor_concentration_low"] == "Buttigieg"
    assert po["reach_endpoints_match"] is True
    # The registered expectation holds on real data.
    assert e["expectation_held"] is True
    assert e["expectation_held"] == (hc["held"] and po["held"])


def test_biden_cap_sensitivity_close_to_primary():
    """The Biden cap selection (benefit-max primary vs chronological) does
    not drive the finding: the two bottom-quintile shares are within a few
    points of each other."""
    art = _artifact()
    bs = art["biden_cap_sensitivity_chronological"]
    primary = bs["primary_bottom_quintile_share_pct"]
    chrono = bs["bottom_quintile_share_pct"]
    assert primary == pytest.approx(
        art["full_sample_metrics"]["Biden"]["bottom_quintile_share_pct"]
    )
    assert abs(primary - chrono) < 5.0


def test_study_population_counts():
    art = _artifact()
    sp = art["study_population"]
    assert sp["n_career_frame"] > 1000
    assert sp["n_scored"] == sp["n_career_frame"]
    assert 0 < sp["n_parents_in_frame"] <= sp["n_career_frame"]
    assert sp["n_birth_records_joined"] > 10000


# =====================================================================
# Pure-helper unit tests (import the builder; no PSID)
# =====================================================================
class _FakeParams:
    """Constant-NAWI params so indexing is the identity and the arithmetic
    is checkable by hand."""

    nawi = {y: 50_000.0 for y in range(1960, 2050)}
    pe_us_revision = "fake"

    def wage_base_for(self, year: int) -> float:
        return 1e12


def _fake_transport() -> dict:
    return {
        "index_nawi": 50_000.0,
        "nawi": {y: 50_000.0 for y in range(1960, 2050)},
        "index_year": 2048,
        "bend_points": (1_000.0, 6_000.0),
        "pia_factors": (0.9, 0.32, 0.15),
    }


def test_transported_aime_top35_identity_indexing():
    b = _import_builder()
    params, transport = _FakeParams(), _fake_transport()
    # One year of $60,000 -> 60000 over 35*12 months, floored.
    aime = b.transported_aime({2000: 60_000.0}, params, transport)
    assert aime == float(int(60_000.0 / (35 * 12)))
    # Two years: top-35 sum is both, others zero.
    aime2 = b.transported_aime(
        {2000: 60_000.0, 2001: 120_000.0}, params, transport
    )
    assert aime2 == float(int(180_000.0 / (35 * 12)))


def test_qualifying_years_age_and_earnings_gate():
    b = _import_builder()
    params, transport = _FakeParams(), _fake_transport()
    plan = b.Plan("Test", 0.5, 6, None)  # 1/2 wage, child < 6, no cap
    # credit level = 0.5 * 50000 = 25000. Child born 1998 -> ages 0-5 in
    # 1998-2003.
    history = {
        2000: 10_000.0,  # child age 2, below credit -> qualifies
        2001: 30_000.0,  # child age 3, above credit -> excluded
        2005: 5_000.0,  # child age 7 -> too old, excluded
    }
    got = b.qualifying_years(history, [1998], plan, params, transport)
    assert got == [(2000, 25_000.0)]


def test_qualifying_years_cap_selection_benefit_max_vs_chronological():
    b = _import_builder()
    params, transport = _FakeParams(), _fake_transport()
    plan = b.Plan("BidenLike", 0.5, 12, 2)  # child < 12, cap 2 years
    # Child born 1990 -> ages 5,6,7 in 1995,1996,1997, all < 12; all below
    # the 25000 credit level. Identity indexing -> indexed earn == earn.
    history = {1995: 5_000.0, 1996: 20_000.0, 1997: 10_000.0}
    bmax = b.qualifying_years(
        history, [1990], plan, params, transport, selection="benefit_max"
    )
    # Benefit-max credits the two LOWEST-earning years (1995, 1997).
    assert sorted(y for y, _ in bmax) == [1995, 1997]
    chrono = b.qualifying_years(
        history, [1990], plan, params, transport, selection="chronological"
    )
    # Chronological credits the two EARLIEST years (1995, 1996).
    assert sorted(y for y, _ in chrono) == [1995, 1996]


def test_reformed_history_tops_up():
    b = _import_builder()
    history = {2000: 10_000.0, 2001: 40_000.0}
    new = b.reformed_history(history, [(2000, 25_000.0)])
    assert new[2000] == 25_000.0
    assert new[2001] == 40_000.0  # untouched
    assert history[2000] == 10_000.0  # original not mutated


def test_credit_raises_aime_for_low_earner_only():
    b = _import_builder()
    params, transport = _FakeParams(), _fake_transport()
    plan = b.Plan("Full", 1.0, 6, None)  # full wage, child < 6
    # Low earner with a qualifying child-year: credit lifts the AIME.
    low = {2000: 5_000.0}
    credited = b.qualifying_years(low, [1999], plan, params, transport)
    base = b.transported_aime(low, params, transport)
    reformed = b.transported_aime(
        b.reformed_history(low, credited), params, transport
    )
    assert reformed > base
    # High earner already above the credit level: no qualifying year.
    high = {2000: 90_000.0}
    assert b.qualifying_years(high, [1999], plan, params, transport) == []


def test_plan_metrics_closed_form():
    import pandas as pd

    b = _import_builder()
    # Five persons, one per quintile, only the bottom-quintile person gains.
    df = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            "base_aime": [100.0, 200.0, 300.0, 400.0, 500.0],
            "gain_X": [10.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    quintile = np.array([0, 1, 2, 3, 4])
    m = b.plan_metrics(df, "gain_X", quintile)
    assert m["bottom_quintile_share_pct"] == pytest.approx(100.0)
    assert m["share_gaining_pct"] == pytest.approx(20.0)  # 1 of 5
    assert m["n_gainers"] == 1
    assert sum(
        q["share_of_aggregate_gain_pct"] for q in m["incidence_by_quintile"]
    ) == pytest.approx(100.0)


def test_plan_metrics_respects_weights():
    import pandas as pd

    b = _import_builder()
    # Two bottom-quintile gainers and one top-quintile gainer; the weighted
    # aggregate is dominated by the heavy bottom person.
    df = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "weight": [3.0, 1.0, 1.0],
            "base_aime": [100.0, 100.0, 900.0],
            "gain_X": [10.0, 10.0, 10.0],
        }
    )
    quintile = np.array([0, 0, 4])
    m = b.plan_metrics(df, "gain_X", quintile)
    # Bottom-quintile weighted gain = (3+1)*10 = 40 of total 50 = 80%.
    assert m["bottom_quintile_share_pct"] == pytest.approx(80.0)


def test_spearman_rho():
    b = _import_builder()
    assert b._spearman_rho([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert b._spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Our concentration vs the anchor (Biden/Warren middle swap) -> 0.8.
    ours = [64.3, 58.2, 70.7, 58.5]  # Biden, Buttigieg, Klobuchar, Warren
    anchor = [54.0, 52.0, 62.0, 55.0]
    assert b._spearman_rho(ours, anchor) == pytest.approx(0.8, abs=1e-9)


def test_assign_quintiles_equal_weights():
    b = _import_builder()
    aime = np.arange(1.0, 11.0)  # 10 values
    weight = np.ones(10)
    q = b.assign_quintiles(aime, weight)
    assert list(q) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_plans_match_anchor_definitions():
    b = _import_builder()
    by_name = {p.name: p for p in b.PLANS}
    assert by_name["Biden"].credit_fraction == 0.5
    assert by_name["Biden"].child_age_limit == 12  # max age 11
    assert by_name["Biden"].year_cap == 5
    assert by_name["Buttigieg"].credit_fraction == 1.0
    assert by_name["Buttigieg"].child_age_limit == 18  # max age 17
    assert by_name["Buttigieg"].year_cap is None
    assert by_name["Klobuchar"].credit_fraction == 0.5
    assert by_name["Klobuchar"].child_age_limit == 6  # max age 5
    assert by_name["Warren"].credit_fraction == 1.0
    assert by_name["Warren"].child_age_limit == 6  # max age 5


# =====================================================================
# Seed-0 reproduction pin (needs PSID; run live)
# =====================================================================
@needs_real_psid
def test_seed0_reproduces_committed_artifact():
    """Rerun the real-data load, scoring, and the seed-0 half-split; match
    the committed shares and counts to float precision."""
    builder = _import_builder()
    art = _artifact()

    params = builder.load_ssa_parameters()
    if params.pe_us_revision != art["revision_pins"]["pe_us_revision"]:
        pytest.skip(
            f"policyengine-us at {params.pe_us_revision} differs from the "
            f"artifact's pinned {art['revision_pins']['pe_us_revision']}"
        )
    transport = builder.build_transport(params)
    study = builder.CaregiverStudy(params, transport)
    df = builder.score_population(study, params, transport)

    sp = art["study_population"]
    assert len(study.careers) == sp["n_career_frame"]
    assert len(df) == sp["n_scored"]

    quintile = builder.assign_quintiles(
        df["base_aime"].to_numpy(dtype=np.float64),
        df["weight"].to_numpy(dtype=np.float64),
    )
    # Full-sample per-plan shares pin.
    for plan in builder.PLANS:
        m = builder.plan_metrics(df, f"gain_{plan.name}", quintile)
        ref = art["full_sample_metrics"][plan.name]
        assert m["bottom_quintile_share_pct"] == pytest.approx(
            ref["bottom_quintile_share_pct"], abs=1e-6
        )
        assert m["share_gaining_pct"] == pytest.approx(
            ref["share_gaining_pct"], abs=1e-6
        )
        assert m["n_gainers"] == ref["n_gainers"]

    # Seed-0 half-split pin (the floor inputs).
    seed0 = builder.seed_half_metrics(df, 0)
    ref0 = next(s for s in art["per_seed"] if s["seed"] == 0)["half_metrics"]
    for plan in PLAN_ORDER:
        for side in ("side_a", "side_b"):
            for metric in (
                "bottom_quintile_share_pct",
                "share_gaining_pct",
            ):
                assert seed0[plan][side][metric] == pytest.approx(
                    ref0[plan][side][metric], abs=1e-6
                )
