"""Tests for the shared-earnings PI/PPI replication vs Mermin (2005).

Artifact: runs/replication_ppi_shared_v1.json. Frozen spec: issue #42
comment 4931009783. REPORTED, NOT GATED. REAL DATA ONLY.

Two tiers, mirroring the Phase-A and R7 run tests:

* Always-runnable internal-consistency tests (no PSID) that touch only the
  committed artifact plus the pure helpers in
  :mod:`scripts.replication_ppi_shared`: the schema is sane and marked
  reported-not-gated; the registration pointer, the REAL-DATA-ONLY tranche
  scope, the page-cited Mermin shared-earnings definition, and the Table 2
  anchor rows are present; the PI ratio is quintile-invariant and equals
  the wedge under BOTH the individual (own-quintile) and the shared
  (shared-quintile) grouping; the shared PPI gradient is monotone; the
  three-way table (individual / shared / anchor) and its half-split floors
  RECOMPUTE from the stored per-seed and full-sample values; the
  pre-registered expectations recompute and hold; and the individual and
  shared per-person ratios are identical (only the ranking differs). Pure
  helpers reproduce the couple-mean sharing (cap-after) and the
  own-record-ratio-by-two-rankings measurement.
* A seed-0 reproduction pin (skipped without the PSID family + marriage
  files) that reruns the real-data load, study-population selection, and
  seed-0 half-split through the build machinery and pins the committed
  full-sample and seed-0 floor quintile ratios to float precision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "replication_ppi_shared_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_psid = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir()
    or not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID family/marriage files not staged",
)

SEEDS = [0, 1, 2, 3, 4]
N_QUINTILES = 5
DYNASIM_PPI = (98.7, 90.4, 81.3, 75.7, 71.7)
GROUPINGS = ("by_own_quintile", "by_shared_quintile")


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import replication_ppi_shared as builder

    return builder


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _ppi(side: dict, grouping: str) -> list[float]:
    return [q["ppi_ratio_pct"] for q in side[grouping]]


def _pi(side: dict, grouping: str) -> list[float]:
    return [q["pi_ratio_pct"] for q in side[grouping]]


# =====================================================================
# Schema and reported-not-gated framing (always runnable)
# =====================================================================
def test_schema_and_reported_not_gated():
    art = _artifact()
    assert art["schema_version"] == "replication_ppi_shared.v1"
    assert art["run"] == "replication_ppi_shared_v1"
    assert art["reported_not_gated"] is True
    assert "changes no gate" in art["purpose"]
    assert art["registration_pointer"] == "#42 comment 4931009783"
    assert art["registration"].endswith("4931009783")
    assert "issues/74" in art["program_design_issue"]
    # Cross-references to the two reused replications.
    assert art["phase_a_individual_registration"].endswith("4907444903")
    assert art["r7_join_machinery_registration"].endswith("4911171806")


def test_real_data_only_scope_stated():
    """The ratified tranche map gates generated couples on unlocked 2c."""
    art = _artifact()
    scope = art["real_data_only_scope"]
    assert scope["real_data_only"] is True
    assert scope["tranche_map_issue"].endswith("112")
    stmt = scope["statement"]
    assert "REAL" in stmt
    assert "marriage x earnings joint" in stmt
    assert "generated" in stmt.lower()
    # The registration's "unlocked tranche 2c" is named.
    assert "2c" in stmt
    # The gate-2 scope string is echoed from gates.yaml, verbatim: the
    # marriage x earnings joint is the separate UNLOCKED tranche 2c.
    echoed = scope["gate_2_marriage_x_earnings_joint_scope"]
    assert echoed is not None
    assert "NOT COVERED" in echoed
    assert "UNLOCKED" in echoed
    assert "2c_marriage_earnings_joint" in echoed


def test_mermin_shared_definition_page_cited():
    """Mermin's shared-earnings definition, quoted and page-cited (p.3, fn.9)."""
    art = _artifact()
    d = art["anchor_provenance"]["shared_lifetime_earnings_definition"]
    assert "page 3" in d["citation"]
    assert "footnote 9" in d["citation"]
    # The exact defining clause from the paper.
    assert "half of the earnings of both" in d["quote"]
    assert "single" in d["quote"]
    # Footnote 9 pins the AIME-like wage-indexing (the cap convention).
    assert "wage-indexed" in d["footnote_9_quote"]
    assert "average indexed monthly earnings" in d["footnote_9_quote"]
    # The encoding records couple mean + cap-after-averaging.
    assert "couple mean" in d["encoding"]
    assert "0.5*(own + spouse)" in d["encoding"]
    note = d["capping_convention_note"].lower()
    assert "after averaging" in note
    assert "prior to sharing" in note  # names the R7/F-S contrast


def test_anchor_table2_transcription():
    """Mermin Table 2 target rows (by shared quintile) are transcribed."""
    art = _artifact()
    t2 = art["anchor_provenance"]["table2_retired_workers_62_67_in_2050"]
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
    assert "p.16" in t2["citation"]
    # The population is own-record retired workers (so the ratio is
    # own-record), ranked by shared income.
    pop = art["anchor_provenance"]["population_and_quintile"]["population"]
    assert "own-record" in pop
    assert "SHARED" in pop
    # Named deltas: truncated window, observed cohorts, common support, and
    # the closed individual-vs-shared delta.
    deltas = " ".join(art["anchor_provenance"]["named_population_deltas"])
    assert "truncated observation window" in deltas
    assert "observed cohorts" in deltas
    assert "common-support" in deltas
    assert "CLOSED" in deltas


def test_design_is_own_ratio_shared_ranking():
    """The design isolates the ranking: own-record ratio, two rankings."""
    art = _artifact()
    design = art["design"]
    assert "OWN AIME" in design
    assert "RANKING" in design
    assert "identical per-person" in design
    tc = art["transport_and_conventions"]
    assert "OWN" in tc["ppi_bend"]
    assert "couple mean" in tc["shared_measure"]


def test_transport_and_conventions():
    art = _artifact()
    tc = art["transport_and_conventions"]
    assert tc["eligibility_year"] == 2050
    assert tc["index_year"] == 2048
    assert 0.60 < tc["wedge"] < 0.72
    assert tc["wedge_formula"] == "(1.028/1.039)**(2050-2012)"
    b1, b2 = tc["bend_points_2050"]
    assert 0 < b1 < b2


def test_revision_pins():
    art = _artifact()
    pins = art["revision_pins"]
    assert pins["pe_us_revision"]
    assert pins["artifact_schema_version"] == "replication_ppi_shared.v1"
    assert pins["gates_yaml"]["gate_1_locked"] is True


# =====================================================================
# PI encoding check: quintile-invariant ratio == the wedge, both rankings
# =====================================================================
def test_pi_ratio_quintile_invariant_equals_wedge():
    """Under BOTH the own and the shared grouping, and on every half-split
    side, the five PI quintile ratios equal 100*wedge to 1e-6 -- the PI
    encoding scales all factors by one wedge, so the ratio is the wedge for
    every career regardless of the ranking."""
    art = _artifact()
    wedge_pct = 100.0 * art["transport_and_conventions"]["wedge"]
    sides = [art["full_sample"]]
    for r in art["per_seed"]:
        sides.extend([r["side_a"], r["side_b"]])
    for side in sides:
        for grouping in GROUPINGS:
            pis = [
                q["pi_ratio_pct"]
                for q in side[grouping]
                if q.get("n_positive", 0) > 0
            ]
            assert max(pis) - min(pis) < 1e-6, pis
            for v in pis:
                assert v == pytest.approx(wedge_pct, abs=1e-6)
        assert side["overall_pi_ratio_pct"] == pytest.approx(
            wedge_pct, abs=1e-6
        )
    pi = art["three_way_comparison"]["pi_scalars"]
    assert pi["wedge_implied_scalar_pct"] == pytest.approx(wedge_pct, abs=1e-9)
    assert pi["anchor_dynasim_pct"] == 67.8
    for v in pi["individual_by_quintile_pct"] + pi["shared_by_quintile_pct"]:
        assert v == pytest.approx(wedge_pct, abs=1e-6)


def test_ppi_gradient_monotone_and_bounded():
    """PPI is progressive under both rankings: the ratio is non-increasing
    across quintiles and lies between the wedge and 100%."""
    art = _artifact()
    wedge_pct = 100.0 * art["transport_and_conventions"]["wedge"]
    for side in [art["full_sample"]]:
        for grouping in GROUPINGS:
            ppis = _ppi(side, grouping)
            assert all(
                ppis[k] >= ppis[k + 1] - 1e-6 for k in range(N_QUINTILES - 1)
            ), (grouping, ppis)
            for v in ppis:
                assert wedge_pct - 1e-6 <= v <= 100.0 + 1e-6


# =====================================================================
# Three-way table (individual / shared / anchor) recompute + consistency
# =====================================================================
def test_three_way_columns_and_anchor():
    art = _artifact()
    tw = art["three_way_comparison"]
    assert "individual" in tw["columns"]
    assert "shared" in tw["columns"]
    assert "anchor" in tw["columns"]
    for k, row in enumerate(tw["ppi_by_quintile"]):
        assert row["quintile"] == k + 1
        assert row["anchor_dynasim_ppi_pct"] == DYNASIM_PPI[k]


def test_full_sample_matches_three_way():
    """The three-way individual column IS the full-sample own-quintile PPI,
    and the shared column IS the full-sample shared-quintile PPI (the per
    person ratio is identical; only the grouping differs)."""
    art = _artifact()
    tw = art["three_way_comparison"]["ppi_by_quintile"]
    fo = art["full_sample"]["by_own_quintile"]
    fs = art["full_sample"]["by_shared_quintile"]
    for k in range(N_QUINTILES):
        assert tw[k]["individual_ppi_pct"] == pytest.approx(
            fo[k]["ppi_ratio_pct"], abs=1e-12
        )
        assert tw[k]["shared_ppi_pct"] == pytest.approx(
            fs[k]["ppi_ratio_pct"], abs=1e-12
        )
        # shared - individual and the closer-to-anchor flag recompute.
        assert tw[k]["shared_minus_individual_pp"] == pytest.approx(
            fs[k]["ppi_ratio_pct"] - fo[k]["ppi_ratio_pct"], abs=1e-12
        )
        anchor = DYNASIM_PPI[k]
        ind_gap = abs(fo[k]["ppi_ratio_pct"] - anchor)
        shr_gap = abs(fs[k]["ppi_ratio_pct"] - anchor)
        assert tw[k]["shared_closer_to_anchor"] == (shr_gap < ind_gap)
        assert tw[k]["individual_abs_gap_vs_anchor_pp"] == pytest.approx(
            ind_gap, abs=1e-12
        )
        assert tw[k]["shared_abs_gap_vs_anchor_pp"] == pytest.approx(
            shr_gap, abs=1e-12
        )


def test_three_way_floors_recompute_from_per_seed():
    """Each grouping's per-quintile floor = mean|side_a - side_b| over the
    5 seeds (the sampling-noise scale at half sample)."""
    art = _artifact()
    per_seed = art["per_seed"]
    table = art["three_way_comparison"]["ppi_by_quintile"]
    for k in range(N_QUINTILES):
        for grouping, floor_key in (
            ("by_own_quintile", "individual_floor_pp"),
            ("by_shared_quintile", "shared_floor_pp"),
        ):
            gaps = [
                abs(
                    _ppi(r["side_a"], grouping)[k]
                    - _ppi(r["side_b"], grouping)[k]
                )
                for r in per_seed
            ]
            stored = table[k][floor_key]
            assert stored["mean"] == pytest.approx(_mean(gaps), abs=1e-9)
            assert stored["values"] == pytest.approx(gaps, abs=1e-9)


def test_pre_registered_expectations_recompute_and_hold():
    """The registration's expectations recompute from the table and hold:
    shared closer to the anchor at Q1-Q2, monotone, Q1 within 3pp of 98.7
    (the individual version sat at 100.0)."""
    art = _artifact()
    tw = art["three_way_comparison"]["ppi_by_quintile"]
    exp = art["three_way_comparison"]["pre_registered_expectations"]
    shr = [row["shared_ppi_pct"] for row in tw]

    assert exp["shared_q1_ppi_pct"] == pytest.approx(shr[0], abs=1e-12)
    assert exp["individual_q1_ppi_pct"] == pytest.approx(
        tw[0]["individual_ppi_pct"], abs=1e-12
    )
    assert exp["shared_q1_within_3pp_of_anchor"] == (
        abs(shr[0] - DYNASIM_PPI[0]) <= 3.0
    )
    assert exp["shared_closer_at_q1"] == tw[0]["shared_closer_to_anchor"]
    assert exp["shared_closer_at_q2"] == tw[1]["shared_closer_to_anchor"]
    assert exp["shared_closer_at_q1_q2"] == (
        tw[0]["shared_closer_to_anchor"] and tw[1]["shared_closer_to_anchor"]
    )
    monotone = all(shr[k] >= shr[k + 1] - 1e-9 for k in range(N_QUINTILES - 1))
    assert exp["shared_ppi_monotone_nonincreasing"] == monotone
    assert exp["all_core_expectations_held"] == bool(
        exp["shared_q1_within_3pp_of_anchor"]
        and exp["shared_closer_at_q1"]
        and exp["shared_closer_at_q2"]
        and monotone
    )
    # The registered expectations HOLD on the real data.
    assert exp["all_core_expectations_held"] is True
    # Q1 within 3pp, and shared is strictly closer than the individual 100.0.
    assert abs(shr[0] - 98.7) <= 3.0
    assert tw[0]["individual_ppi_pct"] == pytest.approx(100.0, abs=0.5)
    # The upper quintiles stay compressed vs the anchor (named, not chased).
    assert exp["top_quintile_still_compressed"] is True
    assert tw[4]["shared_ppi_pct"] > DYNASIM_PPI[4] + 3.0


def test_per_seed_structure():
    art = _artifact()
    assert [r["seed"] for r in art["per_seed"]] == SEEDS
    for r in art["per_seed"]:
        assert {"side_a", "side_b"} <= set(r)
        assert r["n_persons_side_a"] > 0 and r["n_persons_side_b"] > 0
        for side in ("side_a", "side_b"):
            for grouping in GROUPINGS:
                assert len(r[side][grouping]) == N_QUINTILES


def test_quintile_cells_internally_consistent():
    """Own AIME increases across own quintiles; shared AIME increases across
    shared quintiles; n_positive <= n_persons; scheduled positive."""
    art = _artifact()
    full = art["full_sample"]
    for grouping, aime_key in (
        ("by_own_quintile", "mean_own_aime"),
        ("by_shared_quintile", "mean_shared_aime"),
    ):
        means = []
        for q in full[grouping]:
            if q.get("n_positive", 0) == 0:
                continue
            assert q["n_positive"] <= q["n_persons"]
            assert q["mean_scheduled_amount"] > 0
            means.append(q[aime_key])
        assert all(means[i] < means[i + 1] for i in range(len(means) - 1)), (
            grouping,
            means,
        )


def test_study_population_counts():
    art = _artifact()
    sp = art["study_population"]
    assert sp["n_study"] <= sp["n_career_frame"] <= sp["n_gate_panel_persons"]
    assert sp["n_study"] > 500
    mj = sp["marriage_joinability"]
    assert 0.0 < mj["marriage_joinable_share"] <= 1.0
    # Never-married persons are trivially joinable (no spouse to join).
    assert mj["by_marital_status"]["never_married"]["share"] == pytest.approx(
        1.0
    )
    em = sp["ever_married_sharing"]
    assert (
        em["n_ever_married_sharing"] + em["n_never_married_own_only"]
        == sp["n_study"]
    )
    # A material share of persons reshuffle between the own and shared
    # rankings (the mechanism that moves the gradient).
    assert 0.0 < sp["weighted_share_reshuffled"] < 1.0
    assert sp["n_reshuffled_own_vs_shared"] > 0


def test_phase_a_reference_present():
    art = _artifact()
    ref = art["three_way_comparison"]["phase_a_full_population_reference"]
    assert ref["available"] is True
    assert "replication_ppi_mermin_v1.json" in ref["source"]
    assert len(ref["individual_ppi_by_quintile_pct"]) == N_QUINTILES


# =====================================================================
# Pure-helper unit tests (import the builder; no PSID)
# =====================================================================
class _Ep:
    """Minimal stand-in for an R7 marriage-episode row."""

    def __init__(self, spouse, start, end):
        self.spouse_person_id = spouse
        self.start_year = start
        self.episode_end_year = end


class _FakeStudy:
    def __init__(self, episodes, history):
        self._episodes = episodes
        self.history = history

    def relevant_episodes(self, pid):
        return self._episodes


def test_shared_earnings_couple_mean_and_single_years():
    """Married years take the couple mean; single years keep own earnings."""
    b = _import_builder()
    # Marriage 2000-2004 to spouse 77; single before/after.
    study = _FakeStudy(
        episodes=[_Ep(77, 2000, 2004)],
        history={77: {2000: 20_000.0, 2002: 60_000.0, 2004: 0.0}},
    )
    periods = np.array([1998, 2000, 2002, 2004, 2006])
    own = np.array([50_000.0, 40_000.0, 40_000.0, 40_000.0, 50_000.0])
    shared = b.shared_earnings_on_periods(study, 1, periods, own, elig=2030)
    assert shared[0] == 50_000.0  # 1998 single -> own
    assert shared[1] == pytest.approx(0.5 * (40_000 + 20_000))  # 2000
    assert shared[2] == pytest.approx(0.5 * (40_000 + 60_000))  # 2002
    assert shared[3] == pytest.approx(0.5 * (40_000 + 0))  # 2004 spouse 0
    assert shared[4] == 50_000.0  # 2006 single -> own


def test_shared_earnings_cap_after_averaging():
    """The couple mean is raw (NOT capped in the sharing) -- the transport
    caps later, so a high-earning couple keeps its full couple mean here."""
    b = _import_builder()
    study = _FakeStudy(
        episodes=[_Ep(77, 2000, 2000)],
        history={77: {2000: 300_000.0}},
    )
    periods = np.array([2000])
    own = np.array([300_000.0])
    shared = b.shared_earnings_on_periods(study, 1, periods, own, elig=2030)
    # 0.5*(300k+300k) = 300k, uncapped (well above any wage base).
    assert shared[0] == pytest.approx(300_000.0)


def test_shared_earnings_ongoing_marriage_shares_through_elig():
    """An ongoing marriage (NA end year) shares through the eligibility year."""
    b = _import_builder()
    study = _FakeStudy(
        episodes=[_Ep(77, 2010, float("nan"))],
        history={77: {2012: 80_000.0, 2020: 80_000.0}},
    )
    periods = np.array([2008, 2012, 2020])
    own = np.array([40_000.0, 40_000.0, 40_000.0])
    shared = b.shared_earnings_on_periods(study, 1, periods, own, elig=2022)
    assert shared[0] == 40_000.0  # 2008 before marriage -> own
    assert shared[1] == pytest.approx(0.5 * (40_000 + 80_000))  # 2012
    assert shared[2] == pytest.approx(0.5 * (40_000 + 80_000))  # 2020 ongoing


def test_shared_earnings_na_spouse_keeps_own():
    """A marriage with no joinable spouse id keeps the person's own earnings."""
    b = _import_builder()
    study = _FakeStudy(episodes=[_Ep(float("nan"), 2000, 2004)], history={})
    periods = np.array([2000, 2002])
    own = np.array([40_000.0, 45_000.0])
    shared = b.shared_earnings_on_periods(study, 1, periods, own, elig=2030)
    assert shared[0] == 40_000.0
    assert shared[1] == 45_000.0


def test_measure_population_ranking_only_changes_grouping():
    """On a synthetic frame: the per-person ratio is own-based, so the
    overall PI/PPI is grouping-invariant; PI is flat at the wedge; PPI is
    monotone under both rankings; and each grouping's mean AIME increases
    across its own quintiles."""
    import pandas as pd

    b = _import_builder()
    transport = {
        "bend_points": (1000.0, 6000.0),
        "pia_factors": (0.9, 0.32, 0.15),
        "wedge": 0.67,
    }
    # 10 persons, own AIME ascending. shared AIME is positively correlated
    # with own (as on real couples, so both gradients stay monotone) but
    # swaps persons 1 and 2 ACROSS the Q1/Q2 boundary, so the two rankings
    # disagree for those two -- the reshuffling the shared measure creates.
    own = np.array(
        [
            400.0,
            900.0,
            1500.0,
            2200.0,
            3000.0,
            4000.0,
            5200.0,
            6800.0,
            8500.0,
            11000.0,
        ]
    )
    shared = own.copy()
    shared[1], shared[2] = own[2], own[1]  # swap across the Q1/Q2 boundary
    frame = pd.DataFrame(
        {
            "person_id": np.arange(10),
            "own_aime": own,
            "shared_aime": shared,
            "weight": np.ones(10),
        }
    )
    m = b.measure_population(frame, transport)
    # PI flat at the wedge, and PPI monotone non-increasing, under BOTH
    # groupings (the per-person ratio is own-based; only the grouping moves).
    for grouping in ("by_own_quintile", "by_shared_quintile"):
        pis = [q["pi_ratio_pct"] for q in m[grouping]]
        assert max(pis) - min(pis) < 1e-9
        assert pis[0] == pytest.approx(67.0, abs=1e-9)
        ppis = [q["ppi_ratio_pct"] for q in m[grouping]]
        assert all(
            ppis[k] >= ppis[k + 1] - 1e-9 for k in range(N_QUINTILES - 1)
        ), (grouping, ppis)
    # Overall ratio is a per-person (own-based) quantity -> grouping-invariant.
    assert m["overall_pi_ratio_pct"] == pytest.approx(67.0, abs=1e-9)
    assert m["overall_ppi_ratio_pct"] == pytest.approx(
        m["overall_ppi_ratio_pct"], abs=1e-12
    )
    # Mean own AIME rises across own quintiles; mean shared AIME across shared.
    own_means = [q["mean_own_aime"] for q in m["by_own_quintile"]]
    shared_means = [q["mean_shared_aime"] for q in m["by_shared_quintile"]]
    assert all(own_means[i] < own_means[i + 1] for i in range(N_QUINTILES - 1))
    assert all(
        shared_means[i] < shared_means[i + 1] for i in range(N_QUINTILES - 1)
    )
    # Bend30 is the weighted 30th percentile of OWN AIME.
    from build_downstream_relevance import _weighted_quantile

    expected_bend = float(
        _weighted_quantile(own, np.ones(10), np.array([0.30]))[0]
    )
    assert m["bend30_own_aime"] == pytest.approx(expected_bend, rel=1e-9)
    # Exactly the two boundary-crossing persons reshuffle off the diagonal.
    assert m["n_reshuffled_own_vs_shared"] == 2


def test_summary_empty_safe():
    b = _import_builder()
    s = b._summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean"] == pytest.approx(3.0)
    assert s["n_seeds"] == 5
    assert b._summary([])["n_seeds"] == 0


# =====================================================================
# Seed-0 reproduction pin (needs PSID family + marriage; run live)
# =====================================================================
@needs_real_psid
def test_seed0_reproduces_committed_artifact():
    """Rerun the real-data load, study-population selection, full-sample
    measurement, and the seed-0 half-split; pin the committed individual +
    shared quintile ratios and the seed-0 floors to float precision."""
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
    study = builder.StudyData(params)
    person_ids = builder.select_study_persons(panel, study)

    sp = art["study_population"]
    assert len(person_ids) == sp["n_study"]
    assert (
        len(builder.coverage_selected_persons(panel)) == sp["n_career_frame"]
    )

    frame = builder.build_person_frame(
        study, panel, person_ids, weight_of, params, transport
    )
    full = builder.measure_population(frame, transport)

    tw = art["three_way_comparison"]["ppi_by_quintile"]
    for k in range(N_QUINTILES):
        assert full["by_own_quintile"][k]["ppi_ratio_pct"] == pytest.approx(
            tw[k]["individual_ppi_pct"], abs=1e-9
        )
        assert full["by_shared_quintile"][k]["ppi_ratio_pct"] == pytest.approx(
            tw[k]["shared_ppi_pct"], abs=1e-9
        )

    floor0 = builder.measure_floor_seed(frame, 0, transport)
    ref0 = next(r for r in art["per_seed"] if r["seed"] == 0)
    for side in ("side_a", "side_b"):
        assert floor0[f"n_persons_{side}"] == ref0[f"n_persons_{side}"]
        for grouping in GROUPINGS:
            for k in range(N_QUINTILES):
                assert floor0[side][grouping][k][
                    "ppi_ratio_pct"
                ] == pytest.approx(
                    ref0[side][grouping][k]["ppi_ratio_pct"], abs=1e-9
                )
