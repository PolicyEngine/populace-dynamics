"""Tests for the Mermin claiming/COLA-row replication vs Mermin (2005).

Artifact: runs/replication_mermin_rows_v1.json. Frozen spec: issue #42
comment 4911609804. REPORTED, NOT GATED.

Two tiers, mirroring the other run/floor tests:

* Always-runnable internal-consistency tests (no PSID) that touch only the
  committed artifact plus the pure helpers in
  :mod:`scripts.replication_mermin_rows`: the schema is sane and marked
  reported-not-gated; the registration pointer and the anchor provenance
  (the two provisions' mechanics with page citations and the transcribed
  Table 2 / Table 4 rows) are present; the NRA and COLA tables' anchor
  cells equal the transcription and their floors and the pre-registered
  expectation all RECOMPUTE from the stored per-seed and full-sample
  values. Pure helpers reproduce the actuarial factor at an imposed NRA
  (and its equality with claiming.benefit_factor at the FRA-67 cohort),
  the expected NRA factors, the NCHS x PSID-band survival, and the COLA
  coefficients + percent.
* A seed-0 reproduction pin (skipped without the PSID family + marriage
  files) that reruns the real-data load, scoring, and the seed-0
  half-split through the build machinery and pins the committed rows and
  counts to float precision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "replication_mermin_rows_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_psid = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir()
    or not (REAL_DATA / "mh85_23").is_dir(),
    reason="PSID family/marriage files not staged",
)

ANCHOR_NRA_62_67_BY_QUINTILE = (79.4, 79.5, 79.6, 79.8, 79.9)
ANCHOR_NRA_62_67_ALL = 79.7
ANCHOR_COLA_62_67_ALL = 98.9
ANCHOR_COLA_80_85_ALL = 92.4


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _import_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import replication_mermin_rows as builder

    return builder


class _FakeParams:
    """Statutory actuarial constants, hand-checkable and independent of a
    policyengine-us checkout: 402(q) 5/9 of 1% (first 36 months) then 5/12
    of 1%; 402(w) 8% per year; FRA 67; max 48 months of delay."""

    early_monthly_rates = (5.0 / 900.0, 5.0 / 1200.0)
    early_first_bracket_months = 36
    max_delayed_months = 48

    def fra_months(self, birth_year: int) -> int:
        return 67 * 12

    def delayed_credit_annual_rate(self, birth_year: int) -> float:
        return 0.08


# =====================================================================
# Schema and reported-not-gated framing (always runnable)
# =====================================================================
def test_schema_and_reported_not_gated():
    art = _artifact()
    assert art["schema_version"] == "replication_mermin_rows.v1"
    assert art["run"] == "replication_mermin_rows_v1"
    assert art["reported_not_gated"] is True
    assert art["real_data_only"] is True
    assert "changes no gate" in art["purpose"]
    assert art["registration_pointer"] == "#42 comment 4911609804"
    assert art["registration"].endswith("4911609804")
    assert "issues/74" in art["program_design_issue"]


def test_revision_pins():
    art = _artifact()
    pins = art["revision_pins"]
    assert pins["pe_us_revision"]
    assert pins["artifact_schema_version"] == "replication_mermin_rows.v1"
    assert pins["nchs_life_table_vintage"] == 2023
    assert pins["mortality_window"] == "all"


def test_per_seed_present():
    art = _artifact()
    assert [r["seed"] for r in art["per_seed"]] == [0, 1, 2, 3, 4]
    for r in art["per_seed"]:
        hm = r["half_metrics"]
        assert {"side_a", "side_b"} <= set(hm)
        for side in ("side_a", "side_b"):
            assert len(hm[side]["nra_by_quintile_pct"]) == 5
            assert "nra_overall_pct" in hm[side]
            assert "cola_62_67_pct" in hm[side]
            assert "cola_80_85_pct" in hm[side]


# =====================================================================
# Anchor provenance: two provisions + Table 2/4 transcription + cites
# =====================================================================
def test_anchor_provenance_mechanics_and_citations():
    art = _artifact()
    prov = art["anchor_provenance"]
    assert "411260" in prov["paper"]
    assert "Runid 432" in prov["paper"]
    nra = prov["nra_mechanics"]
    assert "p.2" in nra["citation"]
    assert "age 70" in nra["quote_scenario"]
    assert "disability benefits are unaffected" in nra["disability_unaffected"]
    assert "p.7" in nra["disability_unaffected"]
    cola = prov["cola_mechanics"]
    assert "fn.6" in cola["citation"]
    assert "2.8 percent" in cola["quote_rates"]
    assert "2.4 percent" in cola["quote_rates"]
    # The COLA-start convention delta (paper's fn.7 age-62 start) is named.
    assert "fn.7" in cola["paper_cola_start_note"]


def test_anchor_table_transcription():
    art = _artifact()
    prov = art["anchor_provenance"]
    t2 = prov["table2_retired_workers_62_67_in_2050"]
    assert "N=5351" in t2["citation"]
    assert tuple(t2["nra_raised_to_70_pct"]["by_quintile"]) == (
        ANCHOR_NRA_62_67_BY_QUINTILE
    )
    assert t2["nra_raised_to_70_pct"]["all"] == ANCHOR_NRA_62_67_ALL
    assert t2["reduced_cola_pct"]["all"] == ANCHOR_COLA_62_67_ALL
    t4 = prov["table4_retired_workers_80_85_in_2050"]
    assert "N=3088" in t4["citation"]
    assert t4["reduced_cola_pct"]["all"] == ANCHOR_COLA_80_85_ALL
    assert t4["reduced_cola_pct"]["table5_all_beneficiary_types"] == 92.0
    # Named deltas present (individual vs shared quintiles; COLA start).
    deltas = prov["named_population_deltas"]
    assert any("own-record" in d for d in deltas)
    assert any("claim age" in d for d in deltas)


# =====================================================================
# NRA table + internal consistency (always runnable)
# =====================================================================
def test_nra_table_anchor_cells_and_recompute():
    art = _artifact()
    table = art["nra_raise_to_70"]["table"]
    assert [row["quintile"] for row in table["by_quintile"]] == [1, 2, 3, 4, 5]
    for k, row in enumerate(table["by_quintile"]):
        assert row["anchor_pct"] == ANCHOR_NRA_62_67_BY_QUINTILE[k]
        assert row["abs_gap_vs_anchor"] == pytest.approx(
            abs(row["our_pct_of_scheduled"] - row["anchor_pct"]), abs=0.02
        )
    ov = table["overall"]
    assert ov["anchor_pct"] == ANCHOR_NRA_62_67_ALL
    # The overall percent-of-scheduled equals the full-sample value.
    fs = art["nra_raise_to_70"]["full_sample"]
    assert ov["our_pct_of_scheduled"] == pytest.approx(
        fs["overall_pct_of_scheduled"], abs=1e-9
    )
    assert ov["cross_quintile_spread_pp"] == pytest.approx(
        fs["cross_quintile_spread_pp"], abs=1e-9
    )


def test_nra_row_is_near_flat_and_low_gap():
    """The NRA reduction factor is PIA-independent, so the row is nearly
    flat across AIME quintiles and close to the anchor's flat 79.7."""
    art = _artifact()
    table = art["nra_raise_to_70"]["table"]
    shares = [row["our_pct_of_scheduled"] for row in table["by_quintile"]]
    assert max(shares) - min(shares) < 3.0
    assert abs(table["overall"]["our_pct_of_scheduled"] - 79.7) < 3.0


def test_nra_floors_recompute_from_per_seed():
    art = _artifact()
    per_seed = art["per_seed"]
    table = art["nra_raise_to_70"]["table"]
    for k, row in enumerate(table["by_quintile"]):
        gaps = [
            abs(
                s["half_metrics"]["side_a"]["nra_by_quintile_pct"][k]
                - s["half_metrics"]["side_b"]["nra_by_quintile_pct"][k]
            )
            for s in per_seed
        ]
        assert row["floor_mean"] == pytest.approx(
            float(np.mean(gaps)), abs=1e-9
        )
    gaps_ov = [
        abs(
            s["half_metrics"]["side_a"]["nra_overall_pct"]
            - s["half_metrics"]["side_b"]["nra_overall_pct"]
        )
        for s in per_seed
    ]
    assert table["overall"]["floor_mean"] == pytest.approx(
        float(np.mean(gaps_ov)), abs=1e-9
    )


# =====================================================================
# COLA table + internal consistency (always runnable)
# =====================================================================
def test_cola_table_anchor_cells_and_recompute():
    art = _artifact()
    table = art["cola_minus_0_4pp"]["table"]
    assert [row["age_group"] for row in table] == ["62-67", "80-85"]
    by_group = {row["age_group"]: row for row in table}
    assert by_group["62-67"]["anchor_pct"] == ANCHOR_COLA_62_67_ALL
    assert by_group["62-67"]["survival_weighted"] is False
    assert by_group["80-85"]["anchor_pct"] == ANCHOR_COLA_80_85_ALL
    assert by_group["80-85"]["survival_weighted"] is True
    for row in table:
        assert row["abs_gap_vs_anchor"] == pytest.approx(
            abs(row["our_pct_of_scheduled"] - row["anchor_pct"]), abs=0.02
        )


def test_cola_pattern_small_at_claim_compounds_with_age():
    """COLA has little effect at claim (62-67 high) and compounds with age
    (80-85 materially lower) -- the anchor's qualitative pattern."""
    art = _artifact()
    by_group = {
        row["age_group"]: row["our_pct_of_scheduled"]
        for row in art["cola_minus_0_4pp"]["table"]
    }
    assert by_group["62-67"] > by_group["80-85"]
    assert by_group["62-67"] >= 97.5
    assert 90.0 <= by_group["80-85"] <= 94.0


def test_cola_floors_recompute_from_per_seed():
    art = _artifact()
    per_seed = art["per_seed"]
    by_group = {
        row["age_group"]: row for row in art["cola_minus_0_4pp"]["table"]
    }
    for group, key in (
        ("62-67", "cola_62_67_pct"),
        ("80-85", "cola_80_85_pct"),
    ):
        gaps = [
            abs(
                s["half_metrics"]["side_a"][key]
                - s["half_metrics"]["side_b"][key]
            )
            for s in per_seed
        ]
        assert by_group[group]["floor_mean"] == pytest.approx(
            float(np.mean(gaps)), abs=1e-9
        )


# =====================================================================
# Registered expectation (always runnable)
# =====================================================================
def test_registered_expectation_recomputes_and_holds():
    art = _artifact()
    e = art["registered_expectation"]
    nra_mean = art["nra_raise_to_70"]["full_sample"][
        "overall_pct_of_scheduled"
    ]
    spread = art["nra_raise_to_70"]["full_sample"]["cross_quintile_spread_pp"]
    cola = {
        row["age_group"]: row["our_pct_of_scheduled"]
        for row in art["cola_minus_0_4pp"]["table"]
    }
    assert e["nra_mean_77_83_spread_lt_3pp"]["held"] == (
        77.0 <= nra_mean <= 83.0 and spread < 3.0
    )
    assert e["cola_62_67_ge_97_5"]["held"] == (cola["62-67"] >= 97.5)
    assert e["cola_80_85_in_90_94"]["held"] == (90.0 <= cola["80-85"] <= 94.0)
    assert e["all_held"] is True
    assert e["all_held"] == (
        e["nra_mean_77_83_spread_lt_3pp"]["held"]
        and e["cola_62_67_ge_97_5"]["held"]
        and e["cola_80_85_in_90_94"]["held"]
    )


def test_study_population_counts():
    art = _artifact()
    sp = art["study_population"]
    assert sp["n_career_frame"] > 1000
    assert sp["n_scored"] <= sp["n_career_frame"]
    assert sp["n_scored"] == sp["n_by_sex"]["male"] + sp["n_by_sex"]["female"]
    assert sp["n_by_sex"]["male"] > 0 and sp["n_by_sex"]["female"] > 0


# =====================================================================
# Pure-helper unit tests (import the builder; no PSID)
# =====================================================================
def test_benefit_factor_at_fra_equals_claiming_at_fra67():
    """The imposed-NRA helper at FRA 67 equals claiming.benefit_factor at
    the FRA-67 cohort for every drawn claim age -- the faithful reuse."""
    b = _import_builder()
    from populace_dynamics import claiming

    params = _FakeParams()
    for age in range(62, 71):
        assert b.benefit_factor_at_fra(
            age * 12, b.BASELINE_FRA_MONTHS, params
        ) == pytest.approx(
            claiming.benefit_factor(
                age * 12, b.FRA67_COHORT_BIRTH_YEAR, params
            )
        )


def test_benefit_factor_at_fra_known_values():
    b = _import_builder()
    params = _FakeParams()
    # Claim 62 against FRA 70: 96 months early = 36*(5/9%) + 60*(5/12%)
    # = 20% + 25% = 45% reduction -> factor 0.55.
    assert b.benefit_factor_at_fra(62 * 12, 70 * 12, params) == pytest.approx(
        0.55
    )
    # Claim 67 against FRA 70: 36 months early = 20% -> 0.80.
    assert b.benefit_factor_at_fra(67 * 12, 70 * 12, params) == pytest.approx(
        0.80
    )
    # Claim 70 against FRA 70: exactly at the NRA -> 1.0 (no DRC reachable).
    assert b.benefit_factor_at_fra(70 * 12, 70 * 12, params) == 1.0
    # Claim 62 against FRA 67: 60 months early = 20% + 10% = 30% -> 0.70.
    assert b.benefit_factor_at_fra(62 * 12, 67 * 12, params) == pytest.approx(
        0.70
    )
    # Claim 70 against FRA 67: 3 years delayed at 8% -> 1.24.
    assert b.benefit_factor_at_fra(70 * 12, 67 * 12, params) == pytest.approx(
        1.24
    )


def test_reform_factor_below_baseline_for_early_claims():
    """Raising the NRA from 67 to 70 deepens the cut for anyone claiming
    before 70, so the reform factor is below the baseline factor."""
    b = _import_builder()
    params = _FakeParams()
    for age in range(62, 70):
        reform = b.benefit_factor_at_fra(age * 12, 70 * 12, params)
        baseline = b.benefit_factor_at_fra(age * 12, 67 * 12, params)
        assert reform < baseline


def test_expected_nra_factors_matches_pmf_weighted_sum():
    b = _import_builder()
    from populace_dynamics import claiming

    params = _FakeParams()
    pmf = {62: 0.5, 65: 0.3, 70: 0.2}
    baseline, reform = b.expected_nra_factors(pmf, params)
    exp_base = sum(
        p * claiming.benefit_factor(a * 12, b.FRA67_COHORT_BIRTH_YEAR, params)
        for a, p in pmf.items()
    )
    exp_reform = sum(
        p * b.benefit_factor_at_fra(a * 12, 70 * 12, params)
        for a, p in pmf.items()
    )
    assert baseline == pytest.approx(exp_base)
    assert reform == pytest.approx(exp_reform)
    # Percent of scheduled is well inside the pre-registered 77-83 band.
    assert 77.0 <= 100.0 * reform / baseline <= 83.0


def test_survival_from_committed_artifacts():
    b = _import_builder()
    surv = b.Survival()
    # Survival to the start age is 1 (empty product), and it strictly
    # decreases with age.
    for sex in ("male", "female"):
        assert surv.survival(sex, 62, 62) == 1.0
        chain = [surv.survival(sex, 62, a) for a in range(62, 86)]
        assert all(chain[i] > chain[i + 1] for i in range(len(chain) - 1))
        assert 0.0 < chain[-1] < 1.0
    # adjusted_qx = NCHS q_x * PSID/NCHS band ratio (ratio < 1 undercount).
    raw = surv.qx["male"][80]
    adj = surv.adjusted_qx("male", 80)
    ratio = surv.ratios["75-84|male"]
    assert adj == pytest.approx(min(1.0, raw * ratio))
    assert ratio < 1.0


def test_survival_band_mapping():
    b = _import_builder()
    surv = b.Survival()
    assert surv._band(62) == "55-64"
    assert surv._band(70) == "65-74"
    assert surv._band(84) == "75-84"
    assert surv._band(90) == "85+"


def test_cola_group_coefficients_closed_form():
    b = _import_builder()
    surv = b.Survival()
    # Everyone claims at 62; unweighted 62-67 group: at each evaluation age
    # a the only claim age is 62, mass 1, ratio COLA_RATIO**(a-62).
    pmf = {62: 1.0}
    numer, denom = b.cola_group_coefficients(
        pmf, "male", b.AGE_GROUP_62_67, surv, survival_weighted=False
    )
    exp_numer = sum(b.COLA_RATIO ** (a - 62) for a in range(62, 68))
    assert denom == pytest.approx(6.0)
    assert numer == pytest.approx(exp_numer)
    # A later evaluation age compounds the cut further: the 80-85 group's
    # per-unit percent is below the 62-67 group's.
    n2, d2 = b.cola_group_coefficients(
        pmf, "male", b.AGE_GROUP_80_85, surv, survival_weighted=False
    )
    assert 100.0 * n2 / d2 < 100.0 * numer / denom


def test_cola_pct_closed_form():
    import pandas as pd

    b = _import_builder()
    # Two persons sharing one (sex, elig) cell; the percent equals
    # 100 * A/B regardless of weights when the cell is common.
    coeffs = {("male", 2010): {"62_67": (0.99 * 6.0, 6.0)}}
    df = pd.DataFrame(
        {
            "person_id": [1, 2],
            "sex": ["male", "male"],
            "elig_year": [2010, 2010],
            "weight": [3.0, 1.0],
        }
    )
    assert b.cola_pct(df, coeffs, "62_67") == pytest.approx(99.0)


def test_nra_quintile_metrics_closed_form():
    import pandas as pd

    b = _import_builder()
    # Ten persons, AIME 1..10 -> two per quintile; constant factors give a
    # flat 80% of scheduled everywhere.
    df = pd.DataFrame(
        {
            "person_id": list(range(10)),
            "weight": [1.0] * 10,
            "base_aime": [float(i) for i in range(1, 11)],
            "nra_baseline_factor": [0.80] * 10,
            "nra_reform_factor": [0.64] * 10,
        }
    )
    m = b.nra_quintile_metrics(df)
    assert m["overall_pct_of_scheduled"] == pytest.approx(80.0)
    assert m["cross_quintile_spread_pp"] == pytest.approx(0.0)
    aimes = [q["mean_aime"] for q in m["by_quintile"]]
    assert all(aimes[i] < aimes[i + 1] for i in range(4))


def test_cola_ratio_below_one():
    b = _import_builder()
    # The reform COLA (2.4%) is below the baseline (2.8%), so each year
    # since claim the reform benefit is a bit below scheduled.
    assert b.COLA_RATIO == pytest.approx((1.0 + 0.024) / (1.0 + 0.028))
    assert b.COLA_RATIO < 1.0


# =====================================================================
# Seed-0 reproduction pin (needs PSID; run live)
# =====================================================================
@needs_real_psid
def test_seed0_reproduces_committed_artifact():
    """Rerun the real-data load, scoring, and the seed-0 half-split; match
    the committed rows and counts to float precision."""
    builder = _import_builder()
    art = _artifact()

    params = builder.load_ssa_parameters()
    if params.pe_us_revision != art["revision_pins"]["pe_us_revision"]:
        pytest.skip(
            f"policyengine-us at {params.pe_us_revision} differs from the "
            f"artifact's pinned {art['revision_pins']['pe_us_revision']}"
        )
    transport = builder.build_transport(params)
    survival = builder.Survival()
    study = builder.MerminStudy(params, transport)
    df = builder.score_population(study, params, transport)
    coeffs = builder._coefficient_table(df, survival)

    sp = art["study_population"]
    assert len(study.careers) == sp["n_career_frame"]
    assert len(df) == sp["n_scored"]

    # Full-sample NRA overall + COLA age-group percents pin.
    nra = builder.nra_quintile_metrics(df)
    assert nra["overall_pct_of_scheduled"] == pytest.approx(
        art["nra_raise_to_70"]["full_sample"]["overall_pct_of_scheduled"],
        abs=1e-6,
    )
    by_group = {
        row["age_group"]: row["our_pct_of_scheduled"]
        for row in art["cola_minus_0_4pp"]["table"]
    }
    assert builder.cola_pct(df, coeffs, "62_67") == pytest.approx(
        by_group["62-67"], abs=1e-6
    )
    assert builder.cola_pct(df, coeffs, "80_85") == pytest.approx(
        by_group["80-85"], abs=1e-6
    )

    # Seed-0 half-split pin (the floor inputs).
    seed0 = builder.seed_half_metrics(df, coeffs, 0)
    ref0 = next(s for s in art["per_seed"] if s["seed"] == 0)["half_metrics"]
    for side in ("side_a", "side_b"):
        assert seed0[side]["n"] == ref0[side]["n"]
        assert seed0[side]["nra_overall_pct"] == pytest.approx(
            ref0[side]["nra_overall_pct"], abs=1e-6
        )
        for k in range(5):
            assert seed0[side]["nra_by_quintile_pct"][k] == pytest.approx(
                ref0[side]["nra_by_quintile_pct"][k], abs=1e-6
            )
        for key in ("cola_62_67_pct", "cola_80_85_pct"):
            assert seed0[side][key] == pytest.approx(ref0[side][key], abs=1e-6)
