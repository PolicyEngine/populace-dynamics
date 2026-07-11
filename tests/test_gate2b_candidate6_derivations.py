"""Pure delta derivations for gate-2b candidate 6 (synthetic; no PSID).

Unit-tests the four candidate-6 deltas' fitters and apply helpers on synthetic
frames: the 0-4 not-married custodial revert (delta 1), the single-year 18-30
child-age home-exit hazard refit + the maternal own-birth leave-year refit
(delta 2), the female 25-44 cohabitation refit (delta 3), and the count-
conditional non-family bridge (delta 4). Always runnable.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from populace_dynamics.models import household_composition_sim_v6 as hcs6


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
def test_constants_and_stream_tags():
    assert hcs6.GRANDCHILD_LO == 55  # coupling stays 55+ (no downward extend)
    assert hcs6.CORE_SIZE_CAP == 5
    assert hcs6.CUSTODIAL_REVERT_BAND == (0, 4)
    assert hcs6.CHILD_EXIT_REFIT_LO == 18
    assert hcs6.CHILD_EXIT_REFIT_HI == 30
    assert hcs6.COHAB_FEMALE_REFIT_LO == 25
    assert hcs6.COHAB_FEMALE_REFIT_HI == 44
    assert hcs6.DELTA_STREAM_TAG_V5 == 0xC5  # carried
    assert hcs6.DELTA_STREAM_TAG_V6 == 0xC6  # new, isolated
    # the biennial leave-grid ages inside the 18-30 refit window
    assert hcs6._REFIT_GRID_AGES == (19, 21, 23, 25, 27, 29)


# --------------------------------------------------------------------------
# Delta 1: 0-4 not-married custodial revert to the observable basis
# --------------------------------------------------------------------------
def _mock_model(child_record, custodial_overall=0.5, custodial_era=None):
    base_v4 = SimpleNamespace(
        custodial_era=custodial_era or {},
        custodial_age_marital={},
        custodial_band_marital={},
        custodial_overall=custodial_overall,
    )
    return SimpleNamespace(
        custodial_child_record=child_record, base_v4=base_v4
    )


def test_custodial_prob_v6_reverts_zero_four_not_married_to_observable():
    cr = {
        ("0-4", "not_married"): 0.90,  # child-record (artifact-inflated)
        ("5-12", "not_married"): 0.40,
        ("13-17", "not_married"): 0.30,
    }
    # observable at age 2 (era-keyed) is BELOW the child-record 0.90.
    era = hcs6.hcs4.era_of_year(2015)
    model = _mock_model(cr, custodial_era={(2, era, "not_married"): 0.70})
    # delta 1: at 0-4 the not-married cell reverts to the observable basis.
    assert hcs6.custodial_prob_v6(model, 2, era, "not_married") == 0.70
    # 5-17 not-married KEEPS the child-record swap.
    model2 = _mock_model(cr, custodial_overall=0.55)
    assert hcs6.custodial_prob_v6(model2, 8, era, "not_married") == 0.40
    assert hcs6.custodial_prob_v6(model2, 15, era, "not_married") == 0.30
    # married always uses the observable basis (never child-record).
    assert hcs6.custodial_prob_v6(model2, 8, era, "married") == 0.55


# --------------------------------------------------------------------------
# Delta 2: single-year exit hazard + maternal / linked-married leave draws
# --------------------------------------------------------------------------
class _SplineStub:
    """A ParentalExitModel stand-in with a constant hazard."""

    def __init__(self, rate):
        self.rate = rate

    def predict(self, age, is_male):
        return np.full(np.asarray(age).shape, self.rate, dtype=np.float64)


def test_fit_child_exit_single_year_empirical_vs_spline_fallback():
    # Dense 20|male exits at 0.4; sparse 21|male falls back to the spline 0.15.
    rows = []
    for i in range(60):
        rows.append(
            {
                "person_id": i,
                "age": 20,
                "sex": "male",
                "weight": 1.0,
                "coresident_parent": True,
                "has_next": True,
                "next_coresident_parent": i >= 24,  # 24/60 exit => 0.4
            }
        )
    for i in range(5):  # sparse (< _MIN_STRATUM_N) at 21|male
        rows.append(
            {
                "person_id": 100 + i,
                "age": 21,
                "sex": "male",
                "weight": 1.0,
                "coresident_parent": True,
                "has_next": True,
                "next_coresident_parent": True,
            }
        )
    pw = pd.DataFrame(rows)
    table, diag = hcs6.fit_child_exit_single_year(
        pw, _SplineStub(0.15), set(pw["person_id"])
    )
    # 20|male: 24 of 60 exit (next_coresident_parent False) => 0.4 empirical.
    assert abs(table[(20, "male")] - 0.4) < 1e-9
    # 21|male sparse => spline fallback.
    assert abs(table[(21, "male")] - 0.15) < 1e-9
    # every grid age is total (spline-backed).
    for age in range(18, 31):
        for sex in ("female", "male"):
            assert (age, sex) in table


def test_child_leave_years_refit_overrides_only_18_30_window():
    births = pd.DataFrame(
        {"parent_person_id": range(500), "birth_year": [2000] * 500}
    )
    # refit hazard = 1.0 at every grid age in 18-30 => everyone leaves at 19.
    haz = {(a, s): 1.0 for a in range(18, 31) for s in ("female", "male")}
    out = hcs6.child_leave_years_refit(
        births, _SplineStub(0.0), haz, np.random.default_rng(1)
    )
    # first grid age in [18,30] is 19 => leave_year = 2000 + 19.
    assert (out["leave_year"] == 2019).all()
    # empty input is a no-op.
    empty = births.iloc[:0]
    assert (
        len(
            hcs6.child_leave_years_refit(
                empty, _SplineStub(0.0), haz, np.random.default_rng(1)
            )
        )
        == 0
    )


# --------------------------------------------------------------------------
# Delta 3: female single-year cohabitation refit (25-44)
# --------------------------------------------------------------------------
def test_fit_female_cohab_single_year_female_only_and_extends_to_44():
    rows = []
    # 30|female: 100 not-cohabiting at-risk, 20 enter => entry 0.2.
    for i in range(100):
        rows.append(
            {
                "person_id": i,
                "age": 30,
                "band": "25-34",
                "sex": "female",
                "weight": 1.0,
                "has_next": True,
                "cohabiting": False,
                "next_cohabiting": i < 20,
            }
        )
    # 40|female exit: 40 cohabiting, 10 exit => exit 0.25.
    for i in range(40):
        rows.append(
            {
                "person_id": 200 + i,
                "age": 40,
                "band": "35-44",
                "sex": "female",
                "weight": 1.0,
                "has_next": True,
                "cohabiting": True,
                "next_cohabiting": i >= 10,
            }
        )
    pw = pd.DataFrame(rows)
    cohab_flag = pw[["person_id"]].assign(year=0, cohabiting=False)

    def _attach(person_waves, flag):  # bypass the real cohab attach
        return person_waves

    orig = hcs6.hcs2.attach_cohabitation
    hcs6.hcs2.attach_cohabitation = _attach
    try:
        entry, exit_, diag = hcs6.fit_female_cohab_single_year(
            pw, cohab_flag, set(pw["person_id"]), {}, {}
        )
    finally:
        hcs6.hcs2.attach_cohabitation = orig
    assert abs(entry[30] - 0.2) < 1e-9
    assert abs(exit_[40] - 0.25) < 1e-9
    # the refit spans 25-44 (candidate 4 stopped at 34).
    assert 44 in entry and 44 in exit_
    assert set(entry) == set(range(25, 45))


# --------------------------------------------------------------------------
# Delta 4: count-conditional non-family bridge
# --------------------------------------------------------------------------
def _count_frame():
    # core 3 households: half have 0 non-core, quarter 1, quarter 2.
    rows = []
    for i in range(40):
        noncore = 0 if i < 20 else (1 if i < 30 else 2)
        rows.append(
            {
                "person_id": i,
                "year": 0,
                "weight": 1.0,
                "hh_size": 3 + noncore,
                "family_unit_size": 3,
            }
        )
    return pd.DataFrame(rows)


def test_fit_nonfamily_count_by_core_distribution_and_cdf():
    pw = _count_frame()
    fu = pw[["person_id", "year", "family_unit_size"]]
    table, diag = hcs6.fit_nonfamily_count_by_core(
        pw[["person_id", "year", "weight", "hh_size"]],
        fu,
        set(pw["person_id"]),
    )
    vals, cum = table[3]
    # P(0)=0.5, P(1)=0.25, P(2)=0.25 => cumulative 0.5, 0.75, 1.0.
    assert list(vals) == [0, 1, 2]
    assert np.allclose(cum, [0.5, 0.75, 1.0])
    assert abs(diag["noncore_incidence_by_capped_core"]["3"] - 0.5) < 1e-9
    assert cum[-1] == 1.0  # tail guarded against float drift


def test_sample_nonfamily_count_v6_inverse_cdf():
    table = {
        k: (np.array([0], dtype=np.int64), np.array([1.0]))
        for k in range(1, 6)
    }
    # capped core 3: always draw 2 non-core members.
    table[3] = (np.array([0, 2], dtype=np.int64), np.array([0.0, 1.0]))
    model = SimpleNamespace(nonfamily_count_by_core=table)
    pw = pd.DataFrame({"person_id": range(100)})
    core = np.where(np.arange(100) < 50, 3, 1)
    counts = hcs6.sample_nonfamily_count_v6(
        pw, model, np.random.default_rng(0), core
    )
    assert (counts[core == 3] == 2).all()  # core-3 => 2 non-core
    assert (counts[core == 1] == 0).all()  # core-1 => 0 non-core


def test_bridge_feasibility_convolution_matches_hand_computation():
    # core dist: half size-2, half size-3.
    core = {"2": 0.5, "3": 0.5}
    # size-2 core: 0 non-core; size-3 core: half 0, half 2.
    table = {
        2: (np.array([0]), np.array([1.0])),
        3: (np.array([0, 2]), np.array([0.5, 1.0])),
    }
    for k in (1, 4, 5):
        table[k] = (np.array([0]), np.array([1.0]))
    imp = hcs6.bridge_feasibility_convolution(table, core)
    # 0.5 -> hh2; 0.25 -> hh3; 0.25 -> hh5+.
    assert abs(imp["2"] - 0.5) < 1e-9
    assert abs(imp["3"] - 0.25) < 1e-9
    assert abs(imp["5+"] - 0.25) < 1e-9
    assert abs(sum(imp.values()) - 1.0) < 1e-9
