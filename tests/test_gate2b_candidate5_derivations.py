"""Pure delta derivations for gate-2b candidate 5 (synthetic; no PSID).

Unit-tests the three candidate-5 deltas' fitters and apply helpers on synthetic
frames: the multigen--adult-child coupling (delta 1), the not-married
child-record custodial swap (delta 2), and the core-size-conditional non-family
bridge plus the per-ego parent-count composition (delta 3). Always runnable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.models import household_composition_sim_v5 as hcs5


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
def test_constants_and_bands():
    assert hcs5.GRANDCHILD_LO == 55
    assert hcs5.CORE_SIZE_CAP == 5
    assert hcs5.DELTA_STREAM_TAG_V5 == 0xC5
    assert hcs5.COUPLING_AGE_BANDS_55PLUS == ((55, 64), (65, 74), (75, 120))
    assert hcs5.grandchild_age_band(60) == (55, 64)
    assert hcs5.grandchild_age_band(70) == (65, 74)
    assert hcs5.grandchild_age_band(90) == (75, 120)
    assert hcs5.grandchild_age_band(40) is None
    assert hcs5.grandchild_age_band(54) is None


# --------------------------------------------------------------------------
# Delta 1: multigen -- adult-child coupling
# --------------------------------------------------------------------------
def test_fit_multigen_child_coupling_captures_conditional():
    """Multigen 55+ egos carry the coresident child at a far higher rate than
    non-multigen egos -- the coupling the reference exhibits."""
    n = 200
    rows = []
    # 55-64 female: multigen=True -> 90% have a child; multigen=False -> 10%.
    for i in range(n):
        rows.append(
            {
                "person_id": i,
                "age": 60,
                "band": "55-64",
                "sex": "female",
                "multigen": True,
                "coresident_child": i < int(0.9 * n),
                "weight": 1.0,
            }
        )
    for i in range(n):
        rows.append(
            {
                "person_id": n + i,
                "age": 60,
                "band": "55-64",
                "sex": "female",
                "multigen": False,
                "coresident_child": i < int(0.1 * n),
                "weight": 1.0,
            }
        )
    pw = pd.DataFrame(rows)
    table, pooled, diag = hcs5.fit_multigen_child_coupling(
        pw, set(pw["person_id"])
    )
    assert table[("55-64", "female", True)] == pytest.approx(0.9, abs=1e-9)
    assert table[("55-64", "female", False)] == pytest.approx(0.1, abs=1e-9)
    # Coupling: P(child | multigen=True) >> P(child | multigen=False).
    assert (
        table[("55-64", "female", True)]
        > 5 * table[("55-64", "female", False)]
    )
    # Sparse (65-74) falls back to the pooled (sex, multigen) rate.
    assert table[("65-74", "female", True)] == pytest.approx(
        pooled[("female", True)], abs=1e-9
    )
    assert diag["pooled_p_child_given_multigen_true_female"] > 0


def test_coupling_below_55_excluded():
    """Egos below 55 are not in the coupling fit (the 55+ window)."""
    pw = pd.DataFrame(
        {
            "person_id": range(50),
            "age": [40] * 50,
            "band": ["35-44"] * 50,
            "sex": ["female"] * 50,
            "multigen": [True] * 50,
            "coresident_child": [True] * 50,
            "weight": [1.0] * 50,
        }
    )
    table, pooled, _ = hcs5.fit_multigen_child_coupling(
        pw, set(pw["person_id"])
    )
    # No 55+ waves -> pooled rates are 0 and every band falls back to 0.
    assert pooled[("female", True)] == 0.0
    assert table[("55-64", "female", True)] == 0.0


# --------------------------------------------------------------------------
# Delta 2: not-married child-record custodial correction
# --------------------------------------------------------------------------
def test_fit_custodial_child_record_weighted_by_band_marital():
    """The child-record rate is the weighted share coresident with the father,
    by child age band x father marital."""
    expo = pd.DataFrame(
        {
            "father_id": [1, 1, 2, 2, 3, 3],
            "child_person_id": [10, 10, 20, 20, 30, 30],
            "year": [2000, 2001, 2000, 2001, 2000, 2001],
            "child_band": ["5-12"] * 6,
            "weight": [1.0] * 6,
            # not_married: 1 of 4 coresident -> 0.25.
            "with_father": [True, False, False, False, False, True],
            "marital": [
                "not_married",
                "not_married",
                "not_married",
                "not_married",
                "married",
                "married",
            ],
        }
    )
    rates, diag = hcs5.fit_custodial_child_record(expo, {1, 2, 3})
    assert rates[("5-12", "not_married")] == pytest.approx(0.25, abs=1e-9)
    assert rates[("5-12", "married")] == pytest.approx(0.5, abs=1e-9)
    assert diag["not_married_child_record_by_band"]["5-12"] == pytest.approx(
        0.25, abs=1e-5
    )


def test_fit_custodial_child_record_filters_to_train_fathers():
    """Only children whose father is a train person are counted."""
    expo = pd.DataFrame(
        {
            "father_id": [1, 2],
            "child_person_id": [10, 20],
            "year": [2000, 2000],
            "child_band": ["0-4", "0-4"],
            "weight": [1.0, 1.0],
            "with_father": [True, False],
            "marital": ["not_married", "not_married"],
        }
    )
    # Only father 1 in train -> rate is father 1's (coresident) = 1.0.
    rates, _ = hcs5.fit_custodial_child_record(expo, {1})
    assert rates[("0-4", "not_married")] == pytest.approx(1.0, abs=1e-9)


def test_custodial_prob_v5_not_married_swaps_married_keeps():
    """not_married fathers use the child-record band rate; married fathers use
    candidate 4's observable-basis lookup unchanged."""

    class _BaseV4:
        custodial_era = {(8, "2010-2023", "married"): 0.95}
        custodial_age_marital = {(8, "married"): 0.9}
        custodial_band_marital = {("5-12", "married"): 0.8}
        custodial_overall = 0.3

    class _M:
        base_v4 = _BaseV4()
        custodial_child_record = {("5-12", "not_married"): 0.42}

    m = _M()
    # not_married age-8 (band 5-12) -> child-record 0.42, NOT the base overall.
    assert hcs5.custodial_prob_v5(m, 8, "2010-2023", "not_married") == 0.42
    # married age-8 -> candidate 4's observable lookup (age, era, marital) 0.95.
    assert hcs5.custodial_prob_v5(m, 8, "2010-2023", "married") == 0.95
    # not_married with no child-record cell -> falls through to base overall.
    m2 = _M()
    m2.custodial_child_record = {}
    assert hcs5.custodial_prob_v5(m2, 8, "2010-2023", "not_married") == 0.3


# --------------------------------------------------------------------------
# Delta 3a: non-family bridge conditional on core size
# --------------------------------------------------------------------------
def test_fit_nonfamily_by_core_conditions_on_core_size():
    """Size-3 cores get their own non-core incidence; only dense cells are
    returned (sparse fall back to candidate 4's (band, sex) table)."""
    n = 60
    # 60 core-3 waves (family_unit_size=3): 30 with a non-core member (resid 1),
    # 30 without -> P(resid>=1)=0.5, class shares (0.5, 0.5, 0.0).
    pw = pd.DataFrame(
        {
            "person_id": range(n),
            "year": [2000] * n,
            "band": ["25-34"] * n,
            "sex": ["male"] * n,
            "weight": [1.0] * n,
            "hh_size": [4] * 30 + [3] * 30,
        }
    )
    fu = pd.DataFrame(
        {
            "person_id": range(n),
            "year": [2000] * n,
            "family_unit_size": [3] * n,
        }
    )
    table, diag = hcs5.fit_nonfamily_by_core(pw, fu, set(range(n)))
    q0, q1, q2 = table[(3, "25-34", "male")]
    assert q0 == pytest.approx(0.5, abs=1e-9)
    assert q1 == pytest.approx(0.5, abs=1e-9)
    assert q2 == pytest.approx(0.0, abs=1e-9)
    assert diag["p_noncore_member_present_by_core"]["3"] == pytest.approx(
        0.5, abs=1e-5
    )
    # A sparse core (e.g. core 5) with < 20 waves is NOT in the dense table.
    assert (5, "25-34", "male") not in table


def test_sample_nonfamily_v5_uses_core_conditional_thresholds():
    """The class draw is byte-identical in shape to candidate 4, but a size-3
    core uses its (core, band, sex) thresholds where dense."""
    pw = pd.DataFrame({"band": ["25-34"] * 4000, "sex": ["male"] * 4000})
    core = np.where(np.arange(4000) < 2000, 3, 1)  # half core-3, half core-1

    class _BaseV3:
        # Fallback (band, sex): everyone class 0 (no non-core member).
        nonfamily = {("25-34", "male"): (1.0, 0.0, 0.0)}

    class _BaseV4:
        nonfamily_2plus = {("25-34", "male"): (np.array([2]), np.array([1.0]))}

    class _M:
        base_v3 = _BaseV3()
        base_v4 = _BaseV4()
        # Core-3 cells: always class 1 (a non-core member); core-1 uses the
        # (band, sex) fallback (class 0).
        nonfamily_by_core = {(3, "25-34", "male"): (0.0, 1.0, 0.0)}

    contrib = hcs5.sample_nonfamily_v5(
        pw, _M(), np.random.default_rng(0), np.random.default_rng(1), core
    )
    # core-3 -> exactly 1 non-core member; core-1 -> 0.
    assert (contrib[:2000] == 1).all()
    assert (contrib[2000:] == 0).all()


def test_sample_nonfamily_v5_class_draw_byte_identical_shape():
    """The class draw consumes exactly n randoms on class_rng (byte-identical
    to candidate 4's shape) and n on count_rng -- only thresholds differ."""
    pw = pd.DataFrame({"band": ["25-34"] * 100, "sex": ["male"] * 100})
    core = np.full(100, 2)

    class _M:
        base_v3 = type(
            "b", (), {"nonfamily": {("25-34", "male"): (0.5, 0.3, 0.2)}}
        )()
        base_v4 = type(
            "b",
            (),
            {
                "nonfamily_2plus": {
                    ("25-34", "male"): (
                        np.array([2, 3]),
                        np.cumsum([0.5, 0.5]),
                    )
                }
            },
        )()
        nonfamily_by_core: dict = {}

    r1 = np.random.default_rng(7)
    r2 = np.random.default_rng(8)
    hcs5.sample_nonfamily_v5(pw, _M(), r1, r2, core)
    # Each rng advanced by exactly 100 draws (the byte-identical shape claim).
    ref = np.random.default_rng(7)
    ref.random(100)
    assert r1.random() == pytest.approx(ref.random())


# --------------------------------------------------------------------------
# Delta 3b: per-ego coresident-parent count composition
# --------------------------------------------------------------------------
def test_parent_link_counts_counts_parent_links():
    """parent_link_counts tallies the coresident-parent MX8 codes per wave."""
    from populace_dynamics.data import household_composition as hc

    codes = sorted(hc.CORESIDENCE_LINKS["coresident_parent"])
    rel_map = pd.DataFrame(
        {
            "interview_year": [2000, 2000, 2000, 2000],
            "ego_person_id": [1, 1, 2, 3],
            # ego 1 has TWO parent links; ego 2 has one; ego 3 a non-parent.
            "ego_rel_to_alter": [codes[0], codes[1], codes[0], 99],
            "alter_person_id": [11, 12, 21, 31],
        }
    )
    counts = hcs5.parent_link_counts(rel_map).set_index(["person_id", "year"])
    assert int(counts.loc[(1, 2000), "n_parent_links"]) == 2
    assert int(counts.loc[(2, 2000), "n_parent_links"]) == 1
    assert (3, 2000) not in counts.index


def test_fit_parent_count_composition_share_of_two_parents():
    """Among coresident-parent egos, the fitted share with two parents; a
    sparse (band, sex) cell falls back to the pooled rate."""
    n = 100
    pw = pd.DataFrame(
        {
            "person_id": range(n),
            "year": [2000] * n,
            "band": ["25-34"] * n,
            "sex": ["male"] * n,
            "weight": [1.0] * n,
            "coresident_parent": [True] * n,
        }
    )
    # 70 of 100 coresident-parent egos have two parents -> 0.70.
    parent_counts = pd.DataFrame(
        {
            "person_id": range(n),
            "year": [2000] * n,
            "n_parent_links": [2] * 70 + [1] * 30,
        }
    )
    table, pooled, diag = hcs5.fit_parent_count_composition(
        pw, parent_counts, set(range(n))
    )
    assert pooled == pytest.approx(0.70, abs=1e-9)
    assert table[("25-34", "male")] == pytest.approx(0.70, abs=1e-9)
    # A (band, sex) with no coresident-parent waves falls back to pooled.
    assert table[("75+", "female")] == pytest.approx(pooled, abs=1e-9)
    assert diag["n_coresident_parent_waves_train"] == n


# --------------------------------------------------------------------------
# Delta 3 helper: size-3 composition routes
# --------------------------------------------------------------------------
def test_route_of_size3_partition():
    """The five routes are mutually exclusive and cover every configuration."""
    has_spouse = np.array([True, False, True, False, False])
    n_child = np.array([1, 2, 0, 0, 0])
    n_parent = np.array([0, 0, 1, 2, 0])
    routes = hcs5.route_of_size3(has_spouse, n_child, n_parent)
    assert routes["couple_plus_child"].tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert routes["single_parent_plus_two_children"].tolist() == [
        False,
        True,
        False,
        False,
        False,
    ]
    assert routes["couple_plus_parent"].tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert routes["three_adults"].tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert routes["other_family_core"].tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]
    # Exactly one route per ego (a partition).
    stacked = np.vstack([m for m in routes.values()])
    assert (stacked.sum(axis=0) == 1).all()
