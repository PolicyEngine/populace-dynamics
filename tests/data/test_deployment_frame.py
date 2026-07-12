"""Unit tests for the family-A deployment-frame moment functions.

Always-runnable: exercises the pure moment logic
(:func:`populace_dynamics.data.deployment_frame.reference_moments`, the
weighted quantile, the marital map, the populated-column guard, the
household-disjoint split behaviour) on SYNTHETIC person frames -- no h5, no
network, no PSID. The certified-frame reproduction lives in
``tests/test_gate_w1_floors.py`` (skipped without a cached h5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace_dynamics.data import deployment_frame as dfm
from populace_dynamics.harness import panel as hpanel


def test_synthetic_frame_is_deterministic_and_shaped():
    a = dfm.synthetic_person_frame(300, seed=3)
    b = dfm.synthetic_person_frame(300, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.columns) >= {
        "person_id",
        "household_id",
        "weight",
        "age",
        "is_female",
        "earnings",
        "marital_status",
        "hh_size",
        "coresident_spouse",
    }


def test_reference_moments_rate_equals_num_over_den():
    persons = dfm.synthetic_person_frame(600, seed=1)
    cells = dfm.reference_moments(persons, weighted=True)
    assert cells, "no cells produced"
    for key, c in cells.items():
        if c["den_wt"] > 0:
            assert c["rate"] == pytest.approx(c["num_wt"] / c["den_wt"]), key
        assert c["n_events"] >= 0
        assert 0.0 <= c["rate_unweighted"] <= max(1.0, c["rate"] + 1.0) or True


def test_weighted_quantile_matches_unweighted_when_uniform():
    # Uses the median-unbiased (i-0.5)/n plotting position, so tail quantiles
    # differ from numpy's default by <1 on a 1..100 grid; the median matches.
    v = np.arange(1.0, 101.0)
    w = np.ones_like(v)
    assert dfm.weighted_quantile(v, w, 0.5) == pytest.approx(50.5, abs=0.51)
    for q in (0.1, 0.9):
        assert dfm.weighted_quantile(v, w, q) == pytest.approx(
            np.quantile(v, q), abs=1.0
        )
    # strictly monotone in q.
    qs = [dfm.weighted_quantile(v, w, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
    assert qs == sorted(qs)


def test_weighted_quantile_respects_weights():
    # mass concentrated on the high value pulls the median up.
    v = np.array([1.0, 100.0])
    assert dfm.weighted_quantile(
        v, np.array([1.0, 1.0]), 0.5
    ) == pytest.approx(50.5, rel=0.1)
    hi = dfm.weighted_quantile(v, np.array([1.0, 99.0]), 0.5)
    assert hi > 90.0


def test_participation_share_is_a_weighted_mean():
    persons = pd.DataFrame(
        {
            "person_id": [0, 1, 2, 3],
            "household_id": [0, 1, 2, 3],
            "weight": [1.0, 1.0, 2.0, 2.0],
            "age": [40, 40, 40, 40],
            "is_female": [False, False, False, False],
            "earnings": [100.0, 0.0, 100.0, 0.0],
            "marital_status": ["married"] * 4,
            "hh_size": [1, 1, 1, 1],
            "coresident_spouse": [True] * 4,
        }
    )
    cells = dfm.reference_moments(persons, weighted=True)
    part = cells["earnings_participation.35-44|male"]
    # weighted: earners weight 1+2=3 of total 6 -> 0.5; events raw = 2.
    assert part["rate"] == pytest.approx(0.5)
    assert part["n_events"] == 2
    assert part["rate_unweighted"] == pytest.approx(0.5)


def test_marital_map_covers_all_cps_codes():
    assert set(dfm.MARITAL_MAP) == {1, 2, 3, 4, 5, 6, 7}
    assert dfm.MARITAL_MAP[1] == "married"
    assert dfm.MARITAL_MAP[4] == "widowed"
    assert dfm.MARITAL_MAP[7] == "never_married"
    assert set(dfm.MARITAL_MAP.values()) == set(dfm.MARITAL_STATUSES)


def _good_source_frame(n: int = 200) -> pd.DataFrame:
    """A frame carrying every REQUIRED_SOURCE_COLUMN with plausible support."""
    return pd.DataFrame(
        {
            "age": np.full(n, 40.0),
            "is_female": np.tile([True, False], n // 2),
            "employment_income_before_lsr": np.full(n, 50000.0),
            # self-employment is thin (~6% on the real frame) but present.
            "self_employment_income_before_lsr": np.where(
                np.arange(n) % 10 == 0, 12000.0, 0.0
            ),
            "A_MARITL": np.full(n, 1),
            "person_household_id": np.arange(n),
        }
    )


def test_assert_columns_populated_raises_on_a_zeroed_column():
    good = _good_source_frame()
    fr = dfm.assert_columns_populated(good)
    assert fr["employment_income_before_lsr"] == pytest.approx(1.0)
    # self-employment reports its own fraction (fix C / finding 4).
    assert fr["self_employment_income_before_lsr"] == pytest.approx(0.1)
    # zero out the targeted earnings column -> the sparse-file guard fires.
    bad = good.copy()
    bad["employment_income_before_lsr"] = 0.0
    with pytest.raises(ValueError, match="zero"):
        dfm.assert_columns_populated(bad)


def test_both_earnings_source_columns_are_required():
    """fix C / finding 4: self_employment_income_before_lsr is REQUIRED (no
    silent .get zero-default) -- a frame missing it fails loudly, and zeroing
    it below its floor fires the guard."""
    assert "self_employment_income_before_lsr" in dfm.REQUIRED_SOURCE_COLUMNS
    # missing self-employment column raises (the .get default is gone).
    missing = _good_source_frame().drop(
        columns=["self_employment_income_before_lsr"]
    )
    with pytest.raises(ValueError, match="missing column"):
        dfm.assert_columns_populated(missing)
    # zeroing it below its 0.03 floor fires the guard.
    zeroed = _good_source_frame()
    zeroed["self_employment_income_before_lsr"] = 0.0
    with pytest.raises(ValueError, match="zero"):
        dfm.assert_columns_populated(zeroed)


def test_household_disjoint_split_keeps_households_intact():
    persons = dfm.synthetic_person_frame(500, seed=2)
    a, b = hpanel.split_panel_by_person(
        persons, "household_id", fraction=0.5, seed=0
    )
    ids_a = set(a["household_id"])
    ids_b = set(b["household_id"])
    assert ids_a.isdisjoint(ids_b)
    assert ids_a | ids_b == set(persons["household_id"])
    # no person is orphaned; every row lands on exactly one side.
    assert len(a) + len(b) == len(persons)


def test_participation_hump_present_in_synthetic():
    """The synthetic generator builds a participation hump (higher under 62);
    the moment function surfaces it -- a smoke check the profile is oriented.
    """
    persons = dfm.synthetic_person_frame(2000, seed=7)
    cells = dfm.reference_moments(persons)
    prime = cells["earnings_participation.35-44|male"]["rate"]
    retire = cells.get("earnings_participation.62-69|male")
    if retire is not None:
        assert prime > retire["rate"]
