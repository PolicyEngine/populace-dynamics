"""Pure delta derivations for gate-2b candidate 7 (synthetic; no PSID).

Unit-tests the two candidate-7 deltas' pure helpers on synthetic frames: the
enumerated (joinable) linked-child keys (delta 1) and the marginal-preserving,
sibling-synchronized episode-persistence draw + its episode-length histogram
(delta 2). Always runnable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from populace_dynamics.models import household_composition_sim_v7 as hcs7


# --------------------------------------------------------------------------
# Constants + stream tags
# --------------------------------------------------------------------------
def test_constants_and_stream_tags():
    assert hcs7.GRANDCHILD_LO == 55  # coupling stays 55+ (carried)
    assert hcs7.CORE_SIZE_CAP == 5
    assert hcs7.CUSTODIAL_REVERT_BAND == (0, 4)
    assert hcs7.CHILD_CORESIDENCE_MAX_AGE == 60
    assert hcs7.SPELL_CHILD_MAX_AGE == 17
    assert hcs7.DELTA_STREAM_TAG_V6 == 0xC6  # carried
    assert hcs7.DELTA_STREAM_TAG_V7 == 0xC7  # new, isolated
    assert hcs7.EPISODE_BUCKETS == ("1", "2", "3", "4+")
    assert hcs7._CHILD_BANDS == (
        "0-4",
        "5-12",
        "13-17",
        "18-24",
        "25-60",
    )


# --------------------------------------------------------------------------
# Delta 1: enumerated (joinable) linked-child keys
# --------------------------------------------------------------------------
def test_enumerated_joinable_keys():
    flc = pd.DataFrame(
        {
            "parent_person_id": [1, 1, 2, 2],
            "child_person_id": [10, 11, 20, 21],
            "birth_year": [2000, 2002, 2001, 2001],
        }
    )
    keys = hcs7.enumerated_joinable_keys(flc)
    assert keys == frozenset({(1, 2000), (1, 2002), (2, 2001)})
    # a linked (parent, birth_year) NOT in father_links_child is non-joinable.
    assert (3, 1999) not in keys
    assert isinstance(keys, frozenset)


# --------------------------------------------------------------------------
# Delta 2: episode-length histogram (forensics-4 byte-faithful)
# --------------------------------------------------------------------------
def test_episode_length_hist_runs_and_gaps():
    # one father, one child: coresident 2000,2002,2004 (run 3), gap, 2008 (run 1)
    fid = np.array([1, 1, 1, 1, 1])
    ck = np.array([9, 9, 9, 9, 9])
    yr = np.array([2000, 2002, 2004, 2006, 2008])
    cor = np.array([1, 1, 1, 0, 1], bool)
    dist, mean, n = hcs7.episode_length_hist(fid, ck, yr, cor)
    assert n == 2  # a length-3 run and a length-1 run
    assert abs(mean - 2.0) < 1e-9
    assert dist == {"1": 0.5, "2": 0.0, "3": 0.5, "4+": 0.0}
    # empty input is a clean zero.
    d0, m0, n0 = hcs7.episode_length_hist(
        np.array([], np.int64),
        np.array([], np.int64),
        np.array([], np.int64),
        np.array([], bool),
    )
    assert n0 == 0 and m0 == 0.0


def test_episode_length_hist_two_children_same_father_independent_runs():
    # two children of one father, each one contiguous run -> two episodes.
    fid = np.array([1, 1, 1, 1])
    ck = np.array([100, 100, 200, 200])
    yr = np.array([2000, 2002, 2000, 2002])
    cor = np.array([1, 1, 1, 0], bool)
    dist, mean, n = hcs7.episode_length_hist(fid, ck, yr, cor)
    assert n == 2  # child 100: run of 2; child 200: run of 1
    assert abs(mean - 1.5) < 1e-9


# --------------------------------------------------------------------------
# Delta 2: marginal-preserving, sibling-synchronized episode draw
# --------------------------------------------------------------------------
def _synthetic_exposure(n_fathers=4000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    bid = 0
    for f in range(n_fathers):
        for _c in range(int(rng.integers(1, 4))):
            for age in range(0, 18, 2):
                p = 0.8 if age < 5 else (0.55 if age < 13 else 0.3)
                rows.append((f, bid, 2000 + age, age, p))
            bid += 1
    df = pd.DataFrame(
        rows, columns=["parent_person_id", "_birth_id", "year", "age", "p_c"]
    )
    return df.sort_values(["_birth_id", "year"]).reset_index(drop=True)


def test_episode_draw_preserves_marginal_at_every_rho():
    ex = _synthetic_exposure()
    p = ex["p_c"].to_numpy()
    for rho in (0.0, 0.3, 0.6, 1.0):
        cor = hcs7.simulate_linked_episode_coresidence(
            ex["parent_person_id"].to_numpy(np.int64),
            ex["_birth_id"].to_numpy(np.int64),
            p,
            rho,
            np.random.default_rng(7),
        )
        # per-band marginal preserved to Monte-Carlo tolerance (unbiased).
        for lo, hi in ((0, 4), (5, 12), (13, 17)):
            m = (ex["age"].to_numpy() >= lo) & (ex["age"].to_numpy() <= hi)
            assert abs(cor[m].mean() - p[m].mean()) < 0.02, (rho, lo)


def test_episode_draw_mean_length_monotone_in_rho():
    ex = _synthetic_exposure()
    means = []
    for rho in (0.0, 0.5, 1.0):
        cor = hcs7.simulate_linked_episode_coresidence(
            ex["parent_person_id"].to_numpy(np.int64),
            ex["_birth_id"].to_numpy(np.int64),
            ex["p_c"].to_numpy(),
            rho,
            np.random.default_rng(11),
        )
        _d, mean, _n = hcs7.episode_length_hist(
            ex["parent_person_id"].to_numpy(np.int64),
            ex["_birth_id"].to_numpy(np.int64),
            ex["year"].to_numpy(np.int64),
            cor,
        )
        means.append(mean)
    # rho=0 is the candidate-6 independent draw (most fragmented); increasing
    # rho lengthens the episodes monotonically (the fit inverts this).
    assert means[0] < means[1] < means[2]


def test_episode_draw_reduces_occupancy_for_multi_child_fathers():
    # persistence + sibling synchronization concentrates coresidence into fewer
    # father-waves (the spell-channel occupancy reduction).
    ex = _synthetic_exposure()
    fw_key = ex["parent_person_id"].to_numpy(np.int64) * 100000 + ex[
        "year"
    ].to_numpy(np.int64)
    ex = ex.assign(_fw=fw_key)
    occ = {}
    for rho in (0.0, 1.0):
        cor = hcs7.simulate_linked_episode_coresidence(
            ex["parent_person_id"].to_numpy(np.int64),
            ex["_birth_id"].to_numpy(np.int64),
            ex["p_c"].to_numpy(),
            rho,
            np.random.default_rng(3),
        )
        any_cor = (
            pd.DataFrame({"_fw": ex["_fw"], "cor": cor})
            .groupby("_fw")["cor"]
            .max()
        )
        occ[rho] = float(any_cor.mean())
    # fully persistent (rho=1, sibling-synchronized) occupancy < independent.
    assert occ[1.0] < occ[0.0]


def test_episode_draw_rho_zero_matches_independent_marginal():
    ex = _synthetic_exposure(n_fathers=2000, seed=5)
    p = ex["p_c"].to_numpy()
    cor = hcs7.simulate_linked_episode_coresidence(
        ex["parent_person_id"].to_numpy(np.int64),
        ex["_birth_id"].to_numpy(np.int64),
        p,
        0.0,
        np.random.default_rng(9),
    )
    # rho=0 is an independent Bernoulli(p_c) per row -> overall mean ~ mean p_c.
    assert abs(cor.mean() - p.mean()) < 0.01


def test_episode_draw_empty_is_noop():
    cor = hcs7.simulate_linked_episode_coresidence(
        np.array([], np.int64),
        np.array([], np.int64),
        np.array([], np.float64),
        0.5,
        np.random.default_rng(0),
    )
    assert cor.shape == (0,)
