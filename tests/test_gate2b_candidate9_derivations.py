"""Pure delta derivations for gate-2b candidate 9 (synthetic; no PSID).

Unit-tests the ONE candidate-9 delta -- the cohort-scoped fertility-core lift --
and the pre-run analytic-check helpers on synthetic frames: the write-gated
fertility lift (byte-identical to candidate 8 on the deficit cohorts via RNG
parity, reverting every other cohort to the input), and the completed-size
distribution / kernel helpers the analytic check convolves. Always runnable.
"""

from __future__ import annotations

import numpy as np

from populace_dynamics.models import household_composition_sim_v8 as hcs8
from populace_dynamics.models import household_composition_sim_v9 as hcs9


# --------------------------------------------------------------------------
# Constants + scope / collateral / revert partition
# --------------------------------------------------------------------------
def test_scope_constants_and_stream_tags():
    # candidate 9 reuses candidate 8's isolated tags UNCHANGED.
    assert hcs9.DELTA_STREAM_TAG_V7 == 0xC7
    assert hcs9.DELTA_STREAM_TAG_V8 == 0xC8
    assert hcs9.SIZE_BUCKETS == ("0", "1", "2", "3", "4+")
    # the fertility lift is confined to the four forensics-5 deficit cohorts.
    assert hcs9.FERTILITY_LIFT_CELLS == (
        "coresident_child.55-64|male",
        "coresident_child.65-74|male",
        "coresident_child.45-54|female",
        "coresident_child.65-74|female",
    )
    assert hcs9.DEFICIT_COHORTS == {
        ("55-64", "male"),
        ("65-74", "male"),
        ("45-54", "female"),
        ("65-74", "female"),
    }


def test_collateral_cells_are_reverted_and_disjoint_from_scope():
    # the four candidate-8 collateral cells are among the reverted cells.
    assert set(hcs9.COLLATERAL_CELLS) <= set(hcs9.REVERTED_CHILD_CELLS)
    # scope and reverted partition disjointly.
    assert not (
        set(hcs9.FERTILITY_LIFT_CELLS) & set(hcs9.REVERTED_CHILD_CELLS)
    )
    # the collateral cells are exactly the candidate-8 grading's four.
    assert set(hcs9.COLLATERAL_CELLS) == {
        "coresident_child.35-44|male",
        "coresident_child.35-44|female",
        "coresident_child.45-54|male",
        "coresident_child.55-64|female",
    }


def test_analytic_check_cells_cover_priced_held_and_collateral():
    cells = set(hcs9.ANALYTIC_CHECK_CELLS)
    assert "hh_size.5+" in cells
    assert "coresident_child.55-64|male" in cells
    assert "coresident_child.65-74|male" in cells
    assert set(hcs9.COLLATERAL_CELLS) <= cells
    assert len(hcs9.ANALYTIC_CHECK_CELLS) == 7


def test_model_and_fit_alias_candidate8():
    # candidate 9 reuses candidate 8's model + fit verbatim (scope is a
    # simulate-time write gate, not a fit change).
    assert hcs9.HouseholdCompositionModelV9 is hcs8.HouseholdCompositionModelV8
    assert hcs9.fit_household_model_v9 is not None


# --------------------------------------------------------------------------
# The ONE delta: the cohort-scoped fertility-core lift (write gate)
# --------------------------------------------------------------------------
def _model(csd):
    return hcs8.HouseholdCompositionModelV8(
        base_v7=None,
        completed_size_dist_train=csd,
        completed_size_dist_train_all={},
        retention_link_shift={},
    )


def _synthetic_two_cohort(n=8000, seed=0):
    """One SCOPE cohort (55-64|male) + one NON-SCOPE cohort (35-44|male)."""
    rng = np.random.default_rng(seed)
    band = np.array(["55-64", "35-44"] * (n // 2), dtype=object)
    sex = np.array(["male"] * n, dtype=object)
    person_id = np.arange(n, dtype=np.int64)
    weight = np.ones(n)
    child_counts = rng.integers(0, 5, n)
    cor = rng.random(n) < 0.4
    hh = (1 + child_counts).astype(np.int64)
    return person_id, band, sex, weight, child_counts, cor, hh


def test_scoped_lift_byte_identical_to_c8_on_deficit_cohort():
    """The deficit cohort's scoped lift is byte-identical to candidate 8's
    global lift, because every cohort draws the same rng in the same order and
    only the WRITE is gated."""
    pid, band, sex, w, cc, cor, hh = _synthetic_two_cohort()
    d_train = {"0": 0.15, "1": 0.15, "2": 0.25, "3": 0.25, "4+": 0.2}
    m = _model({("55-64", "male"): d_train, ("35-44", "male"): d_train})
    cor8, hh8, _ = hcs8.apply_fertility_core_lift(
        pid, band, sex, w, cc, cor, hh, m, np.random.default_rng(42)
    )
    cor9, hh9, diag = hcs9.apply_scoped_fertility_core_lift(
        pid, band, sex, w, cc, cor, hh, m, np.random.default_rng(42)
    )
    scope = band == "55-64"  # the deficit cohort
    assert (cor9[scope] == cor8[scope]).all()  # byte-identical to candidate 8
    assert (hh9[scope] == hh8[scope]).all()
    # the diagnostic flags the scope cell as in_scope.
    assert diag["per_cell"]["coresident_child.55-64|male"]["in_scope"] is True


def test_scoped_lift_reverts_non_deficit_cohort_to_input():
    """The non-deficit cohort keeps its candidate-7 (input) coresident_child /
    hh_size byte-identically, while candidate 8's global lift would change it.
    """
    pid, band, sex, w, cc, cor, hh = _synthetic_two_cohort()
    d_train = {"0": 0.15, "1": 0.15, "2": 0.25, "3": 0.25, "4+": 0.2}
    m = _model({("55-64", "male"): d_train, ("35-44", "male"): d_train})
    cor8, _hh8, _ = hcs8.apply_fertility_core_lift(
        pid, band, sex, w, cc, cor, hh, m, np.random.default_rng(42)
    )
    cor9, hh9, diag = hcs9.apply_scoped_fertility_core_lift(
        pid, band, sex, w, cc, cor, hh, m, np.random.default_rng(42)
    )
    ns = band == "35-44"  # the non-deficit cohort
    # candidate 9 reverts it to the input exactly.
    assert (cor9[ns] == cor[ns]).all()
    assert (hh9[ns] == hh[ns]).all()
    # candidate 8's global lift DID change it (the collateral mechanism).
    assert (cor8[ns] != cor[ns]).any()
    assert diag["per_cell"]["coresident_child.35-44|male"]["in_scope"] is False


def test_scoped_lift_leaves_non_composition_rows_untouched():
    rng = np.random.default_rng(9)
    n = 2000
    pid = np.arange(n, dtype=np.int64)
    band = np.array([None] * n, dtype=object)  # not a composition band
    sex = np.array(["male"] * n, dtype=object)
    w = np.ones(n)
    cc = rng.integers(0, 3, n)
    cor = cc > 0
    hh = (1 + cc).astype(np.int64)
    cor_new, hh_new, _ = hcs9.apply_scoped_fertility_core_lift(
        pid, band, sex, w, cc, cor, hh, _model({}), rng
    )
    assert (cor_new == cor).all()
    assert (hh_new == hh).all()


def test_scoped_lift_only_the_four_deficit_cohorts_move():
    """Across a frame with all four deficit cohorts and several non-deficit
    ones, only the deficit cohorts' coresidence rate moves off the input."""
    rng = np.random.default_rng(3)
    cohorts = [
        ("55-64", "male"),
        ("65-74", "male"),
        ("45-54", "female"),
        ("65-74", "female"),
        ("35-44", "male"),  # non-deficit
        ("25-34", "female"),  # non-deficit
    ]
    per = 4000
    n = per * len(cohorts)
    band = np.empty(n, dtype=object)
    sex = np.empty(n, dtype=object)
    for i, (b, s) in enumerate(cohorts):
        band[i * per : (i + 1) * per] = b
        sex[i * per : (i + 1) * per] = s
    pid = np.arange(n, dtype=np.int64)
    w = np.ones(n)
    cc = rng.integers(0, 5, n)
    cor = rng.random(n) < 0.4
    hh = (1 + cc).astype(np.int64)
    # a train D that shifts mass to the 3+ tail for every cohort.
    d_train = {"0": 0.1, "1": 0.15, "2": 0.25, "3": 0.25, "4+": 0.25}
    m = _model({c: d_train for c in cohorts})
    cor_new, _hh, _ = hcs9.apply_scoped_fertility_core_lift(
        pid, band, sex, w, cc, cor, hh, m, np.random.default_rng(7)
    )
    for b, s in cohorts:
        mask = (band == b) & (sex == s)
        moved = (cor_new[mask] != cor[mask]).any()
        assert moved == ((b, s) in hcs9.DEFICIT_COHORTS), (b, s)


# --------------------------------------------------------------------------
# Analytic-check helpers (law-of-total-probability convolution kernels)
# --------------------------------------------------------------------------
def test_size_dist_row_is_weighted_distribution():
    bucket = np.array(["0", "1", "2", "2", "4+"], dtype=object)
    w = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
    d = hcs9._size_dist_row(bucket, w)
    assert abs(sum(d.values()) - 1.0) < 1e-12
    assert abs(d["2"] - 2.0 / 6.0) < 1e-12
    assert abs(d["4+"] - 2.0 / 6.0) < 1e-12
    assert d["3"] == 0.0


def test_kernel_row_is_conditional_rate():
    bucket = np.array(["1", "1", "2", "2"], dtype=object)
    hit = np.array([True, False, True, True])
    w = np.ones(4)
    k = hcs9._kernel_row(bucket, hit, w)
    assert abs(k["1"] - 0.5) < 1e-12
    assert abs(k["2"] - 1.0) < 1e-12
    assert k["0"] == 0.0  # empty bucket -> 0


def test_convolution_reconstructs_full_rate():
    """sum_S D[S] K[S] == the marginal hit rate (law of total probability) --
    the identity the scoped/global counterfactuals rest on."""
    rng = np.random.default_rng(1)
    n = 20000
    bucket = rng.choice(
        hcs9.SIZE_BUCKETS, size=n, p=[0.3, 0.2, 0.25, 0.15, 0.1]
    )
    kmap = {"0": 0.0, "1": 0.4, "2": 0.55, "3": 0.7, "4+": 0.85}
    hit = np.array([rng.random() < kmap[b] for b in bucket])
    w = np.ones(n)
    d = hcs9._size_dist_row(bucket, w)
    k = hcs9._kernel_row(bucket, hit, w)
    recon = sum(d[b] * k[b] for b in hcs9.SIZE_BUCKETS)
    assert abs(recon - float(hit.mean())) < 1e-9
    # a train D with more 3+ mass raises the reconstructed rate (the lift).
    d_train = {"0": 0.15, "1": 0.15, "2": 0.2, "3": 0.25, "4+": 0.25}
    cf = sum(d_train[b] * k[b] for b in hcs9.SIZE_BUCKETS)
    assert cf > recon
