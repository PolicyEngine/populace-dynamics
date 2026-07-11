"""Pure delta derivations for gate-2b candidate 8 (synthetic; no PSID).

Unit-tests the three candidate-8 deltas' pure helpers on synthetic frames: the
completed-family-size bucketing + train distribution + Oaxaca telescope (delta 1
/ delta 3 fit), the conditional-flip additive-shift realization (delta 3 / delta
1 kernel), the Bernoulli-superposition cohabitation-overlay lift (delta 2), the
fertility-core lift (swap D holding the kernel, delta 1) and the band-signed
retention + link-coverage refit (delta 3). Always runnable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from populace_dynamics.models import household_composition_sim_v8 as hcs8


# --------------------------------------------------------------------------
# Constants + stream tags
# --------------------------------------------------------------------------
def test_constants_and_stream_tags():
    assert hcs8.GRANDCHILD_LO == 55  # carried
    assert hcs8.CORE_SIZE_CAP == 5
    assert hcs8.CHILD_CORESIDENCE_MAX_AGE == 60
    assert hcs8.SPELL_CHILD_MAX_AGE == 17
    assert hcs8.DELTA_STREAM_TAG_V7 == 0xC7  # carried
    assert hcs8.DELTA_STREAM_TAG_V8 == 0xC8  # new, isolated
    assert hcs8.SIZE_BUCKETS == ("0", "1", "2", "3", "4+")
    assert hcs8.COHAB_OVERLAY_LIFT_BAND == "25-34"
    assert hcs8.COHAB_OVERLAY_LIFT == 0.045
    assert hcs8.RETENTION_EXIT_CELLS == (
        "coresident_child.65-74|male",
        "coresident_child.45-54|female",
        "coresident_child.65-74|female",
    )
    assert hcs8.LINK_COVERAGE_CELLS == (
        "coresident_child.55-64|male",
        "coresident_child.65-74|male",
    )


# --------------------------------------------------------------------------
# Completed-family-size helpers (byte-faithful to forensics-5)
# --------------------------------------------------------------------------
def test_size_bucket_caps_at_4plus():
    out = hcs8.size_bucket(np.array([0, 1, 2, 3, 4, 5, 9]))
    assert list(out) == ["0", "1", "2", "3", "4+", "4+", "4+"]


def test_train_completed_size_is_max_over_waves():
    # parent 1 has 2 children in 2000, 3 in 2002 -> completed size 3.
    pp = pd.DataFrame(
        {
            "parent_person_id": [1, 1, 1, 1, 1, 2, 2],
            "child_person_id": [10, 11, 10, 11, 12, 20, 21],
            "year": [2000, 2000, 2002, 2002, 2002, 2001, 2001],
        }
    )
    size = hcs8.train_completed_size(pp, {1, 2})
    assert size[1] == 3
    assert size[2] == 2
    # a person not in ids_b is absent (caller assigns size 0).
    assert 3 not in size


def test_completed_size_dk_full_rate_is_convolution():
    pw = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "coresident_child": [True, False, True, True],
        }
    )
    size_map = {1: 1, 2: 0, 3: 2, 4: 2}
    d, k, full = hcs8.cell_completed_size_dk(pw, size_map)
    assert abs(sum(d.values()) - 1.0) < 1e-12
    # full rate == sum_S D[S] K[S] exactly (law of total probability).
    recon = sum(d[b] * k[b] for b in hcs8.SIZE_BUCKETS)
    assert abs(full - recon) < 1e-12
    # bucket 0 never coresides (a completed-size-0 parent has no child).
    assert k["0"] == 0.0


def test_oaxaca_terms_reconcile_to_cell_miss():
    # sim cell vs a train D/K: endowment + coefficient == sim_full - ref_full.
    rng = np.random.default_rng(0)
    n = 5000
    buckets = rng.choice(
        hcs8.SIZE_BUCKETS, size=n, p=[0.3, 0.2, 0.25, 0.15, 0.1]
    )
    kmap = {"0": 0.0, "1": 0.4, "2": 0.55, "3": 0.7, "4+": 0.85}
    cor = np.array([rng.random() < kmap[b] for b in buckets])
    w = np.ones(n)
    d_train = {"0": 0.2, "1": 0.18, "2": 0.27, "3": 0.2, "4+": 0.15}
    k_train = {"0": 0.0, "1": 0.42, "2": 0.57, "3": 0.72, "4+": 0.86}
    endow, coef, sim_full, d_sim, k_sim = hcs8._oaxaca_terms(
        buckets, cor, w, d_train, k_train
    )
    ref_full = sum(d_train[b] * k_train[b] for b in hcs8.SIZE_BUCKETS)
    assert abs((endow + coef) - (sim_full - ref_full)) < 1e-9
    assert abs(sum(d_sim.values()) - 1.0) < 1e-12


# --------------------------------------------------------------------------
# Conditional flip: realizes an additive shift (delta 1 kernel / delta 3)
# --------------------------------------------------------------------------
def test_conditional_flip_positive_raises_mean():
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.random(n) < 0.3
    w = np.ones(n)
    subset = np.ones(n, bool)
    out = hcs8._conditional_flip(x, w, subset, 0.15, rng)
    assert abs(out.mean() - (x.mean() + 0.15)) < 0.01
    # only False->True flips (monotone up).
    assert (out | x == out).all()


def test_conditional_flip_negative_lowers_mean():
    rng = np.random.default_rng(2)
    n = 20000
    x = rng.random(n) < 0.7
    w = np.ones(n)
    subset = np.ones(n, bool)
    out = hcs8._conditional_flip(x, w, subset, -0.2, rng)
    assert abs(out.mean() - (x.mean() - 0.2)) < 0.01
    # only True->False flips (monotone down).
    assert (out & x == out).all()


def test_conditional_flip_zero_is_noop():
    rng = np.random.default_rng(3)
    x = rng.random(100) < 0.5
    out = hcs8._conditional_flip(x, np.ones(100), np.ones(100, bool), 0.0, rng)
    assert (out == x).all()


def test_conditional_flip_respects_subset():
    rng = np.random.default_rng(4)
    n = 10000
    x = np.zeros(n, bool)
    w = np.ones(n)
    subset = np.arange(n) < n // 2  # only the first half eligible
    out = hcs8._conditional_flip(x, w, subset, 0.2, rng)
    # the second half is untouched.
    assert not out[~subset].any()
    assert out[subset].mean() > 0.15


# --------------------------------------------------------------------------
# Delta 2: cohabitation-overlay lift (Bernoulli superposition)
# --------------------------------------------------------------------------
def _model(**kw):
    return hcs8.HouseholdCompositionModelV8(
        base_v7=None,
        completed_size_dist_train=kw.get("csd", {}),
        completed_size_dist_train_all=kw.get("csd_all", {}),
        retention_link_shift=kw.get("shift", {}),
        cohab_overlay_lift=kw.get("lift", hcs8.COHAB_OVERLAY_LIFT),
    )


def test_cohab_overlay_lift_only_moves_25_34_female():
    rng = np.random.default_rng(5)
    n = 40000
    band = np.array(["25-34", "35-44"] * (n // 2), dtype=object)
    sex = np.array(["female"] * n, dtype=object)
    weight = np.ones(n)
    spouse = rng.random(n) < 0.55
    model = _model(lift=0.045)
    new, diag = hcs8.apply_cohab_overlay_lift(
        band, sex, weight, spouse, model, rng
    )
    old_2534 = spouse[band == "25-34"].mean()
    new_2534 = new[band == "25-34"].mean()
    # Bernoulli superposition: new = old + 0.045 * (1 - old).
    assert abs(new_2534 - (old_2534 + 0.045 * (1 - old_2534))) < 0.01
    # the 35-44 band is byte-untouched.
    assert (new[band == "35-44"] == spouse[band == "35-44"]).all()
    assert abs(diag["realized_lift"] - (new_2534 - old_2534)) < 1e-9


def test_cohab_overlay_lift_only_moves_female():
    rng = np.random.default_rng(6)
    n = 20000
    band = np.array(["25-34"] * n, dtype=object)
    sex = np.array(["male", "female"] * (n // 2), dtype=object)
    weight = np.ones(n)
    spouse = np.zeros(n, bool)
    model = _model(lift=0.045)
    new, _ = hcs8.apply_cohab_overlay_lift(
        band, sex, weight, spouse, model, rng
    )
    # no male 25-34 row is lifted.
    assert not new[sex == "male"].any()
    assert new[sex == "female"].mean() > 0.02


# --------------------------------------------------------------------------
# Delta 3: band-signed retention + link-coverage refit (additive shift)
# --------------------------------------------------------------------------
def test_retention_link_refit_realizes_band_signed_shift():
    rng = np.random.default_rng(7)
    n = 60000
    cells = [
        ("65-74", "male"),
        ("45-54", "female"),
        ("55-64", "male"),
        ("35-44", "male"),  # not a delta-3 cell -> untouched
    ]
    band = np.empty(n, dtype=object)
    sex = np.empty(n, dtype=object)
    per = n // len(cells)
    for i, (b, s) in enumerate(cells):
        band[i * per : (i + 1) * per] = b
        sex[i * per : (i + 1) * per] = s
    weight = np.ones(n)
    cor = rng.random(n) < 0.3
    shift = {
        "coresident_child.65-74|male": 0.05,  # lift under-retention
        "coresident_child.45-54|female": -0.08,  # reduce over-retention
        "coresident_child.55-64|male": 0.02,  # link coverage
    }
    model = _model(shift=shift)
    new, diag = hcs8.apply_retention_link_refit(
        band, sex, weight, cor, model, rng
    )

    def cell_rate(arr, b, s):
        m = (band == b) & (sex == s)
        return arr[m].mean()

    assert (
        abs(diag["coresident_child.65-74|male"]["realized_shift"] - 0.05)
        < 0.01
    )
    assert (
        abs(diag["coresident_child.45-54|female"]["realized_shift"] - (-0.08))
        < 0.01
    )
    # the 35-44|male rows are untouched (not a delta-3 cell).
    assert cell_rate(new, "35-44", "male") == cell_rate(cor, "35-44", "male")


# --------------------------------------------------------------------------
# Delta 1: fertility-core lift (swap D holding the kernel)
# --------------------------------------------------------------------------
def test_fertility_core_lift_matches_counterfactual_and_holds_kernel():
    rng = np.random.default_rng(8)
    n = 40000
    # one composition band/sex cell; a per-person completed size and a
    # coresidence-given-size kernel the lift must hold.
    person_id = np.arange(n, dtype=np.int64)
    band = np.array(["55-64"] * n, dtype=object)
    sex = np.array(["male"] * n, dtype=object)
    weight = np.ones(n)
    # sim completed-size distribution (under-produces 3+): buckets 0..4+.
    d_sim = np.array([0.3, 0.2, 0.25, 0.15, 0.1])
    sizes = rng.choice([0, 1, 2, 3, 4], size=n, p=d_sim)
    kmap = {0: 0.0, 1: 0.3, 2: 0.45, 3: 0.65, 4: 0.8}
    child_counts = sizes.copy()
    cor = np.array([rng.random() < kmap[s] for s in sizes])
    hh = 1 + child_counts  # a monotone hh_size proxy
    # train D shifts mass to the 3+ tail.
    d_train = {"0": 0.15, "1": 0.15, "2": 0.25, "3": 0.25, "4+": 0.2}
    model = _model(csd={("55-64", "male"): d_train})
    cor_new, hh_new, diag = hcs8.apply_fertility_core_lift(
        person_id, band, sex, weight, child_counts, cor, hh, model, rng
    )
    # the counterfactual = sum_S D_train[S] K_sim[S]; the re-emitted rate
    # matches it (unbiased) and EXCEEDS the sim rate (train has more large
    # families, higher-K buckets).
    k_sim = diag["per_cell"]["coresident_child.55-64|male"]["k_sim_given_size"]
    cf = sum(d_train[b] * k_sim[b] for b in hcs8.SIZE_BUCKETS)
    assert abs(cor_new.mean() - cf) < 0.02
    assert cor_new.mean() > cor.mean()  # the fertility lift raises coresidence
    # hh_size resample lifts the large-household share (more 3+ families).
    assert (hh_new >= 4).mean() > (hh >= 4).mean()


def test_fertility_core_lift_leaves_non_composition_rows_untouched():
    rng = np.random.default_rng(9)
    n = 2000
    person_id = np.arange(n, dtype=np.int64)
    # a band NOT in the composition set (None) is left as-is.
    band = np.array([None] * n, dtype=object)
    sex = np.array(["male"] * n, dtype=object)
    weight = np.ones(n)
    child_counts = rng.integers(0, 3, n)
    cor = child_counts > 0
    hh = 1 + child_counts
    model = _model(csd={})
    cor_new, hh_new, _ = hcs8.apply_fertility_core_lift(
        person_id, band, sex, weight, child_counts, cor, hh, model, rng
    )
    assert (cor_new == cor).all()
    assert (hh_new == hh).all()
