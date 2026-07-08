"""Gate-2 candidate 8 (run 1): candidate 6 + one named delta.

The EIGHTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42
comment 4912995860 (``SPEC_REGISTRATION``): candidate 6's frozen spec
(comment 4912170754) verbatim EXCEPT ONE delta. One-shot; no constant
moves after the registration comment.

The one delta vs candidate 6 (everything else byte-identical)
------------------------------------------------------------
**Order-split remarriage, rescaled to preserve the aggregate count.**
Candidate 6's remarriage is the origin-split weighted empirical hazard by
years-since-dissolution band x origin (divorced/widowed) x sex, add-one
smoothed (candidate 1's table, inherited byte-identically). Candidate 8
ADDITIONALLY conditions it on the order of the marriage being entered --
entering the 2nd (exactly one prior marriage) vs the 3rd-or-higher (two or
more prior) -- with the same YSD bands x origin x sex construction and the
same add-one smoothing. This is EXACTLY candidate 7's delta 1: candidate 7's
estimator (:func:`candidate7.fit_remarriage_ordered`) and its simulation
lookup (:func:`candidate7._remarriage_probs_ordered`) are IMPORTED and reused
byte-for-byte -- nothing about the order split is re-implemented here.

Candidate 7 measured that split moving two things in opposite directions:
it FIXED ``remarriage.after_divorce`` (the compositional fix) but LOWERED
the already-low simulated lifetime marriage counts (higher-order remarriage
rates are lower in the data, and that count-lowering side effect pushed
``mean_lifetime_marriages|male`` the wrong way). Candidate 8 keeps the
compositional fix while NEUTRALIZING the count-lowering side effect with a
single train-side aggregate-preservation constraint: after estimation, BOTH
order-specific tables are multiplied by the ONE scalar that makes the order-
split table's exposure-weighted aggregate remarriage rate over the train
dissolved person-years equal the unsplit candidate-6 table's aggregate over
the SAME exposure. The scalar is computed on TRAIN ONLY (never the holdout),
per seed, and recorded in the artifact. Because a single scalar multiplies
both tables, the 2nd-vs-3rd+ compositional RATIO -- the after-divorce fix --
is preserved exactly, while the overall remarriage level returns to
candidate 6's.

NO fertility delta. Candidate 7's delta 2 (fertility x marital status) was
falsified (it over-produced ``asfr.15-19`` and regressed candidate 6's only
passing seed), so fertility here is BYTE-IDENTICAL to candidate 6: the same
single-year triangular-kernel table, the same marital-state-INDEPENDENT
simulation lookup, the same ``rng.random(n_fertile)`` draw block and
threshold. Everything else -- the source-aligned surviving-spouse marriage-
history widowhood LEVEL, the committed candidate-5 NCHS betas, the knot-at-22
20/22/25/30/40 first-marriage spline, divorce, the spousal-age-gap
DISTRIBUTION draw, the competing-risk step, the RNG rule
``numpy.random.default_rng(4200 + seed)``, the spawned gap-draw stream, one
simulated sequence per person, and the LOCKED gate-2 protocol -- is
byte-identical to candidate 6. This runner IMPORTS candidate 6's machinery
(which chains candidates 5/4/1): the unchanged components come straight from
``candidate6.fit_components``, and only remarriage is recomputed (order-
split via candidate 7's estimator, then rescaled). The scoring path,
precheck, and verdict assembly are candidate 1's, imported unchanged.

Because the delta moves only the remarriage THRESHOLD -- the per-year
uniform blocks (``rng.random(n_active)`` then ``rng.random(n_fertile)``)
are drawn in the same order and size as candidate 6 -- first marriage
(marital-state-independent), the ever-married shares (first-marriage-driven)
AND fertility (marital-state-independent, no delta) are byte-identical to
candidate 6; only remarriage, the marriage counts and the dissolved-state
stocks move.

Hard-stop precheck (identical to candidate 1): the scoring path must
reproduce, bit-for-bit, every committed full-panel reference moment, every
committed per-gate-seed ``rate_a``, and each gate seed's committed
holdout-id sha256, BEFORE any candidate is simulated. Any mismatch is a
hard stop. Run ONCE; publish REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit; no statsmodels). Run from the repository root with the PSID
history files staged::

    .venv/bin/python scripts/run_gate2_candidate8.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Candidate 6 supplies the machinery this build minimally deltas once: its
# source-aligned surviving-spouse widowhood level, the committed NCHS betas,
# its single-year triangular-kernel fertility, its origin-split remarriage,
# its spousal-gap distribution draw, and -- transitively, via candidates
# 5/4/1 -- the knot-at-22 first-marriage fitter, divorce fitter, the
# vectorised simulation helpers, the precheck, the verdict assembly, and the
# report-only summary. Only remarriage is re-implemented, and even its order
# split is candidate 7's estimator (imported, never redefined) plus ONE new
# train-side rescale scalar.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate5 as c5  # noqa: E402
import run_gate2_candidate6 as c6  # noqa: E402
import run_gate2_candidate7 as c7  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v8.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
CANDIDATE7_ARTIFACT = ROOT / "runs" / "gate2_hazard_v7.json"
CANDIDATE6_ARTIFACT = ROOT / "runs" / "gate2_hazard_v6.json"
CANDIDATE5_ARTIFACT = ROOT / "runs" / "gate2_hazard_v5.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v8"
RUN_NAME = "gate2_hazard_v8"

#: This run's frozen-spec registration (issue #42, comment 4912995860).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4912995860"
)
#: The candidate-6 spec this build minimally deltas (comment 4912170754).
CANDIDATE6_REGISTRATION = c6.SPEC_REGISTRATION
CANDIDATE5_REGISTRATION = c5.SPEC_REGISTRATION
CANDIDATE1_REGISTRATION = c1.SPEC_REGISTRATION

#: The one named delta (registration comment 4912995860).
DELTA_VS_CANDIDATE6 = (
    "remarriage hazard is conditioned on the order of the marriage being "
    "entered (entering the 2nd -- exactly one prior marriage -- vs the 3rd-"
    "or-higher -- two or more prior), with the same years-since-dissolution "
    "bands x origin (divorced/widowed) x sex construction and the same add-"
    "one smoothing (wbar_diss) as candidate 1 -- candidate 7's estimator "
    "(fit_remarriage_ordered) and simulation lookup (_remarriage_probs_"
    "ordered), imported and reused byte-for-byte -- THEN both order-specific "
    "tables are multiplied by a SINGLE train-side scalar so the order-split "
    "table's exposure-weighted aggregate remarriage rate over the train "
    "dissolved person-years equals the unsplit candidate-6 table's aggregate "
    "over the SAME exposure (one scalar, computed on train only, recorded per "
    "seed); this preserves the expected aggregate remarriage count while "
    "keeping the 2nd-vs-3rd+ compositional ratio that fixes "
    "remarriage.after_divorce. NO fertility delta (candidate 7's delta 2 was "
    "falsified; fertility is byte-identical to candidate 6)"
)

# --- Frozen dials + band constants + pure helpers, reused (byte-identical;
# imported, never redefined). ---------------------------------------------
GATE_SEEDS = c1.GATE_SEEDS
SIM_SEED_BASE = c1.SIM_SEED_BASE
EXACT_ATOL = c1.EXACT_ATOL
TREND_ANCHOR_YEAR = c6.TREND_ANCHOR_YEAR  # 1995.0 (unchanged)

_ASFR_LO = c1._ASFR_LO
_ASFR_HI = c1._ASFR_HI
_STATE = c1._STATE
_STATE_ABSORB = c1._STATE_ABSORB
FERT_AGE_LO = c5.FERT_AGE_LO  # 15
FERT_AGE_HI = c5.FERT_AGE_HI  # 49

YSD_BANDS = c1.YSD_BANDS  # ((0,4),(5,9),(10,120))
YSD_LOWERS = c1.YSD_LOWERS

_bands_vec = c1._bands_vec
_parity_vec = c1._parity_vec
_divorce_probs = c1._divorce_probs
# Fertility is BYTE-IDENTICAL to candidate 6 (no delta): candidate 5's
# single-year triangular-kernel lookup, marital-state-independent.
_fertility_probs_single = c5._fertility_probs_single
_assemble_sim_panel = c1._assemble_sim_panel
Components = c1.Components

# First marriage + the surviving-spouse widowhood level + the committed NCHS
# betas are candidate 6's, reused unchanged (no delta touches them). Aliased
# for provenance.
fit_first_marriage = c6.fit_first_marriage
FirstMarriageModelC8 = c6.FirstMarriageModelC6
_widow_probs = c6._widow_probs
WIDOW_BANDS = c6.WIDOW_BANDS  # ((45,54),(55,64),(65,74),(75,120))
WIDOW_LOWERS = c6.WIDOW_LOWERS
_committed_beta_v5 = c6._committed_beta_v5

# DELTA: the order-split remarriage estimator and its simulation lookup are
# candidate 7's delta 1, IMPORTED and reused byte-for-byte. Candidate 8 adds
# only the train-side aggregate-preservation rescale (below); nothing about
# the order split itself is re-implemented.
fit_remarriage_ordered = c7.fit_remarriage_ordered
_remarriage_probs_ordered = c7._remarriage_probs_ordered
_remarriage_order_diag = c7._remarriage_order_diag

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate8_run1_cache.json"
)


# --------------------------------------------------------------------------
# DELTA: order-split remarriage (candidate 7's estimator) + the ONE new
# train-side aggregate-preservation rescale scalar
# --------------------------------------------------------------------------
def _remarriage_train_exposure(
    panel: transitions.MaritalPanel,
    order_map: pd.DataFrame,
    train_ids: set[int],
) -> pd.Series:
    """Train dissolved-exposure weights by (ysd_band, origin, sex, order_bit).

    A byte-identical reconstruction of the ``rem_den`` the reused estimator
    (:func:`candidate7.fit_remarriage_ordered`) forms internally: the same
    dissolved person-years (``marital_state`` in {divorced, widowed} with a
    defined ``years_since_dissolution``), the same YSD banding, and the same
    :func:`candidate1._parity_vec` order bit over
    :func:`candidate7._marriage_starts_by`. Returned so the aggregate-
    preservation rescale weights BOTH the order-split table and the unsplit
    candidate-6 table by the identical exposure the estimator saw. Summed over
    the order bit it equals candidate 1/6's unsplit dissolved exposure exactly
    (same ``diss`` filter), so the two aggregates are directly comparable.
    """
    py = panel.person_years
    train_py = py[py["person_id"].isin(train_ids)]
    starts_by = c7._marriage_starts_by(order_map, train_ids)
    diss = train_py[
        train_py["marital_state"].isin(("divorced", "widowed"))
        & train_py["years_since_dissolution"].notna()
    ].copy()
    diss["ysd_band"] = _bands_vec(
        diss["years_since_dissolution"].astype("int64").to_numpy(),
        YSD_LOWERS,
        len(YSD_BANDS),
    )
    diss["order_bit"] = (
        _parity_vec(
            diss["person_id"].to_numpy(),
            diss["year"].to_numpy(),
            starts_by,
        )
        >= 2
    ).astype(int)
    return diss.groupby(["ysd_band", "marital_state", "sex", "order_bit"])[
        "weight"
    ].sum()


def _aggregate_preserving_scalar(
    exposure: pd.Series,
    c6_remarriage: dict[tuple[int, str, str], float],
    remarriage_ordered: dict[tuple[int, str, str, int], float],
) -> tuple[float, dict[str, float]]:
    """The SINGLE train-side rescale scalar (candidate 8's one new operation).

    Over the train dissolved person-years, the *aggregate remarriage rate* a
    hazard table implies is the exposure-weighted mean hazard =
    ``sum(exposure(cell) * hazard(cell)) / sum(exposure(cell))`` -- the
    expected remarriage count over that exposure divided by the exposure. The
    scalar is the ratio that makes the order-split table's aggregate equal the
    unsplit candidate-6 table's aggregate over the identical exposure::

        scalar = (sum_cells E_c6(b,o,s)   * h_c6(b,o,s))
                 / (sum_cells E(b,o,s,ob) * h_split(b,o,s,ob))

    where ``E`` is :func:`_remarriage_train_exposure` (which sums over the
    order bit to ``E_c6``, so both aggregates share one total exposure).
    Multiplying BOTH order tables by ``scalar`` restores the expected
    aggregate remarriage count to candidate 6's while preserving the 2nd-vs-
    3rd+ compositional ratio exactly (a single multiplier cancels in the
    ratio). Computed on TRAIN only; the holdout is never touched.

    Returns ``(scalar, aggregates)``; ``aggregates`` records both aggregate
    rates (candidate-6 and order-split, pre- and post-rescale), the expected
    counts and the total train dissolved exposure -- all train-only.
    """
    total = 0.0
    expected_c6 = 0.0
    expected_split = 0.0
    for (b, ms, sex, ob), w in exposure.items():
        w = float(w)
        total += w
        expected_c6 += w * c6_remarriage[(int(b), ms, sex)]
        expected_split += w * remarriage_ordered[(int(b), ms, sex, int(ob))]
    if total <= 0.0 or expected_split <= 0.0:
        raise RuntimeError(
            "degenerate train remarriage exposure or aggregate; cannot form "
            "the aggregate-preservation scalar."
        )
    scalar = expected_c6 / expected_split
    aggregates = {
        "train_dissolved_exposure_weight": total,
        "candidate6_unsplit_train_aggregate": expected_c6 / total,
        "order_split_train_aggregate_prerescale": expected_split / total,
        "order_split_train_aggregate_rescaled": (scalar * expected_split)
        / total,
        "expected_remarriages_candidate6": expected_c6,
        "expected_remarriages_order_split_prerescale": expected_split,
        "expected_remarriages_order_split_rescaled": scalar * expected_split,
    }
    return scalar, aggregates


# --------------------------------------------------------------------------
# Fitted components (candidate 6's, with the one delta'd remarriage swapped)
# --------------------------------------------------------------------------
def fit_components(
    panel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    mh_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    order_map: pd.DataFrame,
    train_ids: set[int],
) -> Components:
    """Fit all five components on side B, the one delta applied.

    Starts from :func:`candidate6.fit_components` -- so first marriage,
    divorce, the surviving-spouse marriage-history widowhood LEVEL, the
    committed NCHS betas, the spousal-gap distribution draw AND the single-
    year triangular-kernel fertility are byte-identical to candidate 6 by
    construction. Then the ONE delta:

    * remarriage (``base.remarriage``) is replaced by the order-conditioned
      table -- candidate 7's estimator (:func:`fit_remarriage_ordered`),
      reused byte-for-byte -- with BOTH order tables multiplied by the single
      train-side aggregate-preservation scalar
      (:func:`_aggregate_preserving_scalar`). Candidate 6's unsplit remarriage
      table, the pre-rescale order-split cells and the scalar (with its train
      aggregates) are retained under provenance keys.

    Fertility is NOT delta'd (candidate 7's delta 2 was falsified); it stays
    candidate 6's marital-state-independent single-year table verbatim, so its
    reference cells remain byte-identical to candidate 6.
    """
    base = c6.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )

    # DELTA: order-split remarriage (candidate 7's estimator) rescaled by the
    # single train-side scalar that preserves the aggregate remarriage count.
    c6_remarriage = dict(base.remarriage)
    remarriage_ordered, wbar_diss = fit_remarriage_ordered(
        panel, order_map, train_ids
    )
    exposure = _remarriage_train_exposure(panel, order_map, train_ids)
    scalar, aggregates = _aggregate_preserving_scalar(
        exposure, c6_remarriage, remarriage_ordered
    )
    remarriage_rescaled = {
        key: value * scalar for key, value in remarriage_ordered.items()
    }
    base.remarriage = remarriage_rescaled

    base.meta["remarriage_representation"] = (
        "weighted empirical hazard by years-since-dissolution band x origin "
        "(divorced/widowed) x sex x marriage-order bit (entering 2nd vs 3rd+), "
        "add-one smoothed -- candidate 1's origin-split table with one extra "
        "order stratum (candidate 7's estimator, reused) -- THEN both order "
        "tables rescaled by one train-side scalar preserving the unsplit "
        "candidate-6 aggregate remarriage rate over the train dissolved "
        "exposure (DELTA)"
    )
    base.meta["remarriage_order_diagnostics"] = _remarriage_order_diag(
        remarriage_rescaled
    )
    base.meta["remarriage_order_prerescale_cells"] = _remarriage_order_diag(
        remarriage_ordered
    )["cells"]
    base.meta["remarriage_candidate6"] = {
        f"ysd{b}|{origin}|{sex}": rate
        for (b, origin, sex), rate in c6_remarriage.items()
    }
    base.meta["remarriage_rescale"] = {"scalar": float(scalar), **aggregates}
    base.meta["remarriage_mean_dissolved_weight_check"] = wbar_diss
    base.meta["delta_vs_candidate6"] = DELTA_VS_CANDIDATE6
    return base


# --------------------------------------------------------------------------
# Vectorised annual simulation (candidate 5's, with the delta'd widowhood
# LEVEL looked up by the surviving ego's own (age, sex))
# --------------------------------------------------------------------------
@dataclass
class _SimLookupsC8:
    mort_arr: np.ndarray  # [widow_band, sex(0=f,1=m)] surviving-spouse level
    beta_arr: np.ndarray  # [sex(0=f,1=m)] per-sex log-linear period slope
    rem_arr: np.ndarray  # [ysd_band, origin(0=div,1=wid), sex, order_bit]
    fert_arr: np.ndarray  # [age-FERT_AGE_LO, parity_band, decade_idx]
    decade_map: dict[int, int]


def _build_sim_lookups(components: Components) -> _SimLookupsC8:
    """Candidate 6's widowhood level + NCHS slope + single-year fertility,
    with the DELTA'd remarriage lookup gaining an order-bit axis.

    The mortality lookup and per-sex slope are candidate 6's, unchanged; the
    single-year fertility lookup is candidate 6's, unchanged (marital-state-
    independent, no delta). Only the remarriage lookup gains the order-bit
    axis (DELTA 1's estimator, already rescaled in ``components.remarriage``).
    """
    mort_arr = np.zeros((len(WIDOW_BANDS), 2), dtype=np.float64)
    for b, (lo, hi) in enumerate(WIDOW_BANDS):
        band = transitions.band_label(lo, hi)
        for si, sex in enumerate(("female", "male")):
            mort_arr[b, si] = components.mortality.get(f"{band}|{sex}", 0.0)

    beta = components.meta["mortality_beta_by_sex"]
    beta_arr = np.array([beta["female"], beta["male"]], dtype=np.float64)

    # DELTA: remarriage with an order-bit axis (0=entering 2nd, 1=3rd+); the
    # table is already the rescaled order-split one (fit_components).
    rem_arr = np.zeros((len(YSD_BANDS), 2, 2, 2), dtype=np.float64)
    for (b, origin, sex, ob), v in components.remarriage.items():
        oi = 0 if origin == "divorced" else 1
        si = 0 if sex == "female" else 1
        rem_arr[b, oi, si, ob] = v

    decades = sorted({d for (_a, _p, d) in components.fertility})
    decade_map = {d: i for i, d in enumerate(decades)}
    n_age = FERT_AGE_HI - FERT_AGE_LO + 1
    fert_arr = np.zeros((n_age, 4, max(len(decades), 1)), dtype=np.float64)
    for (age, pb, d), v in components.fertility.items():
        fert_arr[age - FERT_AGE_LO, pb, decade_map[d]] = v
    return _SimLookupsC8(mort_arr, beta_arr, rem_arr, fert_arr, decade_map)


# The widowhood-incidence lookup ``_widow_probs`` is candidate 6's, imported
# and reused byte-for-byte (aliased above): no delta touches widowhood.


def simulate_holdout(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: Components,
    sim_seed: int,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Candidate 6's simulation with the DELTA'd order-split remarriage.

    Byte-identical to :func:`candidate6.simulate_holdout` EXCEPT the
    remarriage probability is looked up with the ego's running marriage
    ``order`` (:func:`_remarriage_probs_ordered`, candidate 7's lookup, reused;
    the table is the rescaled order-split one) instead of candidate 6's
    origin-split ``_remarriage_probs``. The per-year uniform blocks
    (``rng.random(n_active)`` then ``rng.random(n_fertile)``) are drawn in the
    same order and size as candidate 6 -- only the remarriage THRESHOLD moves.
    First marriage is marital-state-independent and fertility is candidate 6's
    marital-state-independent single-year lookup (NO delta), so their
    reference cells -- and the ever-married shares -- are byte-identical to
    candidate 6; only remarriage, the marriage counts and the dissolved-state
    stocks move.
    """
    attrs = panel.attrs[panel.attrs["person_id"].isin(holdout_ids)].copy()
    attrs = attrs.sort_values("person_id").reset_index(drop=True)
    n = len(attrs)
    pid = attrs["person_id"].to_numpy(dtype=np.int64)
    by = attrs["birth_year"].to_numpy(dtype=np.float64)
    sex = attrs["sex"].to_numpy()
    is_male = (sex == "male").astype(np.float64)
    sy = attrs["start_exposure_year"].to_numpy(dtype=np.int64)
    ey = attrs["censor_year"].to_numpy(dtype=np.int64)
    decade = (by // 10 * 10).astype(np.int64)

    py = panel.person_years
    entry = (
        py[py["person_id"].isin(holdout_ids)]
        .sort_values("year")
        .groupby("person_id", as_index=False)
        .first()
    )
    entry_state = (
        entry.set_index("person_id")["marital_state"].reindex(pid).to_numpy()
    )
    entry_dur = entry.set_index("person_id")["marriage_duration"].reindex(pid)
    entry_ysd = entry.set_index("person_id")[
        "years_since_dissolution"
    ].reindex(pid)

    state = np.zeros(n, dtype=np.int64)
    cur_start = np.full(n, -1, dtype=np.int64)
    order = np.zeros(n, dtype=np.int64)
    diss_year = np.full(n, -1, dtype=np.int64)
    parity = np.zeros(n, dtype=np.int64)
    open_start = np.full(n, -1, dtype=np.int64)
    open_order = np.zeros(n, dtype=np.int64)

    for i in range(n):
        st = entry_state[i]
        if pd.isna(st) or st == "never_married":
            state[i] = 0
        elif st == "married":
            state[i] = 1
            d = entry_dur.iloc[i]
            d0 = int(d) if not pd.isna(d) else 0
            cur_start[i] = int(sy[i]) - d0
            order[i] = 1
            open_start[i] = cur_start[i]
            open_order[i] = 1
        elif st in ("divorced", "widowed"):
            state[i] = _STATE[st]
            j = entry_ysd.iloc[i]
            j0 = int(j) if not pd.isna(j) else 0
            diss_year[i] = int(sy[i]) - j0
            order[i] = 1
        else:
            state[i] = _STATE_ABSORB

    # Registered simulation RNG + the SPAWNED gap-draw stream (candidate 4's
    # delta 2, retained): the spawn does not advance rng's bit stream.
    rng = np.random.default_rng(sim_seed)
    gap_seed_seq = rng.bit_generator.seed_seq.spawn(1)[0]
    gap_rng = np.random.default_rng(gap_seed_seq)

    gap_dist = components.gap_dist_by_sex
    gap_arr = np.empty(n, dtype=np.float64)
    fem_mask = is_male == 0.0
    male_mask = is_male == 1.0
    n_fem = int(fem_mask.sum())
    n_male = int(male_mask.sum())
    if n_fem:
        gap_arr[fem_mask] = gap_rng.choice(gap_dist["female"], size=n_fem)
    if n_male:
        gap_arr[male_mask] = gap_rng.choice(gap_dist["male"], size=n_male)
    opp_is_male = 1.0 - is_male

    lookups = _build_sim_lookups(components)
    fert_didx = np.array(
        [lookups.decade_map.get(int(d), -1) for d in decade], dtype=np.int64
    )

    ep_person: list[int] = []
    ep_order: list[int] = []
    ep_start: list[int] = []
    ep_end: list[Any] = []
    ep_how: list[str] = []
    bi_person: list[int] = []
    bi_year: list[int] = []
    bi_order: list[int] = []

    def close_ep(idx_arr: np.ndarray, how: str, end_year: int) -> None:
        for i in idx_arr:
            ep_person.append(int(pid[i]))
            ep_order.append(int(open_order[i]))
            ep_start.append(int(open_start[i]))
            ep_end.append(int(end_year))
            ep_how.append(how)

    y0, y1 = int(sy.min()), int(ey.max())
    for y in range(y0, y1 + 1):
        active = (sy <= y) & (y <= ey)
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            continue
        age = y - by[idx]
        u = rng.random(idx.size)
        st = state[idx]

        nm = st == 0
        if nm.any():
            sub = idx[nm]
            p_fm = components.first_marriage.predict(
                age[nm], is_male[sub], decade[sub]
            )
            marry = u[nm] < p_fm
            gi = sub[marry]
            order[gi] += 1
            cur_start[gi] = y
            state[gi] = 1
            open_start[gi] = y
            open_order[gi] = order[gi]

        mar = st == 1
        if mar.any():
            sub = idx[mar]
            dur = (y - cur_start[sub]).astype(np.int64)
            p_div = _divorce_probs(dur, order[sub], components.divorce)
            sp_age = age[mar] + gap_arr[sub]
            # CANDIDATE-6 DELTA: the widowhood LEVEL is the surviving-spouse
            # marriage-history hazard, looked up by the married ego's OWN
            # (age, sex). The candidate-5 spousal-gap draw (sp_age,
            # opp_is_male) is retained byte-identically but no longer enters
            # the level; candidate 5's committed NCHS trend applies unchanged.
            p_wid = _widow_probs(
                age[mar],
                is_male[sub],
                sp_age,
                opp_is_male[sub],
                y,
                lookups.mort_arr,
                lookups.beta_arr,
            )
            um = u[mar]
            div = um < p_div
            wid = (~div) & (um < p_div + p_wid)
            gdi = sub[div]
            close_ep(gdi, "divorce", y)
            state[gdi] = 2
            diss_year[gdi] = y
            gwi = sub[wid]
            close_ep(gwi, "widowhood", y)
            state[gwi] = 3
            diss_year[gwi] = y

        diss = (st == 2) | (st == 3)
        if diss.any():
            sub = idx[diss]
            ysd = (y - diss_year[sub]).astype(np.int64)
            origin = st[diss]
            # CANDIDATE-8 DELTA: remarriage looked up with the ego's running
            # marriage order (>= 2 -> entering 3rd+); same YSD/origin/sex. The
            # table is the rescaled order-split one (candidate 7's estimator +
            # candidate 8's aggregate-preservation scalar). Reuses candidate
            # 7's _remarriage_probs_ordered byte-for-byte.
            p_rm = _remarriage_probs_ordered(
                ysd, origin, is_male[sub], order[sub], lookups.rem_arr
            )
            rm = u[diss] < p_rm
            gri = sub[rm]
            order[gri] += 1
            cur_start[gri] = y
            state[gri] = 1
            diss_year[gri] = -1
            open_start[gri] = y
            open_order[gri] = order[gri]

        # Fertility: women aged 15-49, any marital state (candidate 6's
        # single-year kernel lookup; NO delta -- marital-state-independent, so
        # byte-identical to candidate 6, same uf draw block, same threshold).
        age_all = (y - by).astype(np.int64)
        fert = (
            active
            & (sex == "female")
            & (age_all >= _ASFR_LO)
            & (age_all <= _ASFR_HI)
        )
        fidx = np.nonzero(fert)[0]
        if fidx.size:
            uf = rng.random(fidx.size)
            fage = (y - by[fidx]).astype(np.int64)
            p_birth = _fertility_probs_single(
                fage, parity[fidx], fert_didx[fidx], lookups.fert_arr
            )
            born = uf < p_birth
            gbi = fidx[born]
            for i in gbi:
                bi_person.append(int(pid[i]))
                bi_year.append(int(y))
                bi_order.append(int(parity[i]) + 1)
            parity[gbi] += 1

    still = np.nonzero(state == 1)[0]
    for i in still:
        ep_person.append(int(pid[i]))
        ep_order.append(int(open_order[i]))
        ep_start.append(int(open_start[i]))
        ep_end.append(pd.NA)
        ep_how.append("intact")

    sim_panel = _assemble_sim_panel(
        attrs, ep_person, ep_order, ep_start, ep_end, ep_how
    )
    sim_births = pd.DataFrame(
        {
            "parent_person_id": np.array(bi_person, dtype=np.int64),
            "birth_year": pd.array(bi_year, dtype="Int64"),
            "birth_order": pd.array(bi_order, dtype="Int64"),
            "record_type": pd.array(
                ["birth"] * len(bi_person), dtype="string"
            ),
            "is_event": np.ones(len(bi_person), dtype=bool),
        }
    )
    return sim_panel, sim_births


# --------------------------------------------------------------------------
# Per-seed scoring (candidate 1's, calling the candidate-6 fit + simulate)
# --------------------------------------------------------------------------
def score_seed(
    seed: int,
    panel: transitions.MaritalPanel,
    fert: transitions.FertilityPanel,
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    mh_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    order_map: pd.DataFrame,
    floor: dict[str, Any],
    tol: dict[str, float],
    report_only: list[str],
    verbose: bool,
) -> dict[str, Any]:
    """Fit side B, simulate side A, score every cell against rate_a.

    Identical to :func:`candidate1.score_seed` except it calls the
    candidate-8 :func:`fit_components` and :func:`simulate_holdout`.
    """
    t0 = time.time()
    side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_a = set(int(x) for x in side_a.person_id.unique())
    ids_b = set(int(x) for x in side_b.person_id.unique())

    components = fit_components(
        panel, demo, death_records, mh_records, birth_records, order_map, ids_b
    )
    sim_panel, sim_births = simulate_holdout(
        panel, ids_a, components, SIM_SEED_BASE + seed
    )
    sim_fert = transitions.build_fertility_panel(sim_panel, sim_births)
    cand = transitions.reference_moments(
        sim_panel, sim_fert, ids_a, weighted=True
    )

    committed_cells = {p["seed"]: p for p in floor["noise_floor_per_seed"]}[
        seed
    ]["cells"]

    def score_cell(key: str) -> dict[str, Any]:
        rate_a = float(committed_cells[key]["rate_a"])
        r_cand = float(cand[key]["rate"])
        n_cand = int(cand[key]["n_events"])
        if r_cand > 0 and rate_a > 0:
            s = float(abs(math.log(r_cand / rate_a)))
        else:
            s = float("inf")
        return {
            "r_candidate": r_cand,
            "rate_a": rate_a,
            "n_events_candidate": n_cand,
            "log_ratio_abs": s if math.isfinite(s) else None,
            "score": s,
        }

    gated_cells: dict[str, Any] = {}
    n_gated_pass = 0
    for key in sorted(tol):
        rec = score_cell(key)
        rec["tolerance"] = float(tol[key])
        rec["pass"] = bool(rec["score"] <= tol[key])
        n_gated_pass += rec["pass"]
        gated_cells[key] = rec

    report_cells: dict[str, Any] = {}
    for key in sorted(report_only):
        rec = score_cell(key)
        rec["gated"] = False
        report_cells[key] = rec

    seed_pass = n_gated_pass == len(tol)
    elapsed = round(time.time() - t0, 1)
    if verbose:
        fails = [k for k, v in gated_cells.items() if not v["pass"]]
        print(
            f"seed {seed}: {n_gated_pass}/{len(tol)} gated pass "
            f"(seed_pass={seed_pass}); fails={fails} [{elapsed}s]"
        )
    return {
        "seed": seed,
        "n_holdout_persons": len(ids_a),
        "n_train_persons": len(ids_b),
        "sim_seed": SIM_SEED_BASE + seed,
        "component_meta": components.meta,
        "gated_cells": gated_cells,
        "report_only_cells": report_cells,
        "n_gated": len(tol),
        "n_gated_pass": n_gated_pass,
        "n_gated_fail": len(tol) - n_gated_pass,
        "seed_pass": bool(seed_pass),
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------
# Modal-failure check + targeted-cell / candidate-5 movement
# --------------------------------------------------------------------------
#: The registered modal failure (comment 4912995860): the untargeted
#: lifetime-marriage boundary cell at its very tight 0.047 tolerance, on
#: seeds 3-4 -- the modal failure if candidate 8 fails.
REGISTERED_MODAL_CELL = "mean_lifetime_marriages|male"
#: The cells the one delta targets/protects (registration): the male marriage-
#: count boundary the rescale PROTECTS (does not raise) and the after-divorce
#: remarriage flow the order split FIXES.
TARGETED_CELLS = (
    "mean_lifetime_marriages|male",
    "remarriage.after_divorce",
)
#: The full set of cells whose vs-candidate-6 movement is reported (adds the
#: female lifetime-marriage cell, the persisting c1970s clip, and candidate
#: 7's two seed-0 regressors -- which should RETURN under candidate 8).
MOVEMENT_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
    "remarriage.after_divorce",
    "completed_fertility.c1970s",
    "asfr.15-19",
    "share_divorced.45-54|female",
    "share_widowed.75+|female",
)


def _cells_byte_identical_to_c6(cell: str) -> bool:
    """First-marriage, ever-married AND fertility cells match candidate 6.

    The one delta (order-split remarriage, rescaled) moves only the remarriage
    threshold: it adds no first marriages (so ``first_marriage.*`` and whether
    a person EVER married are unchanged), and fertility is candidate 6's
    marital-state-independent table under the shared RNG stream (NO delta), so
    every ``first_marriage.*``, ``ever_married_by_*``, ``asfr.*`` and
    ``completed_fertility.*`` cell must carry candidate 6's exact
    ``r_candidate``.
    """
    return (
        cell.startswith("first_marriage.")
        or cell.startswith("ever_married_by_")
        or cell.startswith("asfr.")
        or cell.startswith("completed_fertility.")
    )


def _c6_seed0_gated() -> dict[str, Any] | None:
    """Candidate 6's committed seed-0 gated cells (for the regression check)."""
    if not CANDIDATE6_ARTIFACT.exists():
        return None
    a6 = json.loads(CANDIDATE6_ARTIFACT.read_text())
    for s in a6["per_seed"]:
        if s["seed"] == 0:
            return s["gated_cells"]
    return None


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal, targeted cells, seed-0 return, and the decider."""
    fails_by_cell: dict[str, list[int]] = {}
    for f in verdict["all_failing_gated_cells"]:
        fails_by_cell.setdefault(f["cell"], []).append(f["seed"])

    by_seed = {s["seed"]: s for s in per_seed}

    def track(cell: str) -> dict[str, Any]:
        return {
            "tolerance": per_seed[0]["gated_cells"][cell]["tolerance"],
            "per_seed_score": {
                s["seed"]: s["gated_cells"][cell]["score"] for s in per_seed
            },
            "per_seed_pass": {
                s["seed"]: s["gated_cells"][cell]["pass"] for s in per_seed
            },
            "failed_seeds": sorted(fails_by_cell.get(cell, [])),
        }

    def seeds_pass_if_forgiven(forgiven: set[str]) -> int:
        n = 0
        for s in per_seed:
            ok = all(
                rec["pass"]
                for cell, rec in s["gated_cells"].items()
                if cell not in forgiven
            )
            n += ok
        return n

    # Seed-0 return check: the registration EXPECTS seed 0 to return to 46/46
    # (candidate 7's two seed-0 regressions came from the reverted delta 2 and
    # the now-neutralized order-split count effect). A regression is any cell
    # passing for candidate 6 but failing for candidate 8 on seed 0.
    seed0 = by_seed.get(0)
    c6_seed0 = _c6_seed0_gated()
    seed0_regressed: list[str] = []
    if seed0 is not None and c6_seed0 is not None:
        for cell, rec in seed0["gated_cells"].items():
            c6rec = c6_seed0.get(cell)
            if c6rec is not None and c6rec["pass"] and not rec["pass"]:
                seed0_regressed.append(cell)

    n_pass_actual = verdict["n_seeds_pass"]
    n_pass_no_modal = seeds_pass_if_forgiven({REGISTERED_MODAL_CELL})
    n_pass_no_targeted = seeds_pass_if_forgiven(set(TARGETED_CELLS))
    modal_failed = REGISTERED_MODAL_CELL in fails_by_cell
    gate_pass = verdict["gate_2_pass"]

    distinct_fail_cells = {
        f["cell"] for f in verdict["all_failing_gated_cells"]
    }
    if gate_pass:
        decider = "none (gate passed)"
    else:
        modal_flips = n_pass_no_modal >= 4
        targeted_flips = n_pass_no_targeted >= 4
        seed0_held = seed0 is not None and seed0["seed_pass"]
        if modal_flips and targeted_flips:
            decider = (
                "both independently decisive (forgiving either the modal or "
                "the targeted cells alone flips the gate to pass)"
            )
        elif targeted_flips:
            decider = (
                "the targeted cells (forgiving the delta's target/protect "
                "cells flips >= 4 seeds to pass)"
            )
        elif modal_flips:
            decider = (
                "mean_lifetime_marriages|male (the registered modal alone "
                "holds the gate; forgiving it flips >= 4 seeds to pass)"
            )
        else:
            decider = (
                "broader than the modal + targeted cells (other gated cells "
                "also hold the gate below 4 passing seeds)"
            )
        if not seed0_held:
            decider += " [seed 0 did NOT return to a clean sweep]"

    return {
        "registered_modal": (
            f"{REGISTERED_MODAL_CELL} (the untargeted lifetime-marriage "
            "boundary cell at its very tight 0.047 tolerance, on seeds 3-4; "
            "the modal failure if candidate 8 fails)"
        ),
        "modal_cell": REGISTERED_MODAL_CELL,
        "modal_failed": modal_failed,
        "modal_failed_seeds": sorted(
            fails_by_cell.get(REGISTERED_MODAL_CELL, [])
        ),
        "modal_track": track(REGISTERED_MODAL_CELL),
        "modal_is_sole_failing_cell": (
            len(distinct_fail_cells) == 1 and modal_failed
        ),
        "targeted_cells": list(TARGETED_CELLS),
        "targeted_cells_track": {c: track(c) for c in TARGETED_CELLS},
        "seed0_analysis": {
            "seed0_held_all_gated": (
                bool(seed0["seed_pass"]) if seed0 is not None else None
            ),
            "seed0_n_gated_pass": (
                seed0["n_gated_pass"] if seed0 is not None else None
            ),
            "seed0_regressed_cells_vs_candidate6": sorted(seed0_regressed),
            "note": (
                "the registration expects seed 0 to RETURN to 46/46 (its two "
                "candidate-7 regressions came from the reverted delta 2 and "
                "the now-neutralized order-split count effect); any regressed "
                "cell here is a miss of that expectation"
            ),
        },
        "decider_analysis": {
            "n_seeds_pass_actual": n_pass_actual,
            "n_seeds_pass_if_modal_forgiven": n_pass_no_modal,
            "n_seeds_pass_if_targeted_forgiven": n_pass_no_targeted,
            "decider": decider,
            "modal_decided": (not gate_pass) and (n_pass_no_modal >= 4),
        },
    }


def _candidate6_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-cell candidate-6 -> candidate-8 movement for the tracked cells."""
    if not CANDIDATE6_ARTIFACT.exists():
        return {"available": False}
    a6 = json.loads(CANDIDATE6_ARTIFACT.read_text())
    by6 = {s["seed"]: s for s in a6["per_seed"]}
    by8 = {s["seed"]: s for s in per_seed}
    seeds = sorted(by8)
    out: dict[str, Any] = {"available": True, "cells": {}}
    for cell in MOVEMENT_CELLS:
        c6_scores = {s: by6[s]["gated_cells"][cell]["score"] for s in seeds}
        c8_scores = {s: by8[s]["gated_cells"][cell]["score"] for s in seeds}
        c6_pass = sum(by6[s]["gated_cells"][cell]["pass"] for s in seeds)
        c8_pass = sum(by8[s]["gated_cells"][cell]["pass"] for s in seeds)
        out["cells"][cell] = {
            "tolerance": by8[seeds[0]]["gated_cells"][cell]["tolerance"],
            "candidate6_per_seed_score": c6_scores,
            "candidate8_per_seed_score": c8_scores,
            "candidate6_mean_score": float(np.mean(list(c6_scores.values()))),
            "candidate8_mean_score": float(np.mean(list(c8_scores.values()))),
            "candidate6_n_seeds_pass": c6_pass,
            "candidate8_n_seeds_pass": c8_pass,
        }
    return out


def _candidate7_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-cell candidate-7 -> candidate-8 movement for the tracked cells.

    Candidate 7 fired both deltas; candidate 8 keeps only the rescaled order
    split. This records how the tracked cells moved from the falsified two-
    delta build to the rescaled one-delta build -- in particular whether
    candidate 7's seed-0 regressors (asfr.15-19, share_divorced.45-54|female)
    recover.
    """
    if not CANDIDATE7_ARTIFACT.exists():
        return {"available": False}
    a7 = json.loads(CANDIDATE7_ARTIFACT.read_text())
    by7 = {s["seed"]: s for s in a7["per_seed"]}
    by8 = {s["seed"]: s for s in per_seed}
    seeds = sorted(by8)
    out: dict[str, Any] = {"available": True, "cells": {}}
    for cell in MOVEMENT_CELLS:
        c7_scores = {s: by7[s]["gated_cells"][cell]["score"] for s in seeds}
        c8_scores = {s: by8[s]["gated_cells"][cell]["score"] for s in seeds}
        c7_pass = sum(by7[s]["gated_cells"][cell]["pass"] for s in seeds)
        c8_pass = sum(by8[s]["gated_cells"][cell]["pass"] for s in seeds)
        out["cells"][cell] = {
            "candidate7_per_seed_score": c7_scores,
            "candidate8_per_seed_score": c8_scores,
            "candidate7_n_seeds_pass": c7_pass,
            "candidate8_n_seeds_pass": c8_pass,
        }
    return out


def _seed0_full_movement(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Every seed-0 gated cell's candidate-6 -> candidate-8 movement."""
    c6_seed0 = _c6_seed0_gated()
    seed0 = next((s for s in per_seed if s["seed"] == 0), None)
    if c6_seed0 is None or seed0 is None:
        return {"available": False}
    cells: dict[str, Any] = {}
    n_moved = n_regressed = n_improved = 0
    for cell, rec in seed0["gated_cells"].items():
        c6rec = c6_seed0[cell]
        moved = abs(rec["score"] - c6rec["score"]) > EXACT_ATOL
        regressed = c6rec["pass"] and not rec["pass"]
        improved = (not c6rec["pass"]) and rec["pass"]
        n_moved += moved
        n_regressed += regressed
        n_improved += improved
        cells[cell] = {
            "candidate6_score": c6rec["score"],
            "candidate8_score": rec["score"],
            "candidate6_pass": c6rec["pass"],
            "candidate8_pass": rec["pass"],
            "moved": bool(moved),
        }
    return {
        "available": True,
        "note": (
            "seed 0 was candidate 6's only passing seed (46/46); this tracks "
            "how each of its 46 gated cells moved under the one delta and "
            "whether it returned to a clean sweep (the registration's primary "
            "expectation)"
        ),
        "n_cells": len(cells),
        "n_moved": n_moved,
        "n_regressed": n_regressed,
        "n_improved": n_improved,
        "seed0_held_all_gated": bool(seed0["seed_pass"]),
        "cells": cells,
    }


def _identity_vs_candidate6(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """First-marriage, ever-married AND fertility cells match candidate 6.

    The one delta moves only the remarriage threshold; it adds no first
    marriages and fertility is candidate 6's marital-state-independent table
    (NO delta) under the shared RNG stream. So every ``first_marriage.*``,
    ``ever_married_by_*``, ``asfr.*`` and ``completed_fertility.*`` gated cell
    must carry candidate 6's exact ``r_candidate`` -- INCLUDING fertility,
    unlike candidate 7 (whose delta 2 moved it). Records the maximum per-cell
    deviation as the byte-identity attestation.
    """
    if not CANDIDATE6_ARTIFACT.exists():
        return {"available": False}
    a6 = json.loads(CANDIDATE6_ARTIFACT.read_text())
    by6 = {s["seed"]: s for s in a6["per_seed"]}
    max_dev = 0.0
    n_cells = 0
    n_fertility = 0
    for seed in sorted(s["seed"] for s in per_seed):
        s8 = next(s for s in per_seed if s["seed"] == seed)
        for cell, rec in s8["gated_cells"].items():
            if _cells_byte_identical_to_c6(cell):
                dev = abs(
                    rec["r_candidate"]
                    - by6[seed]["gated_cells"][cell]["r_candidate"]
                )
                max_dev = max(max_dev, dev)
                n_cells += 1
                if cell.startswith("asfr.") or cell.startswith(
                    "completed_fertility."
                ):
                    n_fertility += 1
    return {
        "available": True,
        "note": (
            "first_marriage.*, ever_married_by_*, asfr.* and "
            "completed_fertility.* gated cells are byte-identical to candidate "
            "6 (the delta moves only the remarriage threshold; first marriage "
            "and fertility are marital-state-independent under the shared RNG "
            "stream, and there is NO fertility delta -- so fertility, unlike "
            "candidate 7, does not move)"
        ),
        "n_cells_checked": n_cells,
        "n_fertility_cells_checked": n_fertility,
        "max_abs_r_candidate_deviation_vs_candidate6": float(max_dev),
        "byte_identical": bool(max_dev <= EXACT_ATOL),
    }


def _remarriage_rescale_block(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-seed train-side rescale scalars + the aggregate-preservation
    attestation (the DELTA's one new operation).

    For each seed, reads the scalar and the train aggregates from the seed's
    ``component_meta['remarriage_rescale']`` and checks that the order-split
    table's rescaled train aggregate equals the unsplit candidate-6 aggregate
    to float precision -- the property the scalar is defined to enforce.
    """
    per_seed_rows: dict[int, Any] = {}
    max_resid = 0.0
    for s in per_seed:
        rc = s["component_meta"]["remarriage_rescale"]
        seed = s["seed"]
        resid = abs(
            rc["order_split_train_aggregate_rescaled"]
            - rc["candidate6_unsplit_train_aggregate"]
        )
        max_resid = max(max_resid, resid)
        per_seed_rows[seed] = {
            "scalar": rc["scalar"],
            "candidate6_unsplit_train_aggregate": rc[
                "candidate6_unsplit_train_aggregate"
            ],
            "order_split_train_aggregate_prerescale": rc[
                "order_split_train_aggregate_prerescale"
            ],
            "order_split_train_aggregate_rescaled": rc[
                "order_split_train_aggregate_rescaled"
            ],
            "train_dissolved_exposure_weight": rc[
                "train_dissolved_exposure_weight"
            ],
        }
    scalars = {seed: row["scalar"] for seed, row in per_seed_rows.items()}
    return {
        "note": (
            "one scalar per seed, computed on TRAIN only: both order-split "
            "remarriage tables are multiplied by it so their exposure-weighted "
            "aggregate remarriage rate over the train dissolved person-years "
            "equals the unsplit candidate-6 table's aggregate over the same "
            "exposure -- preserving the expected aggregate remarriage count "
            "while keeping the 2nd-vs-3rd+ compositional ratio"
        ),
        "scalar_per_seed": scalars,
        "scalar_mean": float(np.mean(list(scalars.values()))),
        "scalar_min": float(min(scalars.values())),
        "scalar_max": float(max(scalars.values())),
        "per_seed": per_seed_rows,
        "aggregate_preservation_max_abs_residual": float(max_resid),
        "aggregate_preserved": bool(max_resid <= 1e-9),
    }


def _beta_block(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Candidate 5's committed NCHS betas, applied unchanged via candidate 6.

    Byte-identical to candidate 6 (no delta touches widowhood or its trend).
    """
    committed = per_seed[0]["component_meta"]["mortality_beta_by_sex"]
    return {
        "beta_sex_committed_candidate5": committed,
        "beta_sex_source": (
            "candidate 6's fit_components sets the committed candidate-5 NCHS "
            "per-sex slopes (runs/gate2_hazard_v5.json "
            "mortality_trend_beta_comparison.beta_sex_nchs); byte-identical to "
            "candidate 6, no delta touches widowhood or its trend"
        ),
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins, with the candidate-8 schema + c1-c7 + v6/v7 shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for n in (1, 2, 3, 4, 5, 6, 7):
        pins[f"candidate{n}_runner"] = f"scripts/run_gate2_candidate{n}.py"
        pins[f"candidate{n}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{n}.py"
        )
    pins["candidate6_artifact"] = "runs/gate2_hazard_v6.json"
    pins["candidate6_artifact_sha256"] = c1._sha_of_file(CANDIDATE6_ARTIFACT)
    pins["candidate7_artifact"] = "runs/gate2_hazard_v7.json"
    pins["candidate7_artifact_sha256"] = c1._sha_of_file(CANDIDATE7_ARTIFACT)
    pins["delta"] = (
        "order-split remarriage (2nd vs 3rd+; candidate 7's estimator, reused) "
        "rescaled by one train-side aggregate-preservation scalar; NO "
        "fertility delta (byte-identical to candidate 6)"
    )
    return pins


def _model_block() -> dict[str, Any]:
    """Candidate 6's model block, edited for the one candidate-8 delta."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a mortality-"
            "composed widowhood component (candidate 6 + one named delta: "
            "remarriage additionally conditioned on marriage order -- entering "
            "2nd vs 3rd+ -- then rescaled by a single train-side scalar that "
            "preserves the unsplit candidate-6 aggregate remarriage count)"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "delta_vs_candidate6": DELTA_VS_CANDIDATE6,
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic spline on age, "
                "knots 20/22/25/30/40, sex, birth-decade cohort -- BYTE-"
                "IDENTICAL to candidate 6 (no delta touches first marriage; "
                "its reference cells and the ever-married shares are byte-"
                "identical because first marriage is marital-state-"
                "independent)"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x order "
                "(1st vs 2+), add-one smoothed (byte-identical fit to "
                "candidates 1-6; its simulated cells move only through the "
                "delta's effect on the married-state exposure)"
            ),
            "widowhood": (
                "COMPOSED, parametric in period; the spouse-death hazard "
                "LEVEL is candidate 6's train marriage-history surviving-"
                "spouse widowhood incidence over WIDOWHOOD_AGE_BANDS x "
                "surviving-spouse sex, with candidate 5's committed NCHS betas "
                "-- BYTE-IDENTICAL fit to candidate 6 (no delta touches "
                "widowhood; its simulated cells move only through the delta's "
                "effect on the married-state exposure)"
            ),
            "remarriage": (
                "DELTA: weighted empirical hazard by years-since-dissolution "
                "band x origin (divorced/widowed) x sex x MARRIAGE ORDER "
                "(entering 2nd vs 3rd+), add-one smoothed -- candidate 1's "
                "origin-split table with one extra order stratum (candidate "
                "7's estimator, reused byte-for-byte) -- THEN both order tables "
                "multiplied by one train-side scalar so the order-split "
                "aggregate remarriage rate over the train dissolved exposure "
                "equals the unsplit candidate-6 aggregate. Keeps candidate 7's "
                "after-divorce compositional fix while neutralizing its count-"
                "lowering side effect"
            ),
            "fertility": (
                "single-year-of-age rates within parity (0/1/2/3+) x "
                "birth-decade cohort, triangular-kernel smoothed over age "
                "(candidate 5's delta 2, inherited via candidate 6) -- BYTE-"
                "IDENTICAL to candidate 6; NO fertility delta (candidate 7's "
                "marital-status delta 2 was falsified), and its reference "
                "cells are byte-identical because fertility is marital-state-"
                "independent under the shared RNG stream"
            ),
        },
        "registered_ambiguity_resolutions": {
            "remarriage_order_definition": (
                "the order bit is 1 iff the person has >= 2 prior marriages "
                "(entering the 3rd or higher), else 0 (entering the 2nd); the "
                "prior count at each dissolved person-year and remarriage "
                "event is the number of order_map marriages starting strictly "
                "before that year (candidate1._parity_vec); in simulation the "
                "ego's running marriage order counter (>= 2 -> 3rd+) is used "
                "-- candidate 7's estimator and lookup, reused byte-for-byte"
            ),
            "aggregate_preservation_scalar": (
                "ONE scalar per seed = (unsplit candidate-6 table's exposure-"
                "weighted aggregate remarriage rate over the train dissolved "
                "person-years) / (order-split table's exposure-weighted "
                "aggregate over the SAME train exposure); both order tables "
                "are multiplied by it. Computed on TRAIN only, recorded per "
                "seed. Preserves the expected aggregate remarriage count while "
                "leaving the 2nd-vs-3rd+ compositional ratio (the after-"
                "divorce fix) exactly intact, since a single multiplier "
                "cancels in the ratio"
            ),
            "no_fertility_delta": (
                "candidate 7's delta 2 (fertility x marital status) was "
                "falsified and is NOT applied; fertility is byte-identical to "
                "candidate 6 (the same single-year triangular-kernel table and "
                "the same marital-state-independent simulation lookup)"
            ),
            "everything_else": (
                "the surviving-spouse marriage-history widowhood level, the "
                "committed NCHS betas, the knot-at-22 first-marriage spline, "
                "divorce, the single-year fertility, the spousal-gap "
                "distribution draw, the competing-risk step, the RNG rule "
                "default_rng(4200 + seed), the spawned gap-draw stream, one "
                "sequence per person, and the locked protocol are byte-"
                "identical to candidate 6"
            ),
        },
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(
    verbose: bool = True, cache_path: Path | None = None
) -> dict[str, Any]:
    started = time.time()
    cache_path = cache_path or DEFAULT_CACHE
    cache = c1._load_cache(cache_path)

    thresholds = c1.load_gate2_thresholds()
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_2 thresholds are not locked; the pre-registered run may "
            "only execute against locked thresholds."
        )
    tol = c1.gated_tolerances(thresholds)
    if len(tol) != 46:
        raise RuntimeError(
            f"expected 46 gated tolerances, got {len(tol)} from gates.yaml."
        )
    report_only = list(thresholds["report_only"])

    floor = json.loads(FLOOR_RUN.read_text())
    gated_set = set(floor["gate_partition"]["gate_eligible"])
    if set(tol) != gated_set:
        raise RuntimeError(
            "gates.yaml gated tolerances do not match the floor's "
            "gate_partition; refusing to score a mismatched cell set."
        )

    # Preflight: candidate 6 (the movement baseline + the fit base, which
    # chains candidate 5 for the committed NCHS beta), candidate 5, and
    # candidate 7 (the reused-estimator provenance + the vs-c7 movement)
    # artifacts must be present, and the candidate-5 NCHS references
    # (candidate 6's fit imports candidate 5's chain) must be present.
    for name, path in (
        ("candidate-6", CANDIDATE6_ARTIFACT),
        ("candidate-5", CANDIDATE5_ARTIFACT),
        ("candidate-7", CANDIDATE7_ARTIFACT),
    ):
        if not path.exists():
            raise RuntimeError(
                f"{name} artifact missing at {path}; required for the run."
            )
    for year, path in c5.NCHS_LIFE_TABLE_PATHS.items():
        if not path.exists():
            raise RuntimeError(
                f"NCHS life-table reference for {year} missing at {path}; "
                "run scripts/fetch_nchs_life_tables_historical.py first."
            )
    _committed_beta_v5()  # fail fast if the committed betas drifted

    mh_records = marriage.marriage_history()
    birth_records = g2f.births.birth_history()
    death_records = g2f.deaths.read_death_records()
    demo = g2f.panels.demographic_panel()
    panel, fert, data_meta = g2f.load_panels()
    order_map = c1._order_map(mh_records)
    if verbose:
        print(
            f"panel: {data_meta['n_person_years']} person-years, "
            f"{data_meta['panel_persons_weighted']} persons"
        )

    precheck = c1.run_precheck(panel, fert, floor)
    if verbose:
        print(
            "precheck all_reproduced_exactly="
            f"{precheck['all_reproduced_exactly']} "
            f"(ref dev={precheck['reference_moments_max_abs_deviation']:.2e}, "
            f"rate_a dev={precheck['rate_a_max_abs_deviation']:.2e}, "
            f"sha_all={precheck['holdout_sha256_all_match']})"
        )
    if not precheck["all_reproduced_exactly"]:
        raise RuntimeError(
            "Scoring path does not reproduce the committed gate-2 floor "
            "(reference moments / per-seed rate_a / holdout sha256) to bit "
            "precision; refusing to proceed."
        )

    per_seed: list[dict[str, Any]] = []
    for seed in GATE_SEEDS:
        key = f"seed_{seed}"
        if key in cache:
            if verbose:
                print(f"seed {seed}: cached")
            per_seed.append(cache[key])
            continue
        result = score_seed(
            seed,
            panel,
            fert,
            demo,
            death_records,
            mh_records,
            birth_records,
            order_map,
            floor,
            tol,
            report_only,
            verbose,
        )
        cache[key] = json.loads(json.dumps(result, default=c1._json_default))
        c1._save_cache(cache_path, cache)
        per_seed.append(cache[key])

    verdict = c1.build_verdict(per_seed, tol)
    report_block = c1.report_only_summary(per_seed, report_only)
    seed_conjunction = [
        {
            "seed": s["seed"],
            "n_gated_pass": s["n_gated_pass"],
            "n_gated_fail": s["n_gated_fail"],
            "seed_pass": s["seed_pass"],
        }
        for s in per_seed
    ]

    modal = _modal_failure_check(verdict, per_seed)
    candidate6_comparison = _candidate6_comparison(per_seed)
    candidate7_comparison = _candidate7_comparison(per_seed)
    seed0_full_movement = _seed0_full_movement(per_seed)
    identity = _identity_vs_candidate6(per_seed)
    rescale_block = _remarriage_rescale_block(per_seed)
    beta_block = _beta_block(per_seed)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 8",
        "spec_registration": SPEC_REGISTRATION,
        "candidate6_registration": CANDIDATE6_REGISTRATION,
        "candidate5_registration": CANDIDATE5_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "delta_vs_candidate6": DELTA_VS_CANDIDATE6,
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79 + flip #81); "
            "protocol/views/tolerances read at runtime, no threshold moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.35-0.45",
            "conjunction_estimate": 0.4,
            "component_probabilities": {
                "seed0_returns_to_46_of_46": 0.8,
                "remarriage_after_divorce_keeps_fix": 0.85,
                "mean_lifetime_marriages_male_seeds_3_4_clear": 0.45,
                "completed_fertility_c1970s_seed2_absorbed": 0.5,
            },
            "modal_failure": (
                "mean_lifetime_marriages|male at its 0.047 tolerance on seeds "
                "3-4 (the male marriage-count boundary; the rescale PROTECTS "
                "the aggregate count but does not RAISE it, and the under-"
                "production is in first-marriage/divorce/widowhood "
                "composition, untouched)"
            ),
            "component_reads": (
                "seed 0 should return to 46/46 (its two candidate-7 "
                "regressions came from the reverted delta 2 and the now-"
                "neutralized order-split count effect, ~0.8); "
                "remarriage.after_divorce keeps its fix (~0.85); the male "
                "marriage-count boundary cells (seeds 3-4) are only PROTECTED, "
                "not raised -- the under-production is in first-marriage/"
                "divorce/widowhood composition, untouched (~0.45 they clear on "
                "the count-neutral draw); seed 2's completed_fertility.c1970s "
                "clip persists byte-identically (seed 2 passes only if "
                "everything else holds, ~0.5)"
            ),
            "next_step_if_fail": (
                "if candidate 8 fails on mean_lifetime_marriages|male seeds "
                "3-4 again, the next registration must RAISE marriage "
                "production honestly (the widowed-remarriage pathway the "
                "report-only cells show under-producing) rather than protect "
                "counts, still under the one-shot rule"
            ),
            "delta_vs_candidate6": DELTA_VS_CANDIDATE6,
            "registration": SPEC_REGISTRATION,
        },
        "model": _model_block(),
        "protocol": {
            "option": "a (gate-1 mirror; LOCKED gates.yaml gate_2)",
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel.attrs, 'person_id', fraction=0.5, seed=s); side A = "
                "the holdout, side B = the train complement"
            ),
            "gate_seeds": list(GATE_SEEDS),
            "sim_rng_rule": "numpy.random.default_rng(4200 + seed)",
            "one_sequence_per_person": True,
            "scored_against": (
                "side A's own empirical rate (rate_a in "
                "runs/gate2_floors_v2.json noise_floor_per_seed)"
            ),
            "statistic": "|ln(r_candidate / rate_a)| per cell",
            "conjunction": (
                "all 46 gated cells per seed AND >= 4 of 5 gate seeds"
            ),
            "weight_definition": (
                "person-constant most-recent positive PSID cross-sectional "
                "weight; every gated statistic weighted, none unweighted"
            ),
        },
        "data": data_meta,
        "precheck": precheck,
        "per_seed": per_seed,
        "seed_conjunction": seed_conjunction,
        "report_only": report_block,
        "modal_failure_materialized": modal,
        "remarriage_rescale": rescale_block,
        "candidate6_comparison": candidate6_comparison,
        "candidate7_comparison": candidate7_comparison,
        "seed0_full_movement": seed0_full_movement,
        "identity_vs_candidate6": identity,
        "mortality_beta": beta_block,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "candidate6_registration": CANDIDATE6_REGISTRATION,
            "candidate1_registration": CANDIDATE1_REGISTRATION,
            "floor_run": "runs/gate2_floors_v2.json",
            "faithful_candidate_oc": floor["faithful_candidate_oc"][
                "p_gate_pass_4_of_5"
            ],
        },
        "revision_pins": _revision_pins(thresholds),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        v = artifact["verdict"]
        print(
            f"\nVERDICT: gate_2_pass={v['gate_2_pass']} "
            f"({v['n_seeds_pass']}/5 seeds pass)"
        )
        print(f"seed_pass: {v['seed_pass']}")
        print(
            "seed 0 held all gated: "
            f"{modal['seed0_analysis']['seed0_held_all_gated']}; "
            "registered modal (mean_lifetime_marriages|male) failed seeds "
            f"{modal['modal_failed_seeds']}; "
            f"decider={modal['decider_analysis']['decider']}"
        )
        print(
            "remarriage rescale scalars per seed: "
            f"{rescale_block['scalar_per_seed']}; aggregate_preserved="
            f"{rescale_block['aggregate_preserved']}"
        )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE),
        help="Incremental per-seed cache path (outside runs/).",
    )
    args = parser.parse_args()
    artifact = run(verbose=True, cache_path=Path(args.cache))
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
