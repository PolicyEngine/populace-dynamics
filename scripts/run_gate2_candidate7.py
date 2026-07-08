"""Gate-2 candidate 7 (run 1): candidate 6 + two named structural deltas.

The SEVENTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42
comment 4912542742 (``SPEC_REGISTRATION``): candidate 6's frozen spec
(comment 4912170754) verbatim EXCEPT TWO deltas, both standard demographic
structure rather than calibration. One-shot; no constant moves after the
registration comment.

The two deltas vs candidate 6 (everything else byte-identical)
--------------------------------------------------------------
**Delta 1 -- remarriage x marriage order.** Candidate 6's remarriage is the
origin-split weighted empirical hazard by years-since-dissolution band x
origin (divorced/widowed) x sex, add-one smoothed (candidate 1's table,
inherited byte-identically). Candidate 7 ADDITIONALLY conditions it on the
order of the marriage being entered -- entering the 2nd marriage (exactly one
prior marriage) vs the 3rd-or-higher (two or more prior) -- with the same YSD
bands x origin x sex construction and the same add-one smoothing. The prior
marriage count at each dissolved person-year and each remarriage event is the
number of the person's marriages (``order_map``) that started strictly before
that year (:func:`candidate1._parity_vec` over the person's marriage start
years); the order bit is 1 iff >= 2 prior marriages (entering 3rd+), else 0
(entering 2nd). Higher-order remarriage rates are lower in the data, lowering
simulated mean lifetime marriage counts where multi-marriage persons drive
them (the mechanism behind the ``mean_lifetime_marriages`` clips) and moving
``remarriage.after_divorce``. In simulation the level is looked up by the
married ego's running marriage ``order`` counter (>= 2 -> entering 3rd+).

**Delta 2 -- fertility x current simulated marital status.** Candidate 6's
fertility is the single-year-of-age triangular-kernel rate within parity
(0/1/2/3+) x birth-decade cohort (candidate 5's delta 2, inherited byte-
identically). Candidate 7 ADDITIONALLY conditions it on the woman's marital
status in the year (married vs not), with the SAME single-year triangular
kernel (bandwidth 3) applied over age WITHIN each (parity_band, birth_decade,
married) stratum. In the train fit the woman-year and birth marital status is
the panel person-year ``marital_state`` (the state ENTERING the year, married
vs not -- the same discrete-time classification the denominator uses); in
simulation it is the woman's current simulated marital state ENTERING the
year (captured before that year's marital transitions, so the fit and the
simulation classify a woman-year identically). This targets the persistent
``completed_fertility.c1970s`` integration miss (byte-identical failing since
candidate 1).

Everything else -- the source-aligned surviving-spouse marriage-history
widowhood LEVEL (candidate 6's delta), the committed candidate-5 NCHS
per-sex period-trend betas, the knot-at-22 20/22/25/30/40 first-marriage
spline, divorce, the spousal-age-gap DISTRIBUTION draw, the competing-risk
step, the RNG rule ``numpy.random.default_rng(4200 + seed)``, the spawned
gap-draw stream, one simulated sequence per person, and the LOCKED gate-2
protocol -- is byte-identical to candidate 6. This runner IMPORTS candidate
6's machinery (which chains candidates 5/4/1) and reuses every unchanged
function: the unchanged components come straight from
``candidate6.fit_components`` (so first marriage, divorce, the surviving-
spouse widowhood level and the spousal-gap distribution are provably
identical to candidate 6), and only the two delta'd fields are recomputed --
remarriage (now order-conditioned) and fertility (now marital-status-
conditioned). The scoring path, precheck, and verdict assembly are candidate
1's, imported unchanged.

Because both deltas move only THRESHOLDS -- the per-year uniform blocks
(``rng.random(n_active)`` then ``rng.random(n_fertile)``) are drawn in the
same order and size as candidate 6 -- first marriage (marital-state-
independent) and the ever-married shares (first-marriage-driven) are byte-
identical to candidate 6; fertility, remarriage, the marriage-count and the
dissolved-state stocks move.

Hard-stop precheck (identical to candidate 1): the scoring path must
reproduce, bit-for-bit, every committed full-panel reference moment, every
committed per-gate-seed ``rate_a``, and each gate seed's committed
holdout-id sha256, BEFORE any candidate is simulated. Any mismatch is a
hard stop. Run ONCE; publish REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit; no statsmodels). Run from the repository root with the PSID
history files staged::

    .venv/bin/python scripts/run_gate2_candidate7.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Candidate 6 supplies the machinery this build minimally deltas twice: its
# source-aligned surviving-spouse widowhood level, the committed NCHS trend,
# its single-year triangular-kernel fertility, its origin-split remarriage,
# its spousal-gap distribution draw, and -- transitively, via candidates
# 5/4/1 -- the knot-at-22 first-marriage fitter, divorce fitter, the
# vectorised simulation helpers, the precheck, the verdict assembly, and the
# report-only summary. Only the two delta'd fields (remarriage's order
# stratum and fertility's marital-status stratum) are re-implemented.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate5 as c5  # noqa: E402
import run_gate2_candidate6 as c6  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v7.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
CANDIDATE6_ARTIFACT = ROOT / "runs" / "gate2_hazard_v6.json"
CANDIDATE5_ARTIFACT = ROOT / "runs" / "gate2_hazard_v5.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v7"
RUN_NAME = "gate2_hazard_v7"

#: This run's frozen-spec registration (issue #42, comment 4912542742).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4912542742"
)
#: The candidate-6 spec this build minimally deltas (comment 4912170754).
CANDIDATE6_REGISTRATION = c6.SPEC_REGISTRATION
CANDIDATE5_REGISTRATION = c5.SPEC_REGISTRATION
CANDIDATE1_REGISTRATION = c1.SPEC_REGISTRATION

#: The two named deltas (registration comment 4912542742).
DELTA1_VS_CANDIDATE6 = (
    "remarriage hazard is ADDITIONALLY conditioned on the order of the "
    "marriage being entered (entering the 2nd marriage -- exactly one prior "
    "marriage -- vs the 3rd-or-higher -- two or more prior), with the same "
    "years-since-dissolution bands x origin (divorced/widowed) x sex "
    "construction and the same add-one smoothing (wbar_diss) as candidate 1; "
    "the prior-marriage count at each dissolved person-year and remarriage "
    "event is the number of the person's order_map marriages starting "
    "strictly before that year (candidate1._parity_vec), order_bit = 1 iff "
    ">= 2 prior marriages; consumed in simulation by the ego's running "
    "marriage order counter (>= 2 -> entering 3rd+)"
)
DELTA2_VS_CANDIDATE6 = (
    "fertility rate is ADDITIONALLY conditioned on the woman's current "
    "simulated marital status (married vs not), with the same single-year "
    "triangular kernel (bandwidth 3) applied over age WITHIN each "
    "(parity_band, birth-decade, married) stratum; the train marital status "
    "is the panel person-year marital_state (state entering the year) for "
    "both the woman-year denominator and the birth numerator, and the "
    "simulation uses the woman's marital state entering the year (captured "
    "before that year's marital transitions) so fit and simulation classify "
    "a woman-year identically"
)
DELTAS_VS_CANDIDATE6 = [DELTA1_VS_CANDIDATE6, DELTA2_VS_CANDIDATE6]

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

YSD_BANDS = c1.YSD_BANDS
YSD_LOWERS = c1.YSD_LOWERS

_bands_vec = c1._bands_vec
_parity_vec = c1._parity_vec
_divorce_probs = c1._divorce_probs
_assemble_sim_panel = c1._assemble_sim_panel
Components = c1.Components

# The candidate-7 first-marriage model IS candidate 3's (knot-at-22 spline),
# inherited through candidate 6; no delta touches it. Aliased for provenance.
fit_first_marriage = c6.fit_first_marriage
FirstMarriageModelC7 = c6.FirstMarriageModelC6

# The surviving-spouse widowhood level (candidate 6's delta) and its lookup
# are inherited byte-identically; no delta touches widowhood.
_widow_probs = c6._widow_probs
WIDOW_BANDS = c6.WIDOW_BANDS
WIDOW_LOWERS = c6.WIDOW_LOWERS

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate7_run1_cache.json"
)


# --------------------------------------------------------------------------
# DELTA 1: remarriage hazard, additionally conditioned on marriage order
# --------------------------------------------------------------------------
def _marriage_starts_by(
    order_map: pd.DataFrame, train_ids: set[int]
) -> dict[int, np.ndarray]:
    """Per-(train)-person sorted marriage start years, from ``order_map``.

    Fed to :func:`candidate1._parity_vec` as a running "prior marriages
    before year y" count -- the same helper candidate 1 uses for parity.
    """
    om = order_map[order_map["person_id"].isin(train_ids)]
    return {
        int(p): np.sort(g["start_year"].astype("int64").to_numpy())
        for p, g in om.groupby("person_id")
    }


def fit_remarriage_ordered(
    panel: transitions.MaritalPanel,
    order_map: pd.DataFrame,
    train_ids: set[int],
) -> tuple[dict[tuple[int, str, str, int], float], float]:
    """DELTA 1: remarriage hazard by (ysd_band, origin, sex, order_bit).

    Candidate 1's origin-split remarriage construction -- weighted empirical
    hazard by years-since-dissolution band x origin (divorced/widowed) x sex,
    add-one smoothed -- ADDITIONALLY stratified by the order of the marriage
    being entered: ``order_bit`` = 1 iff the person has >= 2 prior marriages
    (entering the 3rd or higher), else 0 (entering the 2nd). The prior count
    at each dissolved person-year / remarriage event is the number of the
    person's ``order_map`` marriages starting strictly before that year
    (:func:`candidate1._parity_vec`). Same YSD bands, same per-cell add-one
    smoothing (``wbar_diss``, computed over all dissolved rows) as candidate
    1 -- one extra stratifying bit, nothing else changes.

    Returns the ``(ysd_band, origin, sex, order_bit) -> rate`` table and the
    mean dissolved weight (byte-identical to candidate 1's ``wbar_diss``).
    """
    py = panel.person_years
    ev = panel.events
    train_py = py[py["person_id"].isin(train_ids)]
    train_ev = ev[ev["person_id"].isin(train_ids)]
    starts_by = _marriage_starts_by(order_map, train_ids)

    diss = train_py[
        train_py["marital_state"].isin(("divorced", "widowed"))
        & train_py["years_since_dissolution"].notna()
    ].copy()
    rem_ev = train_ev[
        (train_ev["transition"] == "remarriage")
        & train_ev["years_since_dissolution"].notna()
    ].copy()
    wbar_diss = float(diss["weight"].mean()) if len(diss) else 1.0

    diss["ysd_band"] = _bands_vec(
        diss["years_since_dissolution"].astype("int64").to_numpy(),
        YSD_LOWERS,
        len(YSD_BANDS),
    )
    rem_ev["ysd_band"] = _bands_vec(
        rem_ev["years_since_dissolution"].astype("int64").to_numpy(),
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
    rem_ev["order_bit"] = (
        _parity_vec(
            rem_ev["person_id"].to_numpy(),
            rem_ev["year"].to_numpy(),
            starts_by,
        )
        >= 2
    ).astype(int)

    rem_den = diss.groupby(["ysd_band", "marital_state", "sex", "order_bit"])[
        "weight"
    ].sum()
    rem_num = rem_ev.groupby(["ysd_band", "origin", "sex", "order_bit"])[
        "weight"
    ].sum()

    remarriage: dict[tuple[int, str, str, int], float] = {}
    for b, origin, sex, ob in itertools.product(
        range(len(YSD_BANDS)),
        ("divorced", "widowed"),
        ("female", "male"),
        (0, 1),
    ):
        wnum = float(rem_num.get((b, origin, sex, ob), 0.0))
        wden = float(rem_den.get((b, origin, sex, ob), 0.0))
        remarriage[(b, origin, sex, ob)] = (wnum + wbar_diss) / (
            wden + 2.0 * wbar_diss
        )
    return remarriage, wbar_diss


def _remarriage_order_diag(
    remarriage: dict[tuple[int, str, str, int], float],
) -> dict[str, Any]:
    """Provenance of the order-conditioned remarriage table."""
    cells = {
        f"ysd{b}|{origin}|{sex}|{'3plus' if ob else '2nd'}": rate
        for (b, origin, sex, ob), rate in remarriage.items()
    }
    return {
        "representation": (
            "weighted empirical remarriage hazard by years-since-dissolution "
            "band x origin (divorced/widowed) x sex x marriage-order bit "
            "(0=entering 2nd, 1=entering 3rd+), add-one smoothed; the "
            "candidate-1 origin-split table with one extra order stratum"
        ),
        "ysd_bands": [list(b) for b in YSD_BANDS],
        "order_bits": {"0": "entering 2nd marriage", "1": "entering 3rd+"},
        "n_cells": len(remarriage),
        "cells": cells,
    }


# --------------------------------------------------------------------------
# DELTA 2: single-year fertility, additionally conditioned on marital status
# --------------------------------------------------------------------------
def fit_fertility_single_marital(
    panel: transitions.MaritalPanel,
    birth_records: pd.DataFrame,
    train_ids: set[int],
    birth_decade: pd.Series,
) -> tuple[dict[tuple[int, int, int, int], float], int]:
    """DELTA 2: single-year fertility within parity x cohort x marital status.

    Candidate 5's single-year-of-age triangular-kernel construction --
    train women aged 15-49, running parity capped at 3, births censored at
    the mother's censor year, numerator and denominator each convolved over
    age with the pre-registered triangular kernel (bandwidth 3) -- with the
    kernel now applied WITHIN each ``(parity_band, birth_decade, married)``
    stratum. The woman-year marital status is the panel person-year
    ``marital_state`` (state entering the year, married vs not); the birth's
    marital status is the mother's ``marital_state`` in the birth year (the
    same person-year classification). Keyed
    ``(age, parity_band, decade, married_bit)`` for ages 15-49.

    Returns the table and the count of births whose birth-year person-year
    marital state could not be joined (assigned not-married; a provenance
    diagnostic).
    """
    py = panel.person_years
    attrs = panel.attrs
    women_ids = set(attrs[attrs["sex"] == "female"]["person_id"]) & train_ids
    lo, hi = FERT_AGE_LO, FERT_AGE_HI
    wy = py[
        py["person_id"].isin(women_ids) & (py["age"] >= lo) & (py["age"] <= hi)
    ][["person_id", "year", "age", "weight", "marital_state"]].copy()
    wy["married_bit"] = (wy["marital_state"] == "married").astype(int)

    be = g2f.births.birth_events(birth_records)
    be = be[
        (be["record_type"] == "birth")
        & be["parent_person_id"].isin(women_ids)
        & be["birth_year"].notna()
    ].copy()
    be = be.rename(columns={"parent_person_id": "person_id"})
    be["birth_year"] = be["birth_year"].astype("int64")
    births_by = {
        int(p): np.sort(g["birth_year"].to_numpy())
        for p, g in be.groupby("person_id")
    }

    wy = wy.reset_index(drop=True)
    wy["parity"] = _parity_vec(
        wy["person_id"].to_numpy(), wy["year"].to_numpy(), births_by
    )
    wy["decade"] = wy["person_id"].map(birth_decade).to_numpy()
    wy["parity_band"] = np.minimum(wy["parity"].to_numpy(), 3)

    attr_by = attrs.set_index("person_id")
    be["mother_birth"] = (
        be["person_id"].map(attr_by["birth_year"]).astype("float64")
    )
    be["mother_censor"] = (
        be["person_id"].map(attr_by["censor_year"]).astype("float64")
    )
    be["mother_age"] = be["birth_year"] - be["mother_birth"]
    be = be[
        (be["mother_age"] >= lo)
        & (be["mother_age"] <= hi)
        & (be["birth_year"] <= be["mother_censor"])
    ].reset_index(drop=True)
    be["decade"] = be["person_id"].map(birth_decade).to_numpy()
    be["weight"] = be["person_id"].map(attr_by["weight"]).to_numpy()
    be["parity"] = _parity_vec(
        be["person_id"].to_numpy(), be["birth_year"].to_numpy(), births_by
    )
    be["parity_band"] = np.minimum(be["parity"].to_numpy(), 3)

    # DELTA 2: the mother's marital status in the birth year -- the same
    # person-year marital_state (state entering the year) the denominator
    # uses. Unjoinable births (no birth-year person-year) get not-married.
    mstate = py[["person_id", "year", "marital_state"]].rename(
        columns={"year": "birth_year", "marital_state": "birth_marital_state"}
    )
    be = be.merge(mstate, on=["person_id", "birth_year"], how="left")
    n_unmatched = int(be["birth_marital_state"].isna().sum())
    be["married_bit"] = (be["birth_marital_state"] == "married").astype(int)

    den = (
        wy.groupby(["age", "parity_band", "decade", "married_bit"])["weight"]
        .sum()
        .to_dict()
    )
    num = (
        be.groupby(["mother_age", "parity_band", "decade", "married_bit"])[
            "weight"
        ]
        .sum()
        .to_dict()
    )

    # Strata present in the denominator (every woman-year cohort/parity/
    # marital cell). The kernel smooths over age within each stratum.
    strata = {(int(pb), int(dec), int(mb)) for (_a, pb, dec, mb) in den}
    kernel = c5._triangular_kernel_weights()
    table: dict[tuple[int, int, int, int], float] = {}
    for pb, dec, mb in strata:
        for age in range(lo, hi + 1):
            num_s = 0.0
            den_s = 0.0
            for d, w in kernel.items():
                a = age + d
                num_s += w * float(num.get((a, pb, dec, mb), 0.0))
                den_s += w * float(den.get((a, pb, dec, mb), 0.0))
            if den_s > 0.0:
                table[(age, pb, dec, mb)] = num_s / den_s
    return table, n_unmatched


def _fertility_marital_diag(
    table: dict[tuple[int, int, int, int], float],
    n_unmatched: int,
) -> dict[str, Any]:
    """Compact provenance summary of the marital single-year fertility table."""
    decades = sorted({d for (_a, _p, d, _m) in table})
    ages = sorted({a for (a, _p, _d, _m) in table})
    married = sorted({m for (_a, _p, _d, m) in table})
    return {
        "representation": (
            "single-year-of-age fertility rate within parity (0/1/2/3+) x "
            "birth-decade cohort x marital status (married vs not), kernel-"
            "smoothed over age (triangular kernel, bandwidth 3) within each "
            "(parity, decade, married) stratum -- candidate 5's single-year "
            "table with one extra marital stratum"
        ),
        "n_cells": len(table),
        "age_resolution": "single_year_of_age",
        "age_range": [FERT_AGE_LO, FERT_AGE_HI],
        "n_ages": len(ages),
        "parity_bands": [0, 1, 2, 3],
        "decades": decades,
        "n_decades": len(decades),
        "married_bits": married,
        "n_births_unjoined_marital_state": n_unmatched,
    }


# --------------------------------------------------------------------------
# Fitted components (candidate 6's, with the two delta'd fields swapped)
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
    """Fit all five components on side B, the two deltas applied.

    Starts from :func:`candidate6.fit_components` -- so first marriage,
    divorce, the surviving-spouse marriage-history widowhood LEVEL, the
    committed NCHS betas and the spousal-gap distribution draw are byte-
    identical to candidate 6 by construction. Then the TWO deltas:

    * DELTA 1 -- remarriage (``base.remarriage``) is replaced by the order-
      conditioned table (:func:`fit_remarriage_ordered`); candidate 6's
      origin-split remarriage is retained under a provenance key.
    * DELTA 2 -- fertility (``base.fertility``) is replaced by the marital-
      status-conditioned single-year table
      (:func:`fit_fertility_single_marital`); candidate 6's single-year
      fertility is retained under a provenance key.
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
    attr_by = panel.attrs.set_index("person_id")
    birth_decade = (attr_by["birth_year"] // 10 * 10).astype("int64")

    # DELTA 1: order-conditioned remarriage replaces the origin-split table.
    c6_remarriage = dict(base.remarriage)
    remarriage_ordered, wbar_diss = fit_remarriage_ordered(
        panel, order_map, train_ids
    )
    base.remarriage = remarriage_ordered

    # DELTA 2: marital-status-conditioned single-year fertility.
    c6_fertility = dict(base.fertility)
    fertility_marital, n_unmatched = fit_fertility_single_marital(
        panel, birth_records, train_ids, birth_decade
    )
    base.fertility = fertility_marital

    base.meta["remarriage_representation"] = (
        "weighted empirical hazard by years-since-dissolution band x origin "
        "(divorced/widowed) x sex x marriage-order bit (entering 2nd vs 3rd+), "
        "add-one smoothed; candidate 1's origin-split table with one extra "
        "order stratum (DELTA 1)"
    )
    base.meta["remarriage_order_diagnostics"] = _remarriage_order_diag(
        remarriage_ordered
    )
    base.meta["remarriage_candidate6"] = {
        f"ysd{b}|{origin}|{sex}": rate
        for (b, origin, sex), rate in c6_remarriage.items()
    }
    base.meta["remarriage_mean_dissolved_weight_check"] = wbar_diss

    base.meta["fertility_representation"] = (
        "single-year-of-age rates within parity (0/1/2/3+) x birth-decade "
        "cohort x marital status (married vs not), triangular-kernel smoothed "
        "over age within each (parity, decade, married) stratum (DELTA 2; "
        "replaces candidate 5/6's marital-status-independent single-year "
        "table)"
    )
    base.meta["fertility_marital_diagnostics"] = _fertility_marital_diag(
        fertility_marital, n_unmatched
    )
    base.meta["fertility_candidate6_n_cells"] = len(c6_fertility)
    base.meta["n_births_unjoined_marital_state"] = n_unmatched

    base.meta["delta1_vs_candidate6"] = DELTA1_VS_CANDIDATE6
    base.meta["delta2_vs_candidate6"] = DELTA2_VS_CANDIDATE6
    base.meta["deltas_vs_candidate6"] = list(DELTAS_VS_CANDIDATE6)
    return base


# --------------------------------------------------------------------------
# Vectorised annual simulation (candidate 6's, with the two delta'd lookups)
# --------------------------------------------------------------------------
@dataclass
class _SimLookupsC7:
    mort_arr: np.ndarray  # [widow_band, sex(0=f,1=m)] surviving-spouse level
    beta_arr: np.ndarray  # [sex(0=f,1=m)] per-sex log-linear period slope
    rem_arr: np.ndarray  # [ysd_band, origin(0=div,1=wid), sex, order_bit]
    fert_arr: np.ndarray  # [age-FERT_AGE_LO, parity_band, decade, married_bit]
    decade_map: dict[int, int]


def _build_sim_lookups(components: Components) -> _SimLookupsC7:
    """Widowhood level + NCHS slope (candidate 6's) + the two delta'd lookups.

    The mortality lookup and per-sex slope are candidate 6's, unchanged; the
    remarriage lookup gains an order-bit axis (DELTA 1) and the fertility
    lookup a married-bit axis (DELTA 2).
    """
    mort_arr = np.zeros((len(WIDOW_BANDS), 2), dtype=np.float64)
    for b, (lo, hi) in enumerate(WIDOW_BANDS):
        band = transitions.band_label(lo, hi)
        for si, sex in enumerate(("female", "male")):
            mort_arr[b, si] = components.mortality.get(f"{band}|{sex}", 0.0)

    beta = components.meta["mortality_beta_by_sex"]
    beta_arr = np.array([beta["female"], beta["male"]], dtype=np.float64)

    # DELTA 1: remarriage with an order-bit axis (0=entering 2nd, 1=3rd+).
    rem_arr = np.zeros((len(YSD_BANDS), 2, 2, 2), dtype=np.float64)
    for (b, origin, sex, ob), v in components.remarriage.items():
        oi = 0 if origin == "divorced" else 1
        si = 0 if sex == "female" else 1
        rem_arr[b, oi, si, ob] = v

    # DELTA 2: fertility with a married-bit axis (0=not married, 1=married).
    decades = sorted({d for (_a, _p, d, _m) in components.fertility})
    decade_map = {d: i for i, d in enumerate(decades)}
    n_age = FERT_AGE_HI - FERT_AGE_LO + 1
    fert_arr = np.zeros((n_age, 4, max(len(decades), 1), 2), dtype=np.float64)
    for (age, pb, d, mb), v in components.fertility.items():
        fert_arr[age - FERT_AGE_LO, pb, decade_map[d], mb] = v
    return _SimLookupsC7(mort_arr, beta_arr, rem_arr, fert_arr, decade_map)


def _remarriage_probs_ordered(
    ysd: np.ndarray,
    origin_state: np.ndarray,
    is_male: np.ndarray,
    order: np.ndarray,
    rem_arr: np.ndarray,
) -> np.ndarray:
    """DELTA 1: remarriage probability by (ysd, origin, sex, order bit).

    ``order`` is the ego's running marriage count entering the year; the
    order bit is 1 iff ``order >= 2`` (entering the 3rd or higher marriage),
    else 0 (entering the 2nd) -- the same 2nd-vs-3rd+ split the train fit
    estimates. Otherwise identical to candidate 1's ``_remarriage_probs``.
    """
    bands = _bands_vec(ysd, YSD_LOWERS, len(YSD_BANDS))
    origin_idx = (origin_state == 3).astype(np.int64)  # 2 div -> 0, 3 wid -> 1
    ob = (order >= 2).astype(np.int64)
    return rem_arr[bands, origin_idx, is_male.astype(np.int64), ob]


def _fertility_probs_single_marital(
    age: np.ndarray,
    parity: np.ndarray,
    didx: np.ndarray,
    married: np.ndarray,
    fert_arr: np.ndarray,
) -> np.ndarray:
    """DELTA 2: single-year fertility probability by marital status.

    ``fert_arr`` is indexed ``[age - FERT_AGE_LO, parity_band, decade_idx,
    married_bit]``; ages clipped into 15-49, parity capped at 3, married bit
    from the woman's current simulated marital state entering the year. A
    person whose birth decade is absent from the train table (``didx < 0``)
    gets probability 0, exactly as candidate 5's lookup.
    """
    ai = np.clip(age - FERT_AGE_LO, 0, fert_arr.shape[0] - 1)
    pb = np.minimum(parity, 3)
    safe = np.where(didx >= 0, didx, 0)
    mb = married.astype(np.int64)
    vals = fert_arr[ai, pb, safe, mb]
    return np.where(didx >= 0, vals, 0.0)


def simulate_holdout(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: Components,
    sim_seed: int,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Candidate 6's simulation with the two delta'd lookups.

    Byte-identical to :func:`candidate6.simulate_holdout` EXCEPT (1) the
    remarriage probability is looked up with the ego's running marriage
    ``order`` (:func:`_remarriage_probs_ordered`) and (2) the fertility
    probability is looked up with the woman's marital state ENTERING the year
    (:func:`_fertility_probs_single_marital`; captured before that year's
    marital transitions to match the train person-year classification). The
    per-year uniform blocks (``rng.random(n_active)`` then
    ``rng.random(n_fertile)``) are drawn in the same order and size as
    candidate 6 -- only the two THRESHOLDS move -- so first marriage and the
    ever-married shares are byte-identical to candidate 6; fertility,
    remarriage, the marriage counts and the dissolved-state stocks move.
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
        # DELTA 2 timing: the woman's marital state ENTERING the year (before
        # this year's marital transitions) is the fertility conditioning bit,
        # matching the train person-year classification (state entering y).
        state_entering = state.copy()

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
            # Widowhood LEVEL is candidate 6's surviving-spouse marriage-
            # history hazard, looked up by the married ego's own (age, sex);
            # byte-identical to candidate 6 (no delta touches widowhood).
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
            # CANDIDATE-7 DELTA 1: remarriage looked up with the ego's running
            # marriage order (>= 2 -> entering 3rd+); same YSD/origin/sex.
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

        # Fertility: women aged 15-49, any marital state. CANDIDATE-7 DELTA 2:
        # the birth threshold is conditioned on the woman's marital state
        # ENTERING the year (state_entering). The uf draw block is the same
        # order and size as candidate 6 -- only the threshold moves.
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
            married_bit = (state_entering[fidx] == 1).astype(np.int64)
            p_birth = _fertility_probs_single_marital(
                fage,
                parity[fidx],
                fert_didx[fidx],
                married_bit,
                lookups.fert_arr,
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
# Per-seed scoring (candidate 1's, calling the candidate-7 fit + simulate)
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
    candidate-7 :func:`fit_components` and :func:`simulate_holdout`.
    """
    import math

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
# Modal / targeted-cell / seed-0 regression / candidate-6 movement
# --------------------------------------------------------------------------
#: The registered secondary modal (comment 4912542742): the untargeted
#: lifetime-marriage boundary cell at its very tight 0.047 tolerance.
REGISTERED_MODAL_CELL = "mean_lifetime_marriages|male"
#: The cells the two deltas target (registration): delta 1's lifetime-
#: marriage clips and after-divorce remarriage, delta 2's c1970s fertility.
TARGETED_CELLS = (
    "mean_lifetime_marriages|male",
    "completed_fertility.c1970s",
    "remarriage.after_divorce",
)
#: The full set of cells whose vs-candidate-6 movement is reported (adds the
#: female lifetime-marriage cell the report-back tracks and c6's widowed
#: cluster, for continuity).
MOVEMENT_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
    "remarriage.after_divorce",
    "completed_fertility.c1970s",
    "share_widowed.65-74|female",
    "share_widowed.75+|female",
    "widowhood.45-64|female",
)


def _cells_byte_identical_to_c6(cell: str) -> bool:
    """First-marriage + ever-married cells are byte-identical to candidate 6.

    Neither delta touches first marriage (marital-state-independent, same RNG
    stream) or whether a person EVER married (remarriage adds marriages but
    creates no first marriages; fertility touches no marital state), so every
    ``first_marriage.*`` and ``ever_married_by_*`` cell must carry candidate
    6's exact ``r_candidate``.
    """
    return cell.startswith("first_marriage.") or cell.startswith(
        "ever_married_by_"
    )


def _is_fertility_cell(cell: str) -> bool:
    return cell.startswith("asfr.") or cell.startswith("completed_fertility.")


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal, targeted cells, seed-0 regression, and the decider."""
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

    # Seed-0 regression check (the registered PRIMARY modal-failure mode): a
    # remarriage or fertility cell that was passing for candidate 6 but fails
    # for candidate 7 on seed 0 (the change-what-works risk).
    seed0 = by_seed.get(0)
    c6_seed0 = _c6_seed0_gated()
    seed0_regressed: list[str] = []
    seed0_regressed_rem_fert: list[str] = []
    if seed0 is not None and c6_seed0 is not None:
        for cell, rec in seed0["gated_cells"].items():
            c6rec = c6_seed0.get(cell)
            if c6rec is not None and c6rec["pass"] and not rec["pass"]:
                seed0_regressed.append(cell)
                if cell.startswith("remarriage.") or _is_fertility_cell(cell):
                    seed0_regressed_rem_fert.append(cell)

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
                "the targeted cells (forgiving the two deltas' target cells "
                "flips >= 4 seeds to pass)"
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
            decider += " [seed 0 regressed -- the primary registered risk]"

    return {
        "registered_modal": (
            f"{REGISTERED_MODAL_CELL} (the untargeted lifetime-marriage "
            "boundary cell at its very tight 0.047 tolerance; the registered "
            "secondary modal, primary being a seed-0 regression)"
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
            "seed0_regressed_remarriage_or_fertility": sorted(
                seed0_regressed_rem_fert
            ),
            "note": (
                "the registered PRIMARY modal-failure mode is a seed-0 "
                "regression on a remarriage or fertility cell previously "
                "passing under candidate 6 (the change-what-works risk)"
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


def _c6_seed0_gated() -> dict[str, Any] | None:
    """Candidate 6's committed seed-0 gated cells (for the regression check)."""
    if not CANDIDATE6_ARTIFACT.exists():
        return None
    a6 = json.loads(CANDIDATE6_ARTIFACT.read_text())
    for s in a6["per_seed"]:
        if s["seed"] == 0:
            return s["gated_cells"]
    return None


def _candidate6_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-cell candidate-6 -> candidate-7 movement for the tracked cells."""
    if not CANDIDATE6_ARTIFACT.exists():
        return {"available": False}
    a6 = json.loads(CANDIDATE6_ARTIFACT.read_text())
    by6 = {s["seed"]: s for s in a6["per_seed"]}
    by7 = {s["seed"]: s for s in per_seed}
    seeds = sorted(by7)
    out: dict[str, Any] = {"available": True, "cells": {}}
    for cell in MOVEMENT_CELLS:
        c6_scores = {s: by6[s]["gated_cells"][cell]["score"] for s in seeds}
        c7_scores = {s: by7[s]["gated_cells"][cell]["score"] for s in seeds}
        c6_pass = sum(by6[s]["gated_cells"][cell]["pass"] for s in seeds)
        c7_pass = sum(by7[s]["gated_cells"][cell]["pass"] for s in seeds)
        out["cells"][cell] = {
            "tolerance": by7[seeds[0]]["gated_cells"][cell]["tolerance"],
            "candidate6_per_seed_score": c6_scores,
            "candidate7_per_seed_score": c7_scores,
            "candidate6_mean_score": float(np.mean(list(c6_scores.values()))),
            "candidate7_mean_score": float(np.mean(list(c7_scores.values()))),
            "candidate6_n_seeds_pass": c6_pass,
            "candidate7_n_seeds_pass": c7_pass,
        }
    return out


def _seed0_full_movement(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Every seed-0 gated cell's candidate-6 -> candidate-7 movement."""
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
            "candidate7_score": rec["score"],
            "candidate6_pass": c6rec["pass"],
            "candidate7_pass": rec["pass"],
            "moved": bool(moved),
        }
    return {
        "available": True,
        "note": (
            "seed 0 was candidate 6's only passing seed (46/46); this tracks "
            "how each of its 46 gated cells moved under the two deltas and "
            "whether any previously-passing cell regressed"
        ),
        "n_cells": len(cells),
        "n_moved": n_moved,
        "n_regressed": n_regressed,
        "n_improved": n_improved,
        "seed0_held_all_gated": bool(seed0["seed_pass"]),
        "cells": cells,
    }


def _first_marriage_identity_vs_candidate6(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """First-marriage + ever-married cells are byte-identical to candidate 6.

    Records the maximum per-cell deviation as an attestation that neither
    delta touches first marriage or the ever-married shares.
    """
    if not CANDIDATE6_ARTIFACT.exists():
        return {"available": False}
    a6 = json.loads(CANDIDATE6_ARTIFACT.read_text())
    by6 = {s["seed"]: s for s in a6["per_seed"]}
    max_dev = 0.0
    n_cells = 0
    for seed in sorted(s["seed"] for s in per_seed):
        s7 = next(s for s in per_seed if s["seed"] == seed)
        for cell, rec in s7["gated_cells"].items():
            if _cells_byte_identical_to_c6(cell):
                dev = abs(
                    rec["r_candidate"]
                    - by6[seed]["gated_cells"][cell]["r_candidate"]
                )
                max_dev = max(max_dev, dev)
                n_cells += 1
    return {
        "available": True,
        "note": (
            "first_marriage.* and ever_married_by_* gated cells are byte-"
            "identical to candidate 6 (first marriage is marital-state-"
            "independent under the shared RNG stream; remarriage adds "
            "marriages but no first marriages, and fertility touches no "
            "marital state -- so whether a person EVER married is unchanged)"
        ),
        "n_cells_checked": n_cells,
        "max_abs_r_candidate_deviation_vs_candidate6": float(max_dev),
        "byte_identical": bool(max_dev <= EXACT_ATOL),
    }


def _fertility_movement_vs_candidate6(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fertility cells MOVED vs candidate 6 -- delta 2 is active (not a no-op).

    Delta 2 conditions fertility on marital status, so ``asfr.*`` and
    ``completed_fertility.*`` are NOT byte-identical to candidate 6 (unlike
    every prior candidate, whose fertility was marital-state-independent).
    """
    if not CANDIDATE6_ARTIFACT.exists():
        return {"available": False}
    a6 = json.loads(CANDIDATE6_ARTIFACT.read_text())
    by6 = {s["seed"]: s for s in a6["per_seed"]}
    max_dev = 0.0
    n_cells = 0
    n_moved = 0
    for seed in sorted(s["seed"] for s in per_seed):
        s7 = next(s for s in per_seed if s["seed"] == seed)
        for cell, rec in s7["gated_cells"].items():
            if _is_fertility_cell(cell):
                dev = abs(
                    rec["r_candidate"]
                    - by6[seed]["gated_cells"][cell]["r_candidate"]
                )
                max_dev = max(max_dev, dev)
                n_cells += 1
                n_moved += dev > EXACT_ATOL
    return {
        "available": True,
        "note": (
            "asfr.* and completed_fertility.* moved vs candidate 6 -- delta 2 "
            "(marital-status-conditioned fertility) is an active structural "
            "change, not a mechanical no-op"
        ),
        "n_cells_checked": n_cells,
        "n_cells_moved": n_moved,
        "max_abs_r_candidate_deviation_vs_candidate6": float(max_dev),
        "delta2_active": bool(max_dev > EXACT_ATOL),
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins, with the candidate-7 schema + c1-c6 + v6 shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for n in (1, 2, 3, 4, 5, 6):
        pins[f"candidate{n}_runner"] = f"scripts/run_gate2_candidate{n}.py"
        pins[f"candidate{n}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{n}.py"
        )
    pins["candidate6_artifact"] = "runs/gate2_hazard_v6.json"
    pins["candidate6_artifact_sha256"] = c1._sha_of_file(CANDIDATE6_ARTIFACT)
    pins["deltas"] = (
        "remarriage x marriage-order (2nd vs 3rd+); fertility x current "
        "simulated marital status (married vs not)"
    )
    return pins


def _model_block() -> dict[str, Any]:
    """Candidate 6's model block, edited for the two candidate-7 deltas."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a mortality-"
            "composed widowhood component (candidate 6 + two named structural "
            "deltas: remarriage additionally conditioned on marriage order, "
            "fertility additionally conditioned on current simulated marital "
            "status)"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "delta1_vs_candidate6": DELTA1_VS_CANDIDATE6,
        "delta2_vs_candidate6": DELTA2_VS_CANDIDATE6,
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
                "deltas' effect on the married-state exposure)"
            ),
            "widowhood": (
                "COMPOSED, parametric in period; the spouse-death hazard "
                "LEVEL is candidate 6's train marriage-history surviving-"
                "spouse widowhood incidence over WIDOWHOOD_AGE_BANDS x "
                "surviving-spouse sex, with candidate 5's committed NCHS betas "
                "-- BYTE-IDENTICAL fit to candidate 6 (no delta touches "
                "widowhood; its simulated cells move only through the deltas' "
                "effect on the married-state exposure)"
            ),
            "remarriage": (
                "DELTA 1: weighted empirical hazard by years-since-dissolution "
                "band x origin (divorced/widowed) x sex x MARRIAGE ORDER "
                "(entering 2nd vs 3rd+), add-one smoothed -- candidate 1's "
                "origin-split table with one extra order stratum. Higher-order "
                "remarriage rates are lower in the data, lowering simulated "
                "lifetime marriage counts and moving remarriage.after_divorce"
            ),
            "fertility": (
                "DELTA 2: single-year-of-age rates within parity (0/1/2/3+) x "
                "birth-decade cohort x MARITAL STATUS (married vs not), "
                "triangular-kernel smoothed over age (bandwidth 3) within each "
                "stratum -- candidate 5/6's single-year table with one extra "
                "marital stratum. The simulation uses the woman's marital "
                "state entering the year"
            ),
        },
        "registered_ambiguity_resolutions": {
            "remarriage_order_definition": (
                "the order bit is 1 iff the person has >= 2 prior marriages "
                "(entering the 3rd or higher), else 0 (entering the 2nd); the "
                "prior count at each dissolved person-year and remarriage "
                "event is the number of order_map marriages starting strictly "
                "before that year (candidate1._parity_vec); in simulation the "
                "ego's running marriage order counter (>= 2 -> 3rd+) is used"
            ),
            "fertility_marital_definition": (
                "married vs not (the residual pools never_married, divorced, "
                "widowed, separated, other); the train classification is the "
                "panel person-year marital_state (state entering the year) for "
                "both the woman-year denominator and the birth-year numerator; "
                "the simulation uses the woman's marital state entering the "
                "year (captured before that year's transitions) so fit and "
                "simulation classify a woman-year identically"
            ),
            "smoothing_and_kernel": (
                "the remarriage add-one smoothing (wbar_diss) and the "
                "fertility triangular kernel (bandwidth 3) are candidate "
                "1's/5's exactly; each is applied per new stratum cell"
            ),
            "everything_else": (
                "the surviving-spouse marriage-history widowhood level, the "
                "committed NCHS betas, the knot-at-22 first-marriage spline, "
                "divorce, the spousal-gap distribution draw, the competing-"
                "risk step, the RNG rule default_rng(4200 + seed), the spawned "
                "gap-draw stream, one sequence per person, and the locked "
                "protocol are byte-identical to candidate 6"
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

    # Preflight: the committed candidate-6 artifact (the movement baseline)
    # and candidate-5 artifact (candidate 6 reads its committed NCHS beta from
    # it) must be present, and the candidate-5 NCHS references (candidate 6's
    # fit imports candidate 5's chain) must be present.
    for name, path in (
        ("candidate-6", CANDIDATE6_ARTIFACT),
        ("candidate-5", CANDIDATE5_ARTIFACT),
    ):
        if not path.exists():
            raise RuntimeError(
                f"{name} artifact missing at {path}; required for the run."
            )
    for year, path in c5.NCHS_LIFE_TABLE_PATHS.items():
        if not path.exists():
            raise RuntimeError(
                f"NCHS life-table reference for {year} missing at {path}."
            )
    c6._committed_beta_v5()  # fail fast if the committed betas drifted

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
    seed0_full_movement = _seed0_full_movement(per_seed)
    fm_identity = _first_marriage_identity_vs_candidate6(per_seed)
    fert_movement = _fertility_movement_vs_candidate6(per_seed)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 7",
        "spec_registration": SPEC_REGISTRATION,
        "candidate6_registration": CANDIDATE6_REGISTRATION,
        "candidate5_registration": CANDIDATE5_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "delta1_vs_candidate6": DELTA1_VS_CANDIDATE6,
        "delta2_vs_candidate6": DELTA2_VS_CANDIDATE6,
        "deltas_vs_candidate6": list(DELTAS_VS_CANDIDATE6),
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79 + flip #81); "
            "protocol/views/tolerances read at runtime, no threshold moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.5-0.6",
            "conjunction_estimate": 0.55,
            "component_probabilities": {
                "mean_lifetime_marriages_male_seeds_3_4_fixed": 0.6,
                "completed_fertility_c1970s_seed_2_fixed": 0.55,
                "seed0_stays_perfect": 0.85,
            },
            "modal_failure": (
                "PRIMARY: a seed-0 regression on a remarriage or fertility "
                "cell previously passing under candidate 6 (the change-what-"
                "works risk, since both deltas touch seed 0's passing cells). "
                "SECONDARY: mean_lifetime_marriages|male persisting at its "
                "0.047 tolerance"
            ),
            "component_reads": (
                "seeds 3/4 need only the mean_lifetime_marriages|male boundary "
                "cell (0.048-0.061 vs 0.047; delta 1 lowers higher-order "
                "remarriage, the mechanism -- ~0.6 each); seed 2 needs only "
                "the c1970s fertility cell (~0.55 under delta 2); seed 0 must "
                "stay perfect (~0.85). The gate can be won by holding seed 0 "
                "and fixing single cells on two of seeds 2/3/4"
            ),
            "deltas_vs_candidate6": list(DELTAS_VS_CANDIDATE6),
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
        "candidate6_comparison": candidate6_comparison,
        "seed0_full_movement": seed0_full_movement,
        "first_marriage_identity_vs_candidate6": fm_identity,
        "fertility_movement_vs_candidate6": fert_movement,
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
