"""Gate-2 candidate 6 (run 1): candidate 5 + one named delta.

The SIXTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42
comment 4912170754 (``SPEC_REGISTRATION``): candidate 5's frozen spec
(comment 4911788302) verbatim EXCEPT ONE delta. One-shot; no constant
moves after the registration comment.

The one delta vs candidate 5 (everything else byte-identical)
------------------------------------------------------------
**Source-aligned spouse-death hazard LEVEL.** Candidate 5's widowhood is
composed from a spouse-death hazard whose LEVEL is candidate 1's pooled
band x sex ``ind2023er`` PSID death-record central rate (the 1995 anchor),
scaled by the external-NCHS per-sex period trend ``exp(beta_sex *
(year - 1995))``. ``runs/mortality_floors_v1.json`` shows the PSID
death-record central rate undercounts the NCHS life-table rate by a median
factor ~0.76, while the gate's *reference* widowhood is measured from the
retrospective marriage-history end-reasons (``mh85_23`` "how ended = death
of spouse"), which do not share that undercount -- so the candidate-5
composition inherits the death-record undercount and produces too few
widows (share_widowed.75+|female stays a level gap at 1/5).

Candidate 6 replaces the LEVEL SOURCE only: the spouse-death hazard level
is estimated from the train half's marriage histories -- spouse-death
marriage endings (``mh85_23`` how-ended = spouse death) over married
person-year exposure, by age band x sex of the SURVIVING spouse -- the
same construction the gate reference uses via
:mod:`populace_dynamics.data.transitions` (``transitions.hazard_cells``
widowhood cells over ``WIDOWHOOD_AGE_BANDS``), reusing that machinery. The
NCHS trend multiplier ``exp(beta_sex * (year - 1995))`` with candidate 5's
COMMITTED beta values (female -0.009234704865961198, male
-0.010643975395626533; read from ``runs/gate2_hazard_v5.json``, NOT
re-fit) applies unchanged. Ego mortality is not simulated (support is
fixed by protocol), so this table's only consumer is the widowhood
composition, and it now shares the reference's source semantics: the
widowhood level is looked up by the married ego's OWN (age band, sex) --
the surviving spouse -- instead of by the dying spouse's (age, sex).

Everything else -- the knot-at-22 20/22/25/30/40 first-marriage spline,
divorce, the single-year triangular-kernel fertility (candidate 5's delta
2), the origin-split remarriage table, the spousal-age-gap DISTRIBUTION
draw (retained byte-identically; it no longer enters the widowhood level,
which now integrates the empirical spouse-age distribution directly), the
NCHS period-trend multiplier, the competing-risk step, the RNG rule
``numpy.random.default_rng(4200 + seed)``, the spawned gap-draw stream, one
simulated sequence per person, and the LOCKED gate-2 protocol -- is
byte-identical to candidate 5. This runner IMPORTS candidate 5's machinery
(which chains candidates 4/3/2/1) and reuses every unchanged function: the
unchanged components come straight from ``candidate5.fit_components`` (so
first marriage, divorce, remarriage, the spousal-gap distribution and the
single-year fertility are provably identical to candidate 5), and only the
one delta'd field is recomputed -- the spouse-death hazard LEVEL SOURCE
(marriage-history surviving-spouse widowhood, replacing the death-record
central rate). The scoring path, precheck, and verdict assembly are
candidate 1's, imported unchanged.

Because widowhood transitions only ever leave the ``married`` state, and
fertility and first marriage are marital-state-independent, the per-year
uniform blocks (``rng.random(n_active)`` then ``rng.random(n_fertile)``)
are drawn in the same order and size as candidate 5 -- only the widowhood
THRESHOLD moves -- so the first-marriage and fertility reference cells are
byte-identical to candidate 5, and only widowhood and the marital states it
feeds (remarriage, dissolved-state stocks, lifetime marriages) move.

Hard-stop precheck (identical to candidate 1): the scoring path must
reproduce, bit-for-bit, every committed full-panel reference moment, every
committed per-gate-seed ``rate_a``, and each gate seed's committed
holdout-id sha256, BEFORE any candidate is simulated. Any mismatch is a
hard stop. Run ONCE; publish REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit; no statsmodels). Run from the repository root with the PSID
history files staged::

    .venv/bin/python scripts/run_gate2_candidate6.py
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

# Candidate 5 supplies the machinery this build minimally deltas once: its
# NCHS-trend widowhood composition, its single-year triangular-kernel
# fertility, its origin-split remarriage, its spousal-gap distribution draw,
# and -- transitively, via candidates 4/3/2/1 -- the knot-at-22 first-marriage
# fitter, divorce fitter, the vectorised simulation helpers, the precheck, the
# verdict assembly, and the report-only summary. Only the one delta'd field
# (the spouse-death hazard LEVEL SOURCE) is re-implemented.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate4 as c4  # noqa: E402
import run_gate2_candidate5 as c5  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v6.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
CANDIDATE5_ARTIFACT = ROOT / "runs" / "gate2_hazard_v5.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v6"
RUN_NAME = "gate2_hazard_v6"

#: This run's frozen-spec registration (issue #42, comment 4912170754).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4912170754"
)
#: The candidate-5 spec this build minimally deltas (comment 4911788302).
CANDIDATE5_REGISTRATION = c5.SPEC_REGISTRATION
CANDIDATE4_REGISTRATION = c4.SPEC_REGISTRATION
CANDIDATE1_REGISTRATION = c1.SPEC_REGISTRATION

#: The one named delta (registration comment 4912170754).
DELTA_VS_CANDIDATE5 = (
    "spouse-death hazard LEVEL is estimated from the train half's marriage "
    "histories -- spouse-death marriage endings (mh85_23 how-ended = spouse "
    "death) over married person-year exposure, by age band x sex of the "
    "SURVIVING spouse (transitions.hazard_cells widowhood cells over "
    "WIDOWHOOD_AGE_BANDS, the gate reference's own construction) -- replacing "
    "candidate 1's ind2023er PSID death-record central-rate table as the "
    "level source; the NCHS period-trend multiplier exp(beta_sex * "
    "(year - 1995)) with candidate 5's committed beta values applies "
    "unchanged; the widowhood level is now looked up by the married ego's own "
    "(age band, sex) -- the surviving spouse -- rather than by the dying "
    "spouse's, and the retained spousal-gap draw no longer enters the level"
)

# --- Frozen dials + band constants + pure helpers, reused (byte-identical;
# imported, never redefined). ---------------------------------------------
GATE_SEEDS = c1.GATE_SEEDS
SIM_SEED_BASE = c1.SIM_SEED_BASE
EXACT_ATOL = c1.EXACT_ATOL
TREND_ANCHOR_YEAR = c4.TREND_ANCHOR_YEAR  # 1995.0 (unchanged)

MORT_BANDS = c1.MORT_BANDS  # death-record table bands (candidate-5 level)
MORT_LOWERS = c1.MORT_LOWERS
_ASFR_LO = c1._ASFR_LO
_ASFR_HI = c1._ASFR_HI
_STATE = c1._STATE
_STATE_ABSORB = c1._STATE_ABSORB
FERT_AGE_LO = c5.FERT_AGE_LO  # 15
FERT_AGE_HI = c5.FERT_AGE_HI  # 49

_bands_vec = c1._bands_vec
_divorce_probs = c1._divorce_probs
_remarriage_probs = c1._remarriage_probs
_fertility_probs_single = c5._fertility_probs_single  # single-year (delta 2)
_assemble_sim_panel = c1._assemble_sim_panel
Components = c1.Components

# The candidate-6 first-marriage model IS candidate 3's (knot-at-22 spline),
# inherited through candidate 5; no delta touches it. Aliased for provenance.
fit_first_marriage = c5.fit_first_marriage
FirstMarriageModelC6 = c5.FirstMarriageModelC5

# DELTA constants: the surviving-spouse widowhood level is banded on the gate
# reference's own widowhood age bands (45-54, 55-64, 65-74, 75+), one slope
# per surviving-spouse sex. Ages below 45 clip into the youngest band exactly
# as candidate 1's mortality lookup clips ages below 25 (same _bands_vec
# mechanism; only the table and its bands change).
WIDOW_BANDS = (
    transitions.WIDOWHOOD_AGE_BANDS
)  # ((45,54),(55,64),(65,74),(75,120))
WIDOW_LOWERS = np.array([lo for lo, _ in WIDOW_BANDS], dtype=np.int64)

#: Candidate 5's committed NCHS per-sex log-linear period slopes, applied
#: unchanged (read from runs/gate2_hazard_v5.json; NOT re-fit).
_BETA_V5_EXPECTED = {
    "female": -0.009234704865961198,
    "male": -0.010643975395626533,
}

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate6_run1_cache.json"
)


# --------------------------------------------------------------------------
# DELTA: source-aligned spouse-death LEVEL from marriage-history widowhood
# --------------------------------------------------------------------------
def _committed_beta_v5() -> dict[str, float]:
    """Candidate 5's committed NCHS per-sex slopes (read; NOT re-fit).

    Reads ``beta_sex_nchs`` from the committed candidate-5 artifact
    (``runs/gate2_hazard_v5.json``) and checks it against the frozen expected
    values -- the trend multiplier is applied unchanged, from the committed
    values rather than a fresh NCHS fit.
    """
    a5 = json.loads(CANDIDATE5_ARTIFACT.read_text())
    beta = a5["mortality_trend_beta_comparison"]["beta_sex_nchs"]
    out = {"female": float(beta["female"]), "male": float(beta["male"])}
    for sex in ("female", "male"):
        if abs(out[sex] - _BETA_V5_EXPECTED[sex]) > 1e-15:
            raise RuntimeError(
                f"committed candidate-5 beta_{sex}={out[sex]!r} does not match "
                f"the frozen expected {_BETA_V5_EXPECTED[sex]!r}."
            )
    return out


def _widowhood_hazard_cells(
    panel: transitions.MaritalPanel, train_ids: set[int]
) -> dict[str, dict[str, float]]:
    """Train marriage-history widowhood hazard cells (surviving spouse).

    The gate reference's own widowhood construction
    (:func:`transitions.hazard_cells`, ``weighted=True``) restricted to the
    train complement: weighted spouse-death marriage endings over weighted
    married person-year exposure, by :data:`WIDOW_BANDS` x sex of the
    surviving spouse. Returned keyed ``"band|sex"`` (the ``widowhood.``
    prefix stripped) to match the mortality-anchor key convention.
    """
    cells = transitions.hazard_cells(panel, train_ids, weighted=True)
    pref = "widowhood."
    return {
        key[len(pref) :]: dict(cell)
        for key, cell in cells.items()
        if key.startswith(pref)
    }


def fit_widowhood_level(
    panel: transitions.MaritalPanel, train_ids: set[int]
) -> dict[str, float]:
    """Surviving-spouse widowhood-incidence LEVEL table (the DELTA).

    ``{band|sex: rate}`` over :data:`WIDOW_BANDS` x surviving-spouse sex,
    the marriage-history widowhood hazard on the train complement -- the
    replacement level source for candidate 5's death-record central rate.
    """
    return {
        key: float(cell["rate"])
        for key, cell in _widowhood_hazard_cells(panel, train_ids).items()
    }


def _widow_level_diag(
    panel: transitions.MaritalPanel, train_ids: set[int]
) -> dict[str, Any]:
    """Provenance of the surviving-spouse widowhood LEVEL table."""
    cells = _widowhood_hazard_cells(panel, train_ids)
    return {
        "source": (
            "train marriage-history widowhood endings (mh85_23 how-ended = "
            "spouse death) over married person-year exposure, by age band x "
            "sex of the surviving spouse"
        ),
        "construction": (
            "transitions.hazard_cells(weighted=True) widowhood cells "
            "restricted to the seed's train complement -- the gate reference's "
            "own machinery, reused"
        ),
        "bands": [list(b) for b in WIDOW_BANDS],
        "n_cells": len(cells),
        "cells": {
            key: {
                "rate": float(cell["rate"]),
                "num_wt": float(cell["num_wt"]),
                "den_wt": float(cell["den_wt"]),
                "n_events": int(cell["n_events"]),
            }
            for key, cell in cells.items()
        },
    }


# --------------------------------------------------------------------------
# Fitted components (candidate 5's, with the one delta'd LEVEL swapped)
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

    Starts from :func:`candidate5.fit_components` -- so first marriage,
    divorce, remarriage, the spousal-gap distribution and the single-year
    triangular-kernel fertility are byte-identical to candidate 5 by
    construction, and ``base.meta['mortality_beta_by_sex']`` carries the NCHS
    slopes. Then the ONE delta:

    * the spouse-death hazard LEVEL (``base.mortality``) is replaced by the
      surviving-spouse marriage-history widowhood table
      (:func:`fit_widowhood_level`); candidate 5's death-record level is
      retained under a provenance key for the old-vs-new comparison; and
    * the NCHS period-trend beta is set from candidate 5's COMMITTED values
      (:func:`_committed_beta_v5`; read, not re-fit) -- identical to the value
      candidate 5's chain recomputes, retained under a provenance key.
    """
    base = c5.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )

    # DELTA: surviving-spouse marriage-history widowhood level replaces the
    # ind2023er death-record central-rate level. Retain candidate 5's
    # death-record level for the old-vs-new-by-band comparison.
    c5_deathrecord_level = dict(base.mortality)
    widow_level = fit_widowhood_level(panel, train_ids)
    base.mortality = widow_level

    # The NCHS period-trend multiplier applies unchanged, from candidate 5's
    # committed values (read, not re-fit). Retain the chain's recomputed value.
    beta_committed = _committed_beta_v5()
    base.meta["mortality_beta_by_sex_candidate5_recomputed"] = dict(
        base.meta["mortality_beta_by_sex"]
    )
    base.meta["mortality_beta_by_sex"] = beta_committed
    base.meta["mortality_beta_source"] = (
        "candidate 5 committed NCHS per-sex log-linear period slopes "
        "(runs/gate2_hazard_v5.json mortality_trend_beta_comparison."
        "beta_sex_nchs); read and applied unchanged, NOT re-fit"
    )

    base.meta["mortality_level_source"] = (
        "train marriage-history widowhood endings (mh85_23 how-ended = spouse "
        "death) over married person-year exposure, by age band x sex of the "
        "surviving spouse -- transitions.hazard_cells widowhood construction "
        "(the gate reference's own machinery), replacing candidate 1's "
        "ind2023er death-record central-rate table"
    )
    base.meta["mortality_level_representation"] = (
        "surviving-spouse widowhood-incidence hazard, keyed (WIDOWHOOD_AGE_"
        "BANDS band, surviving-spouse sex); consumed in the widowhood "
        "composition by the married ego's OWN (age band, sex), with the NCHS "
        "period-trend multiplier exp(beta_sex * (year - 1995)) applied "
        "unchanged; the retained spousal-gap draw no longer enters the level"
    )
    base.meta["mortality_level_bands"] = [list(b) for b in WIDOW_BANDS]
    base.meta["mortality_cells"] = len(widow_level)  # 4 bands x 2 sexes = 8
    base.meta["mortality_level_diagnostics"] = _widow_level_diag(
        panel, train_ids
    )
    base.meta["mortality_level_new_widowhood"] = dict(widow_level)
    base.meta["mortality_level_candidate5_deathrecord"] = c5_deathrecord_level
    base.meta["delta_vs_candidate5"] = DELTA_VS_CANDIDATE5
    return base


# --------------------------------------------------------------------------
# Vectorised annual simulation (candidate 5's, with the delta'd widowhood
# LEVEL looked up by the surviving ego's own (age, sex))
# --------------------------------------------------------------------------
@dataclass
class _SimLookupsC6:
    mort_arr: np.ndarray  # [widow_band, sex(0=f,1=m)] surviving-spouse level
    beta_arr: np.ndarray  # [sex(0=f,1=m)] per-sex log-linear period slope
    rem_arr: np.ndarray  # [ysd_band, origin(0=div,1=wid), sex(0=f,1=m)]
    fert_arr: np.ndarray  # [age-FERT_AGE_LO, parity_band, decade_idx]
    decade_map: dict[int, int]


def _build_sim_lookups(components: Components) -> _SimLookupsC6:
    """Surviving-spouse widowhood level + the NCHS slope + single-year fert.

    The mortality lookup is the delta's surviving-spouse widowhood table over
    :data:`WIDOW_BANDS` (keyed ``"band|sex"``); the per-sex slope ``beta_arr``
    is candidate 5's committed NCHS value; the remarriage and single-year
    fertility lookups are candidate 5's, unchanged.
    """
    mort_arr = np.zeros((len(WIDOW_BANDS), 2), dtype=np.float64)
    for b, (lo, hi) in enumerate(WIDOW_BANDS):
        band = transitions.band_label(lo, hi)
        for si, sex in enumerate(("female", "male")):
            mort_arr[b, si] = components.mortality.get(f"{band}|{sex}", 0.0)

    beta = components.meta["mortality_beta_by_sex"]
    beta_arr = np.array([beta["female"], beta["male"]], dtype=np.float64)

    rem_arr = np.zeros(
        (len(transitions.REMARRIAGE_YSD_BANDS), 2, 2), dtype=np.float64
    )
    for (b, origin, sex), v in components.remarriage.items():
        oi = 0 if origin == "divorced" else 1
        si = 0 if sex == "female" else 1
        rem_arr[b, oi, si] = v

    decades = sorted({d for (_a, _p, d) in components.fertility})
    decade_map = {d: i for i, d in enumerate(decades)}
    n_age = FERT_AGE_HI - FERT_AGE_LO + 1
    fert_arr = np.zeros((n_age, 4, max(len(decades), 1)), dtype=np.float64)
    for (age, pb, d), v in components.fertility.items():
        fert_arr[age - FERT_AGE_LO, pb, decade_map[d]] = v
    return _SimLookupsC6(mort_arr, beta_arr, rem_arr, fert_arr, decade_map)


def _widow_probs(
    ego_age: np.ndarray,
    ego_is_male: np.ndarray,
    spouse_age: np.ndarray,
    spouse_is_male: np.ndarray,
    year: int,
    mort_arr: np.ndarray,
    beta_arr: np.ndarray,
) -> np.ndarray:
    """Surviving-spouse widowhood incidence x NCHS period trend (the DELTA).

    ``rate = widow_level(ego_band, ego_sex) * exp(beta_sex * (year - 1995))``.
    The LEVEL is now the train marriage-history widowhood hazard keyed by the
    SURVIVING spouse's (age band, sex), so it is looked up by the married
    ego's OWN ``(ego_age, ego_is_male)`` -- not the dying spouse's -- with the
    same ``_bands_vec`` clip candidate 1 used (ages below the youngest band
    clip into it). ``spouse_age`` / ``spouse_is_male`` are candidate 5's
    spousal-gap-draw arguments, retained so the spousal-gap draw stays
    byte-identical; they no longer enter the level (the surviving-spouse
    hazard already integrates the empirical spouse-age distribution). The NCHS
    period-trend multiplier is candidate 5's committed value, unchanged.
    """
    bands = _bands_vec(
        np.rint(ego_age).astype(np.int64), WIDOW_LOWERS, len(WIDOW_BANDS)
    )
    sidx = ego_is_male.astype(np.int64)
    level = mort_arr[bands, sidx]
    trend = np.exp(beta_arr[sidx] * (float(year) - TREND_ANCHOR_YEAR))
    return level * trend


def simulate_holdout(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: Components,
    sim_seed: int,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Candidate 5's simulation with the delta'd surviving-spouse widowhood.

    Byte-identical to :func:`candidate5.simulate_holdout` EXCEPT the widowhood
    LEVEL is the surviving-spouse marriage-history hazard looked up by the
    married ego's own ``(age, sex)`` (:func:`_widow_probs`) instead of
    candidate 5's death-record hazard looked up by the dying spouse's
    ``(sp_age, opp_is_male)``. The per-year uniform blocks
    (``rng.random(n_active)`` then ``rng.random(n_fertile)``) are drawn in the
    same order and size as candidate 5 -- only the widowhood THRESHOLD moves;
    the spousal-gap draw is retained byte-identically (it no longer enters the
    widowhood level). First marriage and fertility are marital-state-
    independent, so their reference cells are byte-identical to candidate 5;
    only widowhood and the states it feeds move.
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
            p_rm = _remarriage_probs(
                ysd, origin, is_male[sub], lookups.rem_arr
            )
            rm = u[diss] < p_rm
            gri = sub[rm]
            order[gri] += 1
            cur_start[gri] = y
            state[gri] = 1
            diss_year[gri] = -1
            open_start[gri] = y
            open_order[gri] = order[gri]

        # Fertility: women aged 15-49, any marital state (candidate 5's
        # single-year kernel lookup; marital-state-independent, so identical
        # to candidate 5 -- same uf draw block, same threshold).
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
    candidate-6 :func:`fit_components` and :func:`simulate_holdout`.
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
#: The registered modal failure (comment 4912170754): the untargeted
#: lifetime-marriage sequence statistic at its very tight 0.047 tolerance --
#: again the modal failure if candidate 6 fails.
REGISTERED_MODAL_CELL = "mean_lifetime_marriages|male"
#: The cells the one delta targets (registration): the elderly female widowed
#: stock and the female widowhood incidence the source alignment lifts.
TARGETED_CELLS = (
    "share_widowed.75+|female",
    "widowhood.45-64|female",
)
#: The female widowed-stock cluster tracked since candidate 3.
WIDOWED_STOCK_CLUSTER = (
    "share_widowed.65-74|female",
    "share_widowed.75+|female",
    "widowhood.45-64|female",
)
#: The persisting fertility clip the registration flags as untouched by this
#: delta (single-year fertility is byte-identical to candidate 5).
PERSISTING_CLIP_CELL = "completed_fertility.c1970s"


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal, the targeted cells, and which decided the verdict."""
    fails_by_cell: dict[str, list[int]] = {}
    for f in verdict["all_failing_gated_cells"]:
        fails_by_cell.setdefault(f["cell"], []).append(f["seed"])

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

    n_pass_actual = verdict["n_seeds_pass"]
    n_pass_no_modal = seeds_pass_if_forgiven({REGISTERED_MODAL_CELL})
    n_pass_no_targeted = seeds_pass_if_forgiven(set(TARGETED_CELLS))
    n_pass_no_both = seeds_pass_if_forgiven(
        {REGISTERED_MODAL_CELL, *TARGETED_CELLS}
    )
    modal_failed = REGISTERED_MODAL_CELL in fails_by_cell
    gate_pass = verdict["gate_2_pass"]

    if gate_pass:
        decider = "none (gate passed)"
    else:
        modal_flips = n_pass_no_modal >= 4
        targeted_flips = n_pass_no_targeted >= 4
        modal_is_sole = modal_failed and all(
            (c not in fails_by_cell) or c == REGISTERED_MODAL_CELL
            for c in {f["cell"] for f in verdict["all_failing_gated_cells"]}
        )
        if modal_flips and targeted_flips:
            decider = (
                "both independently decisive (forgiving either the modal or "
                "the targeted cells alone flips the gate to pass)"
            )
        elif modal_flips:
            decider = (
                "mean_lifetime_marriages|male (the registered modal alone "
                "holds the gate; forgiving it flips >=4 seeds to pass)"
            )
        elif targeted_flips:
            decider = "targeted_cells"
        elif n_pass_no_both >= 4:
            decider = (
                "modal AND targeted cells jointly (forgiving both flips the "
                "gate; neither alone suffices)"
            )
        else:
            decider = (
                "broader than the modal + targeted cells (other gated cells "
                "also hold the gate below 4 passing seeds)"
            )
        if modal_is_sole:
            decider += (
                " [mean_lifetime_marriages|male is the SOLE distinct failing "
                "gated cell]"
            )

    return {
        "registered_modal": (
            f"{REGISTERED_MODAL_CELL} (the untargeted lifetime-marriage "
            "sequence statistic at its very tight 0.047 tolerance; the modal "
            "failure if candidate 6 fails)"
        ),
        "modal_cell": REGISTERED_MODAL_CELL,
        "modal_failed": modal_failed,
        "modal_failed_seeds": sorted(
            fails_by_cell.get(REGISTERED_MODAL_CELL, [])
        ),
        "modal_track": track(REGISTERED_MODAL_CELL),
        "modal_is_sole_failing_cell": (
            len({f["cell"] for f in verdict["all_failing_gated_cells"]}) == 1
            and modal_failed
        ),
        "targeted_cells": list(TARGETED_CELLS),
        "targeted_cells_track": {c: track(c) for c in TARGETED_CELLS},
        "widowed_stock_cluster": list(WIDOWED_STOCK_CLUSTER),
        "widowed_stock_cluster_track": {
            c: track(c) for c in WIDOWED_STOCK_CLUSTER
        },
        "persisting_clip_cell": PERSISTING_CLIP_CELL,
        "persisting_clip_track": track(PERSISTING_CLIP_CELL),
        "any_materialized": modal_failed,
        "decider_analysis": {
            "n_seeds_pass_actual": n_pass_actual,
            "n_seeds_pass_if_modal_forgiven": n_pass_no_modal,
            "n_seeds_pass_if_targeted_forgiven": n_pass_no_targeted,
            "n_seeds_pass_if_both_forgiven": n_pass_no_both,
            "decider": decider,
            "modal_decided": (not gate_pass) and modal_flips,
        },
    }


def _candidate5_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-cell candidate-5 -> candidate-6 movement for the tracked cells.

    Reads the committed candidate-5 artifact and records, for the registered
    modal, the two targeted cells, the widowed-stock cluster and the persisting
    fertility clip, each cell's per-seed candidate-5 and candidate-6 scores and
    pass counts -- the vs-candidate-5 movement.
    """
    if not CANDIDATE5_ARTIFACT.exists():
        return {"available": False}
    a5 = json.loads(CANDIDATE5_ARTIFACT.read_text())
    by5 = {s["seed"]: s for s in a5["per_seed"]}
    by6 = {s["seed"]: s for s in per_seed}
    seeds = sorted(by6)
    cells = (
        REGISTERED_MODAL_CELL,
        *TARGETED_CELLS,
        *WIDOWED_STOCK_CLUSTER,
        PERSISTING_CLIP_CELL,
    )
    seen: set[str] = set()
    out: dict[str, Any] = {"available": True, "cells": {}}
    for cell in cells:
        if cell in seen:
            continue
        seen.add(cell)
        c5_scores = {s: by5[s]["gated_cells"][cell]["score"] for s in seeds}
        c6_scores = {s: by6[s]["gated_cells"][cell]["score"] for s in seeds}
        c5_pass = sum(by5[s]["gated_cells"][cell]["pass"] for s in seeds)
        c6_pass = sum(by6[s]["gated_cells"][cell]["pass"] for s in seeds)
        out["cells"][cell] = {
            "tolerance": by6[seeds[0]]["gated_cells"][cell]["tolerance"],
            "candidate5_per_seed_score": c5_scores,
            "candidate6_per_seed_score": c6_scores,
            "candidate5_mean_score": float(np.mean(list(c5_scores.values()))),
            "candidate6_mean_score": float(np.mean(list(c6_scores.values()))),
            "candidate5_n_seeds_pass": c5_pass,
            "candidate6_n_seeds_pass": c6_pass,
        }
    return out


def _fertility_first_marriage_identity(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """First-marriage + fertility cells are byte-identical to candidate 5.

    Because widowhood only leaves the ``married`` state and both first
    marriage and fertility are marital-state-independent under the shared RNG
    stream, every ``first_marriage.*``, ``asfr.*`` and
    ``completed_fertility.*`` gated cell must carry candidate 5's exact
    ``r_candidate``. Records the maximum per-cell deviation as an attestation.
    """
    if not CANDIDATE5_ARTIFACT.exists():
        return {"available": False}
    a5 = json.loads(CANDIDATE5_ARTIFACT.read_text())
    by5 = {s["seed"]: s for s in a5["per_seed"]}
    by6 = {s["seed"]: s for s in per_seed}
    max_dev = 0.0
    n_cells = 0
    for seed in sorted(by6):
        for cell, rec in by6[seed]["gated_cells"].items():
            if (
                cell.startswith("first_marriage.")
                or cell.startswith("asfr.")
                or cell.startswith("completed_fertility.")
            ):
                dev = abs(
                    rec["r_candidate"]
                    - by5[seed]["gated_cells"][cell]["r_candidate"]
                )
                max_dev = max(max_dev, dev)
                n_cells += 1
    return {
        "available": True,
        "note": (
            "first_marriage.*, asfr.* and completed_fertility.* gated cells "
            "are byte-identical to candidate 5 (widowhood only leaves the "
            "married state; first marriage and fertility are "
            "marital-state-independent under the shared RNG stream)"
        ),
        "n_cells_checked": n_cells,
        "max_abs_r_candidate_deviation_vs_candidate5": float(max_dev),
        "byte_identical": bool(max_dev <= EXACT_ATOL),
    }


def _mortality_level_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """The two hazard tables' LEVEL comparison, old (c5) vs new (c6), by band.

    ``old`` = candidate 5's ``ind2023er`` PSID death-record central rate
    (``psid_m``), keyed by :data:`MORT_BANDS` x DYING-spouse sex; ``new`` =
    candidate 6's train marriage-history widowhood-incidence hazard, keyed by
    :data:`WIDOW_BANDS` x SURVIVING-spouse sex. The two are indexed by
    opposite parties (the old composition draws the spouse's death at the
    spouse's age/sex; the new level is the surviving ego's widowhood at the
    ego's own age/sex), so the per-band ratio quantifies the applied
    widow-inflow shift the source alignment introduces, not a like-for-like
    mortality ratio. Reported as the cross-seed mean per (band, sex) plus the
    per-seed tables.
    """
    seeds = sorted(s["seed"] for s in per_seed)
    by = {s["seed"]: s for s in per_seed}

    def mean_over_seeds(getter) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for s in seeds:
            for key, val in getter(by[s]).items():
                acc.setdefault(key, []).append(float(val))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    old_mean = mean_over_seeds(
        lambda s: s["component_meta"]["mortality_level_candidate5_deathrecord"]
    )
    new_mean = mean_over_seeds(
        lambda s: s["component_meta"]["mortality_level_new_widowhood"]
    )

    by_band: list[dict[str, Any]] = []
    for lo, hi in WIDOW_BANDS:
        new_band = transitions.band_label(lo, hi)
        for sex in ("female", "male"):
            new_key = f"{new_band}|{sex}"
            new_val = new_mean.get(new_key)
            # Death-record bands that fall inside this widowhood band.
            old_parts = {
                f"{transitions.band_label(mlo, mhi)}|{sex}": old_mean.get(
                    f"{transitions.band_label(mlo, mhi)}|{sex}"
                )
                for mlo, mhi in MORT_BANDS
                if mlo >= lo and mhi <= hi
            }
            old_here = old_mean.get(new_key)  # exact same-label band, if any
            ratio = (
                float(new_val / old_here)
                if (old_here not in (None, 0.0) and new_val is not None)
                else None
            )
            by_band.append(
                {
                    "band": new_band,
                    "sex": sex,
                    "new_widowhood_level_mean": new_val,
                    "old_death_record_level_mean": old_here,
                    "ratio_new_over_old": ratio,
                    "old_death_record_subbands_mean": old_parts,
                }
            )
    return {
        "note": (
            "old = candidate 5 ind2023er death-record central rate (psid_m), "
            "MORT_BANDS x dying-spouse sex; new = candidate 6 train "
            "marriage-history widowhood incidence, WIDOWHOOD_AGE_BANDS x "
            "surviving-spouse sex; indexed by opposite parties (see the "
            "widowhood model note). Cross-seed means; per-seed tables in each "
            "per_seed component_meta (mortality_level_candidate5_deathrecord "
            "and mortality_level_diagnostics)."
        ),
        "new_widowhood_level_bands": [list(b) for b in WIDOW_BANDS],
        "old_death_record_bands": [list(b) for b in MORT_BANDS],
        "old_death_record_level_mean": old_mean,
        "new_widowhood_level_mean": new_mean,
        "by_band": by_band,
    }


def _beta_block(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Candidate 5's committed NCHS betas, applied unchanged (read, not fit)."""
    committed = per_seed[0]["component_meta"]["mortality_beta_by_sex"]
    recomputed = {
        s["seed"]: s["component_meta"][
            "mortality_beta_by_sex_candidate5_recomputed"
        ]
        for s in per_seed
    }
    return {
        "beta_sex_committed_candidate5": committed,
        "beta_sex_source": (
            "runs/gate2_hazard_v5.json mortality_trend_beta_comparison."
            "beta_sex_nchs; read and applied unchanged (NOT re-fit)"
        ),
        "beta_sex_candidate5_recomputed_per_seed": recomputed,
        "note": (
            "the applied beta is candidate 5's committed NCHS value; the "
            "chain-import recomputes the identical external slope per seed "
            "(retained for the consistency check), but the applied value is "
            "the committed one"
        ),
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins, with the candidate-6 schema + c1-c5 + v5 shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for n in (1, 2, 3, 4, 5):
        pins[f"candidate{n}_runner"] = f"scripts/run_gate2_candidate{n}.py"
        pins[f"candidate{n}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{n}.py"
        )
    pins["candidate5_artifact"] = "runs/gate2_hazard_v5.json"
    pins["candidate5_artifact_sha256"] = c1._sha_of_file(CANDIDATE5_ARTIFACT)
    pins["mortality_level_source"] = (
        "train marriage-history widowhood endings (transitions.hazard_cells "
        "widowhood cells; the gate reference's own construction)"
    )
    return pins


def _model_block() -> dict[str, Any]:
    """Candidate 5's model block, edited for the one candidate-6 delta."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a "
            "mortality-composed widowhood component (candidate 5 + one named "
            "fix: the spouse-death hazard LEVEL is source-aligned to the "
            "reference -- the train marriage-history surviving-spouse "
            "widowhood hazard replaces the ind2023er death-record central "
            "rate)"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "delta_vs_candidate5": DELTA_VS_CANDIDATE5,
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic spline on age, "
                "knots 20/22/25/30/40, sex, birth-decade cohort -- "
                "BYTE-IDENTICAL to candidate 5 (no delta touches first "
                "marriage; its reference cells are byte-identical to candidate "
                "5 because first marriage is marital-state-independent)"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x order "
                "(1st vs 2+), add-one smoothed (byte-identical fit to "
                "candidates 1-5; its simulated cells move only through the "
                "delta's effect on the married-state exposure)"
            ),
            "widowhood": (
                "COMPOSED, parametric in period; DELTA: the spouse-death "
                "hazard LEVEL is the train marriage-history surviving-spouse "
                "widowhood incidence (mh85_23 how-ended = spouse death over "
                "married person-year exposure, by WIDOWHOOD_AGE_BANDS x "
                "surviving-spouse sex -- transitions.hazard_cells, the gate "
                "reference's own construction), replacing candidate 1's "
                "ind2023er death-record central-rate level. Applied as "
                "rate = widow_level(ego_band, ego_sex) * exp(beta_sex * "
                "(year - 1995)) with candidate 5's committed NCHS betas "
                "unchanged, looked up by the married ego's OWN (age, sex). The "
                "spousal age gap is candidate 5's empirical distribution draw, "
                "retained byte-identically but no longer entering the level"
            ),
            "remarriage": (
                "weighted empirical hazard by years-since-dissolution band x "
                "origin (divorced/widowed) x sex (candidate 1's origin-split "
                "table, byte-identical fit; its simulated cells move through "
                "the delta's higher widowed inflow feeding widowed-origin "
                "remarriage)"
            ),
            "fertility": (
                "single-year-of-age rates within parity (0/1/2/3+) x "
                "birth-decade cohort, triangular-kernel smoothed over age "
                "(candidate 5's delta 2) -- BYTE-IDENTICAL to candidate 5, and "
                "its reference cells are byte-identical because fertility is "
                "marital-state-independent under the shared RNG stream"
            ),
        },
        "registered_ambiguity_resolutions": {
            "level_source": (
                "the spouse-death hazard LEVEL is the surviving-spouse "
                "marriage-history widowhood hazard from "
                "transitions.hazard_cells (weighted=True) widowhood cells over "
                "WIDOWHOOD_AGE_BANDS, restricted to the seed's train "
                "complement -- the same construction, machinery and bands the "
                "gate reference uses; keyed (band, surviving-spouse sex) and "
                "consumed by the married ego's own (age, sex)"
            ),
            "below_youngest_band": (
                "married person-years below age 45 clip into the youngest "
                "widowhood band (45-54) via the same _bands_vec clip candidate "
                "1's mortality lookup uses below age 25; only the table and "
                "its bands change, not the clip mechanism"
            ),
            "trend_multiplier": (
                "the NCHS period-trend multiplier exp(beta_sex * "
                "(year - 1995)) with candidate 5's committed betas (female "
                "-0.009234704865961198, male -0.010643975395626533; read from "
                "runs/gate2_hazard_v5.json, NOT re-fit) applies unchanged, "
                "indexed by the surviving-spouse (ego) sex the level is keyed "
                "on"
            ),
            "everything_else": (
                "the knot-at-22 first-marriage spline, divorce, the "
                "single-year triangular-kernel fertility, the origin-split "
                "remarriage table, the spousal-gap distribution draw, the "
                "competing-risk step, the RNG rule default_rng(4200 + seed), "
                "the spawned gap-draw stream, one sequence per person, and the "
                "locked protocol are byte-identical to candidate 5"
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

    # Preflight: the committed candidate-5 artifact must be present (the
    # applied NCHS beta is read from it) and the candidate-5 chain's NCHS
    # references must be present (fit_components imports candidate 5's fitter).
    if not CANDIDATE5_ARTIFACT.exists():
        raise RuntimeError(
            f"candidate-5 artifact missing at {CANDIDATE5_ARTIFACT}; the "
            "committed NCHS beta is read from it."
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
    candidate5_comparison = _candidate5_comparison(per_seed)
    fm_fert_identity = _fertility_first_marriage_identity(per_seed)
    mortality_level_comparison = _mortality_level_comparison(per_seed)
    beta_block = _beta_block(per_seed)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 6",
        "spec_registration": SPEC_REGISTRATION,
        "candidate5_registration": CANDIDATE5_REGISTRATION,
        "candidate4_registration": CANDIDATE4_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "delta_vs_candidate5": DELTA_VS_CANDIDATE5,
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79 + flip #81); "
            "protocol/views/tolerances read at runtime, no threshold moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.5-0.6",
            "conjunction_estimate": 0.55,
            "component_probabilities": {
                "share_widowed_75plus_female_fixed": 0.7,
                "widowhood_45_64_female_fixed": 0.7,
                "mean_lifetime_marriages_male_passes": 0.5,
                "completed_fertility_c1970s_passes": 0.8,
            },
            "modal_failure": (
                "mean_lifetime_marriages|male at its 0.047 tolerance (the "
                "untargeted sequence statistic; higher widow inflow feeds "
                "widowed-origin remarriage, raising marriage counts slightly "
                "-- ~0.5 it clears; the modal failure if candidate 6 fails)"
            ),
            "component_reads": (
                "the source alignment removes the documented ~0.76 level wedge "
                "exactly where the failures sit -- share_widowed.75+|female "
                "needs the widow inflow up ~25-35% and widowhood.45-64|female "
                "the same direction; completed_fertility.c1970s is untouched "
                "(single-year fertility byte-identical to candidate 5, so the "
                "seed-2 clip persists byte-identically -- ~0.8 the seed can "
                "absorb it if its other cells pass)"
            ),
            "next_step_if_fail": (
                "if candidate 6 fails with the widowed stock fixed but "
                "mean_lifetime_marriages|male persisting alone >=3 seeds, the "
                "next registration targets the remarriage level directly, "
                "still under the one-shot rule"
            ),
            "delta_vs_candidate5": DELTA_VS_CANDIDATE5,
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
        "candidate5_comparison": candidate5_comparison,
        "first_marriage_fertility_identity_vs_candidate5": fm_fert_identity,
        "mortality_level_comparison": mortality_level_comparison,
        "mortality_beta": beta_block,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "candidate5_registration": CANDIDATE5_REGISTRATION,
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
            "registered modal (mean_lifetime_marriages|male) materialized: "
            f"{modal['modal_failed']} (seeds {modal['modal_failed_seeds']}); "
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
