"""Gate-2c candidate 2: earnings-conditioned first-marriage timing.

Registered on issue #42 comment 4950370498 (the FROZEN spec). Candidate 2 of
the audited 2c ladder, one targeted delta against candidate 1
(:mod:`populace_dynamics.models.couple_formation_sim_v1`, grading comment
4950370477 -- the sole binding family was ``first_marriage_by_earnings``, all
five misses marginal, 1.02-1.34x tolerance).

THE ONE DELTA -- earnings-axis conditioning of first-marriage timing
--------------------------------------------------------------------
A train-fitted multiplicative hazard modifier ``m(tercile | age band, sex)``
composed onto the certified tranche-2a first-marriage hazard. Candidate 1's
certified core conditions timing on age x sex x cohort but NOT on the earnings
axis, so within an (age band, sex) cell it reproduces the pooled hazard and
only the age/cohort composition differentiates terciles (grading 4950370477).
The modifier supplies the RESIDUAL earnings gradient the earnings-blind core
misses:

    m_raw(t | b, s) = real_train_hazard(t, b, s)
                      / certified_expected_hazard(t, b, s)

where ``certified_expected_hazard`` is the exposure-weighted mean of the
certified ``first_marriage.predict(age, is_male, birth_decade)`` over the
train never-married person-years in the cell (deterministic -- the fitted 2a
hazard is READ, never re-simulated, so the fit consumes no RNG). A cell whose
certified core already matches the real tercile hazard gets ``m_raw ~ 1`` (no
double-counting of the age/cohort gradient the core already carries).

Marginal preservation (the load-bearing constraint)
---------------------------------------------------
The modifier is NORMALIZED so the certified core's age x sex timing marginal
does not move -- the same marginal-preservation constraint class the 2b
coupling used (``household_composition_sim_v5``: a delta that READS a carried
marginal and composes a conditional onto it without moving the marginal).
With the modifier applied to the certified first-marriage event weights, the
pooled-over-tercile band hazard scales by ``sum_t m(t | b, s) * phi_cert(t | b,
s)`` where ``phi_cert`` is the certified expected first-marriage EVENT share by
tercile. Normalizing

    Z(b, s) = sum_t m_shrunk(t | b, s) * phi_cert(t | b, s)
    m(t | b, s) = m_shrunk(t | b, s) / Z(b, s)

makes ``sum_t m * phi_cert = 1`` per band EXACTLY (the frozen constraint:
"sum over terciles of m x train tercile shares = 1 per band"), so the certified
pooled band hazard is preserved in expectation. The realized per-draw pooled
band hazard moves only by the simulated-vs-expected event-share drift, which
:func:`simulate_draw_v2` records per draw; the exact constraint is recorded per
draw too, and a constraint violation is a SPEC violation.

Byte-carry (the c9 / 2b write-gate carry pattern)
-------------------------------------------------
Everything else is byte-carried from candidate 1. The modifier is applied as a
deterministic reweighting of the ``first_marriage`` rows of the ALREADY-built
``marital_events`` frame, AFTER :func:`couple_earnings._build_marital` and
consuming NO random draws, so:

* the certified :func:`simulate`, the assortative-kernel draws, the spousal
  age-gap draws and the event-window draws consume the IDENTICAL RNG streams
  as candidate 1 -- the simulated couples, event windows and never-married
  exposure are bit-identical;
* only the ``first_marriage`` event WEIGHTS change, so only the
  ``first_marriage_by_earnings`` cells move; ``assort_mating``,
  ``remarriage_by_earnings`` (remarriage rows are untouched),
  ``earnings_around_{marriage,divorce}`` and ``shared_earnings_ratio`` are
  bit-identical to candidate 1's per-draw rates.

The run proves this with a per-draw byte-carry regression against
``runs/gate2c_hazard_v1.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from populace_dynamics.data import couple_earnings as ce
from populace_dynamics.data import transitions
from populace_dynamics.models import couple_formation_sim_v1 as v1
from populace_dynamics.models.couple_formation_sim_v1 import (
    CommittedAxis,
    CoupleFormationModelV1,
    build_committed_axis,
    fit_couple_model_v1,
)
from populace_dynamics.models.family_transitions.fitted import (
    FittedFamilyTransitions,
)

__all__ = [
    "MODIFIER_SHRINKAGE_ALPHA",
    "MODIFIER_AGE_BANDS",
    "GATED_MARGINAL_BANDS",
    "CommittedAxis",
    "build_committed_axis",
    "CERTIFIED_SPEC",
    "KERNEL_SMOOTHING_ALPHA",
    "N_DECILES",
    "FirstMarriageEarningsModifier",
    "fit_first_marriage_modifier",
    "CoupleFormationModelV2",
    "fit_couple_model_v2",
    "apply_first_marriage_modifier",
    "simulate_draw_v2",
]

#: Carried candidate-1 pins re-exported so the runner and tests pin the same
#: certified core (byte-carry): the certified 2a spec + its sha, the
#: assortative-kernel smoothing strength and the earnings-axis decile count.
CERTIFIED_SPEC = v1.CERTIFIED_SPEC
KERNEL_SMOOTHING_ALPHA = v1.KERNEL_SMOOTHING_ALPHA
N_DECILES = v1.N_DECILES

#: Empirical-Bayes shrinkage strength for the log-modifier: a cell's residual
#: ``ln(real / certified)`` is shrunk toward 0 (neutral) with weight
#: ``n_events / (n_events + alpha)``. A light regularizer so a thin cell's
#: noisy ratio cannot dominate the normalization; the dense GATED cells
#: (n_events >= 570) are essentially unshrunk (weight >= 0.986). Pinned,
#: seed-independent judgment knob; recorded in the run artifact. Chosen a
#: priori (the gate verdict is invariant to alpha over {0, 8, 20} -- the
#: residual modifier is well estimated for every gated cell).
MODIFIER_SHRINKAGE_ALPHA = 8.0

#: The modifier conditioning age bands -- the certified first-marriage hazard
#: bands (18-24, 25-34, 35-44, 45+), so a modifier band matches the gated
#: ``first_marriage_by_earnings`` cell band exactly.
MODIFIER_AGE_BANDS = ce.FIRST_MARRIAGE_AGE_BANDS

_SEX_INDEX = {"female": 0, "male": 1}
_N_SEX = len(ce.SEXES)
_N_BANDS = len(MODIFIER_AGE_BANDS)
_N_TERC = len(ce.TERCILES)
_BAND_LABELS = tuple(ce.band_label(lo, hi) for lo, hi in MODIFIER_AGE_BANDS)

#: The bands with at least one GATED ``first_marriage_by_earnings`` cell
#: (18-24 all terciles; 25-34 tercile 3). The certified pooled band-hazard the
#: modifier must not move is scored only on these bands; the sparse 35-44 / 45+
#: bands are report-only (POWER), where a thin cell's realized event share can
#: drift from its certified expectation and move the realized pooled hazard
#: (disclosed, never gated).
GATED_MARGINAL_BANDS = ("18-24", "25-34")


# --------------------------------------------------------------------------
# The fitted first-marriage earnings modifier
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FirstMarriageEarningsModifier:
    """The train-fitted, normalized first-marriage earnings-tercile modifier.

    All arrays are indexed ``[sex(2), band(4), tercile(3)]`` on the sex order
    :data:`couple_earnings.SEXES`, the bands :data:`MODIFIER_AGE_BANDS`, and
    the terciles ``1, 2, 3``.

    Attributes:
        m_norm: the APPLIED modifier -- ``m_shrunk`` divided by the
            per-(band, sex) normalizer ``Z`` so ``sum_t m_norm * phi_cert = 1``.
        m_raw: the unshrunk, unnormalized fit ``real_hazard /
            certified_expected_hazard`` (the "raw" modifier).
        m_shrunk: ``exp(w * ln(m_raw))`` with ``w = n_events / (n_events +
            alpha)`` (log-shrunk toward the neutral modifier 1).
        phi_cert: the certified expected first-marriage EVENT share by tercile
            within (band, sex) -- the "train tercile shares" the normalization
            constraint is written against (they sum to 1 over terciles).
        z_norm: ``Z[sex, band] = sum_t m_shrunk * phi_cert`` (the per-band
            normalizer the modifier is divided by).
        n_events: the real train first-marriage event COUNT per cell (the
            shrinkage precision).
        alpha: the pinned shrinkage strength.
        meta: fit diagnostics recorded in the run artifact.
    """

    m_norm: np.ndarray
    m_raw: np.ndarray
    m_shrunk: np.ndarray
    phi_cert: np.ndarray
    z_norm: np.ndarray
    n_events: np.ndarray
    alpha: float
    meta: dict[str, Any]

    def lookup(self, sex: np.ndarray, band: np.ndarray, tercile: np.ndarray):
        """Vectorized ``m_norm`` for arrays of (sex, band-label, tercile)."""
        si = np.array([_SEX_INDEX[s] for s in sex], dtype=int)
        bi = np.array(
            [_BAND_LABELS.index(b) if b in _BAND_LABELS else -1 for b in band],
            dtype=int,
        )
        ti = np.array([int(t) - 1 for t in tercile], dtype=int)
        out = np.ones(len(si), dtype=float)
        valid = (bi >= 0) & (ti >= 0) & (ti < _N_TERC)
        out[valid] = self.m_norm[si[valid], bi[valid], ti[valid]]
        return out

    def constraint_per_band(self) -> np.ndarray:
        """``sum_t m_norm * phi_cert`` per ``[sex, band]`` (each == 1.0)."""
        return (self.m_norm * self.phi_cert).sum(axis=2)

    def constraint_max_abs_dev(self) -> float:
        """The max ``|sum_t m_norm * phi_cert - 1|`` over defined bands."""
        con = self.constraint_per_band()
        defined = self.phi_cert.sum(axis=2) > 0
        if not defined.any():
            return 0.0
        return float(np.max(np.abs(con[defined] - 1.0)))

    def fit_vs_raw_record(self) -> dict[str, Any]:
        """The per-cell fit (m_raw) vs applied (m_norm) disclosure."""
        rows: dict[str, dict[str, float]] = {}
        for si, s in enumerate(ce.SEXES):
            for bi, b in enumerate(_BAND_LABELS):
                for ti, t in enumerate(ce.TERCILES):
                    rows[f"t{t}.{b}|{s}"] = {
                        "m_raw": float(self.m_raw[si, bi, ti]),
                        "m_shrunk": float(self.m_shrunk[si, bi, ti]),
                        "m_norm": float(self.m_norm[si, bi, ti]),
                        "phi_cert": float(self.phi_cert[si, bi, ti]),
                        "n_events": int(self.n_events[si, bi, ti]),
                    }
        return {
            "alpha": self.alpha,
            "z_norm_by_sex_band": {
                f"{s}.{b}": float(self.z_norm[si, bi])
                for si, s in enumerate(ce.SEXES)
                for bi, b in enumerate(_BAND_LABELS)
            },
            "constraint_max_abs_dev_from_one": self.constraint_max_abs_dev(),
            "cells": rows,
        }


def _fm_band_index(ages: np.ndarray) -> np.ndarray:
    """Band index (0..3) of each age, or -1 when below the first band."""
    out = np.full(len(ages), -1, dtype=int)
    for bi, (lo, hi) in enumerate(MODIFIER_AGE_BANDS):
        out[(ages >= lo) & (ages <= hi)] = bi
    return out


def fit_first_marriage_modifier(
    components: FittedFamilyTransitions,
    mpanel: transitions.MaritalPanel,
    axis: CommittedAxis,
    train_ids: set[int],
    *,
    alpha: float = MODIFIER_SHRINKAGE_ALPHA,
) -> FirstMarriageEarningsModifier:
    """Fit ``m(tercile | age band, sex)`` on the seed's train half (side B).

    The real train first-marriage hazard by cell is measured on the certified
    marital panel's never-married person-years and first-marriage events
    (restricted to the earnings supply, so the cells match
    :func:`couple_earnings._first_marriage_cells` exactly); the certified
    expected hazard is the exposure-weighted mean of the certified
    ``first_marriage.predict`` over the SAME train person-years (deterministic;
    no RNG). The modifier is the shrunk, normalized ratio.
    """
    supply = set(axis.earn)
    tmap = {pid: ce._tercile_of(v, axis.cuts) for pid, v in axis.earn.items()}

    # Train never-married person-years in the earnings supply (the hazard
    # denominator + the certified-hazard evaluation set).
    py = mpanel.person_years
    py = py[
        py["person_id"].isin(train_ids)
        & py["person_id"].isin(supply)
        & (py["marital_state"] == "never_married")
    ]
    ages = py["age"].to_numpy(dtype=np.int64)
    years = py["year"].to_numpy(dtype=np.int64)
    band_idx = _fm_band_index(ages)
    keep = band_idx >= 0
    ages = ages[keep]
    years = years[keep]
    band_idx = band_idx[keep]
    pids = py["person_id"].to_numpy()[keep]
    sexes = py["sex"].to_numpy()[keep]
    weights = py["weight"].to_numpy(dtype=np.float64)[keep]
    sex_idx = np.array([_SEX_INDEX.get(s, -1) for s in sexes], dtype=int)
    terc_idx = np.array([tmap[int(p)] - 1 for p in pids], dtype=int)
    decade = ((years - ages) // 10 * 10).astype(np.int64)
    is_male = sex_idx == 1

    # Certified expected first-marriage probability per never-married PY.
    h_cert = components.first_marriage.predict(
        ages.astype(np.float64), is_male, decade
    )
    cert_ev_wt = h_cert * weights  # expected certified event weight per PY

    # Real train first-marriage events by cell (same supply, band, tercile).
    ev = mpanel.events
    ev = ev[
        ev["person_id"].isin(train_ids)
        & ev["person_id"].isin(supply)
        & (ev["transition"] == "first_marriage")
    ]
    ev_ages = ev["age"].to_numpy(dtype=np.int64)
    ev_band = _fm_band_index(ev_ages)
    ev_keep = ev_band >= 0
    ev_band = ev_band[ev_keep]
    ev_sex = np.array(
        [_SEX_INDEX.get(s, -1) for s in ev["sex"].to_numpy()[ev_keep]],
        dtype=int,
    )
    ev_terc = np.array(
        [tmap[int(p)] - 1 for p in ev["person_id"].to_numpy()[ev_keep]],
        dtype=int,
    )
    ev_wt = ev["weight"].to_numpy(dtype=np.float64)[ev_keep]

    shape = (_N_SEX, _N_BANDS, _N_TERC)
    cert_ev = np.zeros(shape)
    real_ev = np.zeros(shape)
    n_ev = np.zeros(shape, dtype=int)
    valid_py = (sex_idx >= 0) & (terc_idx >= 0)
    np.add.at(
        cert_ev,
        (sex_idx[valid_py], band_idx[valid_py], terc_idx[valid_py]),
        cert_ev_wt[valid_py],
    )
    valid_ev = (ev_sex >= 0) & (ev_terc >= 0)
    np.add.at(
        real_ev,
        (ev_sex[valid_ev], ev_band[valid_ev], ev_terc[valid_ev]),
        ev_wt[valid_ev],
    )
    np.add.at(
        n_ev,
        (ev_sex[valid_ev], ev_band[valid_ev], ev_terc[valid_ev]),
        1,
    )

    # Raw modifier real / certified (expected event weights; the shared
    # exposure cancels). Undefined cells (no certified mass or no real event)
    # are neutral (1.0).
    with np.errstate(divide="ignore", invalid="ignore"):
        m_raw = np.where((cert_ev > 0) & (real_ev > 0), real_ev / cert_ev, 1.0)
    # Log-shrink toward neutral by real-event count.
    w = n_ev / (n_ev + alpha) if alpha > 0 else np.ones(shape)
    m_shrunk = np.exp(w * np.log(m_raw))

    # Certified expected event share (the "train tercile shares").
    band_cert = cert_ev.sum(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_cert = np.where(band_cert > 0, cert_ev / band_cert, 0.0)

    # Normalize so sum_t m * phi_cert = 1 per (sex, band).
    z = (m_shrunk * phi_cert).sum(axis=2)
    z_safe = np.where(z > 0, z, 1.0)
    m_norm = m_shrunk / z_safe[:, :, None]

    meta = {
        "alpha": float(alpha),
        "n_train_never_married_person_years": int(valid_py.sum()),
        "n_train_first_marriage_events": int(valid_ev.sum()),
        "certified_hazard_source": (
            "components.first_marriage.predict(age, is_male, birth_decade) "
            "evaluated on the train never-married supply person-years "
            "(deterministic; the fit consumes no RNG)"
        ),
        "normalization_shares": "certified_expected_first_marriage_event_share",
    }
    modifier = FirstMarriageEarningsModifier(
        m_norm=m_norm,
        m_raw=m_raw,
        m_shrunk=m_shrunk,
        phi_cert=phi_cert,
        z_norm=z,
        n_events=n_ev,
        alpha=float(alpha),
        meta=meta,
    )
    return modifier


# --------------------------------------------------------------------------
# The candidate-2 model: candidate 1 + the first-marriage earnings modifier
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CoupleFormationModelV2:
    """Candidate 1's per-seed fit plus the first-marriage earnings modifier.

    Attributes:
        base: the byte-identical candidate-1 fit
            (:class:`couple_formation_sim_v1.CoupleFormationModelV1`) -- the
            certified components, the assortative kernel, the event-window
            pools, all carried unchanged.
        fm_modifier: the train-fitted, normalized first-marriage earnings
            modifier (the ONE delta).
        meta: the base fit meta plus the modifier meta.
    """

    base: CoupleFormationModelV1
    fm_modifier: FirstMarriageEarningsModifier
    meta: dict[str, Any]

    @property
    def components(self) -> FittedFamilyTransitions:
        return self.base.components

    @property
    def assort_kernel(self) -> np.ndarray:
        return self.base.assort_kernel


def fit_couple_model_v2(
    ce_panel: ce.CoupleEarningsPanel,
    mpanel: transitions.MaritalPanel,
    *,
    demographic_panel: pd.DataFrame,
    marriage_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    marriage_order_map: pd.DataFrame,
    axis: CommittedAxis,
    train_ids: set[int],
    alpha: float = MODIFIER_SHRINKAGE_ALPHA,
    kernel_alpha: float = v1.KERNEL_SMOOTHING_ALPHA,
) -> CoupleFormationModelV2:
    """Fit candidate 1 unchanged, then fit the first-marriage modifier.

    The base fit is candidate 1 verbatim (byte-identical certified components,
    assortative kernel and event-window pools -- the modifier fit READS the
    fitted certified hazard and adds no RNG), so the carried families simulate
    bit-identically.
    """
    base = fit_couple_model_v1(
        ce_panel,
        mpanel,
        demographic_panel=demographic_panel,
        marriage_records=marriage_records,
        birth_records=birth_records,
        marriage_order_map=marriage_order_map,
        axis=axis,
        train_ids=train_ids,
        alpha=kernel_alpha,
    )
    fm_modifier = fit_first_marriage_modifier(
        base.components, mpanel, axis, train_ids, alpha=alpha
    )
    meta = dict(base.meta)
    meta["fm_earnings_modifier"] = fm_modifier.meta
    return CoupleFormationModelV2(
        base=base, fm_modifier=fm_modifier, meta=meta
    )


# --------------------------------------------------------------------------
# Apply the modifier + the per-draw marginal-preservation check
# --------------------------------------------------------------------------
def _pooled_band_hazard(
    events: pd.DataFrame, exposure: pd.DataFrame
) -> dict[tuple[str, str], float]:
    """Pooled-over-tercile first-marriage hazard per (band, sex)."""
    ev = events[events["transition"] == "first_marriage"]
    ex = exposure[exposure["marital_state"] == "never_married"]
    out: dict[tuple[str, str], float] = {}
    ev_g = ev.groupby(["fm_band", "sex"])["weight"].sum()
    ex_g = ex.groupby(["fm_band", "sex"])["weight"].sum()
    for key, denom in ex_g.items():
        num = float(ev_g.get(key, 0.0))
        out[key] = num / float(denom) if denom > 0 else 0.0
    return out


def apply_first_marriage_modifier(
    marital_events: pd.DataFrame,
    modifier: FirstMarriageEarningsModifier,
    marital_exposure: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scale first-marriage event weights by ``m(tercile | band, sex)``.

    Returns a NEW events frame (remarriage rows untouched; row order and every
    other column identical to the input, so only the ``first_marriage_by_
    earnings`` cells move) and the per-draw marginal-preservation check: the
    exact fit constraint ``sum_t m * phi_cert = 1`` and the REALIZED pooled
    band-hazard deviation the reweighting induces on this draw.
    """
    events = marital_events.copy()
    is_fm = (events["transition"] == "first_marriage").to_numpy()
    factors = np.ones(len(events), dtype=float)
    if is_fm.any():
        fm = events.loc[is_fm]
        factors[is_fm] = modifier.lookup(
            fm["sex"].to_numpy(),
            fm["fm_band"].to_numpy(),
            fm["tercile"].to_numpy(),
        )
    before = _pooled_band_hazard(events, marital_exposure)
    events["weight"] = events["weight"].to_numpy(dtype=float) * factors
    after = _pooled_band_hazard(events, marital_exposure)

    realized = {}
    max_abs_ln = 0.0
    max_abs_ln_gated = 0.0
    for key in before:
        h0 = before[key]
        h1 = after.get(key, 0.0)
        if h0 > 0 and h1 > 0:
            dev = abs(float(np.log(h1 / h0)))
        else:
            dev = 0.0
        band, sex = key
        realized[f"{band}|{sex}"] = dev
        max_abs_ln = max(max_abs_ln, dev)
        if band in GATED_MARGINAL_BANDS:
            max_abs_ln_gated = max(max_abs_ln_gated, dev)

    check = {
        "constraint_max_abs_dev_from_one": modifier.constraint_max_abs_dev(),
        "constraint_holds": bool(modifier.constraint_max_abs_dev() <= 1e-9),
        "realized_pooled_band_hazard_max_abs_ln": max_abs_ln,
        "realized_pooled_band_hazard_max_abs_ln_gated_bands": (
            max_abs_ln_gated
        ),
        "realized_pooled_band_hazard_abs_ln": realized,
        "n_first_marriage_events_reweighted": int(is_fm.sum()),
    }
    if not check["constraint_holds"]:
        raise RuntimeError(
            "marginal-preservation constraint violated (spec violation): "
            f"sum_t m*phi_cert deviates from 1 by "
            f"{check['constraint_max_abs_dev_from_one']:.2e}"
        )
    return events, check


# --------------------------------------------------------------------------
# Simulate one draw (holdout / side A) -- candidate 1 + the modifier
# --------------------------------------------------------------------------
def simulate_draw_v2(
    ce_panel: ce.CoupleEarningsPanel,
    mpanel: transitions.MaritalPanel,
    model: CoupleFormationModelV2,
    axis: CommittedAxis,
    holdout_ids: set[int],
    draw_seed: int,
) -> tuple[ce.CoupleEarningsPanel, dict[str, Any]]:
    """Simulate the seed's holdout (side A) at draw ``draw_seed``.

    Byte-identical to :func:`couple_formation_sim_v1.simulate_draw_v1` in RNG
    topology and in every carried family -- the certified :func:`simulate`, the
    assortative-kernel / spouse-value / age-gap / event-window draws all
    consume the identical streams. The SOLE difference is that the
    ``first_marriage`` rows of ``marital_events`` are reweighted by the fitted
    earnings modifier AFTER :func:`_build_marital` (consuming no RNG), so only
    the ``first_marriage_by_earnings`` cells move.
    """
    base = model.base
    valid = set(int(v) for v in mpanel.attrs["person_id"])
    sim_ids = holdout_ids & valid
    sim_panel, _sim_births = v1.simulate(
        mpanel, sim_ids, base.components, draw_seed
    )

    seeds = np.random.SeedSequence([draw_seed, 0x2C1]).spawn(4)
    rng_dec = np.random.default_rng(seeds[0])
    rng_val = np.random.default_rng(seeds[1])
    rng_gap = np.random.default_rng(seeds[2])
    rng_ewin = np.random.default_rng(seeds[3])

    marital_events, marital_exposure = ce._build_marital(
        sim_panel, axis.earn, axis.cuts
    )
    # THE ONE DELTA: earnings-condition the first-marriage timing.
    marital_events, marginal_check = apply_first_marriage_modifier(
        marital_events, model.fm_modifier, marital_exposure
    )

    couples, couple_diag = v1._build_simulated_couples(
        sim_panel, base, axis, rng_dec, rng_val, rng_gap
    )
    event_windows = v1._build_simulated_event_windows(
        sim_panel, base, axis, rng_ewin
    )

    sim_ce_panel = ce.CoupleEarningsPanel(
        couples=couples,
        marital_events=marital_events,
        marital_exposure=marital_exposure,
        event_windows=event_windows,
        attrs=ce_panel.attrs,
        earn_tercile_cuts=axis.cuts,
        placebo_deflators=ce_panel.placebo_deflators,
        meta={},
    )
    diag = {
        "draw_seed": draw_seed,
        "n_sim_holdout_persons": len(sim_ids),
        "n_marital_events": int(len(marital_events)),
        "n_marital_exposure_person_years": int(len(marital_exposure)),
        "n_event_windows": int(len(event_windows)),
        "marginal_preservation_check": marginal_check,
        **couple_diag,
    }
    return sim_ce_panel, diag
