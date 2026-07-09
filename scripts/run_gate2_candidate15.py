"""Gate-2 candidate 15 (run 1): candidate 14 + EXACTLY ONE delta, scored under
the amended mean-over-K=20-draws estimator.

The FIFTEENTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42 comment
4928232089 (``SPEC_REGISTRATION``): candidate 14's frozen spec (comment
4927236029, ``scripts/run_gate2_candidate14.py``, merged #105) verbatim EXCEPT
one delta, registered from candidate 14's grading (the composed widowhood
hazard is scored against the untrended PSID reference but carries an NCHS
mortality-decline trend, so it systematically undershoots the gate's own
measurement basis). One-shot; no constant moves after the registration comment;
published REGARDLESS of verdict.

The one delta vs candidate 14 (everything else byte-identical)
-------------------------------------------------------------
**The NCHS period-trend multiplier ``exp(beta_sex * (year - 1995))`` is REMOVED
from the surviving-spouse widowhood hazard.** Candidate 14 applied, at
simulation time, ``rate = widow_level(ego_band, ego_sex) * exp(beta_sex *
(year - 1995))`` -- the train-empirical 7-band x sex level scaled by candidate
5's committed NCHS log-linear period slope. Candidate 15 applies ``rate =
widow_level(ego_band, ego_sex)`` -- the source-aligned train-empirical 7-band
x sex level, period-pooled, with the SAME smoothing convention
(``transitions._hazard_by_band`` weighted hazard num_wt / den_wt, no add-one).
The reference the gate scores against is the untrended PSID panel measurement;
the trend scaled panel-era spouse-death inflow to ~0.92-0.95 of that empirical
rate over the exposure-weighted years, so removing it aligns the composition
with the gate's own measurement basis. The committed NCHS beta values stay
DOCUMENTED in the artifact (``component_meta.mortality_beta_by_sex`` and
``nchs_trend_beta_by_sex_committed``) for deployment-time use (projection under
Trustees-style assumptions); nothing else changes.

Everything else is byte-identical to candidate 14 -- the seven-band
surviving-spouse widowhood LEVEL table itself (candidate 14's delta, refit
identically), the entry-widowed observed initial state (candidate 12's delta
1), the 5-band remarriage current-age table (candidate 11), the observed
undatable-marriage lifetime-count initial state (candidate 9's delta 1), the
single-year triangular-kernel fertility, the RNG, the K=20 mean-of-draws
protocol, and ``fresh_run_artifact_schema`` conformance (per-draw per-cell
rates [20, 46, 5]; undefined draw invalidates; report-only dispersion). The
vestigial spousal-age-gap machinery (candidate 12's delta 2, proven inert) is
left UNTOUCHED per byte-minimality. Runner
``scripts/run_gate2_candidate15.py``, artifact ``runs/gate2_hazard_v15.json``.

Provable byte-identity (code-object reuse)
------------------------------------------
Candidate 14 was a DATA delta (the band table), so every compute code object --
including ``_widow_probs`` -- was reused. Candidate 15 is a COMPUTE delta: the
whole change lives in ``_widow_probs`` (it drops the ``* trend`` factor). So
``_widow_probs`` MOVES from the reused set into the DIVERGED set: candidate 15
RE-IMPLEMENTS ``_widow_probs`` (returning the level only) and REUSES candidate
14's EXACT code objects for the rest of the compute chain
(``_build_sim_lookups``, ``simulate_holdout``, ``_draw_moments``,
``score_seed``, ``fit_remarriage_age_banded``,
``_remarriage_probs_age_banded``), rebound (:func:`_rebind`) so their globals
resolve against THIS module -- so the byte-identical ``simulate_holdout`` calls
candidate 15's untrended ``_widow_probs`` while every other statement is
candidate 14's, guaranteed at the bytecode level
(``candidate15.simulate_holdout.__code__ is
candidate14.simulate_holdout.__code__`` and
``candidate15._widow_probs.__code__ is NOT
candidate14._widow_probs.__code__``). ``fit_components`` is a thin wrapper over
:func:`candidate14.fit_components` -- the FIT is byte-identical (the same
seven-band level, the same committed NCHS betas retained in ``meta``) and the
wrapper only records the trend-removal provenance.

Removing ``* trend`` is RNG-NEUTRAL: the per-year uniform block
(``rng.random(n_active)``) is drawn BEFORE ``_widow_probs`` and the widowhood
threshold array keeps its shape and dtype, so only the competing-risk THRESHOLD
value moves. The scored RNG stream is therefore byte-identical to candidate 14,
and the marital-state-independent cells (``asfr.*``, ``completed_fertility.*``,
``first_marriage.*``) are byte-identical to candidate 14 draw-by-draw. Unlike
candidate 14's split (localised above age 75), candidate 15's removal touches
the widowhood hazard at EVERY band, so all widowhood-incidence and
widowed-stock cells move.

Designed effect (registration)
------------------------------
Removing a ~0.93 multiplier lifts widowhood inflow ~7% across all bands: the
75+ stock ratio moves from ~0.838 toward ~0.89 (seed 0 needs +0.2%, seed 3
+2.6% -- both inside the lift with margin); elderly incidence moves 0.95 ->
~1.02 (the gated cells have ln(1.5)-scale headroom, ~0.95 they stay); counts
gain only near-zero-remarriage elderly widows plus a ~7% young-widow inflow
rise against female-count margins of 0.013-0.026 ln (~0.8 they hold).
``P(pass) ~= 0.65-0.75``; pass path 0, 1, 3, 4 (seed 2's fertility tilt
persists untouched). Modal failure if it fails: the female count cells
re-clipping from the young-inflow rise; secondary: seed-3's stock needing more
than the exposure-weighted lift delivers.

Hard-stop prechecks (inherited): the scoring path must reproduce, bit-for-bit,
every committed full-panel reference moment, every committed per-gate-seed
``rate_a`` and each gate seed's committed holdout-id sha256 BEFORE any candidate
is simulated; candidate 9's delta-1 count reconciliation must close to remainder
0.0; and candidate 12's entry-widowed carried classification must reproduce
forensics 3's committed Q6 initial-state-fixable share to float precision. Any
mismatch is a hard stop. Run ONCE; publish REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit; no statsmodels). Run from the repository root with the PSID
history files staged::

    .venv/bin/python scripts/run_gate2_candidate15.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Candidate 14 supplies the machinery this build deltas ONCE: its compute chain
# (the seven-band surviving-spouse widowhood table x candidate 5's committed
# NCHS trend over candidate 12's entry-widowed initial state + age-band gap over
# candidate 11's 5-band remarriage over candidate 8's fit and candidate 9's
# delta 1), its fit (byte-identical), the fresh-run artifact-schema blocks, and
# -- via its imports -- candidate 1's precheck / verdict assembly and candidate
# 8's vectorised simulation helpers. Only the surviving-spouse widowhood
# APPLICATION changes: the NCHS trend multiplier is removed from _widow_probs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate5 as c5  # noqa: E402
import run_gate2_candidate6 as c6  # noqa: E402
import run_gate2_candidate14 as c14  # noqa: E402

# Candidates 4/8/9/10/11/12/13 in the chain are import-bound THROUGH candidate
# 14 (its ``simulate_holdout`` / gap machinery / delta-1 helpers / schema blocks
# / seven-band fit are candidate 15's, reused), so candidate 15 needs candidate
# 1 (precheck / verdict), candidate 5 (committed NCHS references), candidate 6
# (the trend anchor + committed-beta guard) and candidate 14 directly.
from populace_dynamics.data import marriage, transitions  # noqa: E402

# ``hpanel`` is referenced by candidate 14's ``score_seed`` code object, which
# candidate 15 reuses rebound to THIS module's globals (see ``_rebind`` below);
# it must stay a module global even though candidate 15's own source never
# names it (F401 is silenced, not the runtime need).
from populace_dynamics.harness import panel as hpanel  # noqa: E402, F401

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v15.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
CANDIDATE14_ARTIFACT = ROOT / "runs" / "gate2_hazard_v14.json"
CANDIDATE13_ARTIFACT = ROOT / "runs" / "gate2_hazard_v13.json"
CANDIDATE12_ARTIFACT = ROOT / "runs" / "gate2_hazard_v12.json"
CANDIDATE11_ARTIFACT = ROOT / "runs" / "gate2_hazard_v11.json"
CANDIDATE10_ARTIFACT = ROOT / "runs" / "gate2_hazard_v10.json"
CANDIDATE9_ARTIFACT = ROOT / "runs" / "gate2_hazard_v9.json"
CANDIDATE8_ARTIFACT = ROOT / "runs" / "gate2_hazard_v8.json"
CANDIDATE7_ARTIFACT = ROOT / "runs" / "gate2_hazard_v7.json"
CANDIDATE6_ARTIFACT = ROOT / "runs" / "gate2_hazard_v6.json"
CANDIDATE5_ARTIFACT = ROOT / "runs" / "gate2_hazard_v5.json"
FORENSICS3_ARTIFACT = ROOT / "runs" / "gate2_forensics3_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v15"
RUN_NAME = "gate2_hazard_v15"

#: This run's frozen-spec registration (issue #42, comment 4928232089).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4928232089"
)
#: The bare registration comment id, pinned for the artifact pointer.
REGISTRATION_POINTER = "4928232089"
#: The candidate-14 spec this build deltas ONCE (comment 4927236029, #105).
CANDIDATE14_REGISTRATION = c14.SPEC_REGISTRATION
#: The registration chain candidate 14 carried, threaded through for provenance.
CANDIDATE13_REGISTRATION = c14.CANDIDATE13_REGISTRATION
CANDIDATE12_REGISTRATION = c14.CANDIDATE12_REGISTRATION
CANDIDATE11_REGISTRATION = c14.CANDIDATE11_REGISTRATION
CANDIDATE10_REGISTRATION = c14.CANDIDATE10_REGISTRATION
CANDIDATE9_REGISTRATION = c14.CANDIDATE9_REGISTRATION
CANDIDATE8_REGISTRATION = c14.CANDIDATE8_REGISTRATION
CANDIDATE6_REGISTRATION = c14.CANDIDATE6_REGISTRATION
CANDIDATE5_REGISTRATION = c14.CANDIDATE5_REGISTRATION
CANDIDATE1_REGISTRATION = c14.CANDIDATE1_REGISTRATION
#: Candidate 14's grading (comment 4928232089) isolated the trend-basis
#: mismatch: the composition is scored against the untrended PSID panel but
#: carries the NCHS mortality-decline trend, undershooting by construction.
CANDIDATE14_GRADING_DIAGNOSTIC = (
    "issue #42 comment 4928232089 (candidate 14 grading: FAIL 2/5; the "
    "exposure-preserving 75-84/85+ split recovered elderly incidence "
    "(0.929 -> 0.952 sim/ref) but could not lift the stock (0.841 -> 0.838), "
    "seeds 0 and 3 clipping share_widowed.75+|female. Post-run measurement "
    "isolated the residual: the NCHS trend multiplier exp(beta_sex * "
    "(year - 1995)) scales panel-era spouse-death inflow to ~0.92-0.95 of the "
    "train-empirical rate over the exposure-weighted years -- matching "
    "reality's mortality decline, but the gate's reference is the UNTRENDED "
    "PSID panel measurement, so the composition is scored against a basis it "
    "systematically undershoots by construction)"
)

# --- Amended-estimator draw stream (inherited from candidate 14, unchanged). -
#: The amended 20-draw stream base: draw k uses default_rng(5200 + k), the
#: committed forensics convention (gates.yaml gate_2, amendment 1).
DRAW_SEED_BASE = c14.DRAW_SEED_BASE  # 5200
N_DRAWS = c14.N_DRAWS  # 20

# --------------------------------------------------------------------------
# Frozen dials + pure helpers, reused (byte-identical; import-bound via c14).
# --------------------------------------------------------------------------
GATE_SEEDS = c14.GATE_SEEDS
SIM_SEED_BASE = c14.SIM_SEED_BASE  # 4200 (single-draw stream; provenance only)
EXACT_ATOL = c14.EXACT_ATOL
Components = c14.Components

YSD_BANDS = c14.YSD_BANDS
YSD_LOWERS = c14.YSD_LOWERS
_bands_vec = c14._bands_vec

#: Candidate 6's period-trend anchor year (1995.0). Candidate 15's _widow_probs
#: no longer uses it (the trend is removed); retained as a module global for the
#: trend-multiplier diagnostic that documents what the removal cancels.
TREND_ANCHOR_YEAR = c14.TREND_ANCHOR_YEAR

# Candidate 8's simulation helpers (import-bound via candidate 14, unchanged).
_STATE = c14._STATE
_STATE_ABSORB = c14._STATE_ABSORB
_ASFR_LO = c14._ASFR_LO
_ASFR_HI = c14._ASFR_HI
_assemble_sim_panel = c14._assemble_sim_panel
_divorce_probs = c14._divorce_probs
_fertility_probs_single = c14._fertility_probs_single

# Candidate 9's observed residual + reconciliation (delta 1 of candidate 9),
# inherited via candidate 14, reused byte-for-byte.
observed_residual_counts = c14.observed_residual_counts
_delta1_reconciliation = c14._delta1_reconciliation

#: Candidate 10's simulation-lookup container (band-agnostic dataclass; its
#: ``mort_arr`` is shaped by ``_build_sim_lookups`` from ``len(WIDOW_BANDS)`` and
#: still carries ``beta_arr`` unchanged -- candidate 15 simply does not apply it.
_SimLookupsC10 = c14._SimLookupsC10

# --- The 5-band remarriage table (candidate 11's delta; UNCHANGED here). -----
REM_AGE_BANDS = c14.REM_AGE_BANDS
REM_AGE_LOWERS = c14.REM_AGE_LOWERS
_REM_AGE_LABEL = c14._REM_AGE_LABEL

# --- The vestigial spousal-age-gap machinery (candidate 12's delta 2, proven
# --- INERT; left UNTOUCHED per byte-minimality; import-bound unchanged). ------
GAP_AGE_BANDS = c14.GAP_AGE_BANDS
GAP_AGE_LOWERS = c14.GAP_AGE_LOWERS
_GAP_AGE_LABEL = c14._GAP_AGE_LABEL
FALLBACK_MIN_WEIGHTED_COUPLES = c14.FALLBACK_MIN_WEIGHTED_COUPLES
spousal_gap_distribution_by_band = c14.spousal_gap_distribution_by_band
c11_pooled_gap = c14.c11_pooled_gap
_fallback_group = c14._fallback_group
_gap_band_arrays = c14._gap_band_arrays
_draw_banded_gaps = c14._draw_banded_gaps

# --- Candidate 12's entry-widowed observed initial state (delta 1; UNCHANGED,
# --- import-bound so the reused ``simulate_holdout`` injects it unchanged). ---
observed_support = c14.observed_support
entry_widowed_carried_cells = c14.entry_widowed_carried_cells
_entry_widowed_seed_counts = c14._entry_widowed_seed_counts
_entry_widowed_reconciliation = c14._entry_widowed_reconciliation
_inject_entry_widowed = c14._inject_entry_widowed
_widowed_share_by_age = c14._widowed_share_by_age

# Fresh-run artifact-schema blocks are band-independent (they operate on the
# scored per-seed dicts); import-bound via candidate 14 (identical N_DRAWS /
# DRAW_SEED_BASE), so the [20, 46, 5] cube, the undefined-draw rule and the
# report-only dispersion are candidate 10's exact assembly.
_per_draw_per_cell_rates_block = c14._per_draw_per_cell_rates_block
_undefined_draw_block = c14._undefined_draw_block
_per_draw_dispersion_block = c14._per_draw_dispersion_block

# --- The seven-band surviving-spouse widowhood table (candidate 14's delta;
# --- UNCHANGED here -- the LEVEL is byte-identical, only its APPLICATION at
# --- simulation time changes). -----------------------------------------------
WIDOW_BANDS = c14.WIDOW_BANDS
WIDOW_LOWERS = c14.WIDOW_LOWERS
_WIDOW_BAND_LABEL = c14._WIDOW_BAND_LABEL

# --------------------------------------------------------------------------
# THE ONE DELTA vs candidate 14 (registration comment 4928232089)
# --------------------------------------------------------------------------
#: Candidate 5's committed NCHS per-sex log-linear period slopes. Candidate 15
#: RETAINS these committed values (they remain in the fitted components' meta
#: and are documented in the artifact for deployment-time projection) but does
#: NOT apply them at gate time -- the one delta.
NCHS_BETA_BY_SEX_COMMITTED = dict(c6._committed_beta_v5())

#: The single named delta (registration comment 4928232089).
DELTA_VS_CANDIDATE14 = (
    "EXACTLY ONE delta vs candidate 14 (comment 4927236029, merged #105): the "
    "NCHS period-trend multiplier exp(beta_sex * (year - 1995)) is REMOVED "
    "from the surviving-spouse widowhood hazard. Candidate 14 applied "
    "rate = widow_level(ego_band, ego_sex) * exp(beta_sex * (year - 1995)) at "
    "simulation time; candidate 15 applies rate = widow_level(ego_band, "
    "ego_sex) -- the source-aligned train-empirical seven-band x sex level, "
    "period-pooled, with the SAME smoothing convention "
    "(transitions._hazard_by_band weighted hazard num_wt/den_wt, no add-one). "
    "Registered from candidate 14's grading: the composed widowhood hazard is "
    "scored against the UNTRENDED PSID panel reference but carried the NCHS "
    "mortality-decline trend, which scaled panel-era spouse-death inflow to "
    "~0.92-0.95 of the train-empirical rate over the exposure-weighted years, "
    "undershooting the gate's own measurement basis by construction. Removing "
    "the trend restores the untrended level and lifts widowhood inflow ~7% "
    "across all bands. The committed NCHS beta values (female "
    f"{NCHS_BETA_BY_SEX_COMMITTED['female']!r}, male "
    f"{NCHS_BETA_BY_SEX_COMMITTED['male']!r}) stay documented in the artifact "
    "for deployment-time use (projection under Trustees-style assumptions). "
    "Everything else -- the seven-band widowhood LEVEL table itself (candidate "
    "14's delta, refit bit-identically), the entry-widowed observed initial "
    "state (candidate 12's delta 1), the 5-band remarriage current-age table, "
    "the observed undatable-marriage lifetime-count initial state (candidate "
    "9's delta 1), the single-year triangular-kernel fertility, the RNG, the "
    "K=20 mean-of-draws protocol, the fresh-run artifact schema, and the "
    "vestigial spousal-age-gap machinery (candidate 12's delta 2, left "
    "untouched per byte-minimality) -- is byte-identical to candidate 14"
)

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate15_run1_cache.json"
)

#: The marriage-count cells -- NOT insulated here (unlike candidate 14): the
#: ~7% young-widow inflow rise feeds the widowed-origin remarriage exposure, so
#: the counts are a gated RISK. The registered modal failure is the female
#: count cell re-clipping.
COUNT_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
)
#: The gated remarriage cells the delta reshapes via the widowed trajectory
#: (more widows at every age enter the widowed-origin remarriage exposure).
REMARRIAGE_GATED_CELLS = (
    "remarriage.after_divorce",
    "remarriage.ysd0-4",
    "remarriage.ysd5-9",
    "remarriage.ysd10+",
)
#: The elderly-widow-stock cell the delta is designed to lift (candidate 14's
#: two failing seeds, 0 and 3, are on this cell).
MODAL_CELL = "share_widowed.75+|female"
#: The registered modal failure (comment 4928232089): the female marriage-count
#: cell re-clipping from the young-widow-inflow rise (primary); the secondary is
#: seed 3's 75+ stock needing more than the exposure-weighted lift delivers.
REGISTERED_MODAL_CELLS = ("mean_lifetime_marriages|female",)
#: The registered secondary failure (seed 3's stock).
REGISTERED_SECONDARY_CELLS = ("share_widowed.75+|female",)
#: The gated widowhood-incidence cells; the trend removal lifts every band's
#: inflow (~7%), so all move toward reference from candidate 14's ~0.95.
WIDOWHOOD_INCIDENCE_CELLS = (
    "widowhood.45-64|female",
    "widowhood.65-74|female",
    "widowhood.75+|female",
    "widowhood.45+|male",
)
#: The 75+ cells the delta most directly targets: the widowhood incidence (lifts
#: further toward reference) and the widowed stock (must lift where candidate 14
#: clipped seeds 0 and 3).
ELDERLY_75PLUS_CELLS = (
    "widowhood.75+|female",
    "share_widowed.75+|female",
)
#: The cells the delta most directly touches: the elderly incidence/stock, both
#: marriage counts (the gated risk), the widowhood-incidence bands and the
#: remarriage flows.
TARGETED_CELLS = (
    ELDERLY_75PLUS_CELLS
    + COUNT_CELLS
    + REMARRIAGE_GATED_CELLS
    + (
        "widowhood.45-64|female",
        "widowhood.65-74|female",
        "widowhood.45+|male",
    )
)


# ==========================================================================
# THE DELTA: the surviving-spouse widowhood incidence WITHOUT the NCHS trend.
# Re-implemented (diverged from candidate 14's ``_widow_probs``): the SAME
# level lookup and ``_bands_vec`` clip, but the returned rate is the level
# ONLY -- the ``exp(beta_sex * (year - 1995))`` factor candidate 14 applied is
# removed. Signature byte-identical to candidate 14 so the reused
# ``simulate_holdout`` call site is unchanged; ``year`` / ``beta_arr`` /
# ``spouse_age`` / ``spouse_is_male`` are accepted and IGNORED (documented), so
# no year multiplier enters the widowhood lookup.
# ==========================================================================
def _widow_probs(
    ego_age: np.ndarray,
    ego_is_male: np.ndarray,
    spouse_age: np.ndarray,
    spouse_is_male: np.ndarray,
    year: int,
    mort_arr: np.ndarray,
    beta_arr: np.ndarray,
) -> np.ndarray:
    """Surviving-spouse widowhood incidence, NCHS period trend REMOVED (DELTA).

    ``rate = widow_level(ego_band, ego_sex)`` -- the train marriage-history
    widowhood hazard keyed by the SURVIVING spouse's (age band, sex), looked up
    by the married ego's OWN ``(ego_age, ego_is_male)`` with the same
    ``_bands_vec`` clip candidate 1 used (ages below the youngest band clip into
    it). Candidate 14 multiplied this level by ``exp(beta_arr[sex] * (year -
    1995))``; candidate 15 does NOT -- the source-aligned train-empirical level
    is period-pooled, matching the gate reference's own (untrended) measurement
    basis. ``year`` and ``beta_arr`` are accepted (the signature and the reused
    ``simulate_holdout`` call site stay byte-identical to candidate 14) but do
    NOT enter the returned rate, so the widowhood lookup is year-invariant.
    ``spouse_age`` / ``spouse_is_male`` are candidate 5's spousal-gap-draw
    arguments, retained so the spousal-gap draw stays byte-identical; they no
    longer enter the level (the surviving-spouse hazard already integrates the
    empirical spouse-age distribution). The committed NCHS betas remain in the
    components' meta for deployment-time use.
    """
    bands = _bands_vec(
        np.rint(ego_age).astype(np.int64), WIDOW_LOWERS, len(WIDOW_BANDS)
    )
    sidx = ego_is_male.astype(np.int64)
    level = mort_arr[bands, sidx]
    return level


# ==========================================================================
# Fitted components (candidate 14's fit, UNCHANGED; the delta is at simulation
# time). ``fit_components`` is a thin wrapper over :func:`candidate14.
# fit_components` so the fit -- the seven-band surviving-spouse widowhood LEVEL
# and the committed NCHS betas -- is byte-identical; the wrapper only records
# the trend-removal provenance (and keeps the committed betas explicit for
# deployment-time use).
# ==========================================================================
def fit_components(
    panel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    mh_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    order_map: pd.DataFrame,
    train_ids: set[int],
) -> Components:
    """Candidate 14's components, UNCHANGED (the c15 delta is at simulation
    time, not in the fit).

    Calls :func:`candidate14.fit_components` directly, so first marriage,
    divorce, the 5-band remarriage table, the single-year triangular-kernel
    fertility, the entry-widowed carried cells, the observed marriage-count
    initial state, the spousal-gap distribution, the committed NCHS betas and
    the seven-band surviving-spouse widowhood LEVEL are byte-identical to
    candidate 14. The betas remain in ``base.meta['mortality_beta_by_sex']``
    for deployment-time use, but candidate 15's re-implemented
    :func:`_widow_probs` does NOT apply them (the trend multiplier is removed
    from the gate-time widowhood hazard). The wrapper records the disposition.
    """
    base = c14.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )
    committed_beta = {
        k: float(v) for k, v in base.meta["mortality_beta_by_sex"].items()
    }
    base.meta["nchs_trend_applied_in_gate"] = False
    base.meta["nchs_trend_beta_by_sex_committed"] = committed_beta
    base.meta["nchs_trend_disposition"] = (
        "candidate 5's committed NCHS per-sex log-linear period slopes "
        "exp(beta_sex * (year - 1995)) remain fitted and stored, but are NOT "
        "applied to the widowhood hazard at gate time (the gate reference is "
        "the untrended PSID panel measurement). Retained for deployment-time "
        "projection under Trustees-style mortality assumptions"
    )
    base.meta["mortality_level_representation_candidate15"] = (
        "surviving-spouse widowhood-incidence hazard, keyed (seven-band "
        "WIDOW_BANDS, surviving-spouse sex); consumed by the married ego's OWN "
        "(age band, sex) WITHOUT any period-trend multiplier (candidate 14's "
        "exp(beta_sex * (year - 1995)) factor removed -- the one delta)"
    )
    base.meta["delta_vs_candidate14"] = DELTA_VS_CANDIDATE14
    return base


# ==========================================================================
# Provable byte-identity: reuse candidate 14's EXACT code objects for the
# compute chain EXCEPT ``_widow_probs`` (the delta), rebound so their global
# lookups resolve against THIS module (candidate 15). The bytecode is candidate
# 14's; the only observable change is that ``_widow_probs`` is candidate 15's
# untrended re-implementation -- the one delta -- which the reused
# ``simulate_holdout`` calls by global name.
# ==========================================================================
def _rebind(fn: types.FunctionType) -> types.FunctionType:
    """Return a function sharing ``fn``'s code object but this module's globals.

    Candidate 14's compute chain (minus ``_widow_probs``) is unchanged by the
    trend removal. Reusing its code objects verbatim -- only redirecting global
    name resolution to candidate 15's module -- makes ``everything else
    byte-identical`` provable at the bytecode level
    (``candidate15.f.__code__ is candidate14.f.__code__``) while the reused
    ``simulate_holdout`` reads candidate 15's untrended ``_widow_probs``.
    """
    return types.FunctionType(
        fn.__code__,
        globals(),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


#: Candidate 14's (== candidate 10's) exact lookup builder; ``mort_arr`` and
#: ``beta_arr`` are built unchanged (candidate 15 simply never applies
#: ``beta_arr``). Reused, rebound.
_build_sim_lookups = _rebind(c14._build_sim_lookups)
#: Candidate 14's EXACT vectorised annual simulation (both of candidate 12's
#: deltas -- the entry-widowed injection and the age-band gap draw -- inline and
#: byte-for-byte). It calls ``_widow_probs`` (candidate 15's untrended
#: re-implementation, above) by global name; every other statement is candidate
#: 14's.
simulate_holdout = _rebind(c14.simulate_holdout)
#: Candidate 14's (== candidate 11's == candidate 10's) exact single-draw moment
#: builder and per-seed mean-over-K=20 scorer. Rebound to THIS module's globals,
#: they call candidate 15's ``simulate_holdout`` / ``fit_components``.
_draw_moments = _rebind(c14._draw_moments)
score_seed = _rebind(c14.score_seed)
#: The 5-band remarriage fit / lookup (candidate 14's == candidate 11's code),
#: rebound -- UNCHANGED by candidate 15's delta (kept for the attestation).
fit_remarriage_age_banded = _rebind(c14.fit_remarriage_age_banded)
_remarriage_probs_age_banded = _rebind(c14._remarriage_probs_age_banded)

#: The reused-code-object contract (must share candidate 14's bytecode).
#: ``_widow_probs`` is NO LONGER here (it moved to the diverged set -- the one
#: delta is a COMPUTE delta, unlike candidate 14's data delta).
REUSED_CODE_OBJECT_NAMES = (
    "_build_sim_lookups",
    "simulate_holdout",
    "_draw_moments",
    "score_seed",
    "fit_remarriage_age_banded",
    "_remarriage_probs_age_banded",
)
#: The RE-IMPLEMENTED functions: ``_widow_probs`` (the delta: trend removed) and
#: ``fit_components`` (the provenance wrapper). Neither may share candidate 14's
#: code object.
DIVERGED_CODE_OBJECT_NAMES = ("_widow_probs", "fit_components")


# ==========================================================================
# Diagnostic: the exposure-weighted NCHS trend multiplier the removal cancels
# ==========================================================================
def _trend_multiplier_diagnostic(
    panel: transitions.MaritalPanel,
    components: Components,
) -> dict[str, Any]:
    """The exposure-weighted trend multiplier candidate 15's removal cancels.

    Candidate 14 scaled the widowhood hazard by ``exp(beta_sex * (year -
    1995))``; candidate 15 removes it. This reports that multiplier
    exposure-weighted over four slices of the panel (committed betas, so
    split-independent): all-ages vs 75+, each weighted by the married
    person-year RISK SET and by the widowhood-EVENT (spouse-death inflow) set,
    with the weighted-mean year of each. The headline
    ``pooled_exposure_weighted_multiplier`` is the all-ages married-person-year
    weighting (the simulation-time application basis). The registration
    (comment 4928232089) hypothesised ~0.92-0.95 and a ~7% inflow lift ACROSS
    ALL BANDS; the slices reconcile the actual, band-dependent picture (the
    ~0.92-0.95 is the ELDERLY 75+ exposure, at late panel years past the 1995
    anchor where the multiplier < 1; the all-ages aggregate is > 1 because
    young married exposure sits at early panel years).
    """
    beta = {
        k: float(v)
        for k, v in components.meta["mortality_beta_by_sex"].items()
    }
    anchor = float(TREND_ANCHOR_YEAR)

    def _wstats(df: pd.DataFrame) -> dict[str, Any]:
        per: dict[str, Any] = {}
        tot_num = 0.0
        tot_den = 0.0
        tot_numy = 0.0
        for sex in ("female", "male"):
            sub = df[df["sex"] == sex]
            w = sub["weight"].to_numpy(dtype=np.float64)
            yr = sub["year"].to_numpy(dtype=np.float64)
            mult = np.exp(beta[sex] * (yr - anchor))
            den = float(w.sum())
            num = float((w * mult).sum())
            numy = float((w * yr).sum())
            wmean = num / den if den > 0 else None
            per[sex] = {
                "beta": beta[sex],
                "multiplier": wmean,
                "implied_inflow_uplift": (
                    (1.0 / wmean - 1.0) if wmean else None
                ),
                "weighted_mean_year": (numy / den if den > 0 else None),
                "weighted_exposure": den,
                "n_rows": int(len(sub)),
            }
            tot_num += num
            tot_den += den
            tot_numy += numy
        pooled = tot_num / tot_den if tot_den > 0 else None
        per["pooled"] = {
            "multiplier": pooled,
            "implied_inflow_uplift": (
                (1.0 / pooled - 1.0) if pooled else None
            ),
            "weighted_mean_year": (
                tot_numy / tot_den if tot_den > 0 else None
            ),
            "weighted_exposure": tot_den,
        }
        return per

    py = panel.person_years
    ev = panel.events
    married = py[py["marital_state"] == "married"]
    widow = ev[ev["transition"] == "widowhood"]
    slices = {
        "all_ages_married_person_years": _wstats(married),
        "all_ages_widowhood_events": _wstats(widow),
        "elderly_75plus_married_person_years": _wstats(
            married[married["age"] >= 75]
        ),
        "elderly_75plus_widowhood_events": _wstats(widow[widow["age"] >= 75]),
    }
    headline = slices["all_ages_married_person_years"]["pooled"]["multiplier"]
    elderly_py = slices["elderly_75plus_married_person_years"]["pooled"][
        "multiplier"
    ]
    elderly_evt = slices["elderly_75plus_widowhood_events"]["pooled"][
        "multiplier"
    ]
    return {
        "note": (
            "the NCHS period-trend multiplier exp(beta_sex * (year - 1995)) "
            "candidate 14 applied to the surviving-spouse widowhood hazard and "
            "candidate 15 REMOVES, exposure-weighted over four panel slices. "
            "The headline pooled_exposure_weighted_multiplier is the all-ages "
            "married-person-year weighting (the simulation-time application "
            "basis)."
        ),
        "anchor_year": anchor,
        "beta_by_sex_committed": beta,
        "beta_by_sex_frozen_expected": dict(NCHS_BETA_BY_SEX_COMMITTED),
        "slices": slices,
        "pooled_exposure_weighted_multiplier": headline,
        "pooled_implied_inflow_uplift": (
            (1.0 / headline - 1.0) if headline else None
        ),
        "registration_reconciliation": (
            "the registration (comment 4928232089) hypothesised the trend "
            "scales spouse-death INFLOW to ~0.92-0.95 and lifts inflow ~7% "
            "ACROSS ALL BANDS. Measured: the ~0.92-0.95 matches the ELDERLY "
            f"75+ exposure (75+ married-PY multiplier {elderly_py:.4f}, 75+ "
            f"widowhood-event multiplier {elderly_evt:.4f}), whose exposure "
            "concentrates at late panel years (weighted-mean year ~2007, past "
            "the 1995 anchor, where exp(beta*(year-1995)) < 1). The ALL-AGES "
            f"married-PY aggregate is {headline:.4f} (> 1): young married "
            "exposure dominates and sits at early panel years (weighted-mean "
            "~1994, before the anchor, multiplier > 1). So removing the trend "
            "LIFTS elderly 75+ inflow (~1/0.90 ~ +11%, driving the 75+ "
            "incidence 0.952 -> 1.060 sim/ref) but slightly LOWERS young "
            "inflow -- the registered female-count re-clip did NOT materialise "
            "(the count held 5/5 with margin, because young inflow fell rather "
            "than rose). The 'across all bands' premise was the misprediction"
        ),
    }


# ==========================================================================
# Diagnostics: 75+ incidence & stock sim/ref vs candidate 14 (the designed lift)
# ==========================================================================
def _elderly_75plus_diagnostic(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Candidate 15 vs candidate 14 on the 75+ incidence and stock cells.

    The removal's designed effect: lift the 75+ widowhood INCIDENCE further
    toward reference (candidate 14 sat at ~0.952 sim/ref) and thereby lift the
    75+ widowed STOCK (candidate 14 sat at ~0.838 of reference, clipping seeds
    0 and 3). Both quantities are read from the scored gated cells (side A
    holdout, K=20 draw-mean): sim/ref = rbar / rate_a. Candidate 14's committed
    values come from ``runs/gate2_hazard_v14.json``.
    """
    c14art = (
        json.loads(CANDIDATE14_ARTIFACT.read_text())
        if CANDIDATE14_ARTIFACT.exists()
        else None
    )
    by14 = {s["seed"]: s for s in c14art["per_seed"]} if c14art else {}

    out: dict[str, Any] = {
        "note": (
            "75+ widowhood incidence (widowhood.75+|female) and 75+ widowed "
            "stock (share_widowed.75+|female), sim/ref = rbar / rate_a on the "
            "side-A holdout at the K=20 draw-mean. Removing the NCHS trend "
            "lifts every band's inflow ~7%, so the incidence rises further "
            "toward reference (candidate 14 ~0.952) and the stock lifts "
            "(candidate 14 ~0.838, clipping seeds 0 and 3). Candidate 14 "
            "values committed in runs/gate2_hazard_v14.json"
        ),
        "candidate14_incidence_sim_over_ref": 0.9519,
        "cells": {},
    }
    for cell in ELDERLY_75PLUS_CELLS:
        rows: list[dict[str, Any]] = []
        c15_ratios: list[float] = []
        c14_ratios: list[float] = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            rbar = rec["rbar"]
            rate_a = rec["rate_a"]
            c15_ratio = rbar / rate_a if rate_a > 0 else None
            c14rec = by14[s["seed"]]["gated_cells"][cell] if by14 else None
            c14_ratio = (
                c14rec["rbar"] / c14rec["rate_a"]
                if c14rec and c14rec["rate_a"] > 0
                else None
            )
            if c15_ratio is not None:
                c15_ratios.append(c15_ratio)
            if c14_ratio is not None:
                c14_ratios.append(c14_ratio)
            rows.append(
                {
                    "seed": s["seed"],
                    "c15_rbar": rbar,
                    "rate_a": rate_a,
                    "c15_sim_over_ref": c15_ratio,
                    "c14_rbar": c14rec["rbar"] if c14rec else None,
                    "c14_rate_a": c14rec["rate_a"] if c14rec else None,
                    "c14_sim_over_ref": c14_ratio,
                    "c15_score_abs_ln": rec["score"],
                    "c14_score_abs_ln": c14rec["score"] if c14rec else None,
                    "tolerance": rec["tolerance"],
                    "c15_pass": rec["pass"],
                    "c14_pass": c14rec["pass"] if c14rec else None,
                }
            )
        out["cells"][cell] = {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "c15_sim_over_ref_mean": (
                float(np.mean(c15_ratios)) if c15_ratios else None
            ),
            "c14_sim_over_ref_mean": (
                float(np.mean(c14_ratios)) if c14_ratios else None
            ),
            "c15_n_seeds_pass": sum(1 for r in rows if r["c15_pass"]),
            "c14_n_seeds_pass": (
                sum(1 for r in rows if r["c14_pass"]) if by14 else None
            ),
        }
    inc = out["cells"]["widowhood.75+|female"]
    stock = out["cells"]["share_widowed.75+|female"]
    out["summary"] = {
        "incidence_sim_over_ref": {
            "c14_mean": inc["c14_sim_over_ref_mean"],
            "c15_mean": inc["c15_sim_over_ref_mean"],
            "moved_toward_reference": (
                inc["c15_sim_over_ref_mean"] is not None
                and inc["c14_sim_over_ref_mean"] is not None
                and abs(inc["c15_sim_over_ref_mean"] - 1.0)
                < abs(inc["c14_sim_over_ref_mean"] - 1.0)
            ),
        },
        "stock_sim_over_ref": {
            "c14_mean": stock["c14_sim_over_ref_mean"],
            "c15_mean": stock["c15_sim_over_ref_mean"],
            "lifted_toward_reference": (
                stock["c15_sim_over_ref_mean"] is not None
                and stock["c14_sim_over_ref_mean"] is not None
                and stock["c15_sim_over_ref_mean"]
                > stock["c14_sim_over_ref_mean"]
            ),
        },
    }
    return out


# ==========================================================================
# Count-cell margins (the gated risk) + candidate-14 comparison + decider
# ==========================================================================
def _count_cell_margin_diagnostic(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """The count-cell margins the young-widow-inflow rise threatens.

    Candidate 14 held both marriage-count cells at 5/5 (insulated -- its split
    touched only the oldest band). Candidate 15 removes the trend at EVERY
    band, so the ~7% young-widow inflow rise feeds the widowed-origin
    remarriage exposure and the counts are a gated RISK. Reports candidate 15's
    signed ln-tilt and the margin (tolerance - score, positive = headroom) per
    seed against candidate 14's committed count scores, and whether each count
    still holds >= 4/5.
    """
    c14art = (
        json.loads(CANDIDATE14_ARTIFACT.read_text())
        if CANDIDATE14_ARTIFACT.exists()
        else None
    )
    by14 = {s["seed"]: s for s in c14art["per_seed"]} if c14art else {}

    out: dict[str, Any] = {
        "design_note": (
            "registration comment 4928232089: the trend removal lifts "
            "young-widow inflow ~7% at every band, feeding the widowed-origin "
            "remarriage exposure, so the marriage counts are NOT insulated "
            "(unlike candidate 14). The registered modal failure is the female "
            "count cell re-clipping. Margin = tolerance - score (positive = "
            "headroom); movement reported against candidate 14's committed "
            "count scores."
        ),
        "cells": {},
    }
    for cell in COUNT_CELLS:
        rows = []
        signed = []
        margins = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            rbar = rec["rbar"]
            rate_a = rec["rate_a"]
            tilt = (
                float(math.log(rbar / rate_a))
                if rbar > 0 and rate_a > 0
                else None
            )
            margin = float(rec["tolerance"] - rec["score"])
            if tilt is not None:
                signed.append(tilt)
            margins.append(margin)
            c14rec = by14[s["seed"]]["gated_cells"][cell] if by14 else None
            rows.append(
                {
                    "seed": s["seed"],
                    "rbar": rbar,
                    "rate_a": rate_a,
                    "signed_ln_tilt": tilt,
                    "score_abs_ln": rec["score"],
                    "tolerance": rec["tolerance"],
                    "margin": margin,
                    "pass": rec["pass"],
                    "candidate14_score": (c14rec["score"] if c14rec else None),
                    "candidate14_pass": (c14rec["pass"] if c14rec else None),
                    "delta_score_vs_c14": (
                        float(rec["score"] - c14rec["score"])
                        if c14rec
                        else None
                    ),
                }
            )
        n_pass = sum(r["pass"] for r in rows)
        c14_n_pass = (
            sum(1 for r in rows if r["candidate14_pass"]) if by14 else None
        )
        out["cells"][cell] = {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "mean_signed_ln_tilt": (
                float(np.mean(signed)) if signed else None
            ),
            "min_margin": float(np.min(margins)) if margins else None,
            "mean_abs_ln_score": float(
                np.mean([r["score_abs_ln"] for r in rows])
            ),
            "n_seeds_pass": n_pass,
            "candidate14_n_seeds_pass": c14_n_pass,
            "held_vs_c14": (
                bool(n_pass >= 4) if c14_n_pass is not None else None
            ),
        }
    both_hold = all(out["cells"][c]["n_seeds_pass"] >= 4 for c in COUNT_CELLS)
    out["count_cells_hold"] = bool(both_hold)
    out["summary"] = (
        "both marriage-count cells hold >= 4/5 despite the young-inflow rise"
        if both_hold
        else "at least one marriage-count cell clipped below 4/5"
    )
    return out


def _incidence_headroom_diagnostic(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """The gated widowhood-incidence cells' headroom (tolerance - score).

    The ~7% inflow lift moves every incidence cell; the gated cells have
    ln(1.5)-scale tolerances, so the question is whether the lift stays inside
    the band. Reports per-seed score, tolerance and margin (positive =
    headroom) for the four gated widowhood-incidence cells, and the minimum
    margin per cell.
    """
    out: dict[str, Any] = {
        "note": (
            "gated widowhood-incidence cells: score |ln(rbar / rate_a)|, "
            "tolerance and margin (tolerance - score; positive = headroom). "
            "The trend removal lifts inflow ~7% at every band; the cells hold "
            "iff the lift stays inside the ln(1.5)-scale tolerance"
        ),
        "cells": {},
    }
    for cell in WIDOWHOOD_INCIDENCE_CELLS:
        rows = []
        margins = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            margin = float(rec["tolerance"] - rec["score"])
            margins.append(margin)
            rows.append(
                {
                    "seed": s["seed"],
                    "score": rec["score"],
                    "tolerance": rec["tolerance"],
                    "margin": margin,
                    "pass": rec["pass"],
                }
            )
        out["cells"][cell] = {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "min_margin": float(np.min(margins)) if margins else None,
            "n_seeds_pass": sum(r["pass"] for r in rows),
        }
    return out


def _candidate14_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Movement of the delta targets vs candidate 14 + the trend removed.

    Compares candidate 15's per-seed rbar-scores for the targeted cells against
    candidate 14's committed scores, and confirms the fitted widowhood LEVEL is
    bit-identical to candidate 14 (the delta is the APPLICATION, not the fit):
    ``widowhood_level_identical_to_c14`` tabulates the seed-0 fitted level cells
    (candidate 14 six-band-split table) against candidate 15's (equal by
    construction -- same fit), so the only moving part is the removed trend.
    """
    c14art = (
        json.loads(CANDIDATE14_ARTIFACT.read_text())
        if CANDIDATE14_ARTIFACT.exists()
        else None
    )
    by14 = {s["seed"]: s for s in c14art["per_seed"]} if c14art else {}

    def move(cell: str) -> dict[str, Any]:
        rows = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            c14rec = by14[s["seed"]]["gated_cells"][cell] if by14 else None
            rows.append(
                {
                    "seed": s["seed"],
                    "tolerance": rec["tolerance"],
                    "c14_score": c14rec["score"] if c14rec else None,
                    "c15_score": rec["score"],
                    "c14_rbar": c14rec["rbar"] if c14rec else None,
                    "c15_rbar": rec["rbar"],
                    "rate_a": rec["rate_a"],
                    "c14_pass": c14rec["pass"] if c14rec else None,
                    "c15_pass": rec["pass"],
                }
            )
        c14_np = sum(1 for r in rows if r["c14_pass"]) if by14 else None
        c15_np = sum(1 for r in rows if r["c15_pass"])
        return {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "c14_n_seeds_pass": c14_np,
            "c15_n_seeds_pass": c15_np,
            "improved": (
                bool(c15_np > c14_np) if c14_np is not None else None
            ),
        }

    # The fitted widowhood LEVEL is unchanged (same fit); confirm bit-identity
    # of the seed-0 fitted level cells vs candidate 14.
    c15_level = per_seed[0]["component_meta"]["mortality_level_new_widowhood"]
    c14_level = (
        c14art["per_seed"][0]["component_meta"][
            "mortality_level_new_widowhood"
        ]
        if c14art
        else {}
    )
    level_cells = {}
    for key in sorted(c15_level):
        c15r = float(c15_level[key])
        c14r = float(c14_level[key]) if key in c14_level else None
        level_cells[key] = {
            "candidate14_rate": c14r,
            "candidate15_rate": c15r,
            "bit_identical": (c14r is not None and abs(c14r - c15r) <= 1e-12),
        }

    return {
        "note": (
            "candidate 15 = candidate 14 (comment 4927236029, #105) with "
            "exactly one delta (the NCHS trend multiplier removed from the "
            "surviving-spouse widowhood hazard). Scores compared cell-by-cell "
            "against candidate 14's committed run (runs/gate2_hazard_v14.json)."
        ),
        "modal_cell": {MODAL_CELL: move(MODAL_CELL)},
        "count_cells": {c: move(c) for c in COUNT_CELLS},
        "remarriage_gated_cells": {c: move(c) for c in REMARRIAGE_GATED_CELLS},
        "widowhood_incidence_cells": {
            c: move(c) for c in WIDOWHOOD_INCIDENCE_CELLS
        },
        "widowhood_level_identical_to_c14": {
            "note": (
                "the fitted seven-band surviving-spouse widowhood LEVEL is "
                "byte-identical to candidate 14 (same fit); the one delta is "
                "that candidate 15 does not apply the NCHS trend to it"
            ),
            "all_bit_identical": all(
                v["bit_identical"] for v in level_cells.values()
            ),
            "cells": level_cells,
        },
    }


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal (female count re-clip), secondary (seed-3 stock),
    targets and decider."""
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

    gate_pass = verdict["gate_2_pass"]
    n_pass_actual = verdict["n_seeds_pass"]
    n_pass_no_modal = seeds_pass_if_forgiven(set(REGISTERED_MODAL_CELLS))
    n_pass_no_secondary = seeds_pass_if_forgiven(
        set(REGISTERED_SECONDARY_CELLS)
    )
    n_pass_no_targeted = seeds_pass_if_forgiven(set(TARGETED_CELLS))
    count_cell = "mean_lifetime_marriages|female"
    stock_cell = "share_widowed.75+|female"
    modal_failed_seeds = sorted(fails_by_cell.get(count_cell, []))
    secondary_failed_seeds = sorted(fails_by_cell.get(stock_cell, []))
    # The registered modal materialises if the female count cell re-clips on at
    # least one seed.
    modal_materialized = len(modal_failed_seeds) >= 1
    distinct_fail_cells = {
        f["cell"] for f in verdict["all_failing_gated_cells"]
    }

    if gate_pass:
        decider = "none (gate passed)"
    elif n_pass_no_modal >= 4:
        decider = (
            "the registered modal cell (the female marriage count; forgiving "
            "it flips >= 4 seeds to pass)"
        )
    elif n_pass_no_secondary >= 4:
        decider = (
            "the registered secondary cell (the 75+ widowed stock; forgiving "
            "it flips >= 4 seeds to pass)"
        )
    elif n_pass_no_targeted >= 4:
        decider = (
            "the delta-targeted cells (forgiving the elderly-stock/incidence/"
            "count/remarriage targets flips >= 4 seeds to pass)"
        )
    else:
        decider = (
            "broader than the registered modal + secondary + delta-targeted "
            "cells (other gated cells also hold the gate below 4 passing seeds)"
        )

    return {
        "registered_modal": (
            "the female marriage-count cell (mean_lifetime_marriages|female) "
            "re-clipping from the ~7% young-widow-inflow rise; secondary: seed "
            "3's 75+ widowed stock needing more than the exposure-weighted "
            "lift delivers"
        ),
        "modal_cells": list(REGISTERED_MODAL_CELLS),
        "secondary_cells": list(REGISTERED_SECONDARY_CELLS),
        "modal_failed_seeds": modal_failed_seeds,
        "modal_materialized": modal_materialized,
        "secondary_failed_seeds": secondary_failed_seeds,
        "secondary_seed3_failed": 3 in secondary_failed_seeds,
        "modal_track": {
            c: track(c)
            for c in REGISTERED_MODAL_CELLS + REGISTERED_SECONDARY_CELLS
        },
        "targeted_cells": list(TARGETED_CELLS),
        "targeted_cells_track": {c: track(c) for c in TARGETED_CELLS},
        "distinct_failing_cells": sorted(distinct_fail_cells),
        "decider_analysis": {
            "n_seeds_pass_actual": n_pass_actual,
            "n_seeds_pass_if_modal_forgiven": n_pass_no_modal,
            "n_seeds_pass_if_secondary_forgiven": n_pass_no_secondary,
            "n_seeds_pass_if_targeted_forgiven": n_pass_no_targeted,
            "decider": decider,
        },
    }


# ==========================================================================
# Provenance
# ==========================================================================
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins + the candidate-15 schema, c1-c14 and forensics shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for name in (1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14):
        pins[f"candidate{name}_runner"] = (
            f"scripts/run_gate2_candidate{name}.py"
        )
        pins[f"candidate{name}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{name}.py"
        )
    pins["candidate14_artifact"] = "runs/gate2_hazard_v14.json"
    pins["candidate14_artifact_sha256"] = c1._sha_of_file(CANDIDATE14_ARTIFACT)
    pins["forensics3_runner"] = "scripts/gate2_forensics3.py"
    pins["forensics3_runner_sha256"] = c1._sha_of_file(
        ROOT / "scripts" / "gate2_forensics3.py"
    )
    pins["forensics3_artifact"] = "runs/gate2_forensics3_v1.json"
    pins["forensics3_artifact_sha256"] = c1._sha_of_file(FORENSICS3_ARTIFACT)
    pins["estimator"] = "mean_over_K20_draws (5200 + k, k=0..19)"
    pins["deltas"] = (
        "one delta vs candidate 14: the NCHS period-trend multiplier "
        "exp(beta_sex * (year - 1995)) is removed from the surviving-spouse "
        "widowhood hazard (the source-aligned train-empirical seven-band x sex "
        "level, period-pooled, is applied without the trend; the committed "
        "betas stay documented for deployment-time use). Everything else "
        "byte-identical to candidate 14 (the reused compute chain shares "
        "candidate 14's exact code objects EXCEPT _widow_probs, which is "
        "re-implemented to drop the trend; the fit is candidate 14's, "
        "unchanged)"
    )
    pins["byte_identity_code_objects"] = {
        name: (getattr(c14, name).__code__ is globals()[name].__code__)
        for name in REUSED_CODE_OBJECT_NAMES
    }
    pins["diverged_code_objects_vs_candidate14"] = {
        name: (getattr(c14, name).__code__ is not globals()[name].__code__)
        for name in DIVERGED_CODE_OBJECT_NAMES
    }
    pins["widowhood_bands_candidate14"] = [list(b) for b in c14.WIDOW_BANDS]
    pins["widowhood_bands_candidate15"] = [list(b) for b in WIDOW_BANDS]
    pins["nchs_trend_applied_candidate14"] = True
    pins["nchs_trend_applied_candidate15"] = False
    return pins


def _model_block() -> dict[str, Any]:
    """The model block, edited for the one candidate-15 delta."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a "
            "surviving-spouse widowhood component (candidate 14 with one "
            "delta: the NCHS period-trend multiplier removed from the "
            "widowhood hazard -- the untrended train-empirical seven-band "
            "level), scored under the amended mean-over-K=20-draws estimator"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "delta_vs_candidate14": DELTA_VS_CANDIDATE14,
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic spline on age, "
                "knots 20/22/25/30/40, sex, birth-decade cohort -- "
                "BYTE-IDENTICAL to candidate 14 at a shared draw seed"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x order, "
                "add-one smoothed -- estimator BYTE-IDENTICAL to candidate 14"
            ),
            "widowhood": (
                "COMPOSED; surviving-spouse marriage-history widowhood level "
                "(seven bands {18-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+} "
                "x sex, candidate 14's fit, BYTE-IDENTICAL). THE DELTA: the "
                "NCHS period-trend multiplier exp(beta_sex * (year - 1995)) is "
                "REMOVED from the applied hazard -- candidate 15 applies the "
                "source-aligned train-empirical level ONLY, period-pooled, "
                "matching the gate reference's untrended measurement basis. "
                "The committed NCHS betas stay documented for deployment-time "
                "use but are not applied at gate time"
            ),
            "remarriage": (
                "weighted empirical hazard by ego age band "
                "(18-34/35-49/50-64/65-74/75+) x years-since-dissolution band "
                "x origin x sex, add-one smoothed -- the candidate-11 5-band "
                "table, BYTE-IDENTICAL to candidate 14"
            ),
            "fertility": (
                "single-year-of-age triangular-kernel rates within parity "
                "(0/1/2/3+) x birth-decade cohort -- BYTE-IDENTICAL to "
                "candidate 14 (no fertility delta)"
            ),
            "spousal_age_gap": (
                "candidate 12's age-band-conditioned spousal-gap draw "
                "(vestigial: the composed widowhood hazard keys on the "
                "surviving ego's own age, not the imputed spouse age; proven "
                "inert at candidate 12's grading) -- left UNTOUCHED per "
                "byte-minimality, BYTE-IDENTICAL to candidate 14"
            ),
            "entry_widowed_initial_state": (
                "candidate 12's delta 1 (inherited, unchanged): persons "
                "observed already-widowed at their first observed PSID wave "
                "enter widowed; the reference-carried widowed person-years are "
                "injected onto the simulated marital_state post-assembly. "
                "RNG-neutral, BYTE-IDENTICAL to candidate 14"
            ),
            "lifetime_marriage_count_initial_state": (
                "candidate 9's delta 1 (inherited, unchanged): each holdout "
                "person's simulated lifetime-marriage count initialises at "
                "their OBSERVED residual and accumulates the simulated datable "
                "transitions. An observed initial state; RNG-neutral"
            ),
        },
        "estimator": (
            "AMENDED (gates.yaml gate_2 amendment 1, ratified 2026-07-08): per "
            "cell rbar_candidate,s = mean over K=20 draws (default_rng(5200 + "
            "k), k=0..19) of the cell rate; score |ln(rbar / rate_a,s)| scored "
            "once (NOT the mean of per-draw scores); tolerances and the "
            "46-cell 4-of-5-seed conjunction byte-identical to the lock"
        ),
        "registered_ambiguity_resolutions": {
            "widowhood_trend": (
                "the NCHS period-trend multiplier exp(beta_sex * (year - "
                "1995)) is REMOVED from the widowhood hazard; the "
                "source-aligned train-empirical seven-band x sex level is "
                "applied period-pooled (year-invariant). The committed betas "
                "are read (candidate 5's frozen values) and stored, not "
                "applied"
            ),
            "widowhood_level_fit": (
                "candidate 14's seven-band surviving-spouse widowhood level, "
                "refit bit-identically (train mh85_23 spouse-death endings "
                "over married person-year exposure, transitions._hazard_by_band "
                "weighted, no add-one); UNCHANGED by candidate 15's delta"
            ),
            "byte_identity": (
                "candidate 15 reuses candidate 14's EXACT code objects for "
                "_build_sim_lookups, simulate_holdout, _draw_moments, "
                "score_seed, fit_remarriage_age_banded and "
                "_remarriage_probs_age_banded (rebound to this module's "
                "globals); _widow_probs is RE-IMPLEMENTED (the trend removed) "
                "and fit_components wraps candidate 14's fit "
                "(revision_pins.byte_identity_code_objects / "
                "diverged_code_objects_vs_candidate14)"
            ),
            "everything_else": (
                "the seven-band widowhood level, the entry-widowed observed "
                "initial state, the 5-band remarriage table, the observed "
                "marriage-count initial state (candidate 9 delta 1), the "
                "first-marriage spline, divorce, the single-year "
                "triangular-kernel fertility, the vestigial spousal-gap "
                "machinery, the competing-risk step, one sequence per person "
                "per draw, and the locked protocol are byte-identical to "
                "candidate 14"
            ),
        },
    }


# ==========================================================================
# Driver
# ==========================================================================
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

    # Preflight: candidate 14 (the base), its fit chain and forensics 3 must be
    # present, plus the candidate-5 NCHS references.
    for name, path in (
        ("candidate-14", CANDIDATE14_ARTIFACT),
        ("candidate-13", CANDIDATE13_ARTIFACT),
        ("candidate-12", CANDIDATE12_ARTIFACT),
        ("candidate-11", CANDIDATE11_ARTIFACT),
        ("candidate-10", CANDIDATE10_ARTIFACT),
        ("candidate-9", CANDIDATE9_ARTIFACT),
        ("candidate-8", CANDIDATE8_ARTIFACT),
        ("candidate-7", CANDIDATE7_ARTIFACT),
        ("candidate-6", CANDIDATE6_ARTIFACT),
        ("candidate-5", CANDIDATE5_ARTIFACT),
        ("forensics-3", FORENSICS3_ARTIFACT),
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
    c6._committed_beta_v5()  # fail fast if the committed betas drifted

    # Structural delta guard: candidate 15 removes the NCHS trend from the
    # widowhood hazard; the widowhood band table is candidate 14's, unchanged.
    if tuple(WIDOW_BANDS) != tuple(c14.WIDOW_BANDS):
        raise RuntimeError(
            "candidate 15's widowhood bands differ from candidate 14's; the "
            "delta is the trend removal only, not the band table."
        )
    # The delta must be live: _widow_probs must be year-invariant (no trend),
    # and must DIFFER from candidate 14's away from the anchor year.
    _mort = np.array([[0.01, 0.005]] * len(WIDOW_BANDS), dtype=np.float64)
    _beta = np.array(
        [
            NCHS_BETA_BY_SEX_COMMITTED["female"],
            NCHS_BETA_BY_SEX_COMMITTED["male"],
        ],
        dtype=np.float64,
    )
    _age = np.array([40.0, 80.0, 90.0], dtype=np.float64)
    _egom = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    _sp = np.array([42.0, 82.0, 88.0], dtype=np.float64)
    _spm = 1.0 - _egom
    p_anchor = _widow_probs(_age, _egom, _sp, _spm, 1995, _mort, _beta)
    p_late = _widow_probs(_age, _egom, _sp, _spm, 2020, _mort, _beta)
    if not np.allclose(p_anchor, p_late, rtol=0, atol=0):
        raise RuntimeError(
            "candidate 15's _widow_probs is not year-invariant; the trend was "
            "not fully removed."
        )
    c14_late = c14._widow_probs(_age, _egom, _sp, _spm, 2020, _mort, _beta)
    if np.allclose(c14_late, p_late):
        raise RuntimeError(
            "candidate 15's _widow_probs equals candidate 14's away from the "
            "anchor year; the delta is inert (the trend was not removed)."
        )

    # Byte-identity guard: the reused compute chain MUST share candidate 14's
    # exact code objects; the re-implemented functions MUST NOT. Fail fast.
    for name in REUSED_CODE_OBJECT_NAMES:
        if globals()[name].__code__ is not getattr(c14, name).__code__:
            raise RuntimeError(
                f"{name} does not share candidate 14's code object; the "
                "reused-chain byte-identity contract is violated."
            )
        if globals()[name].__globals__ is not globals():
            raise RuntimeError(
                f"{name} is not rebound to candidate 15's globals; it would "
                "read candidate 14's _widow_probs (with the trend)."
            )
    for name in DIVERGED_CODE_OBJECT_NAMES:
        if globals()[name].__code__ is getattr(c14, name).__code__:
            raise RuntimeError(
                f"{name} shares candidate 14's code object but must be "
                "re-implemented for the candidate-15 delta."
            )

    mh_records = marriage.marriage_history()
    birth_records = g2f.births.birth_history()
    death_records = g2f.deaths.read_death_records()
    demo = g2f.panels.demographic_panel()
    panel, fert, data_meta = g2f.load_panels()
    order_map = c1._order_map(mh_records)
    if verbose:
        print(
            f"panel: {data_meta['n_person_years']} person-years, "
            f"{data_meta['panel_persons_weighted']} persons; "
            f"estimator: mean over K={N_DRAWS} draws (5200 + k)"
        )

    # Hard gate 1: bit-exact reproduction of the committed floor (inherited).
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

    # Hard gate 2: candidate 9's delta-1 count reconciliation (inherited;
    # train-side). Must reconcile to remainder 0.0.
    reconciliation = _delta1_reconciliation(panel, mh_records, GATE_SEEDS)
    if verbose:
        pid_max = reconciliation["per_person_identity_max_abs_residual"]
        agg_max = reconciliation["aggregate_reconciliation_max_abs_remainder"]
        print(
            "delta-1 (count) reconciliation reconciled="
            f"{reconciliation['reconciled']} "
            f"(per-person identity max={pid_max:.2e}, "
            f"aggregate max remainder={agg_max:.2e})"
        )
    if not reconciliation["reconciled"]:
        raise RuntimeError(
            "candidate 9 delta-1 count reconciliation failed; refusing to "
            "proceed."
        )

    # Hard gate 3: candidate 12's entry-widowed carried classification must
    # reproduce forensics 3's committed Q6 to float precision (inherited).
    entry_recon = _entry_widowed_reconciliation(panel, demo, GATE_SEEDS)
    if verbose:
        print(
            "delta-1 (entry-widowed) reconciliation reconciled="
            f"{entry_recon['reconciled']} "
            f"(max remainder={entry_recon['max_abs_remainder']:.2e})"
        )
    if not entry_recon["reconciled"]:
        raise RuntimeError(
            "entry-widowed carried classification does not reproduce forensics "
            "3's committed Q6 initial_state_fixable share; refusing to proceed."
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

    # Fresh-run artifact-schema blocks (amendment 1; candidate 10's assembly).
    per_draw_cube = _per_draw_per_cell_rates_block(per_seed, tol)
    undefined_block = _undefined_draw_block(per_seed)
    dispersion_block = _per_draw_dispersion_block(per_seed, tol)

    if undefined_block["run_invalidated"]:
        raise RuntimeError(
            "RUN INVALIDATED (fresh_run_artifact_schema.undefined_draw_rule): "
            f"{undefined_block['n_undefined_gated_draws']} undefined gated "
            "cell draw(s) (empty simulated denominator); the run must be "
            "re-registered and re-run. No draw may be dropped or substituted."
        )

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
    count_margins = _count_cell_margin_diagnostic(per_seed)
    incidence_headroom = _incidence_headroom_diagnostic(per_seed)
    comparison = _candidate14_comparison(per_seed)
    entry_counts = _entry_widowed_seed_counts(panel, demo, GATE_SEEDS)
    elderly = _elderly_75plus_diagnostic(per_seed)

    # The exposure-weighted trend multiplier the removal cancels, computed on
    # the fitted components (seed-0 train complement betas -- committed, so
    # split-independent). One extra train-side fit, no new outer contact.
    side_a0, side_b0 = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=GATE_SEEDS[0]
    )
    ids_b0 = set(int(x) for x in side_b0.person_id.unique())
    components0 = fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        ids_b0,
    )
    trend_diag = _trend_multiplier_diagnostic(panel, components0)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 15",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate14_registration": CANDIDATE14_REGISTRATION,
        "candidate13_registration": CANDIDATE13_REGISTRATION,
        "candidate12_registration": CANDIDATE12_REGISTRATION,
        "candidate11_registration": CANDIDATE11_REGISTRATION,
        "candidate10_registration": CANDIDATE10_REGISTRATION,
        "candidate9_registration": CANDIDATE9_REGISTRATION,
        "candidate8_registration": CANDIDATE8_REGISTRATION,
        "candidate6_registration": CANDIDATE6_REGISTRATION,
        "candidate5_registration": CANDIDATE5_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "candidate14_grading_diagnostic": CANDIDATE14_GRADING_DIAGNOSTIC,
        "delta_vs_candidate14": DELTA_VS_CANDIDATE14,
        "amended_estimator": (
            "gates.yaml gate_2 amendment 1 (ratified 2026-07-08, flip #97): "
            "per-cell score |ln(rbar / rate_a)| with rbar the mean over K=20 "
            "draws (default_rng(5200 + k), k=0..19) of the cell rate "
            "(inherited from candidate 14, unchanged)"
        ),
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79); amendment 1 flipped "
            "live (#97). Protocol/views/tolerances/schema read at runtime; no "
            "threshold moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.65-0.75",
            "conjunction_estimate": 0.70,
            "pass_path_seeds": [0, 1, 3, 4],
            "modal_failure": (
                "the female marriage-count cell "
                "(mean_lifetime_marriages|female) re-clipping from the ~7% "
                "young-widow-inflow rise; secondary: seed 3's 75+ widowed "
                "stock needing more than the exposure-weighted lift delivers"
            ),
            "registration": SPEC_REGISTRATION,
        },
        "model": _model_block(),
        "protocol": {
            "option": (
                "a (gate-1 mirror; LOCKED gates.yaml gate_2, amendment 1 live)"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel.attrs, 'person_id', fraction=0.5, seed=s); side A = "
                "the holdout, side B = the train complement"
            ),
            "gate_seeds": list(GATE_SEEDS),
            "estimator": "mean_over_K20_draws",
            "n_draws": N_DRAWS,
            "draw_rng_rule": "numpy.random.default_rng(5200 + k), k=0..19",
            "single_draw_provenance_rng_rule": (
                "numpy.random.default_rng(4200 + seed) (candidates 1-9; "
                "retained for provenance only, not scored)"
            ),
            "one_sequence_per_person_per_draw": True,
            "scored_against": (
                "side A's own empirical rate (rate_a in "
                "runs/gate2_floors_v2.json noise_floor_per_seed)"
            ),
            "statistic": (
                "|ln(rbar_candidate,s / rate_a,s)| per cell, rbar the 20-draw "
                "mean rate, scored once (NOT the mean of per-draw scores)"
            ),
            "conjunction": (
                "all 46 gated cells per seed AND >= 4 of 5 gate seeds"
            ),
            "weight_definition": (
                "person-constant most-recent positive PSID cross-sectional "
                "weight; every gated statistic weighted, none unweighted"
            ),
        },
        "fresh_run_artifact_schema": {
            "applies_to": (
                "the fresh candidate-15 one-shot run registered AFTER the "
                "2026-07-08 ratification (registration 4928232089)"
            ),
            "per_draw_per_cell_rates": per_draw_cube,
            "undefined_draw_rule": undefined_block,
            "per_draw_dispersion_disclosure": dispersion_block,
        },
        "data": data_meta,
        "precheck": precheck,
        "delta1_reconciliation": reconciliation,
        "entry_widowed_reconciliation": entry_recon,
        "entry_widowed_seed_counts": entry_counts,
        "trend_multiplier_removed": trend_diag,
        "per_seed": per_seed,
        "seed_conjunction": seed_conjunction,
        "report_only": report_block,
        "count_cell_margins": count_margins,
        "incidence_headroom": incidence_headroom,
        "candidate14_comparison": comparison,
        "elderly_75plus_diagnostic": elderly,
        "modal_failure_materialized": modal,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "registration_pointer": REGISTRATION_POINTER,
            "candidate14_registration": CANDIDATE14_REGISTRATION,
            "candidate14_artifact": "runs/gate2_hazard_v14.json",
            "grading_diagnostic": CANDIDATE14_GRADING_DIAGNOSTIC,
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
        cm = comparison
        mc = cm["modal_cell"][MODAL_CELL]
        print(
            f"  {MODAL_CELL}: c14 {mc['c14_n_seeds_pass']}/5 -> "
            f"c15 {mc['c15_n_seeds_pass']}/5"
        )
        esum = elderly["summary"]
        print(
            "  75+ incidence sim/ref: c14 "
            f"{esum['incidence_sim_over_ref']['c14_mean']:.4f} -> c15 "
            f"{esum['incidence_sim_over_ref']['c15_mean']:.4f}; "
            "75+ stock sim/ref: c14 "
            f"{esum['stock_sim_over_ref']['c14_mean']:.4f} -> c15 "
            f"{esum['stock_sim_over_ref']['c15_mean']:.4f}"
        )
        print(
            "  pooled trend multiplier removed="
            f"{trend_diag['pooled_exposure_weighted_multiplier']:.4f} "
            f"(implied inflow uplift "
            f"{trend_diag['pooled_implied_inflow_uplift']:.4f})"
        )
        for c in COUNT_CELLS:
            b = count_margins["cells"][c]
            print(
                f"  {c}: c14 {b['candidate14_n_seeds_pass']}/5 -> "
                f"c15 {b['n_seeds_pass']}/5 "
                f"(min margin={b['min_margin']:.4f})"
            )
        print(
            "modal (female count re-clip) materialized="
            f"{modal['modal_materialized']} "
            f"(count failed seeds {modal['modal_failed_seeds']}; "
            f"stock failed seeds {modal['secondary_failed_seeds']}); "
            f"decider={modal['decider_analysis']['decider']}"
        )
        print(f"distinct failing cells={modal['distinct_failing_cells']}")
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
