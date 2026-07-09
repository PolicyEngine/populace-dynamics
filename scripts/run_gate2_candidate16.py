"""Gate-2 candidate 16 (run 1): candidate 15 + EXACTLY ONE delta -- the
surviving-spouse widowhood hazard is additionally conditioned on the OBSERVED
binary support-composition stratum.

The SIXTEENTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42 comment
4929419524 (``SPEC_REGISTRATION``): candidate 15's frozen spec (comment
4928232089, ``scripts/run_gate2_candidate15.py``, merged #107 -- the untrended
seven-band surviving-spouse widowhood level, the c13/c14 band structure)
verbatim EXCEPT one delta, registered from gate-2 forensics 4 (#108,
``runs/gate2_forensics4_v1.json``). One-shot; no constant moves after the
registration comment; published REGARDLESS of verdict.

The one delta vs candidate 15 (everything else byte-identical)
-------------------------------------------------------------
**The surviving-spouse widowhood hazard (seven-band x sex, untrended per
candidate 15) is additionally conditioned on ONE observed design covariate:
whether the person's OBSERVED support window extends to at least age 75.** The
stratum is binary (0 = the observed window does NOT reach age 75, 1 = it
reaches >= 75), known ex ante for every person from the observed panel
attributes alone (``min(last_wave, censor_year) >= birth_year + 75``, the
forensics-4 Q9 window-geometry rule; ``scripts/gate2_forensics4.py``, #108) --
the SAME observed-data class as the initial-state fixes (candidate 9's observed
marriage count, candidate 12's entry-widowed initial state). Candidate 15
applied ``rate = widow_level(ego_band, ego_sex)``; candidate 16 applies ``rate =
widow_level(ego_band, ego_sex, ego_support_stratum)``. Both strata are
train-estimated per age band x sex under the EXISTING smoothing convention
(``transitions._hazard_by_band`` weighted hazard num_wt / den_wt, no add-one).

Within each age band x sex, the two strata recombine to candidate 15's band
AGGREGATE train rate by construction -- the exposure-weighted identity
``(den_0 * rate_0 + den_1 * rate_1) / (den_0 + den_1) == (num_0 + num_1) /
(den_0 + den_1) == rate_aggregate`` holds because the binary person-level
stratum PARTITIONS the married person-year exposure and the widowhood events
exactly (each row belongs to exactly one stratum). So the AGGREGATE widowhood
incidence is preserved (candidate 15's untrended level, unchanged in aggregate)
while the event COMPOSITION matches the reference's observed-window correlation:
real widowhood correlates with long observed support (forensics 4 Q9: simulated
50-64-onset widowhoods reach age-75 windows 39% vs the reference's 57%), which a
uniform within-band hazard cannot see. The recombination identity is recorded in
the artifact (``support_stratum_recombination``) and reconciles to 0.0 as a
hard gate.

Everything else is byte-identical to candidate 15 -- the seven-band widowhood
LEVEL bands themselves (c13/c14 structure {18-34, 35-44, 45-54, 55-64, 65-74,
75-84, 85+}), the NCHS-trend removal (candidate 15's delta; the committed betas
stay documented, not applied), the entry-widowed observed initial state
(candidate 12's delta 1), the 5-band remarriage current-age table (candidate
11), the observed undatable-marriage lifetime-count initial state (candidate 9's
delta 1), the single-year triangular-kernel fertility, the RNG, the K=20
mean-of-draws protocol, and ``fresh_run_artifact_schema`` conformance. The
vestigial spousal-age-gap machinery (candidate 12's delta 2, proven inert) is
left UNTOUCHED per byte-minimality. Runner ``scripts/run_gate2_candidate16.py``,
artifact ``runs/gate2_hazard_v16.json``.

Provable byte-identity (code-object reuse)
------------------------------------------
Candidate 15 was a COMPUTE delta INSIDE ``_widow_probs`` that reused the
signature and call site, so ``simulate_holdout`` / ``_build_sim_lookups`` were
REUSED. Candidate 16's delta adds a per-person conditioning covariate that the
widowhood hazard must SEE at simulation time, so it threads a new per-person
``stratum`` argument through ``simulate_holdout`` into ``_widow_probs`` and a
stratum axis through ``_build_sim_lookups``' ``mort_arr`` -- so
``_widow_probs``, ``_build_sim_lookups``, ``simulate_holdout`` and
``fit_components`` all DIVERGE (re-implemented), while candidate 16 REUSES
candidate 15's EXACT code objects for the still-unaffected compute chain
(``_draw_moments``, ``score_seed``, ``fit_remarriage_age_banded``,
``_remarriage_probs_age_banded``), rebound (:func:`_rebind`) so their globals
resolve against THIS module -- the reused ``_draw_moments`` / ``score_seed``
call candidate 16's ``simulate_holdout`` / ``fit_components`` by global name.
Every re-implemented statement in ``simulate_holdout`` is candidate 12's
verbatim EXCEPT (a) the per-person ``stratum`` array assembled from the observed
support windows once before the year loop, and (b) the ``stratum[sub]`` argument
passed to ``_widow_probs`` in the married competing-risk block.

Conditioning on the stratum is RNG-NEUTRAL: the per-year uniform block
(``rng.random(n_active)``) is drawn BEFORE ``_widow_probs`` and the widowhood
threshold array keeps its shape and dtype, so only the competing-risk THRESHOLD
value moves (exactly as candidate 15's trend removal). The scored RNG stream is
therefore byte-identical to candidate 15, and the marital-state-independent
cells (``asfr.*``, ``completed_fertility.*``, ``first_marriage.*``) are
byte-identical to candidate 15 draw-by-draw. The stratum reshapes every
widowhood band's competition, so all widowhood-incidence and widowed-stock cells
move.

Designed effect (registration)
------------------------------
The delta moves the measured dominant term (forensics 4 Q9, the survival-to-75+
yield) with ~7x headroom: seed 3's stock (+0.023 needed) and seed 2's razor edge
(+0.00013) both clear on the Q9 arithmetic if even a third of the yield gap
closes; seeds 0/1/4's stocks move toward reference from below. Counts gain
widowed exposure on long-window persons (margins 0.026-0.049; the added widows
are majority 65+ with near-zero remarriage). Incidence cells preserved by the
recombination identity. ``P(pass) ~= 0.6-0.7``; pass path 0, 1, 3, 4 (seed 2's
completed_fertility.c1970s is an RNG-isolated split artifact, byte-identical to
candidate 15, so it stays failing regardless). Modal failure if it fails: the
count cells clipping from the recomposed widowed exposure; secondary: a stock
overshoot on seed 1 (the high-side split).

Hard-stop prechecks (inherited): the scoring path must reproduce, bit-for-bit,
every committed full-panel reference moment, every committed per-gate-seed
``rate_a`` and each gate seed's committed holdout-id sha256 BEFORE any candidate
is simulated; candidate 9's delta-1 count reconciliation must close to remainder
0.0; candidate 12's entry-widowed carried classification must reproduce
forensics 3's committed Q6 initial-state-fixable share to float precision; AND
candidate 16's exposure-weighted recombination identity must close to 0.0 (the
two strata recombine to candidate 15's band aggregate). Any mismatch is a hard
stop. Run ONCE; publish REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit; no statsmodels). Run from the repository root with the PSID
history files staged::

    .venv/bin/python scripts/run_gate2_candidate16.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Candidate 15 supplies the machinery this build deltas ONCE: its compute chain
# (the untrended seven-band surviving-spouse widowhood level over candidate 12's
# entry-widowed initial state + age-band gap over candidate 11's remarriage over
# candidate 9's count residual), its fresh-run artifact-schema blocks, and --
# via its imports -- candidate 1's precheck / verdict assembly and candidate 8's
# vectorised simulation helpers. Only the surviving-spouse widowhood hazard
# gains the observed support-composition stratum.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import gate2_forensics3 as gf3  # noqa: E402
import gate2_forensics4 as gf4  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate5 as c5  # noqa: E402
import run_gate2_candidate6 as c6  # noqa: E402
import run_gate2_candidate14 as c14  # noqa: E402
import run_gate2_candidate15 as c15  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v16.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
CANDIDATE15_ARTIFACT = ROOT / "runs" / "gate2_hazard_v15.json"
CANDIDATE14_ARTIFACT = ROOT / "runs" / "gate2_hazard_v14.json"
CANDIDATE12_ARTIFACT = ROOT / "runs" / "gate2_hazard_v12.json"
CANDIDATE9_ARTIFACT = ROOT / "runs" / "gate2_hazard_v9.json"
CANDIDATE6_ARTIFACT = ROOT / "runs" / "gate2_hazard_v6.json"
CANDIDATE5_ARTIFACT = ROOT / "runs" / "gate2_hazard_v5.json"
FORENSICS3_ARTIFACT = ROOT / "runs" / "gate2_forensics3_v1.json"
FORENSICS4_ARTIFACT = ROOT / "runs" / "gate2_forensics4_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v16"
RUN_NAME = "gate2_hazard_v16"

#: This run's frozen-spec registration (issue #42, comment 4929419524).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4929419524"
)
#: The bare registration comment id, pinned for the artifact pointer.
REGISTRATION_POINTER = "4929419524"
#: The candidate-15 spec this build deltas ONCE (comment 4928232089, #107).
CANDIDATE15_REGISTRATION = c15.SPEC_REGISTRATION
CANDIDATE15_POINTER = c15.REGISTRATION_POINTER
#: The registration chain candidate 15 carried, threaded through for provenance.
CANDIDATE14_REGISTRATION = c15.CANDIDATE14_REGISTRATION
CANDIDATE12_REGISTRATION = c15.CANDIDATE12_REGISTRATION
CANDIDATE9_REGISTRATION = c15.CANDIDATE9_REGISTRATION
CANDIDATE6_REGISTRATION = c15.CANDIDATE6_REGISTRATION
CANDIDATE5_REGISTRATION = c15.CANDIDATE5_REGISTRATION
CANDIDATE1_REGISTRATION = c15.CANDIDATE1_REGISTRATION
#: The forensics-4 diagnostic (#108) candidate 16 registers from and cites.
FORENSICS4_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4928761676"
)
#: Forensics 4's finding: the ~17% elderly-widow-stock leak is the
#: survival-to-75+ YIELD -- real widowhood correlates with long observed
#: support, which a uniform within-band hazard cannot see.
FORENSICS4_DIAGNOSTIC = (
    "issue #42 comment 4928761676 / runs/gate2_forensics4_v1.json (#108): "
    "candidate 15's binding failure is the 75+ widowed STOCK (share_widowed."
    "75+|female failing seeds 2 and 3), and its ~17%-of-reachable leak is the "
    "survival-to-75+ YIELD -- the realized 75+ widowed person-years per onset "
    "within observed support, concentrated at 50-64/65-74 onset. The yield "
    "leak is SUPPORT-WINDOW TRUNCATION, not over-remarriage: survival-in-"
    "widowhood tracks the reference, but only ~39% of the simulated reachable "
    "50-64-onset widowhoods have an observed window reaching age 75 vs ~57% in "
    "the reference. A uniform within-band widowhood hazard produces widows in "
    "proportion to married exposure regardless of observed support length; the "
    "reference's widows concentrate among long-observed-support persons. "
    "Conditioning the hazard on the observed support-composition stratum keeps "
    "long-window mid-age onsets reachable. Seed 2's completed_fertility.c1970s "
    "is a single-seed split artifact (RNG-isolated), not a fertility deficit "
    "-- no fertility delta warranted"
)

# --- Amended-estimator draw stream (inherited from candidate 15, unchanged). -
DRAW_SEED_BASE = c15.DRAW_SEED_BASE  # 5200
N_DRAWS = c15.N_DRAWS  # 20

# --------------------------------------------------------------------------
# Frozen dials + pure helpers, reused (byte-identical; import-bound via c15).
# --------------------------------------------------------------------------
GATE_SEEDS = c15.GATE_SEEDS
SIM_SEED_BASE = c15.SIM_SEED_BASE  # 4200 (single-draw stream; provenance only)
EXACT_ATOL = c15.EXACT_ATOL
Components = c15.Components

YSD_BANDS = c15.YSD_BANDS
YSD_LOWERS = c15.YSD_LOWERS
_bands_vec = c15._bands_vec

#: Candidate 6's period-trend anchor year (1995.0); retained for the artifact
#: provenance (candidate 16, like candidate 15, does NOT apply the trend).
TREND_ANCHOR_YEAR = c15.TREND_ANCHOR_YEAR

# Candidate 8's simulation helpers (import-bound via candidate 15, unchanged).
_STATE = c15._STATE
_STATE_ABSORB = c15._STATE_ABSORB
_ASFR_LO = c15._ASFR_LO
_ASFR_HI = c15._ASFR_HI
_assemble_sim_panel = c15._assemble_sim_panel
_divorce_probs = c15._divorce_probs
_fertility_probs_single = c15._fertility_probs_single

# Candidate 9's observed residual + reconciliation (delta 1 of candidate 9),
# inherited via candidate 15, reused byte-for-byte.
observed_residual_counts = c15.observed_residual_counts
_delta1_reconciliation = c15._delta1_reconciliation

# --- The 5-band remarriage table (candidate 11's delta; UNCHANGED here). -----
REM_AGE_BANDS = c15.REM_AGE_BANDS
REM_AGE_LOWERS = c15.REM_AGE_LOWERS
_REM_AGE_LABEL = c15._REM_AGE_LABEL

# --- The vestigial spousal-age-gap machinery (candidate 12's delta 2, proven
# --- INERT; left UNTOUCHED per byte-minimality; import-bound unchanged). ------
GAP_AGE_BANDS = c15.GAP_AGE_BANDS
GAP_AGE_LOWERS = c15.GAP_AGE_LOWERS
_GAP_AGE_LABEL = c15._GAP_AGE_LABEL
FALLBACK_MIN_WEIGHTED_COUPLES = c15.FALLBACK_MIN_WEIGHTED_COUPLES
spousal_gap_distribution_by_band = c15.spousal_gap_distribution_by_band
c11_pooled_gap = c15.c11_pooled_gap
_fallback_group = c15._fallback_group
_gap_band_arrays = c15._gap_band_arrays
_draw_banded_gaps = c15._draw_banded_gaps

# --- Candidate 12's entry-widowed observed initial state (delta 1; UNCHANGED,
# --- import-bound so the re-implemented ``simulate_holdout`` injects it
# --- unchanged). -------------------------------------------------------------
observed_support = c15.observed_support
entry_widowed_carried_cells = c15.entry_widowed_carried_cells
_entry_widowed_seed_counts = c15._entry_widowed_seed_counts
_entry_widowed_reconciliation = c15._entry_widowed_reconciliation
_inject_entry_widowed = c15._inject_entry_widowed
_widowed_share_by_age = c15._widowed_share_by_age

# Fresh-run artifact-schema blocks are band-independent (they operate on the
# scored per-seed dicts); import-bound via candidate 15 (identical N_DRAWS /
# DRAW_SEED_BASE), so the [20, 46, 5] cube, the undefined-draw rule and the
# report-only dispersion are candidate 10's exact assembly.
_per_draw_per_cell_rates_block = c15._per_draw_per_cell_rates_block
_undefined_draw_block = c15._undefined_draw_block
_per_draw_dispersion_block = c15._per_draw_dispersion_block

# --- The seven-band surviving-spouse widowhood table (candidate 13/14
# --- structure, untrended per candidate 15; UNCHANGED here -- the BANDS are
# --- byte-identical, only the hazard gains the support-composition stratum). --
WIDOW_BANDS = c15.WIDOW_BANDS
WIDOW_LOWERS = c15.WIDOW_LOWERS
_WIDOW_BAND_LABEL = c15._WIDOW_BAND_LABEL

#: Candidate 5's committed NCHS per-sex log-linear period slopes (retained,
#: documented, NOT applied -- candidate 15's trend removal is inherited).
NCHS_BETA_BY_SEX_COMMITTED = dict(c15.NCHS_BETA_BY_SEX_COMMITTED)

# --------------------------------------------------------------------------
# THE ONE DELTA vs candidate 15: the observed support-composition stratum
# --------------------------------------------------------------------------
#: The support-composition stratum threshold age. The observed support window
#: "extends to at least age 75" iff the observed window end reaches this age.
SUPPORT_STRATUM_AGE = 75
#: The two strata: 0 = the observed support window does NOT reach age 75;
#: 1 = it reaches >= 75. Binary, per-person, known ex ante.
SUPPORT_STRATA = (0, 1)
_STRATUM_LABEL = {
    0: "window_below_75",
    1: "window_reaches_75plus",
}

#: The single named delta (registration comment 4929419524).
DELTA_VS_CANDIDATE15 = (
    "EXACTLY ONE delta vs candidate 15 (comment 4928232089, merged #107): the "
    "surviving-spouse widowhood hazard is additionally conditioned on the "
    "OBSERVED binary support-composition stratum -- whether the person's "
    "observed support window extends to at least age 75 (0 = below 75, 1 = "
    ">= 75), computed ex ante from the observed panel attributes alone "
    "(min(last_wave, censor_year) >= birth_year + 75, the forensics-4 Q9 "
    "window-geometry rule). Candidate 15 applied rate = widow_level(ego_band, "
    "ego_sex); candidate 16 applies rate = widow_level(ego_band, ego_sex, "
    "ego_support_stratum). Both strata are train-estimated per age band x sex "
    "under the EXISTING smoothing convention (transitions._hazard_by_band "
    "weighted hazard num_wt/den_wt, no add-one). The binary person-level "
    "stratum PARTITIONS the married person-year exposure and the widowhood "
    "events exactly, so within each age band x sex the two strata recombine to "
    "candidate 15's band AGGREGATE train rate by the exposure-weighted "
    "identity (den_0*rate_0 + den_1*rate_1)/(den_0+den_1) == "
    "(num_0+num_1)/(den_0+den_1) == rate_aggregate: aggregate incidence is "
    "PRESERVED while the event composition matches the reference's observed-"
    "window correlation (forensics 4 Q9: real widowhood correlates with long "
    "observed support, which a uniform within-band hazard cannot see). The "
    "recombination identity is recorded (support_stratum_recombination) and "
    "reconciles to 0.0. Everything else -- the seven-band widowhood LEVEL "
    "bands (c13/c14 structure), the NCHS-trend removal (candidate 15's delta; "
    "committed betas documented, not applied), the entry-widowed observed "
    "initial state (candidate 12 delta 1), the 5-band remarriage table, the "
    "observed marriage-count initial state (candidate 9 delta 1), the single-"
    "year triangular-kernel fertility, the RNG, the K=20 mean-of-draws "
    "protocol, the fresh-run artifact schema, and the vestigial spousal-age-"
    "gap machinery (candidate 12 delta 2, untouched per byte-minimality) -- is "
    "byte-identical to candidate 15"
)

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate16_run1_cache.json"
)
#: The train-side yield-ledger cache (the forensics-4 before/after), separate so
#: a crash there never re-scores a completed outer seed.
DEFAULT_YIELD_CACHE = (
    Path.home()
    / ".claude-worktrees"
    / "_gate2_candidate16_run1_yield_cache.json"
)

#: The marriage-count cells -- a gated RISK: the recomposed widowed exposure
#: (more widows among long-observed-support persons) feeds the widowed-origin
#: remarriage exposure. The registered modal failure is the count cells
#: clipping.
COUNT_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
)
#: The gated remarriage cells the delta reshapes via the widowed trajectory.
REMARRIAGE_GATED_CELLS = (
    "remarriage.after_divorce",
    "remarriage.ysd0-4",
    "remarriage.ysd5-9",
    "remarriage.ysd10+",
)
#: The elderly-widow-stock cell the delta is designed to lift (candidate 15's
#: two failing seeds, 2 and 3, are on this cell).
MODAL_CELL = "share_widowed.75+|female"
#: The registered modal failure (comment 4929419524): the count cells clipping
#: from the recomposed widowed exposure (primary); secondary: a stock overshoot
#: on seed 1 (the high-side split).
REGISTERED_MODAL_CELLS = COUNT_CELLS
#: The registered secondary failure (a stock overshoot on seed 1).
REGISTERED_SECONDARY_CELLS = ("share_widowed.75+|female",)
#: The gated widowhood-incidence cells; the recombination identity preserves the
#: band aggregate, so these are expected to hold (~0.95).
WIDOWHOOD_INCIDENCE_CELLS = (
    "widowhood.45-64|female",
    "widowhood.65-74|female",
    "widowhood.75+|female",
    "widowhood.45+|male",
)
#: The 75+ cells the delta most directly targets: incidence (preserved) and the
#: widowed stock (must lift where candidate 15 clipped seeds 2 and 3).
ELDERLY_75PLUS_CELLS = (
    "widowhood.75+|female",
    "share_widowed.75+|female",
)
#: The cells the delta most directly touches.
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
# The observed support-composition stratum (the delta's covariate)
# Known ex ante for every person from the observed panel attributes alone;
# the SAME observed-data class as candidate 9's / candidate 12's initial-state
# fixes. The window-end rule is forensics 4's Q9 window geometry
# (``scripts/gate2_forensics4.py``, #108): the observed window end is
# ``min(last_wave, censor_year)`` and it "reaches age 75" iff that end is at or
# past ``birth_year + 75``.
# ==========================================================================
def support_stratum_map(
    panel: transitions.MaritalPanel, demo: pd.DataFrame
) -> dict[str, Any]:
    """Per-person observed support-composition stratum (0/1) + provenance.

    For every panel person, the observed support window end is ``min(last_wave,
    censor_year)`` (``last_wave`` from :func:`gate2_forensics3.observed_support`
    on the demographic panel, ``censor_year`` from ``panel.attrs``), and the
    stratum is 1 iff that end is at or past ``birth_year + 75`` (the person's
    observed support extends to at least age 75), else 0. Persons with no
    observed support window (absent from the demographic panel) get stratum 0.
    Returns the fast lookup Series (person_id -> {0, 1}) plus the weighted /
    unweighted stratum counts. Split-independent (read once per reference
    panel); the SAME observed-data class as the initial-state fixes.
    """
    support = observed_support(demo)
    attrs = panel.attrs
    pid = attrs["person_id"].to_numpy(dtype=np.int64)
    birth = attrs["birth_year"].to_numpy(dtype=np.float64)
    censor = attrs["censor_year"].to_numpy(dtype=np.float64)
    last_wave = (
        attrs["person_id"].map(support["last_wave"]).to_numpy(dtype=np.float64)
    )
    # min(last_wave, censor): NaN last_wave (no observed support) -> no window,
    # so the observed end cannot reach 75 (stratum 0).
    obs_end = np.where(
        np.isnan(last_wave), -np.inf, np.minimum(last_wave, censor)
    )
    reaches_75 = obs_end >= (birth + float(SUPPORT_STRATUM_AGE))
    stratum = reaches_75.astype(np.int64)
    series = pd.Series(stratum, index=pd.Index(pid, name="person_id"))

    n_no_support = int(np.isnan(last_wave).sum())
    weight = (
        attrs["weight"].to_numpy(dtype=np.float64)
        if "weight" in attrs.columns
        else np.ones(len(attrs), dtype=np.float64)
    )
    provenance = {
        "definition": (
            "binary per-person observed support-composition stratum: 1 iff the "
            "observed support window end min(last_wave, censor_year) is at or "
            "past birth_year + 75 (the observed support extends to at least "
            "age 75), else 0. last_wave from gate2_forensics3.observed_support "
            "(the observed PSID wave window), censor_year / birth_year from "
            "panel.attrs. The window-end rule is forensics 4's Q9 window "
            "geometry (scripts/gate2_forensics4.py, #108). Known ex ante from "
            "observed attributes alone -- the SAME observed-data class as "
            "candidate 9's marriage-count and candidate 12's entry-widowed "
            "initial states"
        ),
        "threshold_age": SUPPORT_STRATUM_AGE,
        "window_end_rule": "min(last_wave, censor_year) >= birth_year + 75",
        "n_persons": int(len(attrs)),
        "n_window_reaches_75plus": int(stratum.sum()),
        "n_window_below_75": int((stratum == 0).sum()),
        "n_no_observed_support": n_no_support,
        "share_window_reaches_75plus": float(stratum.mean()),
        "weighted_share_window_reaches_75plus": (
            float((weight * stratum).sum() / weight.sum())
            if weight.sum() > 0
            else 0.0
        ),
    }
    return {"stratum_by_person": series, "provenance": provenance}


def _person_stratum_array(
    support_stratum: dict[str, Any], pid: np.ndarray
) -> np.ndarray:
    """Per-person stratum (0/1) aligned to ``pid`` (absent persons -> 0)."""
    return (
        pd.Series(pid)
        .map(support_stratum["stratum_by_person"])
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )


# ==========================================================================
# THE DELTA fit: the surviving-spouse widowhood hazard, stratified by the
# observed support-composition stratum. Each stratum's per-band x sex hazard is
# candidate 6/13/14's construction (``transitions._hazard_by_band`` weighted
# hazard, no add-one) restricted to the stratum's train rows. The two strata's
# num_wt / den_wt sum to the aggregate's, so the exposure-weighted recombination
# is exact.
# ==========================================================================
def _widowhood_hazard_cells_by_stratum(
    panel: transitions.MaritalPanel,
    train_ids: set[int],
    stratum_series: pd.Series,
) -> dict[int, dict[str, dict[str, float]]]:
    """Train widowhood hazard cells over the seven bands, per stratum.

    Partitions the train married person-year exposure and the train widowhood
    events by the per-person stratum and runs candidate 14's
    :func:`transitions._hazard_by_band` (weighted, no add-one) on each stratum
    over :data:`WIDOW_BANDS` x sex -- the gate reference's own machinery,
    unchanged, restricted to each stratum. Returned keyed ``stratum -> {"band|
    sex": cell}`` with the ``widowhood.`` prefix stripped.
    """
    py = panel.person_years
    ev = panel.events
    py = py[py["person_id"].isin(train_ids)]
    ev = ev[ev["person_id"].isin(train_ids)]
    married = py[py["marital_state"] == "married"].copy()
    widow = ev[ev["transition"] == "widowhood"].copy()
    married["stratum"] = (
        married["person_id"].map(stratum_series).fillna(0).astype(np.int64)
    )
    widow["stratum"] = (
        widow["person_id"].map(stratum_series).fillna(0).astype(np.int64)
    )
    pref = "widowhood."
    out: dict[int, dict[str, dict[str, float]]] = {}
    for s in SUPPORT_STRATA:
        cells = transitions._hazard_by_band(
            widow[widow["stratum"] == s],
            married[married["stratum"] == s],
            "age",
            WIDOW_BANDS,
            prefix="widowhood",
            by_sex=True,
            weighted=True,
        )
        out[s] = {
            key[len(pref) :]: dict(cell)
            for key, cell in cells.items()
            if key.startswith(pref)
        }
    return out


def _recombination_identity(
    strat_cells: dict[int, dict[str, dict[str, float]]],
    agg_cells: dict[str, dict[str, float]],
    aggregate_level: dict[str, float],
) -> dict[str, Any]:
    """The exposure-weighted recombination of the two strata to the aggregate.

    For each band x sex cell the two strata's weighted-event / weighted-exposure
    counts sum to the aggregate's (the binary person-level stratum partitions
    the rows exactly), so the exposure-weighted recombination ``(den_0 * rate_0
    + den_1 * rate_1) / (den_0 + den_1) == (num_0 + num_1) / (den_0 + den_1)``
    equals candidate 15's aggregate band rate to float precision. Reports the
    per-cell strata rates + exposures, the recombined rate, and its residual vs
    both the aggregate cells' own rate and candidate 15's applied
    ``aggregate_level`` (``components.mortality``). ``reconciled`` (max residual
    <= 1e-9) is a hard gate.
    """
    cells: dict[str, Any] = {}
    max_rate_resid = 0.0
    max_num_resid = 0.0
    max_den_resid = 0.0
    max_vs_applied = 0.0
    for key in sorted(agg_cells):
        agg = agg_cells[key]
        num0 = float(strat_cells[0][key]["num_wt"])
        den0 = float(strat_cells[0][key]["den_wt"])
        rate0 = float(strat_cells[0][key]["rate"])
        num1 = float(strat_cells[1][key]["num_wt"])
        den1 = float(strat_cells[1][key]["den_wt"])
        rate1 = float(strat_cells[1][key]["rate"])
        agg_num = float(agg["num_wt"])
        agg_den = float(agg["den_wt"])
        agg_rate = float(agg["rate"])
        applied_rate = float(aggregate_level.get(key, agg_rate))
        den_sum = den0 + den1
        recombined = (
            (den0 * rate0 + den1 * rate1) / den_sum if den_sum > 0 else 0.0
        )
        rate_resid = abs(recombined - agg_rate)
        num_resid = abs((num0 + num1) - agg_num)
        den_resid = abs(den_sum - agg_den)
        vs_applied = abs(recombined - applied_rate)
        max_rate_resid = max(max_rate_resid, rate_resid)
        max_num_resid = max(max_num_resid, num_resid)
        max_den_resid = max(max_den_resid, den_resid)
        max_vs_applied = max(max_vs_applied, vs_applied)
        cells[key] = {
            "stratum0_window_below_75": {
                "rate": rate0,
                "num_wt": num0,
                "den_wt": den0,
                "n_events": int(strat_cells[0][key]["n_events"]),
            },
            "stratum1_window_reaches_75plus": {
                "rate": rate1,
                "num_wt": num1,
                "den_wt": den1,
                "n_events": int(strat_cells[1][key]["n_events"]),
            },
            "aggregate_rate": agg_rate,
            "candidate15_applied_rate": applied_rate,
            "recombined_rate": recombined,
            "exposure_weight_stratum1_share": (
                den1 / den_sum if den_sum > 0 else 0.0
            ),
            "abs_residual_recombined_vs_aggregate": rate_resid,
            "abs_residual_recombined_vs_candidate15_applied": vs_applied,
        }
    reconciled = bool(
        max_rate_resid <= 1e-9
        and max_num_resid <= 1e-6
        and max_den_resid <= 1e-6
        and max_vs_applied <= 1e-9
    )
    return {
        "note": (
            "the two support-composition strata recombine to candidate 15's "
            "band aggregate by the exposure-weighted identity (den_0*rate_0 + "
            "den_1*rate_1)/(den_0+den_1) == (num_0+num_1)/(den_0+den_1) == "
            "rate_aggregate. The binary person-level stratum partitions the "
            "married person-year exposure and the widowhood events exactly, so "
            "the aggregate widowhood incidence is preserved while the event "
            "composition matches the reference's observed-window correlation"
        ),
        "n_cells": len(cells),
        "max_abs_residual_recombined_vs_aggregate": max_rate_resid,
        "max_abs_residual_num_wt": max_num_resid,
        "max_abs_residual_den_wt": max_den_resid,
        "max_abs_residual_recombined_vs_candidate15_applied": max_vs_applied,
        "reconciled": reconciled,
        "cells": cells,
    }


# ==========================================================================
# Fitted components (candidate 15's fit, PLUS the stratified widowhood level and
# the per-person support-composition stratum). ``fit_components`` wraps
# :func:`candidate15.fit_components` (so the aggregate widowhood LEVEL, the
# entry-widowed cells, the remarriage / fertility / count machinery and the
# committed betas are byte-identical) and adds the DELTA: the two-stratum
# widowhood level and the observed stratum map.
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
    """Candidate 15's components PLUS the support-composition stratum (DELTA).

    Calls :func:`candidate15.fit_components` for the aggregate untrended seven-
    band surviving-spouse widowhood LEVEL (``base.mortality``, the recombination
    target) and every inherited component, then fits the DELTA: the two-stratum
    widowhood level (:func:`_widowhood_hazard_cells_by_stratum`) and the
    per-person observed support-composition stratum map
    (:func:`support_stratum_map`). Records the exposure-weighted recombination
    identity and attaches the stratum lookup / stratified level for
    :func:`_build_sim_lookups` and :func:`simulate_holdout`.
    """
    base = c15.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )
    # The observed support-composition stratum (split-independent covariate).
    stratum = support_stratum_map(panel, demo)
    base.support_stratum = stratum
    base.support_stratum_series = stratum["stratum_by_person"]

    # The DELTA: the two-stratum widowhood level over the train complement.
    strat_cells = _widowhood_hazard_cells_by_stratum(
        panel, train_ids, base.support_stratum_series
    )
    level_by_stratum = {
        s: {key: float(cell["rate"]) for key, cell in strat_cells[s].items()}
        for s in SUPPORT_STRATA
    }
    base.widow_level_by_stratum = level_by_stratum

    # The aggregate cells (candidate 14's machinery over ALL train rows) supply
    # the num_wt / den_wt for the exposure-weighted recombination identity;
    # base.mortality (candidate 15's applied rate) is the recombination target.
    agg_cells = c14._widowhood_hazard_cells(panel, train_ids)
    recomb = _recombination_identity(strat_cells, agg_cells, base.mortality)

    base.meta["support_stratum_definition"] = stratum["provenance"][
        "definition"
    ]
    base.meta["support_stratum_provenance"] = stratum["provenance"]
    base.meta["mortality_level_new_widowhood_by_stratum"] = {
        str(s): dict(level_by_stratum[s]) for s in SUPPORT_STRATA
    }
    base.meta["support_stratum_recombination"] = recomb
    base.meta["mortality_level_representation_candidate16"] = (
        "surviving-spouse widowhood-incidence hazard keyed (seven-band "
        "WIDOW_BANDS, surviving-spouse sex, observed support-composition "
        "stratum in {0: window below 75, 1: window reaches >= 75}); consumed "
        "by the married ego's OWN (age band, sex, support stratum) WITHOUT any "
        "period-trend multiplier (candidate 15's trend removal inherited). The "
        "two strata recombine to candidate 15's band aggregate by the "
        "exposure-weighted identity (support_stratum_recombination)"
    )
    base.meta["delta_vs_candidate15"] = DELTA_VS_CANDIDATE15
    return base


# ==========================================================================
# Provable byte-identity: reuse candidate 15's EXACT code objects for the
# still-unaffected compute chain (``_draw_moments``, ``score_seed``,
# ``fit_remarriage_age_banded``, ``_remarriage_probs_age_banded``), rebound so
# their global lookups resolve against THIS module (candidate 16). The delta
# adds a per-person conditioning covariate the widowhood hazard must SEE, so
# ``_widow_probs``, ``_build_sim_lookups`` and ``simulate_holdout`` DIVERGE.
# ==========================================================================
def _rebind(fn: types.FunctionType) -> types.FunctionType:
    """Return a function sharing ``fn``'s code object but this module's globals.

    Candidate 15's still-unaffected compute chain is unchanged by the stratum;
    reusing its code objects verbatim -- only redirecting global name resolution
    to candidate 16's module -- makes ``everything else byte-identical`` provable
    at the bytecode level while the reused ``_draw_moments`` / ``score_seed``
    read candidate 16's ``simulate_holdout`` / ``fit_components``.
    """
    return types.FunctionType(
        fn.__code__,
        globals(),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


# ==========================================================================
# THE DELTA (compute): the widowhood incidence lookup, stratified. Re-banded to
# a [band, sex, stratum] ``mort_arr``; the returned rate is the level ONLY (no
# trend, candidate 15's removal inherited), indexed by the married ego's OWN
# (age band, sex) AND observed support-composition stratum.
# ==========================================================================
@dataclass
class _SimLookupsC16:
    mort_arr: np.ndarray  # [widow_band, sex, stratum] surviving-spouse level
    beta_arr: np.ndarray  # [sex] committed NCHS slope (retained, NOT applied)
    rem_arr: np.ndarray  # [age_band, ysd_band, origin, sex]
    fert_arr: np.ndarray  # [age-FERT_AGE_LO, parity_band, decade_idx]
    decade_map: dict[int, int]


def _build_sim_lookups(components: Components) -> _SimLookupsC16:
    """Candidate 15's lookups with the widowhood level gaining a stratum axis.

    ``mort_arr`` is [widow_band, sex, stratum] (the DELTA), built from the two-
    stratum train level (``components.widow_level_by_stratum``); the remarriage
    (candidate 11's 5-band table) and single-year fertility lookups are
    candidate 10/14's, unchanged. ``beta_arr`` is retained (documented) but not
    applied -- candidate 15's trend removal is inherited.
    """
    mort_arr = np.zeros((len(WIDOW_BANDS), 2, len(SUPPORT_STRATA)), np.float64)
    for b, (lo, hi) in enumerate(WIDOW_BANDS):
        band = transitions.band_label(lo, hi)
        for si, sex in enumerate(("female", "male")):
            for st in SUPPORT_STRATA:
                mort_arr[b, si, st] = components.widow_level_by_stratum[
                    st
                ].get(f"{band}|{sex}", 0.0)

    beta = components.meta["mortality_beta_by_sex"]
    beta_arr = np.array([beta["female"], beta["male"]], dtype=np.float64)

    rem_arr = np.zeros(
        (len(REM_AGE_BANDS), len(YSD_BANDS), 2, 2), dtype=np.float64
    )
    for (ab, b, origin, sex), v in components.remarriage.items():
        oi = 0 if origin == "divorced" else 1
        si = 0 if sex == "female" else 1
        rem_arr[ab, b, oi, si] = v

    decades = sorted({d for (_a, _p, d) in components.fertility})
    decade_map = {d: i for i, d in enumerate(decades)}
    n_age = c5.FERT_AGE_HI - c5.FERT_AGE_LO + 1
    fert_arr = np.zeros((n_age, 4, max(len(decades), 1)), dtype=np.float64)
    for (age, pb, d), v in components.fertility.items():
        fert_arr[age - c5.FERT_AGE_LO, pb, decade_map[d]] = v
    return _SimLookupsC16(mort_arr, beta_arr, rem_arr, fert_arr, decade_map)


def _widow_probs(
    ego_age: np.ndarray,
    ego_is_male: np.ndarray,
    spouse_age: np.ndarray,
    spouse_is_male: np.ndarray,
    year: int,
    mort_arr: np.ndarray,
    beta_arr: np.ndarray,
    stratum: np.ndarray,
) -> np.ndarray:
    """Surviving-spouse widowhood incidence, support-stratum conditioned (DELTA).

    ``rate = widow_level(ego_band, ego_sex, ego_support_stratum)`` -- the train
    marriage-history widowhood hazard keyed by the married ego's OWN (age band,
    sex) AND the observed support-composition stratum (``stratum``, 0/1). The
    ``_bands_vec`` clip is candidate 1's (ages below the youngest band clip into
    it). No period-trend multiplier (candidate 15's removal inherited); ``year``
    / ``beta_arr`` are accepted (the committed betas stay documented) but do NOT
    enter the returned rate, so the widowhood lookup is year-invariant.
    ``spouse_age`` / ``spouse_is_male`` are candidate 5's spousal-gap-draw
    arguments, retained so the spousal-gap draw stays byte-identical; they no
    longer enter the level. ``stratum`` is the one delta -- the per-person
    observed support-composition index.
    """
    bands = _bands_vec(
        np.rint(ego_age).astype(np.int64), WIDOW_LOWERS, len(WIDOW_BANDS)
    )
    sidx = ego_is_male.astype(np.int64)
    return mort_arr[bands, sidx, stratum]


def simulate_holdout(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: Components,
    sim_seed: int,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Candidate 12's simulation with candidate 16's support-stratum delta.

    Byte-identical to :func:`candidate12.simulate_holdout` (the reused chain
    through candidate 15) EXCEPT (a) a per-person observed support-composition
    ``stratum_arr`` is assembled once before the year loop from
    ``components.support_stratum`` (an observed covariate; RNG-neutral), and (b)
    the married competing-risk block passes ``stratum_arr[sub]`` to
    :func:`_widow_probs` so the widowhood hazard is stratum-conditioned. The
    per-year uniform block is drawn BEFORE ``_widow_probs`` and keeps its shape,
    so the scored RNG stream is byte-identical to candidate 15 -- only the
    competing-risk widowhood THRESHOLD value moves. Candidate 12's two deltas
    (entry-widowed injection, age-band gap draw) and candidate 9's count add are
    applied unchanged.
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

    # THE DELTA: the per-person observed support-composition stratum (0/1),
    # aligned to ``pid``. An observed covariate (RNG-neutral); the widowhood
    # hazard reads ``stratum_arr[sub]`` in the married block below.
    stratum_arr = _person_stratum_array(components.support_stratum, pid)

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

    rng = np.random.default_rng(sim_seed)
    gap_seed_seq = rng.bit_generator.seed_seq.spawn(1)[0]
    gap_rng = np.random.default_rng(gap_seed_seq)

    gap_band = _gap_band_arrays(components)
    gap_arr = np.zeros(n, dtype=np.float64)
    entry_married = np.nonzero(state == 1)[0]
    if entry_married.size:
        gap_arr[entry_married] = _draw_banded_gaps(
            gap_rng,
            entry_married,
            cur_start[entry_married].astype(np.float64) - by[entry_married],
            is_male,
            gap_band,
        )
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
            if gi.size:
                gap_arr[gi] = _draw_banded_gaps(
                    gap_rng, gi, y - by[gi], is_male, gap_band
                )

        mar = st == 1
        if mar.any():
            sub = idx[mar]
            dur = (y - cur_start[sub]).astype(np.int64)
            p_div = _divorce_probs(dur, order[sub], components.divorce)
            sp_age = age[mar] + gap_arr[sub]
            p_wid = _widow_probs(
                age[mar],
                is_male[sub],
                sp_age,
                opp_is_male[sub],
                y,
                lookups.mort_arr,
                lookups.beta_arr,
                stratum_arr[sub],
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
            p_rm = _remarriage_probs_age_banded(
                age[diss], ysd, origin, is_male[sub], lookups.rem_arr
            )
            rm = u[diss] < p_rm
            gri = sub[rm]
            order[gri] += 1
            cur_start[gri] = y
            state[gri] = 1
            diss_year[gri] = -1
            open_start[gri] = y
            open_order[gri] = order[gri]
            if gri.size:
                gap_arr[gri] = _draw_banded_gaps(
                    gap_rng, gri, y - by[gri], is_male, gap_band
                )

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

    _inject_entry_widowed(sim_panel, components.entry_widowed_cells)

    residual = observed_residual_counts(panel)
    add = (
        sim_panel.attrs["person_id"]
        .map(residual)
        .fillna(0.0)
        .to_numpy(dtype="float64")
    )
    sim_panel.attrs["n_marriages"] = (
        sim_panel.attrs["n_marriages"].to_numpy(dtype="float64") + add
    )
    return sim_panel, sim_births


#: Candidate 15's (== candidate 10's) exact single-draw moment builder and
#: per-seed mean-over-K=20 scorer, rebound to THIS module's globals so they call
#: candidate 16's ``simulate_holdout`` / ``fit_components`` by global name.
_draw_moments = _rebind(c15._draw_moments)
score_seed = _rebind(c15.score_seed)
#: The 5-band remarriage fit / lookup (candidate 15's == candidate 11's code),
#: rebound -- UNCHANGED by candidate 16's delta.
fit_remarriage_age_banded = _rebind(c15.fit_remarriage_age_banded)
_remarriage_probs_age_banded = _rebind(c15._remarriage_probs_age_banded)

#: The reused-code-object contract (must share candidate 15's bytecode).
#: ``simulate_holdout`` and ``_build_sim_lookups`` are NO LONGER here (they
#: moved to the diverged set -- the delta threads a per-person covariate).
REUSED_CODE_OBJECT_NAMES = (
    "_draw_moments",
    "score_seed",
    "fit_remarriage_age_banded",
    "_remarriage_probs_age_banded",
)
#: The RE-IMPLEMENTED functions: ``_widow_probs`` (stratum lookup),
#: ``_build_sim_lookups`` (stratum axis), ``simulate_holdout`` (stratum thread)
#: and ``fit_components`` (the stratified fit). None may share candidate 15's
#: code object.
DIVERGED_CODE_OBJECT_NAMES = (
    "_widow_probs",
    "_build_sim_lookups",
    "simulate_holdout",
    "fit_components",
)


# ==========================================================================
# Diagnostic: the two-strata rate table per band x sex + recombination identity
# ==========================================================================
def _support_stratum_diagnostic(
    components0: Components,
) -> dict[str, Any]:
    """The two-strata widowhood rate table + the recombination identity.

    Read from a seed-0 train fit's stored meta: the per band x sex rate for each
    stratum (0 = observed window below 75, 1 = window reaches >= 75), the
    exposure-weighted recombination back to candidate 15's band aggregate, and
    the max recombination residual (a hard gate at 0.0).
    """
    meta = components0.meta
    recomb = meta["support_stratum_recombination"]
    prov = meta["support_stratum_provenance"]
    rows: list[dict[str, Any]] = []
    for key in sorted(recomb["cells"]):
        cell = recomb["cells"][key]
        band, sex = key.split("|")
        rows.append(
            {
                "band": band,
                "sex": sex,
                "stratum0_window_below_75_rate": cell[
                    "stratum0_window_below_75"
                ]["rate"],
                "stratum1_window_reaches_75plus_rate": cell[
                    "stratum1_window_reaches_75plus"
                ]["rate"],
                "stratum1_exposure_share": cell[
                    "exposure_weight_stratum1_share"
                ],
                "candidate15_aggregate_rate": cell["aggregate_rate"],
                "recombined_rate": cell["recombined_rate"],
                "abs_residual": cell["abs_residual_recombined_vs_aggregate"],
            }
        )
    return {
        "note": (
            "the surviving-spouse widowhood hazard, stratified by the observed "
            "support-composition stratum (0 = observed window below age 75, 1 "
            "= window reaches >= 75), per band x sex. Each stratum is train-"
            "estimated under the existing smoothing (transitions._hazard_by_"
            "band weighted hazard, no add-one); the two strata recombine to "
            "candidate 15's band aggregate by the exposure-weighted identity"
        ),
        "support_stratum_provenance": prov,
        "recombination_identity": recomb,
        "rate_table": rows,
    }


# ==========================================================================
# Diagnostics: 75+ incidence & stock sim/ref vs candidate 15 (the designed lift)
# ==========================================================================
def _elderly_75plus_diagnostic(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Candidate 16 vs candidate 15 on the 75+ incidence and stock cells.

    The delta's designed effect: hold the 75+ widowhood INCIDENCE (the
    recombination identity preserves the band aggregate) while lifting the 75+
    widowed STOCK (candidate 15 sat at ~0.841 of reference, clipping seeds 2 and
    3) by recomposing widowhood events toward long-observed-support persons.
    Both quantities are read from the scored gated cells (side A holdout, K=20
    draw-mean): sim/ref = rbar / rate_a. Candidate 15's committed values come
    from ``runs/gate2_hazard_v15.json``.
    """
    c15art = (
        json.loads(CANDIDATE15_ARTIFACT.read_text())
        if CANDIDATE15_ARTIFACT.exists()
        else None
    )
    by15 = {s["seed"]: s for s in c15art["per_seed"]} if c15art else {}
    out: dict[str, Any] = {
        "note": (
            "75+ widowhood incidence (widowhood.75+|female) and 75+ widowed "
            "stock (share_widowed.75+|female), sim/ref = rbar / rate_a on the "
            "side-A holdout at the K=20 draw-mean. The support-composition "
            "stratum preserves the band-aggregate incidence and recomposes the "
            "widowed stock toward long-observed-support persons. Candidate 15 "
            "values committed in runs/gate2_hazard_v15.json"
        ),
        "cells": {},
    }
    for cell in ELDERLY_75PLUS_CELLS:
        rows: list[dict[str, Any]] = []
        c16_ratios: list[float] = []
        c15_ratios: list[float] = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            rbar = rec["rbar"]
            rate_a = rec["rate_a"]
            c16_ratio = rbar / rate_a if rate_a > 0 else None
            c15rec = by15[s["seed"]]["gated_cells"][cell] if by15 else None
            c15_ratio = (
                c15rec["rbar"] / c15rec["rate_a"]
                if c15rec and c15rec["rate_a"] > 0
                else None
            )
            if c16_ratio is not None:
                c16_ratios.append(c16_ratio)
            if c15_ratio is not None:
                c15_ratios.append(c15_ratio)
            rows.append(
                {
                    "seed": s["seed"],
                    "c16_rbar": rbar,
                    "rate_a": rate_a,
                    "c16_sim_over_ref": c16_ratio,
                    "c15_rbar": c15rec["rbar"] if c15rec else None,
                    "c15_rate_a": c15rec["rate_a"] if c15rec else None,
                    "c15_sim_over_ref": c15_ratio,
                    "c16_score_abs_ln": rec["score"],
                    "c15_score_abs_ln": c15rec["score"] if c15rec else None,
                    "tolerance": rec["tolerance"],
                    "c16_pass": rec["pass"],
                    "c15_pass": c15rec["pass"] if c15rec else None,
                }
            )
        out["cells"][cell] = {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "c16_sim_over_ref_mean": (
                float(np.mean(c16_ratios)) if c16_ratios else None
            ),
            "c15_sim_over_ref_mean": (
                float(np.mean(c15_ratios)) if c15_ratios else None
            ),
            "c16_n_seeds_pass": sum(1 for r in rows if r["c16_pass"]),
            "c15_n_seeds_pass": (
                sum(1 for r in rows if r["c15_pass"]) if by15 else None
            ),
        }
    inc = out["cells"]["widowhood.75+|female"]
    stock = out["cells"]["share_widowed.75+|female"]
    out["summary"] = {
        "incidence_sim_over_ref": {
            "c15_mean": inc["c15_sim_over_ref_mean"],
            "c16_mean": inc["c16_sim_over_ref_mean"],
            "preserved": (
                inc["c16_sim_over_ref_mean"] is not None
                and inc["c15_sim_over_ref_mean"] is not None
                and abs(
                    inc["c16_sim_over_ref_mean"] - inc["c15_sim_over_ref_mean"]
                )
                < 0.05
            ),
        },
        "stock_sim_over_ref": {
            "c15_mean": stock["c15_sim_over_ref_mean"],
            "c16_mean": stock["c16_sim_over_ref_mean"],
            "lifted_toward_reference": (
                stock["c16_sim_over_ref_mean"] is not None
                and stock["c15_sim_over_ref_mean"] is not None
                and stock["c16_sim_over_ref_mean"]
                > stock["c15_sim_over_ref_mean"]
            ),
        },
    }
    return out


def _stock_per_seed_vs_c15(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-seed 75+ widowed-stock ratio (sim/ref) and score, c16 vs c15.

    The binding cell. Reports, per seed, candidate 16's stock sim/ref and gate
    score against candidate 15's committed values, and whether the seed's stock
    cell flipped pass<->fail.
    """
    c15art = json.loads(CANDIDATE15_ARTIFACT.read_text())
    by15 = {s["seed"]: s for s in c15art["per_seed"]}
    rows: list[dict[str, Any]] = []
    for s in per_seed:
        rec = s["gated_cells"][MODAL_CELL]
        c15rec = by15[s["seed"]]["gated_cells"][MODAL_CELL]
        rows.append(
            {
                "seed": s["seed"],
                "c15_sim_over_ref": (
                    c15rec["rbar"] / c15rec["rate_a"]
                    if c15rec["rate_a"] > 0
                    else None
                ),
                "c16_sim_over_ref": (
                    rec["rbar"] / rec["rate_a"] if rec["rate_a"] > 0 else None
                ),
                "c15_score": c15rec["score"],
                "c16_score": rec["score"],
                "tolerance": rec["tolerance"],
                "c15_pass": c15rec["pass"],
                "c16_pass": rec["pass"],
                "delta_score_vs_c15": float(rec["score"] - c15rec["score"]),
                "flipped": bool(rec["pass"] != c15rec["pass"]),
            }
        )
    return {
        "cell": MODAL_CELL,
        "note": (
            "per-seed 75+ female widowed-stock ratio (sim/ref) and gate score "
            "|ln(rbar/rate_a)|, candidate 16 vs candidate 15 (committed "
            "runs/gate2_hazard_v15.json). The binding cell; candidate 15 "
            "failed seeds 2 and 3"
        ),
        "per_seed": rows,
        "c15_n_seeds_pass": sum(1 for r in rows if r["c15_pass"]),
        "c16_n_seeds_pass": sum(1 for r in rows if r["c16_pass"]),
        "seeds_flipped_to_pass": sorted(
            r["seed"] for r in rows if r["c16_pass"] and not r["c15_pass"]
        ),
        "seeds_flipped_to_fail": sorted(
            r["seed"] for r in rows if not r["c16_pass"] and r["c15_pass"]
        ),
    }


# ==========================================================================
# Count-cell margins (the gated risk) + candidate-15 comparison + decider
# ==========================================================================
def _count_cell_margin_diagnostic(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """The count-cell margins the recomposed widowed exposure threatens.

    The stratum recomposes the widowed stock toward long-observed-support
    persons, feeding the widowed-origin remarriage exposure, so the marriage
    counts are a gated RISK (the registered modal failure). Reports candidate
    16's signed ln-tilt and the margin (tolerance - score) per seed against
    candidate 15's committed count scores, and whether each count holds >= 4/5.
    """
    c15art = (
        json.loads(CANDIDATE15_ARTIFACT.read_text())
        if CANDIDATE15_ARTIFACT.exists()
        else None
    )
    by15 = {s["seed"]: s for s in c15art["per_seed"]} if c15art else {}
    out: dict[str, Any] = {
        "design_note": (
            "registration comment 4929419524: the recomposed widowed exposure "
            "(more widows among long-observed-support persons) feeds the "
            "widowed-origin remarriage exposure, so the marriage counts are a "
            "gated risk. The registered modal failure is the count cells "
            "clipping. Margin = tolerance - score (positive = headroom); "
            "movement reported against candidate 15's committed count scores."
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
            c15rec = by15[s["seed"]]["gated_cells"][cell] if by15 else None
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
                    "candidate15_score": (c15rec["score"] if c15rec else None),
                    "candidate15_pass": (c15rec["pass"] if c15rec else None),
                    "delta_score_vs_c15": (
                        float(rec["score"] - c15rec["score"])
                        if c15rec
                        else None
                    ),
                }
            )
        n_pass = sum(r["pass"] for r in rows)
        c15_n_pass = (
            sum(1 for r in rows if r["candidate15_pass"]) if by15 else None
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
            "candidate15_n_seeds_pass": c15_n_pass,
            "held_vs_c15": (
                bool(n_pass >= 4) if c15_n_pass is not None else None
            ),
        }
    both_hold = all(out["cells"][c]["n_seeds_pass"] >= 4 for c in COUNT_CELLS)
    out["count_cells_hold"] = bool(both_hold)
    out["summary"] = (
        "both marriage-count cells hold >= 4/5 despite the recomposed exposure"
        if both_hold
        else "at least one marriage-count cell clipped below 4/5"
    )
    return out


def _incidence_headroom_diagnostic(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """The gated widowhood-incidence cells' headroom (tolerance - score).

    The recombination identity preserves each band's aggregate incidence, so the
    incidence cells are expected to hold near candidate 15's values. Reports
    per-seed score, tolerance and margin for the four gated widowhood-incidence
    cells, and the minimum margin per cell.
    """
    out: dict[str, Any] = {
        "note": (
            "gated widowhood-incidence cells: score |ln(rbar / rate_a)|, "
            "tolerance and margin (tolerance - score; positive = headroom). "
            "The recombination identity preserves the band aggregate, so the "
            "incidence cells hold near candidate 15"
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


def _candidate15_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Movement of the delta targets vs candidate 15 + the aggregate preserved.

    Compares candidate 16's per-seed rbar-scores for the targeted cells against
    candidate 15's committed scores, and confirms the widowhood-level AGGREGATE
    is preserved (the recombination identity): the seed-0 stored recombination
    max residual is pinned.
    """
    c15art = (
        json.loads(CANDIDATE15_ARTIFACT.read_text())
        if CANDIDATE15_ARTIFACT.exists()
        else None
    )
    by15 = {s["seed"]: s for s in c15art["per_seed"]} if c15art else {}

    def move(cell: str) -> dict[str, Any]:
        rows = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            c15rec = by15[s["seed"]]["gated_cells"][cell] if by15 else None
            rows.append(
                {
                    "seed": s["seed"],
                    "tolerance": rec["tolerance"],
                    "c15_score": c15rec["score"] if c15rec else None,
                    "c16_score": rec["score"],
                    "c15_rbar": c15rec["rbar"] if c15rec else None,
                    "c16_rbar": rec["rbar"],
                    "rate_a": rec["rate_a"],
                    "c15_pass": c15rec["pass"] if c15rec else None,
                    "c16_pass": rec["pass"],
                }
            )
        c15_np = sum(1 for r in rows if r["c15_pass"]) if by15 else None
        c16_np = sum(1 for r in rows if r["c16_pass"])
        return {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "c15_n_seeds_pass": c15_np,
            "c16_n_seeds_pass": c16_np,
            "improved": (
                bool(c16_np > c15_np) if c15_np is not None else None
            ),
        }

    recomb = per_seed[0]["component_meta"]["support_stratum_recombination"]
    return {
        "note": (
            "candidate 16 = candidate 15 (comment 4928232089, #107) with "
            "exactly one delta (the surviving-spouse widowhood hazard "
            "additionally conditioned on the observed support-composition "
            "stratum). Scores compared cell-by-cell against candidate 15's "
            "committed run (runs/gate2_hazard_v15.json)."
        ),
        "modal_cell": {MODAL_CELL: move(MODAL_CELL)},
        "count_cells": {c: move(c) for c in COUNT_CELLS},
        "remarriage_gated_cells": {c: move(c) for c in REMARRIAGE_GATED_CELLS},
        "widowhood_incidence_cells": {
            c: move(c) for c in WIDOWHOOD_INCIDENCE_CELLS
        },
        "widowhood_aggregate_preserved": {
            "note": (
                "the two support-composition strata recombine to candidate "
                "15's band-aggregate widowhood rate by the exposure-weighted "
                "identity; the max recombination residual (seed-0 train fit)"
            ),
            "max_abs_residual_recombined_vs_aggregate": recomb[
                "max_abs_residual_recombined_vs_aggregate"
            ],
            "max_abs_residual_recombined_vs_candidate15_applied": recomb[
                "max_abs_residual_recombined_vs_candidate15_applied"
            ],
            "reconciled": recomb["reconciled"],
        },
    }


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal (count cells), secondary (seed-1 stock overshoot),
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
    modal_failed_seeds = sorted(
        {
            seed
            for cell in REGISTERED_MODAL_CELLS
            for seed in fails_by_cell.get(cell, [])
        }
    )
    stock_cell = "share_widowed.75+|female"
    secondary_failed_seeds = sorted(fails_by_cell.get(stock_cell, []))
    modal_materialized = len(modal_failed_seeds) >= 1
    distinct_fail_cells = {
        f["cell"] for f in verdict["all_failing_gated_cells"]
    }

    if gate_pass:
        decider = "none (gate passed)"
    elif n_pass_no_modal >= 4:
        decider = (
            "the registered modal cells (the marriage counts; forgiving them "
            "flips >= 4 seeds to pass)"
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
            "the marriage-count cells (mean_lifetime_marriages|male/female) "
            "clipping from the recomposed widowed exposure; secondary: a stock "
            "overshoot on seed 1 (the high-side split)"
        ),
        "modal_cells": list(REGISTERED_MODAL_CELLS),
        "secondary_cells": list(REGISTERED_SECONDARY_CELLS),
        "modal_failed_seeds": modal_failed_seeds,
        "modal_materialized": modal_materialized,
        "secondary_failed_seeds": secondary_failed_seeds,
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
# The forensics-4 yield ledger, BEFORE (candidate 15) vs AFTER (candidate 16)
# Train-side only (side B); reuses ``gate2_forensics4.stock_ledger`` verbatim,
# so the yield / window-reaches-75 terms are directly comparable to the
# committed candidate-15 forensics-4 numbers.
# ==========================================================================
def _train_yield_seed(
    seed: int,
    data: dict[str, Any],
    support: pd.DataFrame,
    verbose: bool,
) -> dict[str, Any]:
    """Candidate 16's train-side (side B) yield ledger at K=20 draws.

    Mirrors ``gate2_forensics4.compute_seed``'s Q9 path with candidate 16's fit:
    fit on side B, simulate side B at the 20 draw seeds, average the reachable-
    stock ledger (``gate2_forensics4.stock_ledger``). The reference is side B's
    own observed marital panel (deterministic).
    """
    t0 = time.time()
    panel = data["panel"]
    _side_a, side_b = hpanel.split_panel_by_person(
        panel.attrs, "person_id", fraction=0.5, seed=seed
    )
    ids_b = set(int(x) for x in side_b.person_id.unique())
    components = fit_components(
        panel,
        data["demo"],
        data["death_records"],
        data["mh_records"],
        data["birth_records"],
        data["order_map"],
        ids_b,
    )
    ref_ledger = gf4.stock_ledger(panel, ids_b, support)
    sim_ledger_acc: list[dict[str, Any]] = []
    for k in range(N_DRAWS):
        sim_panel, _sim_births = simulate_holdout(
            panel, ids_b, components, DRAW_SEED_BASE + k
        )
        sim_ledger_acc.append(gf4.stock_ledger(sim_panel, ids_b, support))
    sim_ledger_mean = gf4._mean_ledger(sim_ledger_acc)
    elapsed = round(time.time() - t0, 1)
    if verbose:
        print(
            f"  yield seed {seed}: ref_stock "
            f"{ref_ledger['share_widowed_75plus']:.3f} sim "
            f"{sim_ledger_mean['share_widowed_75plus']:.3f} "
            f"(reach {sim_ledger_mean['W_reachable'] / ref_ledger['W_reachable']:.2f}x) "
            f"[{elapsed}s]"
        )
    return {
        "seed": seed,
        "ref_ledger": ref_ledger,
        "sim_ledger_mean": sim_ledger_mean,
        "elapsed_seconds": elapsed,
    }


def _yield_before_after(
    yield_seed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """The forensics-4 yield ledger, BEFORE (candidate 15) vs AFTER (c16).

    "Before" = candidate 15's committed train-side ledger
    (``runs/gate2_forensics4_v1.json`` per-seed ``sim_ledger_mean``). "After" =
    candidate 16's freshly computed train-side ledger. "Reference" = side B's
    own observed ledger (recomputed here; reconciled to forensics 4's committed
    reference). Reports, per forensics-4 onset band, the survival-to-75+ yield
    and the share-of-inflow-with-observed-window-reaching-75, and the aggregate
    75+ widowed-stock ratio -- the term the delta targets.
    """
    f4 = json.loads(FORENSICS4_ARTIFACT.read_text())
    before = {s["seed"]: s for s in f4["per_seed"]}
    band_labels = [
        transitions.band_label(lo, hi) for lo, hi in gf4.LEDGER_ONSET_BANDS
    ]

    def bmean(rows, getter):
        vals = [getter(r) for r in rows]
        return float(np.mean(vals)) if vals else 0.0

    # Aggregate 75+ widowed-stock ratio (sim/ref), before vs after.
    ref_stock = bmean(
        yield_seed_rows, lambda r: r["ref_ledger"]["share_widowed_75plus"]
    )
    after_stock = bmean(
        yield_seed_rows,
        lambda r: r["sim_ledger_mean"]["share_widowed_75plus"],
    )
    before_stock = bmean(
        [before[r["seed"]] for r in yield_seed_rows],
        lambda s: s["sim_ledger_mean"]["share_widowed_75plus"],
    )
    ref_reach = bmean(
        yield_seed_rows, lambda r: r["ref_ledger"]["W_reachable"]
    )
    after_reach = bmean(
        yield_seed_rows, lambda r: r["sim_ledger_mean"]["W_reachable"]
    )
    before_reach = bmean(
        [before[r["seed"]] for r in yield_seed_rows],
        lambda s: s["sim_ledger_mean"]["W_reachable"],
    )
    # Reconcile the recomputed reference to forensics 4's committed reference.
    ref_before = bmean(
        [before[r["seed"]] for r in yield_seed_rows],
        lambda s: s["ref_ledger"]["share_widowed_75plus"],
    )
    ref_reconciliation = abs(ref_stock - ref_before)

    bands: dict[str, Any] = {}
    for b in band_labels:
        ref_yield = bmean(
            yield_seed_rows,
            lambda r, bb=b: r["ref_ledger"]["bands"][bb]["yield_b"],
        )
        after_yield = bmean(
            yield_seed_rows,
            lambda r, bb=b: r["sim_ledger_mean"]["bands"][bb]["yield_b"],
        )
        before_yield = bmean(
            [before[r["seed"]] for r in yield_seed_rows],
            lambda s, bb=b: s["sim_ledger_mean"]["bands"][bb]["yield_b"],
        )
        ref_w75 = bmean(
            yield_seed_rows,
            lambda r, bb=b: r["ref_ledger"]["bands"][bb][
                "share_window_reaches_75"
            ],
        )
        after_w75 = bmean(
            yield_seed_rows,
            lambda r, bb=b: r["sim_ledger_mean"]["bands"][bb][
                "share_window_reaches_75"
            ],
        )
        before_w75 = bmean(
            [before[r["seed"]] for r in yield_seed_rows],
            lambda s, bb=b: s["sim_ledger_mean"]["bands"][bb][
                "share_window_reaches_75"
            ],
        )
        bands[b] = {
            "yield_b": {
                "reference": ref_yield,
                "before_candidate15": before_yield,
                "after_candidate16": after_yield,
                "before_sim_over_ref": (
                    before_yield / ref_yield if ref_yield > 0 else None
                ),
                "after_sim_over_ref": (
                    after_yield / ref_yield if ref_yield > 0 else None
                ),
            },
            "share_window_reaches_75": {
                "reference": ref_w75,
                "before_candidate15": before_w75,
                "after_candidate16": after_w75,
            },
        }
    return {
        "note": (
            "the forensics-4 (#108) reachable-stock yield ledger, BEFORE "
            "(candidate 15, committed runs/gate2_forensics4_v1.json) vs AFTER "
            "(candidate 16, freshly computed), train-side (side B), K=20 "
            "draws. yield_b = realized 75+ widowed person-years per onset "
            "within observed support; share_window_reaches_75 = the share of "
            "each onset band's reachable widowhood inflow whose observed window "
            "reaches age 75. The delta conditions the hazard on exactly this "
            "window covariate, so the after yield should move toward reference"
        ),
        "aggregate_stock_share_75plus": {
            "reference": ref_stock,
            "before_candidate15": before_stock,
            "after_candidate16": after_stock,
            "before_sim_over_ref": (
                before_stock / ref_stock if ref_stock > 0 else None
            ),
            "after_sim_over_ref": (
                after_stock / ref_stock if ref_stock > 0 else None
            ),
        },
        "reachable_weight": {
            "reference": ref_reach,
            "before_candidate15": before_reach,
            "after_candidate16": after_reach,
        },
        "reference_reconciliation_vs_forensics4": {
            "recomputed_reference_stock": ref_stock,
            "committed_forensics4_reference_stock": ref_before,
            "abs_residual": ref_reconciliation,
            "reconciled": bool(ref_reconciliation <= 1e-9),
        },
        "by_onset_band": bands,
        "yield_fix_bands": list(gf4.YIELD_FIX_BANDS),
    }


# ==========================================================================
# Provenance
# ==========================================================================
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins + the candidate-16 schema, c1-c15 and forensics shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for name in (1, 5, 6, 9, 11, 12, 13, 14, 15):
        pins[f"candidate{name}_runner"] = (
            f"scripts/run_gate2_candidate{name}.py"
        )
        pins[f"candidate{name}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{name}.py"
        )
    pins["candidate15_artifact"] = "runs/gate2_hazard_v15.json"
    pins["candidate15_artifact_sha256"] = c1._sha_of_file(CANDIDATE15_ARTIFACT)
    pins["forensics4_runner"] = "scripts/gate2_forensics4.py"
    pins["forensics4_runner_sha256"] = c1._sha_of_file(
        ROOT / "scripts" / "gate2_forensics4.py"
    )
    pins["forensics4_artifact"] = "runs/gate2_forensics4_v1.json"
    pins["forensics4_artifact_sha256"] = c1._sha_of_file(FORENSICS4_ARTIFACT)
    pins["estimator"] = "mean_over_K20_draws (5200 + k, k=0..19)"
    pins["deltas"] = (
        "one delta vs candidate 15: the surviving-spouse widowhood hazard is "
        "additionally conditioned on the observed support-composition stratum "
        "(window reaches age 75, binary), both strata train-estimated per band "
        "x sex under the existing smoothing, recombining to candidate 15's "
        "band aggregate by the exposure-weighted identity. Everything else "
        "byte-identical to candidate 15 (the reused compute chain shares "
        "candidate 15's exact code objects EXCEPT _widow_probs, "
        "_build_sim_lookups and simulate_holdout, which are re-implemented to "
        "thread the per-person stratum, and fit_components, which fits the "
        "stratified level)"
    )
    pins["byte_identity_code_objects"] = {
        name: (getattr(c15, name).__code__ is globals()[name].__code__)
        for name in REUSED_CODE_OBJECT_NAMES
    }
    pins["diverged_code_objects_vs_candidate15"] = {
        name: (getattr(c15, name).__code__ is not globals()[name].__code__)
        for name in DIVERGED_CODE_OBJECT_NAMES
    }
    pins["widowhood_bands_candidate15"] = [list(b) for b in c15.WIDOW_BANDS]
    pins["widowhood_bands_candidate16"] = [list(b) for b in WIDOW_BANDS]
    pins["support_stratum_threshold_age"] = SUPPORT_STRATUM_AGE
    pins["nchs_trend_applied_candidate16"] = False
    return pins


def _model_block() -> dict[str, Any]:
    """The model block, edited for the one candidate-16 delta."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a surviving-"
            "spouse widowhood component (candidate 15 with one delta: the "
            "widowhood hazard additionally conditioned on the observed support-"
            "composition stratum), scored under the amended mean-over-K=20-"
            "draws estimator"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "delta_vs_candidate15": DELTA_VS_CANDIDATE15,
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic spline on age, "
                "knots 20/22/25/30/40, sex, birth-decade cohort -- "
                "BYTE-IDENTICAL to candidate 15 at a shared draw seed"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x order, "
                "add-one smoothed -- estimator BYTE-IDENTICAL to candidate 15"
            ),
            "widowhood": (
                "COMPOSED; surviving-spouse marriage-history widowhood level "
                "(seven bands {18-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+} "
                "x sex, untrended, candidate 15's level in AGGREGATE). THE "
                "DELTA: the hazard is additionally conditioned on the observed "
                "support-composition stratum (whether the person's observed "
                "support window reaches age 75, binary), both strata train-"
                "estimated per band x sex under the existing smoothing "
                "(transitions._hazard_by_band weighted hazard, no add-one). "
                "The two strata recombine to candidate 15's band aggregate by "
                "the exposure-weighted identity, so aggregate incidence is "
                "preserved while event composition matches the reference's "
                "observed-window correlation. No period-trend multiplier "
                "(candidate 15's removal inherited)"
            ),
            "remarriage": (
                "weighted empirical hazard by ego age band "
                "(18-34/35-49/50-64/65-74/75+) x years-since-dissolution band "
                "x origin x sex, add-one smoothed -- the candidate-11 5-band "
                "table, BYTE-IDENTICAL to candidate 15"
            ),
            "fertility": (
                "single-year-of-age triangular-kernel rates within parity "
                "(0/1/2/3+) x birth-decade cohort -- BYTE-IDENTICAL to "
                "candidate 15 (no fertility delta)"
            ),
            "spousal_age_gap": (
                "candidate 12's age-band-conditioned spousal-gap draw "
                "(vestigial; proven inert at candidate 12's grading) -- left "
                "UNTOUCHED per byte-minimality, BYTE-IDENTICAL to candidate 15"
            ),
            "entry_widowed_initial_state": (
                "candidate 12's delta 1 (inherited, unchanged): persons "
                "observed already-widowed at their first observed PSID wave "
                "enter widowed. RNG-neutral, BYTE-IDENTICAL to candidate 15"
            ),
            "lifetime_marriage_count_initial_state": (
                "candidate 9's delta 1 (inherited, unchanged): each holdout "
                "person's simulated lifetime-marriage count initialises at "
                "their OBSERVED residual and accumulates the simulated datable "
                "transitions. An observed initial state; RNG-neutral"
            ),
            "support_composition_stratum": (
                "THE DELTA: a per-person binary observed covariate -- whether "
                "the observed support window end min(last_wave, censor_year) "
                "reaches age 75 (birth_year + 75). Known ex ante from the "
                "observed panel attributes alone (the same observed-data class "
                "as the initial-state fixes); conditions the widowhood hazard "
                "at gate time"
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
            "support_stratum": (
                "the surviving-spouse widowhood hazard is additionally "
                "conditioned on the observed support-composition stratum "
                "(min(last_wave, censor_year) >= birth_year + 75, binary), "
                "both strata train-estimated per band x sex under the existing "
                "transitions._hazard_by_band smoothing (no add-one)"
            ),
            "recombination_identity": (
                "the binary person-level stratum partitions the married "
                "person-year exposure and the widowhood events exactly, so the "
                "two strata recombine to candidate 15's band aggregate by the "
                "exposure-weighted identity (den_0*rate_0 + den_1*rate_1)/"
                "(den_0+den_1) == (num_0+num_1)/(den_0+den_1) == rate_"
                "aggregate; recorded in support_stratum_recombination, "
                "reconciled to 0.0"
            ),
            "byte_identity": (
                "candidate 16 reuses candidate 15's EXACT code objects for "
                "_draw_moments, score_seed, fit_remarriage_age_banded and "
                "_remarriage_probs_age_banded (rebound to this module's "
                "globals); _widow_probs, _build_sim_lookups and "
                "simulate_holdout are RE-IMPLEMENTED (the per-person stratum "
                "threaded) and fit_components wraps candidate 15's fit + the "
                "stratified level (revision_pins.byte_identity_code_objects / "
                "diverged_code_objects_vs_candidate15)"
            ),
            "everything_else": (
                "the seven-band widowhood level bands, the NCHS-trend removal, "
                "the entry-widowed observed initial state, the 5-band "
                "remarriage table, the observed marriage-count initial state, "
                "the first-marriage spline, divorce, the single-year "
                "triangular-kernel fertility, the vestigial spousal-gap "
                "machinery, the competing-risk step, one sequence per person "
                "per draw, and the locked protocol are byte-identical to "
                "candidate 15"
            ),
        },
    }


# ==========================================================================
# Driver
# ==========================================================================
def run(
    verbose: bool = True,
    cache_path: Path | None = None,
    yield_cache_path: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    cache_path = cache_path or DEFAULT_CACHE
    yield_cache_path = yield_cache_path or DEFAULT_YIELD_CACHE
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

    # Preflight: candidate 15 (the base), its fit chain and forensics 4 present.
    for name, path in (
        ("candidate-15", CANDIDATE15_ARTIFACT),
        ("candidate-14", CANDIDATE14_ARTIFACT),
        ("candidate-12", CANDIDATE12_ARTIFACT),
        ("candidate-9", CANDIDATE9_ARTIFACT),
        ("candidate-6", CANDIDATE6_ARTIFACT),
        ("candidate-5", CANDIDATE5_ARTIFACT),
        ("forensics-3", FORENSICS3_ARTIFACT),
        ("forensics-4", FORENSICS4_ARTIFACT),
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

    # Structural delta guard: candidate 16 keeps candidate 15's band table; the
    # delta is the support-composition stratum, not the bands.
    if tuple(WIDOW_BANDS) != tuple(c15.WIDOW_BANDS):
        raise RuntimeError(
            "candidate 16's widowhood bands differ from candidate 15's; the "
            "delta is the support-composition stratum only, not the band table."
        )
    # The delta must be live: c16 _widow_probs is stratum-conditioned (differs
    # between strata when the strata differ) and year-invariant (no trend).
    _mort = np.zeros((len(WIDOW_BANDS), 2, len(SUPPORT_STRATA)), np.float64)
    _mort[:, :, 0] = 0.01
    _mort[:, :, 1] = 0.05
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
    _s0 = np.zeros(3, dtype=np.int64)
    _s1 = np.ones(3, dtype=np.int64)
    p_s0 = _widow_probs(_age, _egom, _sp, _spm, 2020, _mort, _beta, _s0)
    p_s1 = _widow_probs(_age, _egom, _sp, _spm, 2020, _mort, _beta, _s1)
    if np.allclose(p_s0, p_s1):
        raise RuntimeError(
            "candidate 16's _widow_probs does not distinguish the support "
            "strata; the delta is inert."
        )
    p_anchor = _widow_probs(_age, _egom, _sp, _spm, 1995, _mort, _beta, _s1)
    if not np.allclose(p_s1, p_anchor, rtol=0, atol=0):
        raise RuntimeError(
            "candidate 16's _widow_probs is not year-invariant; the trend "
            "removal (inherited from candidate 15) is not honoured."
        )

    # Byte-identity guard: the reused compute chain MUST share candidate 15's
    # exact code objects; the re-implemented functions MUST NOT. Fail fast.
    for name in REUSED_CODE_OBJECT_NAMES:
        if globals()[name].__code__ is not getattr(c15, name).__code__:
            raise RuntimeError(
                f"{name} does not share candidate 15's code object; the "
                "reused-chain byte-identity contract is violated."
            )
        if globals()[name].__globals__ is not globals():
            raise RuntimeError(
                f"{name} is not rebound to candidate 16's globals; it would "
                "read candidate 15's simulate_holdout (no stratum)."
            )
    for name in DIVERGED_CODE_OBJECT_NAMES:
        if globals()[name].__code__ is getattr(c15, name).__code__:
            raise RuntimeError(
                f"{name} shares candidate 15's code object but must be "
                "re-implemented for the candidate-16 delta."
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

    # Hard gate 2: candidate 9's delta-1 count reconciliation (inherited).
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

    # Hard gate 4 (the DELTA's identity): candidate 16's two strata must
    # recombine to candidate 15's band aggregate to 0.0 (seed-0 train fit).
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
    recomb0 = components0.meta["support_stratum_recombination"]
    if verbose:
        print(
            "recombination identity reconciled="
            f"{recomb0['reconciled']} "
            f"(max |recombined - aggregate|="
            f"{recomb0['max_abs_residual_recombined_vs_aggregate']:.2e}, "
            "max |recombined - c15 applied|="
            f"{recomb0['max_abs_residual_recombined_vs_candidate15_applied']:.2e})"
        )
    if not recomb0["reconciled"]:
        raise RuntimeError(
            "candidate 16's support-composition strata do not recombine to "
            "candidate 15's band aggregate (exposure-weighted identity); "
            "refusing to proceed."
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
    comparison = _candidate15_comparison(per_seed)
    entry_counts = _entry_widowed_seed_counts(panel, demo, GATE_SEEDS)
    elderly = _elderly_75plus_diagnostic(per_seed)
    stock_vs_c15 = _stock_per_seed_vs_c15(per_seed)
    support_diag = _support_stratum_diagnostic(components0)

    # The forensics-4 yield ledger, BEFORE (candidate 15) vs AFTER (c16), on the
    # train half (side B). No outer holdout contact -- side A is never simulated
    # here. Cached separately so a crash never re-scores a completed seed.
    ycache = c1._load_cache(yield_cache_path)
    data_bundle = {
        "panel": panel,
        "demo": demo,
        "death_records": death_records,
        "mh_records": mh_records,
        "birth_records": birth_records,
        "order_map": order_map,
    }
    support = gf3.observed_support(demo)
    yield_rows: list[dict[str, Any]] = []
    for seed in GATE_SEEDS:
        key = f"yield_seed_{seed}"
        if key in ycache:
            if verbose:
                print(f"  yield seed {seed}: cached")
            yield_rows.append(ycache[key])
            continue
        row = _train_yield_seed(seed, data_bundle, support, verbose)
        ycache[key] = json.loads(json.dumps(row, default=c1._json_default))
        c1._save_cache(yield_cache_path, ycache)
        yield_rows.append(ycache[key])
    yield_before_after = _yield_before_after(yield_rows)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 16",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate15_registration": CANDIDATE15_REGISTRATION,
        "candidate14_registration": CANDIDATE14_REGISTRATION,
        "candidate12_registration": CANDIDATE12_REGISTRATION,
        "candidate9_registration": CANDIDATE9_REGISTRATION,
        "candidate6_registration": CANDIDATE6_REGISTRATION,
        "candidate5_registration": CANDIDATE5_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "forensics4_registration": FORENSICS4_REGISTRATION,
        "forensics4_diagnostic": FORENSICS4_DIAGNOSTIC,
        "delta_vs_candidate15": DELTA_VS_CANDIDATE15,
        "amended_estimator": (
            "gates.yaml gate_2 amendment 1 (ratified 2026-07-08, flip #97): "
            "per-cell score |ln(rbar / rate_a)| with rbar the mean over K=20 "
            "draws (default_rng(5200 + k), k=0..19) of the cell rate "
            "(inherited from candidate 15, unchanged)"
        ),
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79); amendment 1 flipped "
            "live (#97). Protocol/views/tolerances/schema read at runtime; no "
            "threshold moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.6-0.7",
            "conjunction_estimate": 0.65,
            "pass_path_seeds": [0, 1, 3, 4],
            "modal_failure": (
                "the marriage-count cells clipping from the recomposed widowed "
                "exposure; secondary: a stock overshoot on seed 1 (the high-"
                "side split). Seed 2's completed_fertility.c1970s is an RNG-"
                "isolated split artifact (byte-identical to candidate 15), so "
                "it stays failing regardless -- the pass path is seeds 0,1,3,4"
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
                "the fresh candidate-16 one-shot run registered AFTER the "
                "2026-07-08 ratification (registration 4929419524)"
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
        "support_stratum_diagnostic": support_diag,
        "recombination_identity": recomb0,
        "per_seed": per_seed,
        "seed_conjunction": seed_conjunction,
        "report_only": report_block,
        "count_cell_margins": count_margins,
        "incidence_headroom": incidence_headroom,
        "candidate15_comparison": comparison,
        "elderly_75plus_diagnostic": elderly,
        "stock_per_seed_vs_candidate15": stock_vs_c15,
        "yield_before_after": yield_before_after,
        "modal_failure_materialized": modal,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "registration_pointer": REGISTRATION_POINTER,
            "candidate15_registration": CANDIDATE15_REGISTRATION,
            "candidate15_artifact": "runs/gate2_hazard_v15.json",
            "forensics4_registration": FORENSICS4_REGISTRATION,
            "forensics4_artifact": "runs/gate2_forensics4_v1.json",
            "forensics4_diagnostic": FORENSICS4_DIAGNOSTIC,
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
            f"  {MODAL_CELL}: c15 {mc['c15_n_seeds_pass']}/5 -> "
            f"c16 {mc['c16_n_seeds_pass']}/5"
        )
        esum = elderly["summary"]
        print(
            "  75+ incidence sim/ref: c15 "
            f"{esum['incidence_sim_over_ref']['c15_mean']:.4f} -> c16 "
            f"{esum['incidence_sim_over_ref']['c16_mean']:.4f}; "
            "75+ stock sim/ref: c15 "
            f"{esum['stock_sim_over_ref']['c15_mean']:.4f} -> c16 "
            f"{esum['stock_sim_over_ref']['c16_mean']:.4f}"
        )
        yba = yield_before_after["aggregate_stock_share_75plus"]
        print(
            "  train yield stock sim/ref: before(c15) "
            f"{yba['before_sim_over_ref']:.4f} -> after(c16) "
            f"{yba['after_sim_over_ref']:.4f}"
        )
        for c in COUNT_CELLS:
            b = count_margins["cells"][c]
            print(
                f"  {c}: c15 {b['candidate15_n_seeds_pass']}/5 -> "
                f"c16 {b['n_seeds_pass']}/5 "
                f"(min margin={b['min_margin']:.4f})"
            )
        print(
            "modal (count re-clip) materialized="
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
    parser.add_argument(
        "--yield-cache",
        default=str(DEFAULT_YIELD_CACHE),
        help="Incremental train-side yield-ledger cache path (outside runs/).",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="lbfgs failed to converge")
    artifact = run(
        verbose=True,
        cache_path=Path(args.cache),
        yield_cache_path=Path(args.yield_cache),
    )
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
