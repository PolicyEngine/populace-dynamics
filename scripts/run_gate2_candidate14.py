"""Gate-2 candidate 14 (run 1): candidate 13 + EXACTLY ONE delta, scored under
the amended mean-over-K=20-draws estimator.

The FOURTEENTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42 comment
4927236029 (``SPEC_REGISTRATION``): candidate 13's frozen spec (comment
4925748151, ``scripts/run_gate2_candidate13.py``, merged #104) verbatim EXCEPT
one delta, registered from candidate 13's grading (the pooled 75+ band averages
away the 75-84 vs 85+ mortality gradient). One-shot; no constant moves after the
registration comment; published REGARDLESS of verdict.

The one delta vs candidate 13 (everything else byte-identical)
--------------------------------------------------------------
**The surviving-spouse widowhood hazard's oldest band, 75+, splits into 75-84
and 85+.** Candidate 13's six-band table ``{18-34, 35-44, 45-54, 55-64, 65-74,
75+}`` x sex pools ages 75-120 into a single 75+ rate (~0.056/yr female), so an
85-year-old married ego widows at the same rate as a 76-year-old. Candidate 13's
grading isolated this as the last unpulled lever: forensics 3 measured the 75+
widowhood incidence at 0.91 of reference, and the pooled band averages away the
75-84 vs 85+ hazard gradient (0.031 vs 0.110 in the committed mortality tables)
across a married pool whose age composition differs from the reference's, which
starved the elderly widowed stock (``share_widowed.75+|female`` failing seeds 0
and 3 at 2/5). Candidate 14's full widowhood table is
``{18-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+}`` x sex, train-estimated from
``mh85_23`` spouse-death endings over married person-year exposure with the
EXISTING smoothing convention (``transitions._hazard_by_band``'s weighted
hazard -- num_wt / den_wt, no add-one), the NCHS period-trend multiplier applied
unchanged. Letting 85+ egos widow at their own higher hazard recovers elderly
incidence toward reference; the five inherited bands' rates (18-34 ... 65-74)
are bit-identical to candidate 13 (same events, same exposure, same band edges).

Everything else is byte-identical to candidate 13 -- the entry-widowed observed
initial state (candidate 12's delta 1), the 5-band remarriage current-age table
(candidate 11), the observed undatable-marriage lifetime-count initial state
(candidate 9's delta 1), the single-year triangular-kernel fertility, the RNG,
the K=20 mean-of-draws protocol, and ``fresh_run_artifact_schema`` conformance
(per-draw per-cell rates [20, 46, 5]; undefined draw invalidates; report-only
dispersion). The vestigial spousal-age-gap machinery (candidate 12's delta 2,
proven inert) is left UNTOUCHED per byte-minimality. Runner
``scripts/run_gate2_candidate14.py``, artifact ``runs/gate2_hazard_v14.json``.

Provable byte-identity (code-object reuse)
------------------------------------------
Because the delta changes ONLY the widowhood age-band table (the constants
``WIDOW_BANDS`` / ``WIDOW_LOWERS`` and its train fit), and candidate 13's
compute chain is band-count-agnostic (its widowhood lookup / lookup-builder
loops and array shapes derive from ``len(WIDOW_BANDS)``), candidate 14 REUSES
candidate 13's EXACT code objects for the band-dependent compute
(``_widow_probs``, ``_build_sim_lookups``, ``simulate_holdout``,
``_draw_moments``, ``score_seed``, ``fit_remarriage_age_banded``,
``_remarriage_probs_age_banded`` -- themselves candidate 12's, threaded
through), rebound (:func:`_rebind`) to resolve their globals against THIS module
-- so the byte-identical simulation calls candidate 14's seven-band widowhood
table (the one delta) while every other statement is candidate 13's, guaranteed
at the bytecode level (``candidate14.simulate_holdout.__code__ is
candidate13.simulate_holdout.__code__``). Only ``fit_components`` is
re-implemented (to install the seven-band level and emit the delta's
provenance); the three fit helpers ``_widowhood_hazard_cells`` /
``fit_widowhood_level`` / ``_widow_level_diag`` re-band candidate 6's
construction over the seven bands (the five inherited bands' num_wt / den_wt /
rate are bit-identical to candidate 13, the pooled 75+ band splits in two).

Designed effect (registration)
------------------------------
The split recovers the elderly widowhood incidence toward reference (the 0.91
shortfall), lifting the 75+ stock where seeds 0 and 3 need +0.002 and +0.019 ln;
seeds 1 and 4 gain stock margin in the same direction. Counts are insulated
(elderly widows remarry at the near-zero split rates from candidate 11); the
gated 45+ incidence cells have ln(1.5)-scale headroom for a sub-10% rate move.
``P(pass) ~= 0.6-0.7``; pass path 0, 1, 3, 4 (seed 2 keeps failing the fertility
tilt). Modal failure if it fails: seed 3's stock clearing only partially;
secondary: thin 85+ male cells misfiring under the split (add-one smoothing at
low exposure -- the candidate-2 lesson bounds this: the smoothing convention
operates at slice-weight scale, disclosed in the artifact).

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

    .venv/bin/python scripts/run_gate2_candidate14.py
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

# Candidate 13 supplies the machinery this build deltas ONCE: its compute chain
# (the six-band surviving-spouse widowhood table over candidate 12's
# entry-widowed initial state + age-band gap over candidate 11's 5-band
# remarriage over candidate 8's fit and candidate 9's delta 1), the fresh-run
# artifact-schema blocks, and -- via its imports -- candidate 1's precheck /
# verdict assembly and candidate 8's vectorised simulation helpers. Only the
# surviving-spouse widowhood age-band table changes (6 bands -> 7 bands: the
# pooled 75+ band splits into 75-84 and 85+).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate5 as c5  # noqa: E402
import run_gate2_candidate6 as c6  # noqa: E402
import run_gate2_candidate13 as c13  # noqa: E402

# Candidates 4/8/9/10/11/12 in the chain are import-bound THROUGH candidate 13
# (its ``simulate_holdout`` / gap machinery / delta-1 helpers / schema blocks
# are candidate 14's, rebound), so candidate 14 needs candidate 1 (precheck /
# verdict), candidate 5 (committed NCHS references), candidate 6 (the trend
# anchor + committed-beta guard) and candidate 13 directly; candidate 12's
# provenance (its registration, the four-band table) is threaded through
# candidate 13.
from populace_dynamics.data import marriage, transitions  # noqa: E402

# ``hpanel`` is referenced by candidate 13's ``score_seed`` code object, which
# candidate 14 reuses rebound to THIS module's globals (see ``_rebind`` below);
# it must stay a module global even though candidate 14's own source never
# names it (F401 is silenced, not the runtime need).
from populace_dynamics.harness import panel as hpanel  # noqa: E402, F401

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v14.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
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
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v14"
RUN_NAME = "gate2_hazard_v14"

#: This run's frozen-spec registration (issue #42, comment 4927236029).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4927236029"
)
#: The bare registration comment id, pinned for the artifact pointer.
REGISTRATION_POINTER = "4927236029"
#: The candidate-13 spec this build deltas ONCE (comment 4925748151, #104).
CANDIDATE13_REGISTRATION = c13.SPEC_REGISTRATION
#: The registration chain candidate 13 carried, threaded through for provenance.
CANDIDATE12_REGISTRATION = c13.CANDIDATE12_REGISTRATION
CANDIDATE11_REGISTRATION = c13.CANDIDATE11_REGISTRATION
CANDIDATE10_REGISTRATION = c13.CANDIDATE10_REGISTRATION
CANDIDATE9_REGISTRATION = c13.CANDIDATE9_REGISTRATION
CANDIDATE8_REGISTRATION = c13.CANDIDATE8_REGISTRATION
CANDIDATE6_REGISTRATION = c13.CANDIDATE6_REGISTRATION
CANDIDATE5_REGISTRATION = c13.CANDIDATE5_REGISTRATION
CANDIDATE1_REGISTRATION = c13.CANDIDATE1_REGISTRATION
#: Candidate 13's grading (comment 4927236029) located the pooled-75+ average.
CANDIDATE13_GRADING_DIAGNOSTIC = (
    "issue #42 comment 4927236029 (candidate 13 grading: FAIL 2/5; the pooled "
    "75+ surviving-spouse band averages ages 75-120, holding the 75+ widowhood "
    "incidence at 0.91 of reference -- forensics 3 -- and averaging away the "
    "75-84 vs 85+ mortality gradient (0.031 vs 0.110) across a married pool "
    "whose age composition differs from the reference's, starving the elderly "
    "widowed stock so share_widowed.75+|female failed seeds 0 and 3)"
)

# --- Amended-estimator draw stream (inherited from candidate 13, unchanged). -
#: The amended 20-draw stream base: draw k uses default_rng(5200 + k), the
#: committed forensics convention (gates.yaml gate_2, amendment 1).
DRAW_SEED_BASE = c13.DRAW_SEED_BASE  # 5200
N_DRAWS = c13.N_DRAWS  # 20

# --------------------------------------------------------------------------
# Frozen dials + pure helpers, reused (byte-identical; import-bound).
# --------------------------------------------------------------------------
GATE_SEEDS = c1.GATE_SEEDS
SIM_SEED_BASE = c1.SIM_SEED_BASE  # 4200 (single-draw stream; provenance only)
EXACT_ATOL = c1.EXACT_ATOL
Components = c1.Components

YSD_BANDS = c1.YSD_BANDS
YSD_LOWERS = c1.YSD_LOWERS
_bands_vec = c1._bands_vec

#: Candidate 6's period-trend anchor year (1995.0), needed by the rebound
#: ``_widow_probs`` code object resolving its globals against THIS module.
TREND_ANCHOR_YEAR = c6.TREND_ANCHOR_YEAR

# Candidate 8's simulation helpers (import-bound via candidate 13, unchanged).
_STATE = c13._STATE
_STATE_ABSORB = c13._STATE_ABSORB
_ASFR_LO = c13._ASFR_LO
_ASFR_HI = c13._ASFR_HI
_assemble_sim_panel = c13._assemble_sim_panel
_divorce_probs = c13._divorce_probs
_fertility_probs_single = c13._fertility_probs_single

# Candidate 9's observed residual + reconciliation (delta 1 of candidate 9),
# inherited via candidate 13, reused byte-for-byte.
observed_residual_counts = c13.observed_residual_counts
_delta1_reconciliation = c13._delta1_reconciliation

#: Candidate 10's simulation-lookup container (band-agnostic dataclass; its
#: ``mort_arr`` is shaped by ``_build_sim_lookups`` from ``len(WIDOW_BANDS)``,
#: so it holds candidate 14's seven-band table unchanged). Reused as-is.
_SimLookupsC10 = c13._SimLookupsC10

# --- The 5-band remarriage table (candidate 11's delta; UNCHANGED here). -----
REM_AGE_BANDS = c13.REM_AGE_BANDS
REM_AGE_LOWERS = c13.REM_AGE_LOWERS
_REM_AGE_LABEL = c13._REM_AGE_LABEL

# --- The vestigial spousal-age-gap machinery (candidate 12's delta 2, proven
# --- INERT; left UNTOUCHED per byte-minimality; import-bound unchanged). ------
GAP_AGE_BANDS = c13.GAP_AGE_BANDS
GAP_AGE_LOWERS = c13.GAP_AGE_LOWERS
_GAP_AGE_LABEL = c13._GAP_AGE_LABEL
FALLBACK_MIN_WEIGHTED_COUPLES = c13.FALLBACK_MIN_WEIGHTED_COUPLES
spousal_gap_distribution_by_band = c13.spousal_gap_distribution_by_band
c11_pooled_gap = c13.c11_pooled_gap
_fallback_group = c13._fallback_group
_gap_band_arrays = c13._gap_band_arrays
_draw_banded_gaps = c13._draw_banded_gaps

# --- Candidate 12's entry-widowed observed initial state (delta 1; UNCHANGED,
# --- import-bound so the reused ``simulate_holdout`` injects it unchanged). ---
observed_support = c13.observed_support
entry_widowed_carried_cells = c13.entry_widowed_carried_cells
_entry_widowed_seed_counts = c13._entry_widowed_seed_counts
_entry_widowed_reconciliation = c13._entry_widowed_reconciliation
_inject_entry_widowed = c13._inject_entry_widowed
_widowed_share_by_age = c13._widowed_share_by_age

# Fresh-run artifact-schema blocks are band-independent (they operate on the
# scored per-seed dicts); import-bound via candidate 13 (identical N_DRAWS /
# DRAW_SEED_BASE), so the [20, 46, 5] cube, the undefined-draw rule and the
# report-only dispersion are candidate 10's exact assembly.
_per_draw_per_cell_rates_block = c13._per_draw_per_cell_rates_block
_undefined_draw_block = c13._undefined_draw_block
_per_draw_dispersion_block = c13._per_draw_dispersion_block

# --------------------------------------------------------------------------
# THE ONE DELTA vs candidate 13 (registration comment 4927236029)
# --------------------------------------------------------------------------
#: Candidate 13's widowhood table pools ages 75-120 into a single 75+ band.
#: Candidate 14 splits that oldest band into 75-84 and 85+, so the full
#: surviving-spouse widowhood table is banded 18-34 / 35-44 / 45-54 / 55-64 /
#: 65-74 / 75-84 / 85+ x sex. The five inherited bands (18-34 ... 65-74) are
#: bit-identical to candidate 13; only the pooled 75+ band splits in two.
WIDOW_BANDS: tuple[tuple[int, int], ...] = (
    (18, 34),
    (35, 44),
    (45, 54),
    (55, 64),
    (65, 74),
    (75, 84),
    (85, 120),
)
WIDOW_LOWERS = np.array([lo for lo, _ in WIDOW_BANDS], dtype=np.int64)
_WIDOW_BAND_LABEL = {
    0: "18-34",
    1: "35-44",
    2: "45-54",
    3: "55-64",
    4: "65-74",
    5: "75-84",
    6: "85+",
}
#: The two bands candidate 14 splits candidate 13's pooled 75+ band into.
SPLIT_WIDOW_BAND_INDICES = (5, 6)
#: The candidate-13 pooled band the split replaces (its rate is what 75-84 and
#: 85+ egos shared under candidate 13).
POOLED_75PLUS_BAND = (75, 120)

#: The single named delta (registration comment 4927236029).
DELTA_VS_CANDIDATE13 = (
    "EXACTLY ONE delta vs candidate 13 (comment 4925748151, merged #104): the "
    "surviving-spouse widowhood hazard's oldest band, 75+, splits into 75-84 "
    "and 85+. Full structure: {18-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+} "
    "x sex, train-estimated from mh85_23 spouse-death endings over married "
    "person-year exposure with the EXISTING smoothing convention "
    "(transitions._hazard_by_band weighted hazard num_wt/den_wt -- the gate "
    "reference's own machinery, no add-one), the NCHS period-trend multiplier "
    "applied unchanged; no other change. Registered from candidate 13's "
    "grading: the pooled 75+ band averages ages 75-120 into one rate, holding "
    "the 75+ widowhood incidence at 0.91 of reference (forensics 3) and "
    "averaging away the 75-84 vs 85+ mortality gradient (0.031 vs 0.110 in the "
    "committed tables) across a married pool whose age composition differs "
    "from the reference's, which starved the elderly widowed stock "
    "(share_widowed.75+|female failed seeds 0 and 3). The split lets 85+ egos "
    "widow at their own higher hazard, recovering elderly incidence and "
    "lifting the 75+ stock; the five inherited bands (18-34 ... 65-74) are "
    "bit-identical to candidate 13. Everything else -- the entry-widowed "
    "observed initial state (candidate 12's delta 1), the 5-band remarriage "
    "current-age table, the observed undatable-marriage lifetime-count initial "
    "state (candidate 9's delta 1), the single-year triangular-kernel "
    "fertility, the RNG, the K=20 mean-of-draws protocol, the fresh-run "
    "artifact schema, and the vestigial spousal-age-gap machinery (candidate "
    "12's delta 2, left untouched per byte-minimality) -- is byte-identical to "
    "candidate 13"
)

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate14_run1_cache.json"
)

#: The marriage-count cells (insulated by construction -- the split touches only
#: the oldest widowhood band, whose new widows remarry at the near-zero
#: candidate-11 elderly rates; the check is that they HOLD candidate 13's 5/5).
COUNT_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
)
#: The gated remarriage cells the delta can reshape via the elderly widowed
#: trajectory (new 85+ widows enter the widowed-origin remarriage exposure).
REMARRIAGE_GATED_CELLS = (
    "remarriage.after_divorce",
    "remarriage.ysd0-4",
    "remarriage.ysd5-9",
    "remarriage.ysd10+",
)
#: The elderly-widow-stock cell the delta is designed to lift (candidate 13's
#: two failing seeds, 0 and 3, are on this cell).
MODAL_CELL = "share_widowed.75+|female"
#: The registered modal failure (comment 4927236029): seed 3's stock clearing
#: only partially (the incidence recovery not fully reaching the stock).
REGISTERED_MODAL_CELLS = ("share_widowed.75+|female",)
#: The gated widowhood-incidence cells; the 75+ female cell is where the split
#: recovers the elderly incidence toward reference (the 0.91 shortfall).
WIDOWHOOD_INCIDENCE_CELLS = (
    "widowhood.45-64|female",
    "widowhood.65-74|female",
    "widowhood.75+|female",
    "widowhood.45+|male",
)
#: The 75+ cells the delta most directly touches: the widowhood incidence (must
#: recover) and the widowed stock (must lift where candidate 13 clipped).
ELDERLY_75PLUS_CELLS = (
    "widowhood.75+|female",
    "share_widowed.75+|female",
)
#: The cells the delta most directly touches: the elderly incidence/stock, both
#: marriage counts (insulated), and the remarriage flows.
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
# THE DELTA fit: seven-band surviving-spouse widowhood hazard (train)
# Re-banded (not rebound) because ``transitions.hazard_cells`` bands widowhood
# on the module-level ``WIDOWHOOD_AGE_BANDS`` (four bands, the committed
# reference); candidate 14 calls the same ``transitions._hazard_by_band``
# machinery directly over its own seven bands, so the five inherited bands are
# bit-identical and the pooled 75+ band splits in two, WITHOUT touching the
# shared reference moment.
# ==========================================================================
def _widowhood_hazard_cells(
    panel: transitions.MaritalPanel, train_ids: set[int]
) -> dict[str, dict[str, float]]:
    """Train marriage-history widowhood hazard cells over the seven bands.

    Candidate 6's construction (:func:`candidate6._widowhood_hazard_cells`) --
    weighted spouse-death marriage endings over weighted married person-year
    exposure, the gate reference's own :func:`transitions._hazard_by_band`
    machinery with the existing smoothing convention (rate = num_wt / den_wt,
    0 where the denominator is empty; NO add-one) -- restricted to the train
    complement and banded on candidate 14's :data:`WIDOW_BANDS` (the seven-band
    table) instead of candidate 13's six-band table. The band boundaries below
    75 are identical, so those five bands' num_wt / den_wt / rate are
    bit-identical to candidate 13; the pooled 75+ band splits into 75-84 and
    85+ (each its own num_wt / den_wt / rate). Returned keyed ``"band|sex"``
    (the ``widowhood.`` prefix stripped).
    """
    py = panel.person_years
    ev = panel.events
    if train_ids is not None:
        py = py[py["person_id"].isin(train_ids)]
        ev = ev[ev["person_id"].isin(train_ids)]
    cells = transitions._hazard_by_band(
        ev[ev["transition"] == "widowhood"],
        py[py["marital_state"] == "married"],
        "age",
        WIDOW_BANDS,
        prefix="widowhood",
        by_sex=True,
        weighted=True,
    )
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

    ``{band|sex: rate}`` over :data:`WIDOW_BANDS` x surviving-spouse sex, the
    marriage-history widowhood hazard on the train complement -- candidate 6's
    level source re-banded onto the seven-band table.
    """
    return {
        key: float(cell["rate"])
        for key, cell in _widowhood_hazard_cells(panel, train_ids).items()
    }


def _widow_level_diag(
    panel: transitions.MaritalPanel, train_ids: set[int]
) -> dict[str, Any]:
    """Provenance of the seven-band surviving-spouse widowhood LEVEL table."""
    cells = _widowhood_hazard_cells(panel, train_ids)
    return {
        "source": (
            "train marriage-history widowhood endings (mh85_23 how-ended = "
            "spouse death) over married person-year exposure, by age band x "
            "sex of the surviving spouse"
        ),
        "construction": (
            "transitions._hazard_by_band(weighted=True) widowhood cells over "
            "candidate 14's seven-band WIDOW_BANDS restricted to the seed's "
            "train complement -- the gate reference's own machinery, re-banded "
            "(the existing smoothing convention: rate = num_wt/den_wt, no "
            "add-one)"
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


def _widowhood_split_band_table(
    new_level: dict[str, float],
    c13_level_6band: dict[str, float],
    diag_cells: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Old (candidate 13, pooled 75+) vs new (candidate 14, split) table.

    For the two split bands (75-84, 85+) the "candidate-13 applied rate" is the
    pooled 75+ rate their egos shared; for the five inherited bands candidate 13
    and candidate 14 are bit-identical. Carries the split bands' exposures
    (num_wt = weighted spouse-death events, den_wt = weighted married
    person-years, n_events) so the reallocation is auditable.
    """
    pooled_label = transitions.band_label(*POOLED_75PLUS_BAND)  # "75+"
    cells: dict[str, Any] = {}
    for b, (lo, hi) in enumerate(WIDOW_BANDS):
        band = transitions.band_label(lo, hi)
        for sex in ("female", "male"):
            key = f"{band}|{sex}"
            c14_rate = float(new_level[key])
            if b in SPLIT_WIDOW_BAND_INDICES:
                applied = float(c13_level_6band[f"{pooled_label}|{sex}"])
                dcell = diag_cells[key]
                cells[key] = {
                    "split_band": True,
                    "candidate13_pooled_rate": applied,
                    "candidate13_pooled_via": (
                        f"shared candidate 13's pooled {pooled_label} band "
                        "(ages 75-120)"
                    ),
                    "candidate14_rate": c14_rate,
                    "ratio_c14_over_c13_pooled": (
                        c14_rate / applied if applied > 0 else None
                    ),
                    "num_wt": float(dcell["num_wt"]),
                    "den_wt": float(dcell["den_wt"]),
                    "n_events": int(dcell["n_events"]),
                }
            else:
                c13_rate = float(c13_level_6band[key])
                cells[key] = {
                    "split_band": False,
                    "candidate13_rate": c13_rate,
                    "candidate14_rate": c14_rate,
                    "bit_identical": abs(c13_rate - c14_rate) <= 1e-12,
                }
    return {
        "note": (
            "surviving-spouse widowhood LEVEL by band x sex: candidate 13's "
            "six-band table (ages 75-120 pooled into one 75+ rate) vs "
            "candidate 14's seven-band table. The pooled 75+ band splits into "
            "75-84 and 85+ (each its own train rate + exposure); the five "
            "inherited bands (18-34 ... 65-74) are bit-identical"
        ),
        "bands": [list(b) for b in WIDOW_BANDS],
        "pooled_band": list(POOLED_75PLUS_BAND),
        "pooled_label": pooled_label,
        "cells": cells,
    }


# ==========================================================================
# Fitted components (candidate 13's, with the widowhood LEVEL re-banded)
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
    """Candidate 13's components with the widowhood LEVEL re-banded (the DELTA).

    Starts from :func:`candidate13.fit_components` -- so first marriage,
    divorce, the 5-band remarriage table, the single-year triangular-kernel
    fertility, the committed NCHS betas, the entry-widowed carried cells
    (candidate 12's delta 1), the observed marriage-count initial state
    (candidate 9's delta 1) and the age-band spousal-gap distribution
    (candidate 12's vestigial delta 2) are byte-identical to candidate 13 by
    construction, and ``base.mortality`` is candidate 13's six-band
    surviving-spouse widowhood level. Then THE ONE DELTA replaces
    ``base.mortality`` with candidate 14's seven-band table
    (:func:`fit_widowhood_level`), retaining candidate 13's six-band level
    under a provenance key for the old-vs-new-by-band comparison. The NCHS
    period-trend beta is untouched (candidate 5's committed value, applied
    unchanged).
    """
    base = c13.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )

    # THE DELTA: re-band the surviving-spouse widowhood LEVEL onto seven bands.
    # ``base.mortality`` at this point is candidate 13's six-band table.
    c13_widow_level_6band = dict(base.mortality)
    widow_level = fit_widowhood_level(panel, train_ids)
    base.mortality = widow_level
    diag = _widow_level_diag(panel, train_ids)

    base.meta["mortality_level_candidate13_6band"] = c13_widow_level_6band
    base.meta["mortality_level_new_widowhood"] = dict(widow_level)
    base.meta["mortality_level_bands"] = [list(b) for b in WIDOW_BANDS]
    base.meta["mortality_cells"] = len(widow_level)  # 7 bands x 2 sexes = 14
    base.meta["mortality_level_diagnostics"] = diag
    base.meta["widowhood_split_band_table"] = _widowhood_split_band_table(
        widow_level, c13_widow_level_6band, diag["cells"]
    )
    base.meta["mortality_level_source"] = (
        "train marriage-history widowhood endings (mh85_23 how-ended = spouse "
        "death) over married person-year exposure, by age band x sex of the "
        "surviving spouse -- transitions._hazard_by_band widowhood "
        "construction (the gate reference's own machinery, the existing "
        "smoothing convention), re-banded onto candidate 14's seven-band table "
        "{18-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+} (candidate 13 pooled "
        "ages 75-120 into one 75+ band)"
    )
    base.meta["mortality_level_representation"] = (
        "surviving-spouse widowhood-incidence hazard, keyed (candidate 14 "
        "seven-band WIDOW_BANDS, surviving-spouse sex); consumed by the "
        "married ego's OWN (age band, sex) with the NCHS period-trend "
        "multiplier exp(beta_sex * (year - 1995)) applied unchanged"
    )
    base.meta["delta_vs_candidate13"] = DELTA_VS_CANDIDATE13
    return base


# ==========================================================================
# Provable byte-identity: reuse candidate 13's EXACT code objects, rebound so
# their global lookups resolve against THIS module (candidate 14). The bytecode
# is candidate 13's (== candidate 12's, threaded through); the only observable
# change is that ``WIDOW_BANDS`` (and the widowhood lookup / lookup-builder
# keyed on it) are candidate 14's seven-band table -- the one delta. Only
# ``fit_components`` is re-implemented (above).
# ==========================================================================
def _rebind(fn: types.FunctionType) -> types.FunctionType:
    """Return a function sharing ``fn``'s code object but this module's globals.

    Candidate 13's widowhood-band-dependent compute is band-count-agnostic (the
    ``_build_sim_lookups`` ``mort_arr`` shape and the ``_widow_probs`` clip both
    derive from ``len(WIDOW_BANDS)``). Reusing its code object verbatim -- only
    redirecting global name resolution to candidate 14's module -- makes
    ``everything else byte-identical`` provable at the bytecode level
    (``candidate14.f.__code__ is candidate13.f.__code__``) while the reused code
    reads candidate 14's seven-band widowhood table.
    """
    return types.FunctionType(
        fn.__code__,
        globals(),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


#: THE DELTA's lookup: candidate 13's (== candidate 6's) exact widowhood
#: incidence x NCHS-trend code, reading candidate 14's seven-band WIDOW_BANDS /
#: WIDOW_LOWERS -- so ages 85+ now index the 85+ row instead of pooling into
#: the shared 75+ rate.
_widow_probs = _rebind(c13._widow_probs)
#: Candidate 13's (== candidate 10's) exact lookup builder; ``mort_arr`` gains
#: candidate 14's split 85+ row automatically (shape from ``len(WIDOW_BANDS)``).
_build_sim_lookups = _rebind(c13._build_sim_lookups)
#: Candidate 13's EXACT vectorised annual simulation (both of candidate 12's
#: deltas -- the entry-widowed injection and the age-band gap draw -- inline and
#: byte-for-byte). It calls ``_widow_probs`` / ``_build_sim_lookups`` (candidate
#: 14's seven-band, above); every other statement is candidate 13's.
simulate_holdout = _rebind(c13.simulate_holdout)
#: Candidate 13's (== candidate 11's == candidate 10's) exact single-draw moment
#: builder and per-seed mean-over-K=20 scorer. Rebound to THIS module's globals,
#: they call candidate 14's ``simulate_holdout`` / ``fit_components``.
_draw_moments = _rebind(c13._draw_moments)
score_seed = _rebind(c13.score_seed)
#: The 5-band remarriage fit / lookup (candidate 13's == candidate 11's code),
#: rebound -- UNCHANGED by candidate 14's delta (kept for the attestation).
fit_remarriage_age_banded = _rebind(c13.fit_remarriage_age_banded)
_remarriage_probs_age_banded = _rebind(c13._remarriage_probs_age_banded)

#: The reused-code-object contract (must share candidate 13's bytecode).
REUSED_CODE_OBJECT_NAMES = (
    "_widow_probs",
    "_build_sim_lookups",
    "simulate_holdout",
    "_draw_moments",
    "score_seed",
    "fit_remarriage_age_banded",
    "_remarriage_probs_age_banded",
)
#: The RE-IMPLEMENTED function (the delta installs the seven-band level and
#: emits its provenance): it must NOT share candidate 13's code object.
DIVERGED_CODE_OBJECT_NAMES = ("fit_components",)


# ==========================================================================
# Diagnostics: 75+ incidence & stock sim/ref vs candidate 13 (the designed
# recovery)
# ==========================================================================
def _elderly_75plus_diagnostic(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Candidate 14 vs candidate 13 on the 75+ incidence and stock cells.

    The split's designed effect: raise the 75+ widowhood INCIDENCE toward
    reference (candidate 13 sat at ~0.91-0.93 of reference; forensics 3
    measured 0.91) and thereby lift the 75+ widowed STOCK (candidate 13 sat at
    ~0.84 of reference, clipping seeds 0 and 3). Both quantities are read from
    the scored gated cells (side A holdout, K=20 draw-mean): sim/ref =
    rbar / rate_a. Candidate 13's committed values come from
    ``runs/gate2_hazard_v13.json``.
    """
    c13art = (
        json.loads(CANDIDATE13_ARTIFACT.read_text())
        if CANDIDATE13_ARTIFACT.exists()
        else None
    )
    by13 = {s["seed"]: s for s in c13art["per_seed"]} if c13art else {}

    out: dict[str, Any] = {
        "note": (
            "75+ widowhood incidence (widowhood.75+|female) and 75+ widowed "
            "stock (share_widowed.75+|female), sim/ref = rbar / rate_a on the "
            "side-A holdout at the K=20 draw-mean. The split recovers the "
            "elderly incidence (candidate 13 ~0.91-0.93 of reference; "
            "forensics 3 0.91) toward reference and lifts the stock (candidate "
            "13 ~0.84, clipping seeds 0 and 3). Candidate 13 values committed "
            "in runs/gate2_hazard_v13.json"
        ),
        "forensics3_incidence_shortfall": 0.91,
        "cells": {},
    }
    for cell in ELDERLY_75PLUS_CELLS:
        rows: list[dict[str, Any]] = []
        c14_ratios: list[float] = []
        c13_ratios: list[float] = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            rbar = rec["rbar"]
            rate_a = rec["rate_a"]
            c14_ratio = rbar / rate_a if rate_a > 0 else None
            c13rec = by13[s["seed"]]["gated_cells"][cell] if by13 else None
            c13_ratio = (
                c13rec["rbar"] / c13rec["rate_a"]
                if c13rec and c13rec["rate_a"] > 0
                else None
            )
            if c14_ratio is not None:
                c14_ratios.append(c14_ratio)
            if c13_ratio is not None:
                c13_ratios.append(c13_ratio)
            rows.append(
                {
                    "seed": s["seed"],
                    "c14_rbar": rbar,
                    "rate_a": rate_a,
                    "c14_sim_over_ref": c14_ratio,
                    "c13_rbar": c13rec["rbar"] if c13rec else None,
                    "c13_rate_a": c13rec["rate_a"] if c13rec else None,
                    "c13_sim_over_ref": c13_ratio,
                    "c14_score_abs_ln": rec["score"],
                    "c13_score_abs_ln": c13rec["score"] if c13rec else None,
                    "tolerance": rec["tolerance"],
                    "c14_pass": rec["pass"],
                    "c13_pass": c13rec["pass"] if c13rec else None,
                }
            )
        out["cells"][cell] = {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "c14_sim_over_ref_mean": (
                float(np.mean(c14_ratios)) if c14_ratios else None
            ),
            "c13_sim_over_ref_mean": (
                float(np.mean(c13_ratios)) if c13_ratios else None
            ),
            "c14_n_seeds_pass": sum(1 for r in rows if r["c14_pass"]),
            "c13_n_seeds_pass": (
                sum(1 for r in rows if r["c13_pass"]) if by13 else None
            ),
        }
    inc = out["cells"]["widowhood.75+|female"]
    stock = out["cells"]["share_widowed.75+|female"]
    out["summary"] = {
        "incidence_sim_over_ref": {
            "c13_mean": inc["c13_sim_over_ref_mean"],
            "c14_mean": inc["c14_sim_over_ref_mean"],
            "recovered_toward_reference": (
                inc["c14_sim_over_ref_mean"] is not None
                and inc["c13_sim_over_ref_mean"] is not None
                and abs(inc["c14_sim_over_ref_mean"] - 1.0)
                < abs(inc["c13_sim_over_ref_mean"] - 1.0)
            ),
        },
        "stock_sim_over_ref": {
            "c13_mean": stock["c13_sim_over_ref_mean"],
            "c14_mean": stock["c14_sim_over_ref_mean"],
            "lifted_toward_reference": (
                stock["c14_sim_over_ref_mean"] is not None
                and stock["c13_sim_over_ref_mean"] is not None
                and stock["c14_sim_over_ref_mean"]
                > stock["c13_sim_over_ref_mean"]
            ),
        },
    }
    return out


# ==========================================================================
# Candidate-13 comparison, count stability, modal / decider
# ==========================================================================
def _count_cell_stability(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """The count-cell stability the insulated split is designed to preserve.

    Candidate 13 cleared both marriage-count cells to 5/5. The split touches
    only the oldest widowhood band; new 85+ widows remarry at the near-zero
    candidate-11 elderly rates, so the marriage counts should HOLD candidate
    13's 5/5. Reports candidate 14's realised signed ln-tilt per seed against
    candidate 13's committed count scores, and whether the count is stable
    (n_seeds_pass unchanged and still >= 4/5).
    """
    c13art = (
        json.loads(CANDIDATE13_ARTIFACT.read_text())
        if CANDIDATE13_ARTIFACT.exists()
        else None
    )
    by13 = {s["seed"]: s for s in c13art["per_seed"]} if c13art else {}

    out: dict[str, Any] = {
        "design_note": (
            "registration comment 4927236029: the split touches only the "
            "oldest widowhood band; new 85+ widows remarry at the near-zero "
            "candidate-11 elderly rates, so the marriage counts are insulated "
            "and should HOLD candidate 13's 5/5. Movement reported against "
            "candidate 13's committed count scores."
        ),
        "cells": {},
    }
    for cell in COUNT_CELLS:
        rows = []
        signed = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            rbar = rec["rbar"]
            rate_a = rec["rate_a"]
            tilt = (
                float(math.log(rbar / rate_a))
                if rbar > 0 and rate_a > 0
                else None
            )
            if tilt is not None:
                signed.append(tilt)
            c13rec = by13[s["seed"]]["gated_cells"][cell] if by13 else None
            rows.append(
                {
                    "seed": s["seed"],
                    "rbar": rbar,
                    "rate_a": rate_a,
                    "signed_ln_tilt": tilt,
                    "score_abs_ln": rec["score"],
                    "tolerance": rec["tolerance"],
                    "pass": rec["pass"],
                    "candidate13_score": (c13rec["score"] if c13rec else None),
                    "candidate13_pass": (c13rec["pass"] if c13rec else None),
                    "delta_score_vs_c13": (
                        float(rec["score"] - c13rec["score"])
                        if c13rec
                        else None
                    ),
                }
            )
        n_pass = sum(r["pass"] for r in rows)
        c13_n_pass = (
            sum(1 for r in rows if r["candidate13_pass"]) if by13 else None
        )
        out["cells"][cell] = {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "mean_signed_ln_tilt": (
                float(np.mean(signed)) if signed else None
            ),
            "mean_abs_ln_score": float(
                np.mean([r["score_abs_ln"] for r in rows])
            ),
            "n_seeds_pass": n_pass,
            "candidate13_n_seeds_pass": c13_n_pass,
            "stable_vs_c13": (
                bool(n_pass == c13_n_pass and n_pass >= 4)
                if c13_n_pass is not None
                else None
            ),
        }
    both_hold = all(out["cells"][c]["n_seeds_pass"] >= 4 for c in COUNT_CELLS)
    both_stable = all(out["cells"][c]["stable_vs_c13"] for c in COUNT_CELLS)
    out["count_cells_hold"] = bool(both_hold)
    out["count_cells_stable_vs_c13"] = bool(both_stable)
    out["summary"] = (
        "both marriage-count cells hold >= 4/5 and are stable vs candidate 13"
        if both_stable
        else "at least one marriage-count cell moved off candidate 13's count"
    )
    return out


def _candidate13_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Movement of the delta targets vs candidate 13 + the split-band table.

    Compares candidate 14's per-seed rbar-scores for the targeted cells against
    candidate 13's committed scores, and tabulates candidate 13's six-band
    widowhood level (pooled 75+) against candidate 14's seven-band split (from
    the seed-0 fit) so the 75+ reallocation is visible in the fit.
    """
    c13art = (
        json.loads(CANDIDATE13_ARTIFACT.read_text())
        if CANDIDATE13_ARTIFACT.exists()
        else None
    )
    by13 = {s["seed"]: s for s in c13art["per_seed"]} if c13art else {}

    def move(cell: str) -> dict[str, Any]:
        rows = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            c13rec = by13[s["seed"]]["gated_cells"][cell] if by13 else None
            rows.append(
                {
                    "seed": s["seed"],
                    "tolerance": rec["tolerance"],
                    "c13_score": c13rec["score"] if c13rec else None,
                    "c14_score": rec["score"],
                    "c13_rbar": c13rec["rbar"] if c13rec else None,
                    "c14_rbar": rec["rbar"],
                    "rate_a": rec["rate_a"],
                    "c13_pass": c13rec["pass"] if c13rec else None,
                    "c14_pass": rec["pass"],
                }
            )
        c13_np = sum(1 for r in rows if r["c13_pass"]) if by13 else None
        c14_np = sum(1 for r in rows if r["c14_pass"])
        return {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "c13_n_seeds_pass": c13_np,
            "c14_n_seeds_pass": c14_np,
            "improved": (
                bool(c14_np > c13_np) if c13_np is not None else None
            ),
        }

    split_table = per_seed[0]["component_meta"]["widowhood_split_band_table"]

    return {
        "note": (
            "candidate 14 = candidate 13 (comment 4925748151, #104) with "
            "exactly one delta (the surviving-spouse widowhood table's 75+ "
            "band splits into 75-84 and 85+). Scores compared cell-by-cell "
            "against candidate 13's committed run (runs/gate2_hazard_v13.json)."
        ),
        "modal_cell": {MODAL_CELL: move(MODAL_CELL)},
        "count_cells": {c: move(c) for c in COUNT_CELLS},
        "remarriage_gated_cells": {c: move(c) for c in REMARRIAGE_GATED_CELLS},
        "widowhood_incidence_cells": {
            c: move(c) for c in WIDOWHOOD_INCIDENCE_CELLS
        },
        "widowhood_split_band_table_seed0": split_table,
    }


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal (seed 3's stock partial clear), targets, and decider."""
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
    n_pass_no_targeted = seeds_pass_if_forgiven(set(TARGETED_CELLS))
    stock_cell = "share_widowed.75+|female"
    modal_failed_seeds = sorted(fails_by_cell.get(stock_cell, []))
    # The registered modal materialises if the 75+ stock clears only partially
    # -- i.e. it still fails on at least one seed (seed 3 the registered one).
    modal_materialized = len(modal_failed_seeds) >= 1
    distinct_fail_cells = {
        f["cell"] for f in verdict["all_failing_gated_cells"]
    }

    if gate_pass:
        decider = "none (gate passed)"
    elif n_pass_no_modal >= 4:
        decider = (
            "the registered modal cell (the 75+ widowed stock; forgiving it "
            "flips >= 4 seeds to pass)"
        )
    elif seeds_pass_if_forgiven(set(TARGETED_CELLS)) >= 4:
        decider = (
            "the delta-targeted cells (forgiving the elderly-stock/incidence/"
            "count/remarriage targets flips >= 4 seeds to pass)"
        )
    else:
        decider = (
            "broader than the registered modal + delta-targeted cells "
            "(other gated cells also hold the gate below 4 passing seeds)"
        )

    return {
        "registered_modal": (
            "seed 3's 75+ widowed stock (share_widowed.75+|female) clearing "
            "only partially -- the incidence recovery not fully reaching the "
            "stock; secondary: thin 85+ male cells misfiring under the split"
        ),
        "modal_cells": list(REGISTERED_MODAL_CELLS),
        "modal_failed_seeds": modal_failed_seeds,
        "modal_materialized": modal_materialized,
        "modal_seed3_failed": 3 in modal_failed_seeds,
        "modal_track": {c: track(c) for c in REGISTERED_MODAL_CELLS},
        "targeted_cells": list(TARGETED_CELLS),
        "targeted_cells_track": {c: track(c) for c in TARGETED_CELLS},
        "distinct_failing_cells": sorted(distinct_fail_cells),
        "decider_analysis": {
            "n_seeds_pass_actual": n_pass_actual,
            "n_seeds_pass_if_modal_forgiven": n_pass_no_modal,
            "n_seeds_pass_if_targeted_forgiven": n_pass_no_targeted,
            "decider": decider,
        },
    }


# ==========================================================================
# Provenance
# ==========================================================================
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins + the candidate-14 schema, c1-c13 and forensics shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for name in (1, 5, 6, 7, 8, 9, 10, 11, 12, 13):
        pins[f"candidate{name}_runner"] = (
            f"scripts/run_gate2_candidate{name}.py"
        )
        pins[f"candidate{name}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{name}.py"
        )
    pins["candidate13_artifact"] = "runs/gate2_hazard_v13.json"
    pins["candidate13_artifact_sha256"] = c1._sha_of_file(CANDIDATE13_ARTIFACT)
    pins["forensics3_runner"] = "scripts/gate2_forensics3.py"
    pins["forensics3_runner_sha256"] = c1._sha_of_file(
        ROOT / "scripts" / "gate2_forensics3.py"
    )
    pins["forensics3_artifact"] = "runs/gate2_forensics3_v1.json"
    pins["forensics3_artifact_sha256"] = c1._sha_of_file(FORENSICS3_ARTIFACT)
    pins["estimator"] = "mean_over_K20_draws (5200 + k, k=0..19)"
    pins["deltas"] = (
        "one delta vs candidate 13: the surviving-spouse widowhood hazard "
        "table's oldest band, 75+, splits into 75-84 and 85+ (seven bands "
        "{18-34,35-44,45-54,55-64,65-74,75-84,85+} x sex, train-estimated with "
        "the existing smoothing convention, NCHS trend unchanged). Everything "
        "else byte-identical to candidate 13 (the reused compute chain shares "
        "candidate 13's exact code objects; the vestigial spousal-gap "
        "machinery is untouched)"
    )
    pins["byte_identity_code_objects"] = {
        name: (getattr(c13, name).__code__ is globals()[name].__code__)
        for name in REUSED_CODE_OBJECT_NAMES
    }
    pins["diverged_code_objects_vs_candidate13"] = {
        name: (getattr(c13, name).__code__ is not globals()[name].__code__)
        for name in DIVERGED_CODE_OBJECT_NAMES
    }
    pins["widowhood_bands_candidate13"] = [list(b) for b in c13.WIDOW_BANDS]
    pins["widowhood_bands_candidate14"] = [list(b) for b in WIDOW_BANDS]
    return pins


def _model_block() -> dict[str, Any]:
    """The model block, edited for the one candidate-14 delta."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a "
            "mortality-composed widowhood component (candidate 13 with one "
            "delta: the surviving-spouse widowhood hazard table's 75+ band "
            "splits into 75-84 and 85+), scored under the amended "
            "mean-over-K=20-draws estimator"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "delta_vs_candidate13": DELTA_VS_CANDIDATE13,
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic spline on age, "
                "knots 20/22/25/30/40, sex, birth-decade cohort -- "
                "BYTE-IDENTICAL to candidate 13 at a shared draw seed"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x order, "
                "add-one smoothed -- estimator BYTE-IDENTICAL to candidate 13"
            ),
            "widowhood": (
                "COMPOSED, parametric in period; surviving-spouse "
                "marriage-history widowhood level x candidate 5's committed "
                "NCHS betas. THE DELTA: the level table's oldest band splits, "
                "75+ -> 75-84 and 85+ (seven bands {18-34, 35-44, 45-54, "
                "55-64, 65-74, 75-84, 85+} x sex) so an 85+ married ego widows "
                "at its own higher hazard instead of the pooled 75+ rate; the "
                "five inherited bands (18-34 ... 65-74) are bit-identical to "
                "candidate 13; the NCHS trend multiplier is unchanged"
            ),
            "remarriage": (
                "weighted empirical hazard by ego age band "
                "(18-34/35-49/50-64/65-74/75+) x years-since-dissolution band "
                "x origin x sex, add-one smoothed -- the candidate-11 5-band "
                "table, BYTE-IDENTICAL to candidate 13"
            ),
            "fertility": (
                "single-year-of-age triangular-kernel rates within parity "
                "(0/1/2/3+) x birth-decade cohort -- BYTE-IDENTICAL to "
                "candidate 13 (no fertility delta)"
            ),
            "spousal_age_gap": (
                "candidate 12's age-band-conditioned spousal-gap draw "
                "(vestigial: the composed widowhood hazard keys on the "
                "surviving ego's own age, not the imputed spouse age; proven "
                "inert at candidate 12's grading) -- left UNTOUCHED per "
                "byte-minimality, BYTE-IDENTICAL to candidate 13"
            ),
            "entry_widowed_initial_state": (
                "candidate 12's delta 1 (inherited, unchanged): persons "
                "observed already-widowed at their first observed PSID wave "
                "enter widowed; the reference-carried widowed person-years are "
                "injected onto the simulated marital_state post-assembly. "
                "RNG-neutral, BYTE-IDENTICAL to candidate 13"
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
            "widowhood_band_edges": (
                "the pooled 75+ band splits into 75-84 and 85+ (full table "
                "18-34 / 35-44 / 45-54 / 55-64 / 65-74 / 75-84 / 85+ x sex); "
                "the _bands_vec clip maps ages 85+ into the 85+ band"
            ),
            "widowhood_level_fit": (
                "train mh85_23 spouse-death marriage endings over married "
                "person-year exposure, transitions._hazard_by_band "
                "(weighted=True) over the seven bands x surviving-spouse sex "
                "-- the gate reference's own machinery, the existing smoothing "
                "convention (rate = num_wt/den_wt, no add-one); the five "
                "inherited bands are bit-identical to candidate 13"
            ),
            "nchs_trend": (
                "candidate 5's committed per-sex NCHS log-linear period slopes "
                "exp(beta_sex * (year - 1995)), applied unchanged (read, not "
                "re-fit)"
            ),
            "byte_identity": (
                "candidate 14 reuses candidate 13's EXACT code objects for "
                "_widow_probs, _build_sim_lookups, simulate_holdout, "
                "_draw_moments, score_seed, fit_remarriage_age_banded and "
                "_remarriage_probs_age_banded (rebound to this module's "
                "globals); only fit_components is re-implemented to install "
                "the seven-band level "
                "(revision_pins.byte_identity_code_objects / "
                "diverged_code_objects_vs_candidate13)"
            ),
            "everything_else": (
                "the entry-widowed observed initial state, the 5-band "
                "remarriage table, the observed marriage-count initial state "
                "(candidate 9 delta 1), the first-marriage spline, divorce, "
                "the single-year triangular-kernel fertility, the vestigial "
                "spousal-gap machinery, the competing-risk step, one sequence "
                "per person per draw, and the locked protocol are "
                "byte-identical to candidate 13"
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

    # Preflight: candidate 13 (the base), its fit chain and forensics 3 must be
    # present, plus the candidate-5 NCHS references.
    for name, path in (
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

    # Structural delta guard: candidate 14 splits candidate 13's oldest band
    # (75+) into 75-84 and 85+, keeping the five bands below 75 unchanged.
    if len(WIDOW_BANDS) != 7:
        raise RuntimeError(
            f"expected 7 widowhood bands, got {len(WIDOW_BANDS)}."
        )
    if tuple(WIDOW_BANDS) == tuple(c13.WIDOW_BANDS):
        raise RuntimeError(
            "candidate 14's widowhood bands are identical to candidate 13's; "
            "the one delta is absent."
        )
    if tuple(WIDOW_BANDS[:5]) != tuple(c13.WIDOW_BANDS[:5]):
        raise RuntimeError(
            "candidate 14's five inherited bands (below 75) must equal "
            "candidate 13's; only the pooled 75+ band may split."
        )
    if tuple(c13.WIDOW_BANDS[5]) != tuple(POOLED_75PLUS_BAND):
        raise RuntimeError(
            "candidate 13's oldest band is not the pooled 75+ (75, 120); the "
            "split precondition is violated."
        )
    if WIDOW_BANDS[5] != (75, 84) or WIDOW_BANDS[6] != (85, 120):
        raise RuntimeError(
            "candidate 14's split bands must be exactly 75-84 and 85+ "
            f"(got {WIDOW_BANDS[5]} and {WIDOW_BANDS[6]})."
        )
    if WIDOW_BANDS[5][0] != POOLED_75PLUS_BAND[0] or (
        WIDOW_BANDS[6][1] != POOLED_75PLUS_BAND[1]
    ):
        raise RuntimeError(
            "the split bands must cover exactly candidate 13's pooled 75+ "
            "range (75 lower edge, 120 upper edge)."
        )

    # Byte-identity guard: the reused compute chain MUST share candidate 13's
    # exact code objects; the re-implemented function MUST NOT. Fail fast.
    for name in REUSED_CODE_OBJECT_NAMES:
        if globals()[name].__code__ is not getattr(c13, name).__code__:
            raise RuntimeError(
                f"{name} does not share candidate 13's code object; the "
                "reused-chain byte-identity contract is violated."
            )
        if globals()[name].__globals__ is not globals():
            raise RuntimeError(
                f"{name} is not rebound to candidate 14's globals; it would "
                "read candidate 13's widowhood table."
            )
    for name in DIVERGED_CODE_OBJECT_NAMES:
        if globals()[name].__code__ is getattr(c13, name).__code__:
            raise RuntimeError(
                f"{name} shares candidate 13's code object but must be "
                "re-implemented for the candidate-14 delta."
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
    count_stability = _count_cell_stability(per_seed)
    comparison = _candidate13_comparison(per_seed)
    entry_counts = _entry_widowed_seed_counts(panel, demo, GATE_SEEDS)
    elderly = _elderly_75plus_diagnostic(per_seed)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 14",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate13_registration": CANDIDATE13_REGISTRATION,
        "candidate12_registration": CANDIDATE12_REGISTRATION,
        "candidate11_registration": CANDIDATE11_REGISTRATION,
        "candidate10_registration": CANDIDATE10_REGISTRATION,
        "candidate9_registration": CANDIDATE9_REGISTRATION,
        "candidate8_registration": CANDIDATE8_REGISTRATION,
        "candidate6_registration": CANDIDATE6_REGISTRATION,
        "candidate5_registration": CANDIDATE5_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "candidate13_grading_diagnostic": CANDIDATE13_GRADING_DIAGNOSTIC,
        "delta_vs_candidate13": DELTA_VS_CANDIDATE13,
        "amended_estimator": (
            "gates.yaml gate_2 amendment 1 (ratified 2026-07-08, flip #97): "
            "per-cell score |ln(rbar / rate_a)| with rbar the mean over K=20 "
            "draws (default_rng(5200 + k), k=0..19) of the cell rate "
            "(inherited from candidate 13, unchanged)"
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
                "seed 3's 75+ widowed stock clearing only partially (the "
                "incidence recovery not fully reaching the stock); secondary: "
                "thin 85+ male cells misfiring under the split (add-one "
                "smoothing at low exposure)"
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
                "the fresh candidate-14 one-shot run registered AFTER the "
                "2026-07-08 ratification (registration 4927236029)"
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
        "per_seed": per_seed,
        "seed_conjunction": seed_conjunction,
        "report_only": report_block,
        "count_cell_stability": count_stability,
        "candidate13_comparison": comparison,
        "elderly_75plus_diagnostic": elderly,
        "modal_failure_materialized": modal,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "registration_pointer": REGISTRATION_POINTER,
            "candidate13_registration": CANDIDATE13_REGISTRATION,
            "candidate13_artifact": "runs/gate2_hazard_v13.json",
            "grading_diagnostic": CANDIDATE13_GRADING_DIAGNOSTIC,
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
            f"  {MODAL_CELL}: c13 {mc['c13_n_seeds_pass']}/5 -> "
            f"c14 {mc['c14_n_seeds_pass']}/5"
        )
        esum = elderly["summary"]
        print(
            "  75+ incidence sim/ref: c13 "
            f"{esum['incidence_sim_over_ref']['c13_mean']:.4f} -> c14 "
            f"{esum['incidence_sim_over_ref']['c14_mean']:.4f}; "
            "75+ stock sim/ref: c13 "
            f"{esum['stock_sim_over_ref']['c13_mean']:.4f} -> c14 "
            f"{esum['stock_sim_over_ref']['c14_mean']:.4f}"
        )
        for c in COUNT_CELLS:
            b = count_stability["cells"][c]
            print(
                f"  {c}: c13 {b['candidate13_n_seeds_pass']}/5 -> "
                f"c14 {b['n_seeds_pass']}/5 "
                f"(stable={b['stable_vs_c13']})"
            )
        print(
            "modal (seed 3 stock partial clear) materialized="
            f"{modal['modal_materialized']} "
            f"(stock failed seeds {modal['modal_failed_seeds']}); "
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
