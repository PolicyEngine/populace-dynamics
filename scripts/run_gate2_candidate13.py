"""Gate-2 candidate 13 (run 1): candidate 12 + EXACTLY ONE delta, scored under
the amended mean-over-K=20-draws estimator.

The THIRTEENTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42 comment
4925748151 (``SPEC_REGISTRATION``): candidate 12's frozen spec (comment
4925020986, ``scripts/run_gate2_candidate12.py``, merged #103) verbatim EXCEPT
one delta, registered from candidate 12's grading (the corrected mechanism for
the young-widow surplus). One-shot; no constant moves after the registration
comment; published REGARDLESS of verdict.

The one delta vs candidate 12 (everything else byte-identical)
--------------------------------------------------------------
**The surviving-spouse widowhood hazard table gains younger bands -- 18-34 and
35-44.** Candidate 12's composed widowhood level is banded on the gate
reference's own four widowhood age bands ``{45-54, 55-64, 65-74, 75+}`` x sex;
married egos below 45 CLIP into the youngest band and so inherit the 45-54 rate
(female ~0.005/yr), an order above the true young-widowhood risk. Candidate 12's
grading proved delta 2 (the spousal-age gap) inert and isolated this
youngest-band edge as the SOLE remaining source of the young widowed-pool
inflation (3.16x reference at 15-49) and thence the female lifetime-marriage
count over-production. Candidate 13's full widowhood table is
``{18-34, 35-44, 45-54, 55-64, 65-74, 75+}`` x sex, train-estimated from
``mh85_23`` spouse-death endings over married person-year exposure with the
EXISTING smoothing convention (``transitions._hazard_by_band``'s weighted
hazard, the gate reference's own machinery -- num_wt / den_wt, no add-one), the
NCHS period-trend multiplier applied unchanged. The two new bands deflate the
young widowed pool toward reference; the four inherited bands' rates are
bit-identical to candidate 12 (same events, same exposure, same band edges).

Everything else is byte-identical to candidate 12 -- the entry-widowed observed
initial state (candidate 12's delta 1), the 5-band remarriage current-age table,
the observed undatable-marriage lifetime-count initial state (candidate 9's
delta 1), the single-year triangular-kernel fertility, the RNG, the K=20
mean-of-draws protocol, and ``fresh_run_artifact_schema`` conformance (per-draw
per-cell rates [20, 46, 5]; undefined draw invalidates; report-only
dispersion). The vestigial spousal-age-gap machinery (candidate 12's delta 2,
proven inert) is left UNTOUCHED per byte-minimality. Runner
``scripts/run_gate2_candidate13.py``, artifact ``runs/gate2_hazard_v13.json``.

Provable byte-identity (code-object reuse)
------------------------------------------
Because the delta changes ONLY the widowhood age-band table (the constants
``WIDOW_BANDS`` / ``WIDOW_LOWERS`` and its train fit), and candidate 12's
compute chain is band-count-agnostic (its widowhood lookup / lookup-builder
loops and array shapes derive from ``len(WIDOW_BANDS)``), candidate 13 REUSES
candidate 12's EXACT code objects for the band-dependent compute
(``_widow_probs``, ``_build_sim_lookups``, ``simulate_holdout``,
``_draw_moments``, ``score_seed``, ``fit_remarriage_age_banded``,
``_remarriage_probs_age_banded``), rebound (:func:`_rebind`) to resolve their
globals against THIS module -- so the byte-identical simulation calls candidate
13's six-band widowhood table (the one delta) while every other statement is
candidate 12's, guaranteed at the bytecode level
(``candidate13.simulate_holdout.__code__ is
candidate12.simulate_holdout.__code__``). Only ``fit_components`` is
re-implemented (to install the six-band level and emit the delta's provenance);
the three fit helpers ``_widowhood_hazard_cells`` / ``fit_widowhood_level`` /
``_widow_level_diag`` re-band candidate 6's construction over the six bands.

Designed effect (registration)
------------------------------
The delta removes an order-of-magnitude rate error feeding the young widowed
pool; deflating it toward reference removes young-widow remarriages of the size
the failing cells need (seed 3 needs 0.003 ln, seed 4 needs 0.022 ln on
``mean_lifetime_marriages|female``; the excess young-widow remarriage exposure
is comfortably larger). ``P(pass) ~= 0.6-0.7``; modal failure if it fails: the
female count clearing one of seeds 3/4 but not both; secondary: an occupancy
cell shifting as the young marital mix changes. Seed 2 is expected to keep
failing (fertility tilt + its count cluster); the gate passes without it at 4/5.

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

    .venv/bin/python scripts/run_gate2_candidate13.py
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

# Candidate 12 supplies the machinery this build deltas ONCE: its compute chain
# (entry-widowed initial state + age-band gap over candidate 11's 5-band
# remarriage over candidate 8's fit and candidate 9's delta 1), the fresh-run
# artifact-schema blocks, and -- via its imports -- candidate 1's precheck /
# verdict assembly and candidate 8's vectorised simulation helpers. Only the
# surviving-spouse widowhood age-band table changes (4 bands -> 6 bands).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import gate2_forensics3 as f3  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate5 as c5  # noqa: E402
import run_gate2_candidate6 as c6  # noqa: E402
import run_gate2_candidate12 as c12  # noqa: E402

# Candidates 4/8/9/10/11 in the chain are import-bound THROUGH candidate 12
# (its ``simulate_holdout`` / gap machinery / delta-1 helpers / schema blocks
# are candidate 13's, rebound), so candidate 13 needs only candidate 1 (precheck
# / verdict), candidate 5 (committed NCHS references), candidate 6 (the trend
# anchor + committed-beta guard) and candidate 12 directly.
from populace_dynamics.data import marriage, transitions  # noqa: E402

# ``hpanel`` is referenced by candidate 12's ``score_seed`` code object, which
# candidate 13 reuses rebound to THIS module's globals (see ``_rebind`` below);
# it must stay a module global even though candidate 13's own source only names
# it in the young-pool diagnostic (kept explicit for the F401 audit).
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v13.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
CANDIDATE12_ARTIFACT = ROOT / "runs" / "gate2_hazard_v12.json"
CANDIDATE11_ARTIFACT = ROOT / "runs" / "gate2_hazard_v11.json"
CANDIDATE10_ARTIFACT = ROOT / "runs" / "gate2_hazard_v10.json"
CANDIDATE9_ARTIFACT = ROOT / "runs" / "gate2_hazard_v9.json"
CANDIDATE8_ARTIFACT = ROOT / "runs" / "gate2_hazard_v8.json"
CANDIDATE7_ARTIFACT = ROOT / "runs" / "gate2_hazard_v7.json"
CANDIDATE6_ARTIFACT = ROOT / "runs" / "gate2_hazard_v6.json"
CANDIDATE5_ARTIFACT = ROOT / "runs" / "gate2_hazard_v5.json"
FORENSICS3_ARTIFACT = ROOT / "runs" / "gate2_forensics3_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v13"
RUN_NAME = "gate2_hazard_v13"

#: This run's frozen-spec registration (issue #42, comment 4925748151).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4925748151"
)
#: The bare registration comment id, pinned for the artifact pointer.
REGISTRATION_POINTER = "4925748151"
#: The candidate-12 spec this build deltas ONCE (comment 4925020986, #103).
CANDIDATE12_REGISTRATION = c12.SPEC_REGISTRATION
#: The registration chain candidate 12 carried, threaded through for provenance.
CANDIDATE11_REGISTRATION = c12.CANDIDATE11_REGISTRATION
CANDIDATE10_REGISTRATION = c12.CANDIDATE10_REGISTRATION
CANDIDATE9_REGISTRATION = c12.CANDIDATE9_REGISTRATION
CANDIDATE8_REGISTRATION = c12.CANDIDATE8_REGISTRATION
CANDIDATE6_REGISTRATION = c12.CANDIDATE6_REGISTRATION
CANDIDATE5_REGISTRATION = c12.CANDIDATE5_REGISTRATION
CANDIDATE1_REGISTRATION = c12.CANDIDATE1_REGISTRATION
#: Candidate 12's grading (comment 4925748151) located this youngest-band edge.
CANDIDATE12_GRADING_DIAGNOSTIC = (
    "issue #42 comment 4925748151 (candidate 12 grading: delta 2 inert; the "
    "45-54 youngest-band edge is the sole remaining young-widow-surplus source)"
)

# --- Amended-estimator draw stream (inherited from candidate 12, unchanged). -
#: The amended 20-draw stream base: draw k uses default_rng(5200 + k), the
#: committed forensics convention (gates.yaml gate_2, amendment 1).
DRAW_SEED_BASE = c12.DRAW_SEED_BASE  # 5200
N_DRAWS = c12.N_DRAWS  # 20

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

# Candidate 8's simulation helpers (import-bound via candidate 12, unchanged).
_STATE = c12._STATE
_STATE_ABSORB = c12._STATE_ABSORB
_ASFR_LO = c12._ASFR_LO
_ASFR_HI = c12._ASFR_HI
_assemble_sim_panel = c12._assemble_sim_panel
_divorce_probs = c12._divorce_probs
_fertility_probs_single = c12._fertility_probs_single

# Candidate 9's observed residual + reconciliation (delta 1 of candidate 9),
# inherited via candidate 12, reused byte-for-byte.
observed_residual_counts = c12.observed_residual_counts
_delta1_reconciliation = c12._delta1_reconciliation

#: Candidate 10's simulation-lookup container (band-agnostic dataclass; its
#: ``mort_arr`` is shaped by ``_build_sim_lookups`` from ``len(WIDOW_BANDS)``,
#: so it holds candidate 13's six-band table unchanged). Reused as-is.
_SimLookupsC10 = c12._SimLookupsC10

# --- The 5-band remarriage table (candidate 11's delta; UNCHANGED here). -----
REM_AGE_BANDS = c12.REM_AGE_BANDS
REM_AGE_LOWERS = c12.REM_AGE_LOWERS
_REM_AGE_LABEL = c12._REM_AGE_LABEL

# --- The vestigial spousal-age-gap machinery (candidate 12's delta 2, proven
# --- INERT; left UNTOUCHED per byte-minimality; import-bound unchanged). ------
GAP_AGE_BANDS = c12.GAP_AGE_BANDS
GAP_AGE_LOWERS = c12.GAP_AGE_LOWERS
_GAP_AGE_LABEL = c12._GAP_AGE_LABEL
FALLBACK_MIN_WEIGHTED_COUPLES = c12.FALLBACK_MIN_WEIGHTED_COUPLES
spousal_gap_distribution_by_band = c12.spousal_gap_distribution_by_band
c11_pooled_gap = c12.c11_pooled_gap
_fallback_group = c12._fallback_group
_gap_band_arrays = c12._gap_band_arrays
_draw_banded_gaps = c12._draw_banded_gaps

# --- Candidate 12's entry-widowed observed initial state (delta 1; UNCHANGED,
# --- import-bound so the reused ``simulate_holdout`` injects it unchanged). ---
observed_support = c12.observed_support
entry_widowed_carried_cells = c12.entry_widowed_carried_cells
_entry_widowed_seed_counts = c12._entry_widowed_seed_counts
_entry_widowed_reconciliation = c12._entry_widowed_reconciliation
_inject_entry_widowed = c12._inject_entry_widowed
_widowed_share_by_age = c12._widowed_share_by_age

# Fresh-run artifact-schema blocks are band-independent (they operate on the
# scored per-seed dicts); import-bound via candidate 12 (identical N_DRAWS /
# DRAW_SEED_BASE), so the [20, 46, 5] cube, the undefined-draw rule and the
# report-only dispersion are candidate 10's exact assembly.
_per_draw_per_cell_rates_block = c12._per_draw_per_cell_rates_block
_undefined_draw_block = c12._undefined_draw_block
_per_draw_dispersion_block = c12._per_draw_dispersion_block

# --------------------------------------------------------------------------
# THE ONE DELTA vs candidate 12 (registration comment 4925748151)
# --------------------------------------------------------------------------
#: Candidate 12's widowhood table bands on the gate reference's four widowhood
#: age bands (45-54 / 55-64 / 65-74 / 75+; married egos below 45 clip into the
#: youngest). Candidate 13 gains two younger bands -- 18-34 and 35-44 -- so the
#: full surviving-spouse widowhood table is banded 18-34 / 35-44 / 45-54 /
#: 55-64 / 65-74 / 75+ x sex. The ``_bands_vec`` clip now maps ages below 18
#: into the youngest band (essentially no married person-years there).
WIDOW_BANDS: tuple[tuple[int, int], ...] = (
    (18, 34),
    (35, 44),
    (45, 54),
    (55, 64),
    (65, 74),
    (75, 120),
)
WIDOW_LOWERS = np.array([lo for lo, _ in WIDOW_BANDS], dtype=np.int64)
_WIDOW_BAND_LABEL = {
    0: "18-34",
    1: "35-44",
    2: "45-54",
    3: "55-64",
    4: "65-74",
    5: "75+",
}
#: The two bands candidate 13 adds below candidate 12's youngest (45-54) band.
YOUNG_WIDOW_BAND_INDICES = (0, 1)

#: The single named delta (registration comment 4925748151).
DELTA_VS_CANDIDATE12 = (
    "EXACTLY ONE delta vs candidate 12 (comment 4925020986, merged #103): the "
    "surviving-spouse widowhood hazard table gains younger bands -- 18-34 and "
    "35-44. Full structure: {18-34, 35-44, 45-54, 55-64, 65-74, 75+} x sex, "
    "train-estimated from mh85_23 spouse-death endings over married "
    "person-year exposure with the EXISTING smoothing convention "
    "(transitions._hazard_by_band weighted hazard num_wt/den_wt -- the gate "
    "reference's own machinery, no add-one), the NCHS period-trend multiplier "
    "applied unchanged; no other change. Registered from candidate 12's "
    "grading: with the spousal-age gap proven inert, married egos below 45 "
    "inheriting the 45-54 rate (~0.005/yr female, an order above the true "
    "young-widowhood risk) is the sole remaining source of the young "
    "widowed-pool inflation (3.16x reference at 15-49) and thence the female "
    "lifetime-marriage count over-production. The two new bands deflate the "
    "young pool toward reference; the four inherited bands' rates are "
    "bit-identical to candidate 12. Everything else -- the entry-widowed "
    "observed initial state (candidate 12's delta 1), the 5-band remarriage "
    "current-age table, the observed undatable-marriage lifetime-count initial "
    "state (candidate 9's delta 1), the single-year triangular-kernel "
    "fertility, the RNG, the K=20 mean-of-draws protocol, the fresh-run "
    "artifact schema, and the vestigial spousal-age-gap machinery (candidate "
    "12's delta 2, left untouched per byte-minimality) -- is byte-identical to "
    "candidate 12"
)

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate13_run1_cache.json"
)

#: The marriage-count cells (the young-widowhood deflation reshapes the
#: young-widow remarriage exposure that feeds these).
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
#: The elderly-widow-stock cell candidate 12's delta 1 recovered (unchanged
#: here; the young-band delta must not regress it).
MODAL_CELL = "share_widowed.75+|female"
#: The registered modal failure (comment 4925748151): the female marriage count
#: clearing one of the failing seeds but not both.
REGISTERED_MODAL_CELLS = ("mean_lifetime_marriages|female",)
#: The gated widowhood-incidence cells (untouched by construction -- the delta
#: adds bands strictly below the measured 45+ incidence bands).
WIDOWHOOD_INCIDENCE_CELLS = (
    "widowhood.45-64|female",
    "widowhood.65-74|female",
    "widowhood.75+|female",
    "widowhood.45+|male",
)
#: The cells the delta most directly touches: both marriage counts (the young-
#: widow remarriage exposure), the remarriage flows, the widow stock (must
#: hold), and the widowhood incidence (must hold).
TARGETED_CELLS = (
    (MODAL_CELL,)
    + COUNT_CELLS
    + REMARRIAGE_GATED_CELLS
    + WIDOWHOOD_INCIDENCE_CELLS
)
#: The young widowed-stock pools the delta is designed to deflate (diagnostic,
#: NOT gated -- reference_moments produces share_widowed only at 65-74 and 75+).
YOUNG_POOL_BANDS = ((15, 49), (50, 64))
WIDOWED_AGE_BANDS = f3.WIDOWED_AGE_BANDS


# ==========================================================================
# THE DELTA fit: six-band surviving-spouse widowhood hazard (train)
# Re-banded (not rebound) because ``transitions.hazard_cells`` bands widowhood
# on the module-level ``WIDOWHOOD_AGE_BANDS`` (four bands, the committed
# reference); candidate 13 calls the same ``transitions._hazard_by_band``
# machinery directly over its own six bands, so the four inherited bands are
# bit-identical and the two young bands are added, WITHOUT touching the shared
# reference moment.
# ==========================================================================
def _widowhood_hazard_cells(
    panel: transitions.MaritalPanel, train_ids: set[int]
) -> dict[str, dict[str, float]]:
    """Train marriage-history widowhood hazard cells over the six bands.

    Candidate 6's construction (:func:`candidate6._widowhood_hazard_cells`) --
    weighted spouse-death marriage endings over weighted married person-year
    exposure, the gate reference's own :func:`transitions._hazard_by_band`
    machinery with the existing smoothing convention (rate = num_wt / den_wt,
    0 where the denominator is empty; NO add-one) -- restricted to the train
    complement and banded on candidate 13's :data:`WIDOW_BANDS` (the six-band
    table) instead of the four-band ``transitions.WIDOWHOOD_AGE_BANDS``. The
    band boundaries at 45+ are identical, so those four bands' num_wt / den_wt
    / rate are bit-identical to candidate 6/12; the two new bands (18-34,
    35-44) add the young-widowhood cells candidate 12 clipped into 45-54.
    Returned keyed ``"band|sex"`` (the ``widowhood.`` prefix stripped).
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
    level source re-banded onto the six-band table.
    """
    return {
        key: float(cell["rate"])
        for key, cell in _widowhood_hazard_cells(panel, train_ids).items()
    }


def _widow_level_diag(
    panel: transitions.MaritalPanel, train_ids: set[int]
) -> dict[str, Any]:
    """Provenance of the six-band surviving-spouse widowhood LEVEL table."""
    cells = _widowhood_hazard_cells(panel, train_ids)
    return {
        "source": (
            "train marriage-history widowhood endings (mh85_23 how-ended = "
            "spouse death) over married person-year exposure, by age band x "
            "sex of the surviving spouse"
        ),
        "construction": (
            "transitions._hazard_by_band(weighted=True) widowhood cells over "
            "candidate 13's six-band WIDOW_BANDS restricted to the seed's train "
            "complement -- the gate reference's own machinery, re-banded (the "
            "existing smoothing convention: rate = num_wt/den_wt, no add-one)"
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


def _widowhood_band_rate_table(
    new_level: dict[str, float], c12_level_4band: dict[str, float]
) -> dict[str, Any]:
    """Old (candidate 12, four-band) vs new (candidate 13, six-band) table.

    For the two new young bands the "candidate-12 applied rate" is the 45-54
    rate their egos clipped into; for the four inherited bands candidate 12 and
    candidate 13 are bit-identical. Makes the young-band deflation auditable.
    """
    cells: dict[str, Any] = {}
    for b, (lo, hi) in enumerate(WIDOW_BANDS):
        band = transitions.band_label(lo, hi)
        for sex in ("female", "male"):
            key = f"{band}|{sex}"
            c13_rate = float(new_level[key])
            if b in YOUNG_WIDOW_BAND_INDICES:
                applied = float(c12_level_4band[f"45-54|{sex}"])
                cells[key] = {
                    "new_band": True,
                    "candidate12_applied_rate": applied,
                    "candidate12_applied_via": (
                        "clipped into the 45-54 band (candidate 12's youngest)"
                    ),
                    "candidate13_rate": c13_rate,
                    "ratio_c13_over_c12_applied": (
                        c13_rate / applied if applied > 0 else None
                    ),
                }
            else:
                c12_rate = float(c12_level_4band[key])
                cells[key] = {
                    "new_band": False,
                    "candidate12_rate": c12_rate,
                    "candidate13_rate": c13_rate,
                    "bit_identical": abs(c12_rate - c13_rate) <= 1e-12,
                }
    return {
        "note": (
            "surviving-spouse widowhood LEVEL by band x sex: candidate 12's "
            "four-band table (married egos below 45 clip into 45-54) vs "
            "candidate 13's six-band table. The two new bands (18-34, 35-44) "
            "deflate the young-widowhood rate toward reference; the four "
            "inherited bands are bit-identical"
        ),
        "bands": [list(b) for b in WIDOW_BANDS],
        "cells": cells,
    }


# ==========================================================================
# Fitted components (candidate 12's, with the widowhood LEVEL re-banded)
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
    """Candidate 12's components with the widowhood LEVEL re-banded (the DELTA).

    Starts from :func:`candidate12.fit_components` -- so first marriage,
    divorce, the 5-band remarriage table, the single-year triangular-kernel
    fertility, the committed NCHS betas, the entry-widowed carried cells
    (candidate 12's delta 1), the observed marriage-count initial state
    (candidate 9's delta 1) and the age-band spousal-gap distribution
    (candidate 12's vestigial delta 2) are byte-identical to candidate 12 by
    construction, and ``base.mortality`` is candidate 12's four-band
    surviving-spouse widowhood level. Then THE ONE DELTA replaces
    ``base.mortality`` with candidate 13's six-band table
    (:func:`fit_widowhood_level`), retaining candidate 12's four-band level
    under a provenance key for the old-vs-new-by-band comparison. The NCHS
    period-trend beta is untouched (candidate 5's committed value, applied
    unchanged).
    """
    base = c12.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )

    # THE DELTA: re-band the surviving-spouse widowhood LEVEL onto six bands.
    # ``base.mortality`` at this point is candidate 12's four-band table.
    c12_widow_level_4band = dict(base.mortality)
    widow_level = fit_widowhood_level(panel, train_ids)
    base.mortality = widow_level

    base.meta["mortality_level_candidate12_4band"] = c12_widow_level_4band
    base.meta["mortality_level_new_widowhood"] = dict(widow_level)
    base.meta["mortality_level_bands"] = [list(b) for b in WIDOW_BANDS]
    base.meta["mortality_cells"] = len(widow_level)  # 6 bands x 2 sexes = 12
    base.meta["mortality_level_diagnostics"] = _widow_level_diag(
        panel, train_ids
    )
    base.meta["widowhood_young_band_table"] = _widowhood_band_rate_table(
        widow_level, c12_widow_level_4band
    )
    base.meta["mortality_level_source"] = (
        "train marriage-history widowhood endings (mh85_23 how-ended = spouse "
        "death) over married person-year exposure, by age band x sex of the "
        "surviving spouse -- transitions._hazard_by_band widowhood "
        "construction (the gate reference's own machinery, the existing "
        "smoothing convention), re-banded onto candidate 13's six-band table "
        "{18-34, 35-44, 45-54, 55-64, 65-74, 75+} (candidate 12 had four bands "
        "45-54/55-64/65-74/75+; married egos below 45 clipped into 45-54)"
    )
    base.meta["mortality_level_representation"] = (
        "surviving-spouse widowhood-incidence hazard, keyed (candidate 13 "
        "six-band WIDOW_BANDS, surviving-spouse sex); consumed by the married "
        "ego's OWN (age band, sex) with the NCHS period-trend multiplier "
        "exp(beta_sex * (year - 1995)) applied unchanged"
    )
    base.meta["delta_vs_candidate12"] = DELTA_VS_CANDIDATE12
    return base


# ==========================================================================
# Provable byte-identity: reuse candidate 12's EXACT code objects, rebound so
# their global lookups resolve against THIS module (candidate 13). The bytecode
# is candidate 12's; the only observable change is that ``WIDOW_BANDS`` (and the
# widowhood lookup / lookup-builder keyed on it) are candidate 13's six-band
# table -- the one delta. Only ``fit_components`` is re-implemented (above).
# ==========================================================================
def _rebind(fn: types.FunctionType) -> types.FunctionType:
    """Return a function sharing ``fn``'s code object but this module's globals.

    Candidate 12's widowhood-band-dependent compute is band-count-agnostic (the
    ``_build_sim_lookups`` ``mort_arr`` shape and the ``_widow_probs`` clip both
    derive from ``len(WIDOW_BANDS)``). Reusing its code object verbatim -- only
    redirecting global name resolution to candidate 13's module -- makes
    ``everything else byte-identical`` provable at the bytecode level
    (``candidate13.f.__code__ is candidate12.f.__code__``) while the reused code
    reads candidate 13's six-band widowhood table.
    """
    return types.FunctionType(
        fn.__code__,
        globals(),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


#: THE DELTA's lookup: candidate 12's (== candidate 6's) exact widowhood
#: incidence x NCHS-trend code, reading candidate 13's six-band WIDOW_BANDS /
#: WIDOW_LOWERS -- so ages below 45 now index the 18-34 / 35-44 rows instead of
#: clipping into 45-54.
_widow_probs = _rebind(c12._widow_probs)
#: Candidate 12's (== candidate 10's) exact lookup builder; ``mort_arr`` gains
#: candidate 13's two young rows automatically (shape from ``len(WIDOW_BANDS)``).
_build_sim_lookups = _rebind(c12._build_sim_lookups)
#: Candidate 12's EXACT vectorised annual simulation (both of candidate 12's
#: deltas -- the entry-widowed injection and the age-band gap draw -- inline and
#: byte-for-byte). It calls ``_widow_probs`` / ``_build_sim_lookups`` (candidate
#: 13's six-band, above); every other statement is candidate 12's.
simulate_holdout = _rebind(c12.simulate_holdout)
#: Candidate 12's (== candidate 11's == candidate 10's) exact single-draw moment
#: builder and per-seed mean-over-K=20 scorer. Rebound to THIS module's globals,
#: they call candidate 13's ``simulate_holdout`` / ``fit_components``.
_draw_moments = _rebind(c12._draw_moments)
score_seed = _rebind(c12.score_seed)
#: The 5-band remarriage fit / lookup (candidate 12's == candidate 11's code),
#: rebound -- UNCHANGED by candidate 13's delta (kept for the attestation).
fit_remarriage_age_banded = _rebind(c12.fit_remarriage_age_banded)
_remarriage_probs_age_banded = _rebind(c12._remarriage_probs_age_banded)

#: The reused-code-object contract (must share candidate 12's bytecode).
REUSED_CODE_OBJECT_NAMES = (
    "_widow_probs",
    "_build_sim_lookups",
    "simulate_holdout",
    "_draw_moments",
    "score_seed",
    "fit_remarriage_age_banded",
    "_remarriage_probs_age_banded",
)
#: The RE-IMPLEMENTED function (the delta installs the six-band level and emits
#: its provenance): it must NOT share candidate 12's code object.
DIVERGED_CODE_OBJECT_NAMES = ("fit_components",)


# ==========================================================================
# Diagnostics: young-pool widowed shares vs candidate 12 (the designed defl'n)
# ==========================================================================
def _young_pool_diagnostic(
    panel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    mh_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    order_map: pd.DataFrame,
    seeds: tuple[int, ...],
    verbose: bool,
) -> dict[str, Any]:
    """Candidate 13 female widowed-stock shares by age band vs candidate 12.

    Mirrors candidate 12's young-pool convention (fit + simulate side B, K=20
    draws, draw-mean) so candidate 13's young widowed pools are directly
    comparable to candidate 12's committed pools in
    ``runs/gate2_hazard_v12.json``. The reference share is side B's own observed
    panel; the simulated share is the K=20 draw-mean. Reports the sim/ref ratio
    at 15-49 and 50-64 (the pools the young-band delta is designed to deflate)
    alongside candidate 12's committed ratios.
    """
    c12art = json.loads(CANDIDATE12_ARTIFACT.read_text())
    c12_yp = {
        r["seed"]: r["bands"]
        for r in c12art["young_pool_diagnostic"]["per_seed"]
    }
    band_labels = [
        transitions.band_label(lo, hi) for lo, hi in WIDOWED_AGE_BANDS
    ]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        _side_a, side_b = hpanel.split_panel_by_person(
            panel.attrs, "person_id", fraction=0.5, seed=seed
        )
        ids_b = set(int(x) for x in side_b.person_id.unique())
        components = fit_components(
            panel,
            demo,
            death_records,
            mh_records,
            birth_records,
            order_map,
            ids_b,
        )
        ref = _widowed_share_by_age(panel, ids_b)
        sim_acc = {lab: [] for lab in band_labels}
        for k in range(N_DRAWS):
            sim_panel, _b = simulate_holdout(
                panel, ids_b, components, DRAW_SEED_BASE + k
            )
            share = _widowed_share_by_age(sim_panel, ids_b)
            for lab in band_labels:
                sim_acc[lab].append(share[lab])
        sim_mean = {lab: float(np.mean(sim_acc[lab])) for lab in band_labels}
        c12b = c12_yp[seed]
        cells: dict[str, Any] = {}
        for lab in band_labels:
            ref_share = ref[lab]
            c13_ratio = sim_mean[lab] / ref_share if ref_share > 0 else None
            cells[lab] = {
                "ref_widowed_share": ref_share,
                "c13_sim_widowed_share_mean": sim_mean[lab],
                "c12_sim_widowed_share_mean": c12b[lab][
                    "c12_sim_widowed_share_mean"
                ],
                "c13_sim_over_ref": c13_ratio,
                "c12_sim_over_ref": c12b[lab]["c12_sim_over_ref"],
            }
        rows.append({"seed": seed, "bands": cells})
        if verbose:
            y1 = cells["15-49"]
            y2 = cells["50-64"]
            print(
                f"  young-pool seed {seed}: 15-49 c12 "
                f"{y1['c12_sim_over_ref']:.3f} -> c13 "
                f"{y1['c13_sim_over_ref']:.3f}; 50-64 c12 "
                f"{y2['c12_sim_over_ref']:.3f} -> c13 "
                f"{y2['c13_sim_over_ref']:.3f}"
            )

    def _mean_ratio(label: str, which: str) -> float:
        vals = [
            r["bands"][label][which]
            for r in rows
            if r["bands"][label][which] is not None
        ]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        lab: {
            "c12_sim_over_ref_mean": _mean_ratio(lab, "c12_sim_over_ref"),
            "c13_sim_over_ref_mean": _mean_ratio(lab, "c13_sim_over_ref"),
        }
        for lab in band_labels
    }
    return {
        "note": (
            "female widowed-stock share by age band (fit + simulate side B, "
            "K=20 draw-mean). c12 ratios from runs/gate2_hazard_v12.json; the "
            "young-band delta is designed to deflate the 15-49 (c12 ~3.16x) "
            "and 50-64 (c12 ~1.41x) pools toward reference"
        ),
        "bands": band_labels,
        "per_seed": rows,
        "seed_mean_sim_over_ref": summary,
    }


# ==========================================================================
# Candidate-12 comparison, count tilt, modal / decider
# ==========================================================================
def _count_cell_tilt(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """The count-cell movement the young-band delta is designed to produce.

    Candidate 12 left the female count at 2/5 (seeds 3 and 4 failing
    ``mean_lifetime_marriages|female`` at 0.054 and 0.073 ln vs 0.051). The
    young-band delta removes the young-widow remarriage over-exposure, designed
    to pull the female net back toward the residual-only ~0.00 the male side
    enjoys. Reports candidate 13's realised signed ln-tilt per seed against
    candidate 12's committed count scores.
    """
    c12art = (
        json.loads(CANDIDATE12_ARTIFACT.read_text())
        if CANDIDATE12_ARTIFACT.exists()
        else None
    )
    by12 = {s["seed"]: s for s in c12art["per_seed"]} if c12art else {}

    out: dict[str, Any] = {
        "design_note": (
            "registration comment 4925748151: the young-band delta deflates "
            "the young widowed pool, removing the young-widow remarriage "
            "over-exposure, designed to pull the female marriage count net back "
            "from candidate 12's regressed 2/5 (seeds 3/4 failing at 0.054 / "
            "0.073 ln) toward the residual-only ~0.00 the male side enjoys. "
            "Movement reported against candidate 12's committed count scores."
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
            c12rec = by12[s["seed"]]["gated_cells"][cell] if by12 else None
            rows.append(
                {
                    "seed": s["seed"],
                    "rbar": rbar,
                    "rate_a": rate_a,
                    "signed_ln_tilt": tilt,
                    "score_abs_ln": rec["score"],
                    "tolerance": rec["tolerance"],
                    "pass": rec["pass"],
                    "candidate12_score": (c12rec["score"] if c12rec else None),
                    "candidate12_pass": (c12rec["pass"] if c12rec else None),
                    "delta_score_vs_c12": (
                        float(rec["score"] - c12rec["score"])
                        if c12rec
                        else None
                    ),
                }
            )
        n_pass = sum(r["pass"] for r in rows)
        c12_n_pass = (
            sum(1 for r in rows if r["candidate12_pass"]) if by12 else None
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
            "candidate12_n_seeds_pass": c12_n_pass,
        }
    both_pass = all(out["cells"][c]["n_seeds_pass"] >= 4 for c in COUNT_CELLS)
    out["count_cells_cleared"] = bool(both_pass)
    out["summary"] = (
        "both marriage-count cells clear >= 4/5 seeds"
        if both_pass
        else "at least one marriage-count cell holds below 4/5 seeds"
    )
    return out


def _candidate12_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Movement of the delta targets vs candidate 12 + the widowhood-band table.

    Compares candidate 13's per-seed rbar-scores for the targeted cells against
    candidate 12's committed scores, and tabulates candidate 12's four-band
    widowhood level (with the 45-54 rate under-45 egos clipped into) against
    candidate 13's six-band split (from the seed-0 fit) so the young-band
    deflation is visible in the fit.
    """
    c12art = (
        json.loads(CANDIDATE12_ARTIFACT.read_text())
        if CANDIDATE12_ARTIFACT.exists()
        else None
    )
    by12 = {s["seed"]: s for s in c12art["per_seed"]} if c12art else {}

    def move(cell: str) -> dict[str, Any]:
        rows = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            c12rec = by12[s["seed"]]["gated_cells"][cell] if by12 else None
            rows.append(
                {
                    "seed": s["seed"],
                    "tolerance": rec["tolerance"],
                    "c12_score": c12rec["score"] if c12rec else None,
                    "c13_score": rec["score"],
                    "c12_rbar": c12rec["rbar"] if c12rec else None,
                    "c13_rbar": rec["rbar"],
                    "rate_a": rec["rate_a"],
                    "c12_pass": c12rec["pass"] if c12rec else None,
                    "c13_pass": rec["pass"],
                }
            )
        c12_np = sum(1 for r in rows if r["c12_pass"]) if by12 else None
        c13_np = sum(1 for r in rows if r["c13_pass"])
        return {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "c12_n_seeds_pass": c12_np,
            "c13_n_seeds_pass": c13_np,
            "improved": (
                bool(c13_np > c12_np) if c12_np is not None else None
            ),
        }

    widow_table = per_seed[0]["component_meta"]["widowhood_young_band_table"]

    return {
        "note": (
            "candidate 13 = candidate 12 (comment 4925020986, #103) with "
            "exactly one delta (the surviving-spouse widowhood table gains the "
            "18-34 and 35-44 bands). Scores compared cell-by-cell against "
            "candidate 12's committed run (runs/gate2_hazard_v12.json)."
        ),
        "modal_cell": {MODAL_CELL: move(MODAL_CELL)},
        "count_cells": {c: move(c) for c in COUNT_CELLS},
        "remarriage_gated_cells": {c: move(c) for c in REMARRIAGE_GATED_CELLS},
        "widowhood_incidence_cells": {
            c: move(c) for c in WIDOWHOOD_INCIDENCE_CELLS
        },
        "widowhood_band_rate_table_seed0": widow_table,
    }


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal (the female marriage count), targets, and decider."""
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
    female_count = "mean_lifetime_marriages|female"
    modal_failed_seeds = sorted(fails_by_cell.get(female_count, []))
    # The registered modal materialises if the female count clears one of the
    # failing seeds but not both -- i.e. it still fails on exactly one seed.
    modal_materialized = len(modal_failed_seeds) == 1
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
    elif seeds_pass_if_forgiven(set(TARGETED_CELLS)) >= 4:
        decider = (
            "the delta-targeted cells (forgiving the count/remarriage/"
            "widow-stock/widowhood-incidence targets flips >= 4 seeds to pass)"
        )
    else:
        decider = (
            "broader than the registered modal + delta-targeted cells "
            "(other gated cells also hold the gate below 4 passing seeds)"
        )

    return {
        "registered_modal": (
            "the female marriage count clearing one of the failing seeds (3, "
            "4) but not both after the young-band delta deflates the "
            "young-widow remarriage exposure"
        ),
        "modal_cells": list(REGISTERED_MODAL_CELLS),
        "modal_failed_seeds": modal_failed_seeds,
        "modal_materialized": modal_materialized,
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
    """Candidate 1's pins + the candidate-13 schema, c1-c12 and forensics shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for name in (1, 5, 6, 7, 8, 9, 10, 11, 12):
        pins[f"candidate{name}_runner"] = (
            f"scripts/run_gate2_candidate{name}.py"
        )
        pins[f"candidate{name}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{name}.py"
        )
    pins["candidate12_artifact"] = "runs/gate2_hazard_v12.json"
    pins["candidate12_artifact_sha256"] = c1._sha_of_file(CANDIDATE12_ARTIFACT)
    pins["forensics3_runner"] = "scripts/gate2_forensics3.py"
    pins["forensics3_runner_sha256"] = c1._sha_of_file(
        ROOT / "scripts" / "gate2_forensics3.py"
    )
    pins["forensics3_artifact"] = "runs/gate2_forensics3_v1.json"
    pins["forensics3_artifact_sha256"] = c1._sha_of_file(FORENSICS3_ARTIFACT)
    pins["estimator"] = "mean_over_K20_draws (5200 + k, k=0..19)"
    pins["deltas"] = (
        "one delta vs candidate 12: the surviving-spouse widowhood hazard "
        "table gains younger bands 18-34 and 35-44 (six bands "
        "{18-34,35-44,45-54,55-64,65-74,75+} x sex, train-estimated with the "
        "existing smoothing convention, NCHS trend unchanged). Everything else "
        "byte-identical to candidate 12 (the reused compute chain shares "
        "candidate 12's exact code objects; the vestigial spousal-gap "
        "machinery is untouched)"
    )
    pins["byte_identity_code_objects"] = {
        name: (getattr(c12, name).__code__ is globals()[name].__code__)
        for name in REUSED_CODE_OBJECT_NAMES
    }
    pins["diverged_code_objects_vs_candidate12"] = {
        name: (getattr(c12, name).__code__ is not globals()[name].__code__)
        for name in DIVERGED_CODE_OBJECT_NAMES
    }
    pins["widowhood_bands_candidate12"] = [list(b) for b in c12.WIDOW_BANDS]
    pins["widowhood_bands_candidate13"] = [list(b) for b in WIDOW_BANDS]
    return pins


def _model_block() -> dict[str, Any]:
    """The model block, edited for the one candidate-13 delta."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a "
            "mortality-composed widowhood component (candidate 12 with one "
            "delta: the surviving-spouse widowhood hazard table gains the "
            "18-34 and 35-44 bands), scored under the amended "
            "mean-over-K=20-draws estimator"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "delta_vs_candidate12": DELTA_VS_CANDIDATE12,
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic spline on age, "
                "knots 20/22/25/30/40, sex, birth-decade cohort -- "
                "BYTE-IDENTICAL to candidate 12 at a shared draw seed"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x order, "
                "add-one smoothed -- estimator BYTE-IDENTICAL to candidate 12"
            ),
            "widowhood": (
                "COMPOSED, parametric in period; surviving-spouse "
                "marriage-history widowhood level x candidate 5's committed "
                "NCHS betas. THE DELTA: the level table gains the 18-34 and "
                "35-44 bands (six bands {18-34, 35-44, 45-54, 55-64, 65-74, "
                "75+} x sex) so married egos below 45 use their own "
                "young-widowhood rate instead of clipping into 45-54; the four "
                "inherited bands are bit-identical to candidate 12; the NCHS "
                "trend multiplier is unchanged"
            ),
            "remarriage": (
                "weighted empirical hazard by ego age band "
                "(18-34/35-49/50-64/65-74/75+) x years-since-dissolution band "
                "x origin x sex, add-one smoothed -- the candidate-11 5-band "
                "table, BYTE-IDENTICAL to candidate 12"
            ),
            "fertility": (
                "single-year-of-age triangular-kernel rates within parity "
                "(0/1/2/3+) x birth-decade cohort -- BYTE-IDENTICAL to "
                "candidate 12 (no fertility delta)"
            ),
            "spousal_age_gap": (
                "candidate 12's age-band-conditioned spousal-gap draw "
                "(vestigial: the composed widowhood hazard keys on the "
                "surviving ego's own age, not the imputed spouse age; proven "
                "inert at candidate 12's grading) -- left UNTOUCHED per "
                "byte-minimality, BYTE-IDENTICAL to candidate 12"
            ),
            "entry_widowed_initial_state": (
                "candidate 12's delta 1 (inherited, unchanged): persons "
                "observed already-widowed at their first observed PSID wave "
                "enter widowed; the reference-carried widowed person-years are "
                "injected onto the simulated marital_state post-assembly. "
                "RNG-neutral, BYTE-IDENTICAL to candidate 12"
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
                "the two new bands are 18-34 and 35-44 (full table 18-34 / "
                "35-44 / 45-54 / 55-64 / 65-74 / 75+ x sex); the _bands_vec "
                "clip maps ages below 18 into the youngest band"
            ),
            "widowhood_level_fit": (
                "train mh85_23 spouse-death marriage endings over married "
                "person-year exposure, transitions._hazard_by_band "
                "(weighted=True) over the six bands x surviving-spouse sex -- "
                "the gate reference's own machinery, the existing smoothing "
                "convention (rate = num_wt/den_wt, no add-one); the four "
                "inherited bands are bit-identical to candidate 12"
            ),
            "nchs_trend": (
                "candidate 5's committed per-sex NCHS log-linear period slopes "
                "exp(beta_sex * (year - 1995)), applied unchanged (read, not "
                "re-fit)"
            ),
            "byte_identity": (
                "candidate 13 reuses candidate 12's EXACT code objects for "
                "_widow_probs, _build_sim_lookups, simulate_holdout, "
                "_draw_moments, score_seed, fit_remarriage_age_banded and "
                "_remarriage_probs_age_banded (rebound to this module's "
                "globals); only fit_components is re-implemented to install the "
                "six-band level (revision_pins.byte_identity_code_objects / "
                "diverged_code_objects_vs_candidate12)"
            ),
            "everything_else": (
                "the entry-widowed observed initial state, the 5-band "
                "remarriage table, the observed marriage-count initial state "
                "(candidate 9 delta 1), the first-marriage spline, divorce, "
                "the single-year triangular-kernel fertility, the vestigial "
                "spousal-gap machinery, the competing-risk step, one sequence "
                "per person per draw, and the locked protocol are "
                "byte-identical to candidate 12"
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

    # Preflight: candidate 12 (the base), its fit chain and forensics 3 must be
    # present, plus the candidate-5 NCHS references.
    for name, path in (
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

    # Structural delta guard: candidate 13 gains exactly the two young bands and
    # keeps candidate 12's four bands unchanged.
    if len(WIDOW_BANDS) != 6:
        raise RuntimeError(
            f"expected 6 widowhood bands, got {len(WIDOW_BANDS)}."
        )
    if tuple(WIDOW_BANDS) == tuple(c12.WIDOW_BANDS):
        raise RuntimeError(
            "candidate 13's widowhood bands are identical to candidate 12's; "
            "the one delta is absent."
        )
    if tuple(WIDOW_BANDS[2:]) != tuple(c12.WIDOW_BANDS):
        raise RuntimeError(
            "candidate 13's four inherited bands must equal candidate 12's "
            "widowhood bands; only the two young bands may be added."
        )

    # Byte-identity guard: the reused compute chain MUST share candidate 12's
    # exact code objects; the re-implemented function MUST NOT. Fail fast.
    for name in REUSED_CODE_OBJECT_NAMES:
        if globals()[name].__code__ is not getattr(c12, name).__code__:
            raise RuntimeError(
                f"{name} does not share candidate 12's code object; the "
                "reused-chain byte-identity contract is violated."
            )
        if globals()[name].__globals__ is not globals():
            raise RuntimeError(
                f"{name} is not rebound to candidate 13's globals; it would "
                "read candidate 12's widowhood table."
            )
    for name in DIVERGED_CODE_OBJECT_NAMES:
        if globals()[name].__code__ is getattr(c12, name).__code__:
            raise RuntimeError(
                f"{name} shares candidate 12's code object but must be "
                "re-implemented for the candidate-13 delta."
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
    count_tilt = _count_cell_tilt(per_seed)
    comparison = _candidate12_comparison(per_seed)
    entry_counts = _entry_widowed_seed_counts(panel, demo, GATE_SEEDS)
    if verbose:
        print("young-pool diagnostic (the young-band delta's deflation):")
    young_pool = _young_pool_diagnostic(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        GATE_SEEDS,
        verbose,
    )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 13",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate12_registration": CANDIDATE12_REGISTRATION,
        "candidate11_registration": CANDIDATE11_REGISTRATION,
        "candidate10_registration": CANDIDATE10_REGISTRATION,
        "candidate9_registration": CANDIDATE9_REGISTRATION,
        "candidate8_registration": CANDIDATE8_REGISTRATION,
        "candidate6_registration": CANDIDATE6_REGISTRATION,
        "candidate5_registration": CANDIDATE5_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "candidate12_grading_diagnostic": CANDIDATE12_GRADING_DIAGNOSTIC,
        "delta_vs_candidate12": DELTA_VS_CANDIDATE12,
        "amended_estimator": (
            "gates.yaml gate_2 amendment 1 (ratified 2026-07-08, flip #97): "
            "per-cell score |ln(rbar / rate_a)| with rbar the mean over K=20 "
            "draws (default_rng(5200 + k), k=0..19) of the cell rate "
            "(inherited from candidate 12, unchanged)"
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
                "the female marriage count clearing one of the failing seeds "
                "(3, 4) but not both; secondary: an occupancy cell shifting as "
                "the young marital mix changes"
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
                "the fresh candidate-13 one-shot run registered AFTER the "
                "2026-07-08 ratification (registration 4925748151)"
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
        "count_cell_tilt": count_tilt,
        "candidate12_comparison": comparison,
        "young_pool_diagnostic": young_pool,
        "modal_failure_materialized": modal,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "registration_pointer": REGISTRATION_POINTER,
            "candidate12_registration": CANDIDATE12_REGISTRATION,
            "candidate12_artifact": "runs/gate2_hazard_v12.json",
            "grading_diagnostic": CANDIDATE12_GRADING_DIAGNOSTIC,
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
            f"  {MODAL_CELL}: c12 {mc['c12_n_seeds_pass']}/5 -> "
            f"c13 {mc['c13_n_seeds_pass']}/5"
        )
        for c in COUNT_CELLS:
            b = cm["count_cells"][c]
            print(
                f"  {c}: c12 {b['c12_n_seeds_pass']}/5 -> "
                f"c13 {b['c13_n_seeds_pass']}/5"
            )
        print(
            "modal (female count clears one of 3/4 not both) materialized="
            f"{modal['modal_materialized']} "
            f"(female-count failed seeds {modal['modal_failed_seeds']}); "
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
