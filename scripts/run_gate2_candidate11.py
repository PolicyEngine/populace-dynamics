"""Gate-2 candidate 11 (run 1): candidate 10 + EXACTLY ONE delta, scored under
the amended mean-over-K=20-draws estimator.

The ELEVENTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42 comment
4919417729 (``SPEC_REGISTRATION``): candidate 10's frozen spec (comment
4917059482, ``scripts/run_gate2_candidate10.py``, merged #98) verbatim EXCEPT
one delta, registered from gate-2 forensics 2 (#99,
``runs/gate2_forensics2_v1.json``). One-shot; no constant moves after the
registration comment; published REGARDLESS of verdict.

The one delta vs candidate 10 (everything else byte-identical)
--------------------------------------------------------------
**The remarriage current-age conditioning splits the pooled 50+ band into
50-64 / 65-74 / 75+.** Candidate 10 conditions remarriage on the ego age band
``{18-34, 35-49, 50+}`` x years-since-dissolution band x origin
(divorced/widowed) x sex. Candidate 11's full structure is
``{18-34, 35-49, 50-64, 65-74, 75+}`` x duration x origin x sex, with the SAME
add-one smoothing at the same weight convention (``wbar_diss``, byte-identical
to candidate 1) and NO other change. The thin elderly-widow cells this creates
are exactly the cells whose true rates are near zero, so even generously
smoothed estimates are an order of magnitude below the pooled band's
~9x-too-high rate. Forensics 2 (#99) located ONE mechanism behind both
remaining level cells: candidate 10's pooled 50+ current-age remarriage band
applies a rate ~9x too high to 75+ widows, depleting the 75+ widowed stock
from the outflow side (68% of that gap) AND carrying the female count
over-production (+0.043/person in after-widowhood remarriage).

Everything else is byte-identical to candidate 10 -- the observed
undatable-marriage lifetime-count initial state (candidate 9's delta 1, which
candidate 10 inherited), kernel fertility, source-aligned widowhood, the NCHS
trend, the RNG, the K=20 mean-of-draws protocol, and
``fresh_run_artifact_schema`` conformance (per-draw per-cell rates [20, 46, 5];
undefined draw invalidates; report-only dispersion). Runner
``scripts/run_gate2_candidate11.py``, artifact ``runs/gate2_hazard_v11.json``.

Provable byte-identity (code-object reuse)
------------------------------------------
Because the delta changes ONLY the age-band constants, and candidate 10's
compute chain is band-count-agnostic (its loops and array shapes derive from
``len(REM_AGE_BANDS)``), candidate 11 REUSES candidate 10's EXACT code objects
for the pure-compute band-dependent functions (``simulate_holdout``,
``_draw_moments``, ``score_seed``, ``fit_remarriage_age_banded``,
``_build_sim_lookups``, ``_remarriage_probs_age_banded``), rebound
(:func:`_rebind`) to resolve their globals against THIS module -- so the
byte-identical simulation calls candidate 11's five age-band helpers (the one
delta) while every other statement is candidate 10's, guaranteed at the
bytecode level (``candidate11.simulate_holdout.__code__ is
candidate10.simulate_holdout.__code__``). This mirrors how candidate 10 itself
import-bound candidate 8's ``_assemble_sim_panel`` / ``_divorce_probs`` /
``_widow_probs`` / ``_fertility_probs_single``. Only the functions that emit
candidate-specific PROVENANCE strings into the artifact
(``fit_components``, ``_remarriage_age_band_diag``) are re-derived, so the
artifact truthfully reports the 5-band structure and the c11 delta.

Designed effect (registration)
------------------------------
The delta attacks the measured mechanism at both faces: ``share_widowed.75+``
| female (the outflow fix recovers most of the 68% aging-in margin, though the
~10% inflow shortfall remains untouched) and the female count (removing the
+0.043 after-widowhood over-production pulls the net from +0.044 toward the
residual-only ~0.00 the male side already enjoys). ``P(pass) ~= 0.55-0.65``;
modal failure if it fails: ``share_widowed.75+|female`` persisting on 2 seeds
via the untouched inflow shortfall.

Hard-stop precheck (inherited from candidate 1, unchanged): the scoring path
must reproduce, bit-for-bit, every committed full-panel reference moment, every
committed per-gate-seed ``rate_a``, and each gate seed's committed holdout-id
sha256, BEFORE any candidate is simulated. A SECOND hard gate (candidate 9's
delta-1 reconciliation, inherited via candidate 10, train-side, licensed) then
verifies the observed-residual reconciliation to remainder 0.0. Any mismatch
is a hard stop. Run ONCE; publish REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit; no statsmodels). Run from the repository root with the PSID
history files staged::

    .venv/bin/python scripts/run_gate2_candidate11.py
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

# Candidate 10 supplies the machinery this build deltas ONCE: its compute chain
# (which itself chains candidate 8's fit and candidate 9's delta 1), the
# fresh-run artifact-schema blocks, and -- via its imports -- candidate 1's
# precheck / verdict assembly and candidate 8's vectorised simulation helpers.
# Only the age-band conditioning of remarriage changes (50+ -> 50-64/65-74/75+).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate5 as c5  # noqa: E402
import run_gate2_candidate6 as c6  # noqa: E402
import run_gate2_candidate8 as c8  # noqa: E402
import run_gate2_candidate10 as c10  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402

# ``hpanel`` is referenced by candidate 10's ``score_seed`` code object, which
# candidate 11 reuses rebound to THIS module's globals (see ``_rebind`` below);
# it must stay a module global even though candidate 11's own source never
# names it directly (hence the F401 suppression).
from populace_dynamics.harness import panel as hpanel  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v11.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
CANDIDATE10_ARTIFACT = ROOT / "runs" / "gate2_hazard_v10.json"
CANDIDATE9_ARTIFACT = ROOT / "runs" / "gate2_hazard_v9.json"
CANDIDATE8_ARTIFACT = ROOT / "runs" / "gate2_hazard_v8.json"
CANDIDATE7_ARTIFACT = ROOT / "runs" / "gate2_hazard_v7.json"
CANDIDATE6_ARTIFACT = ROOT / "runs" / "gate2_hazard_v6.json"
CANDIDATE5_ARTIFACT = ROOT / "runs" / "gate2_hazard_v5.json"
FORENSICS_ARTIFACT = ROOT / "runs" / "gate2_forensics_v1.json"
FORENSICS2_ARTIFACT = ROOT / "runs" / "gate2_forensics2_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v11"
RUN_NAME = "gate2_hazard_v11"

#: This run's frozen-spec registration (issue #42, comment 4919417729).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4919417729"
)
#: The bare registration comment id, pinned for the artifact pointer.
REGISTRATION_POINTER = "4919417729"
#: The candidate-10 spec this build deltas ONCE (comment 4917059482, #98).
CANDIDATE10_REGISTRATION = c10.SPEC_REGISTRATION
#: The registration chain candidate 10 carried, threaded through for provenance.
CANDIDATE9_REGISTRATION = c10.CANDIDATE9_REGISTRATION
CANDIDATE8_REGISTRATION = c10.CANDIDATE8_REGISTRATION
CANDIDATE6_REGISTRATION = c10.CANDIDATE6_REGISTRATION
CANDIDATE5_REGISTRATION = c10.CANDIDATE5_REGISTRATION
CANDIDATE1_REGISTRATION = c10.CANDIDATE1_REGISTRATION
#: The forensics-1 diagnostic candidate 10's deltas cited (#94).
FORENSICS_REGISTRATION = c10.FORENSICS_REGISTRATION
#: Forensics 2 (#99) located the elderly-widow mechanism this delta attacks.
FORENSICS2_DIAGNOSTIC = "runs/gate2_forensics2_v1.json (#99)"

# --- Amended-estimator draw stream (inherited from candidate 10, unchanged). -
#: The amended 20-draw stream base: draw k uses default_rng(5200 + k), the
#: committed forensics convention (gates.yaml gate_2, amendment 1).
DRAW_SEED_BASE = 5200
N_DRAWS = 20

# --------------------------------------------------------------------------
# THE ONE DELTA vs candidate 10 (registration comment 4919417729)
# --------------------------------------------------------------------------
#: Candidate 10's remarriage age bands are (18-34 / 35-49 / 50+). Candidate 11
#: splits the pooled 50+ band into 50-64 / 65-74 / 75+ -- the SOLE change. The
#: searchsorted band index clips ages below 18 into the youngest band (there
#: are essentially no dissolved person-years there), identical to candidate 10.
REM_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (18, 34),
    (35, 49),
    (50, 64),
    (65, 74),
    (75, 120),
)
REM_AGE_LOWERS = np.array([lo for lo, _ in REM_AGE_BANDS], dtype=np.int64)
_REM_AGE_LABEL = {0: "18-34", 1: "35-49", 2: "50-64", 3: "65-74", 4: "75+"}

#: The single named delta (registration comment 4919417729).
DELTA_VS_CANDIDATE10 = (
    "EXACTLY ONE delta vs candidate 10 (comment 4917059482, merged #98): the "
    "remarriage current-age conditioning splits candidate 10's pooled 50+ band "
    "into 50-64 / 65-74 / 75+. Full structure: {18-34, 35-49, 50-64, 65-74, "
    "75+} x years-since-dissolution band x origin (divorced/widowed) x sex, "
    "SAME add-one smoothing (wbar_diss) at the same weight convention as "
    "candidate 1; no other change. Registered from forensics 2 (#99): "
    "candidate 10's pooled 50+ rate is ~9x too high for 75+ widows, depleting "
    "the 75+ widowed stock from the outflow side (68% of the gap) and carrying "
    "the female-count over-production (+0.043/person after-widowhood "
    "remarriage). Everything else -- the observed undatable-marriage "
    "lifetime-count initial state (candidate 9's delta 1, inherited), kernel "
    "fertility, source-aligned widowhood, the NCHS trend, the RNG, the K=20 "
    "mean-of-draws protocol and the fresh-run artifact schema -- is "
    "byte-identical to candidate 10"
)

# --- Frozen dials + pure helpers, reused (byte-identical; import-bound). ----
GATE_SEEDS = c1.GATE_SEEDS
SIM_SEED_BASE = c1.SIM_SEED_BASE  # 4200 (single-draw stream; provenance only)
EXACT_ATOL = c1.EXACT_ATOL
Components = c1.Components

YSD_BANDS = c1.YSD_BANDS
YSD_LOWERS = c1.YSD_LOWERS
_bands_vec = c1._bands_vec

# Candidate 8's simulation helpers (import-bound exactly as candidate 10 does).
_STATE = c8._STATE
_STATE_ABSORB = c8._STATE_ABSORB
_ASFR_LO = c8._ASFR_LO
_ASFR_HI = c8._ASFR_HI
_assemble_sim_panel = c8._assemble_sim_panel
_divorce_probs = c8._divorce_probs
_widow_probs = c8._widow_probs
_fertility_probs_single = c8._fertility_probs_single
WIDOW_BANDS = c8.WIDOW_BANDS
WIDOW_LOWERS = c8.WIDOW_LOWERS

# Delta 1 (candidate 9's observed residual + reconciliation), inherited via
# candidate 10, reused byte-for-byte.
observed_residual_counts = c10.observed_residual_counts
_delta1_reconciliation = c10._delta1_reconciliation

#: Candidate 10's simulation-lookup container (band-agnostic dataclass; its
#: ``rem_arr`` is shaped by ``_build_sim_lookups`` from ``len(REM_AGE_BANDS)``,
#: so it holds candidate 11's 5-band table unchanged). Reused as-is.
_SimLookupsC10 = c10._SimLookupsC10

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate11_run1_cache.json"
)

#: The marriage-count cells (delta 1's residual + the after-widowhood
#: remarriage the elderly split reshapes both feed these).
COUNT_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
)
#: The gated remarriage cells the delta directly reshapes.
REMARRIAGE_GATED_CELLS = (
    "remarriage.after_divorce",
    "remarriage.ysd0-4",
    "remarriage.ysd5-9",
    "remarriage.ysd10+",
)
#: The elderly-widow-stock cell the outflow fix is designed to recover.
MODAL_CELL = "share_widowed.75+|female"
#: The registered modal failure (comment 4919417729): share_widowed.75+|female
#: persisting via the untouched inflow shortfall.
REGISTERED_MODAL_CELLS = (MODAL_CELL,)
#: The three cells the registration names as the delta's targets: the widow
#: stock (outflow fix) and both marriage counts (after-widowhood remarriage).
THREE_TARGET_CELLS = (MODAL_CELL,) + COUNT_CELLS
#: All cells the delta touches: the three targets plus the remarriage flows.
TARGETED_CELLS = THREE_TARGET_CELLS + REMARRIAGE_GATED_CELLS


# --------------------------------------------------------------------------
# THE DELTA: age-band-conditioned remarriage, 50+ split into 50-64/65-74/75+.
# Re-derived (not rebound) because it emits the 5-band provenance string into
# the artifact; the COMPUTE is candidate 10's, band-count-agnostic.
# --------------------------------------------------------------------------
def _remarriage_age_band_diag(
    panel: transitions.MaritalPanel,
    train_ids: set[int],
    remarriage: dict[tuple[int, int, str, str], float],
) -> dict[str, Any]:
    """Provenance of the age-conditioned remarriage table + per-cell exposure.

    Records each cell's hazard, its train weighted exposure and its raw
    person-year count, and flags the thin cells (< 20 train person-years) the
    add-one smoothing pulls toward 0.5 -- the elderly-widow cells the 50+ split
    creates, whose true rates are near zero.
    """
    py = panel.person_years
    train_py = py[py["person_id"].isin(train_ids)]
    diss = train_py[
        train_py["marital_state"].isin(("divorced", "widowed"))
        & train_py["years_since_dissolution"].notna()
    ].copy()
    diss["ysd_band"] = _bands_vec(
        diss["years_since_dissolution"].astype("int64").to_numpy(),
        YSD_LOWERS,
        len(YSD_BANDS),
    )
    diss["age_band"] = _bands_vec(
        np.rint(diss["age"].to_numpy()).astype(np.int64),
        REM_AGE_LOWERS,
        len(REM_AGE_BANDS),
    )
    wexp = diss.groupby(["age_band", "ysd_band", "marital_state", "sex"])[
        "weight"
    ].sum()
    npy = diss.groupby(["age_band", "ysd_band", "marital_state", "sex"]).size()

    cells: dict[str, Any] = {}
    thin: list[str] = []
    for (ab, b, origin, sex), h in sorted(remarriage.items()):
        label = f"age{_REM_AGE_LABEL[ab]}|ysd{b}|{origin}|{sex}"
        n = int(npy.get((ab, b, origin, sex), 0))
        cells[label] = {
            "hazard": float(h),
            "train_person_years": n,
            "train_weighted_exposure": float(
                wexp.get((ab, b, origin, sex), 0.0)
            ),
        }
        if n < 20:
            thin.append(label)
    return {
        "representation": (
            "weighted empirical remarriage hazard by ego age band "
            "(18-34/35-49/50-64/65-74/75+) x years-since-dissolution band x "
            "origin (divorced/widowed) x sex, add-one smoothed at wbar_diss; "
            "candidate 10's pooled 50+ band split into 50-64/65-74/75+ (THE "
            "ONE DELTA vs candidate 10)"
        ),
        "age_bands": [list(b) for b in REM_AGE_BANDS],
        "ysd_bands": [list(b) for b in YSD_BANDS],
        "n_cells": len(remarriage),
        "n_thin_cells_lt_20_py": len(thin),
        "thin_cells_lt_20_py": thin,
        "cells": cells,
    }


def fit_components(
    panel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    mh_records: pd.DataFrame,
    birth_records: pd.DataFrame,
    order_map: pd.DataFrame,
    train_ids: set[int],
) -> Components:
    """Candidate 8's components with the remarriage table re-stratified.

    Starts from :func:`candidate8.fit_components` -- so first marriage,
    divorce, the surviving-spouse widowhood level, the committed NCHS betas,
    the spousal-gap draw AND the single-year triangular-kernel fertility are
    byte-identical to candidate 8 (hence to candidate 10) by construction.
    Then THE DELTA replaces ``remarriage`` with candidate 11's age-band table
    (:func:`fit_remarriage_age_banded`, candidate 10's exact code reading
    candidate 11's 5-band constants), splitting the pooled 50+ band into
    50-64 / 65-74 / 75+. Candidate 8's order-split / rescale provenance is
    scrubbed (as candidate 10 did) and the 5-band table's provenance recorded.
    The observed undatable-marriage lifetime-count initial state (candidate 9's
    delta 1) is applied at simulation-assembly time, unchanged.
    """
    base = c8.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )

    # THE DELTA: candidate 10's age-band remarriage with the 50+ band split.
    remarriage, wbar_diss = fit_remarriage_age_banded(panel, train_ids)
    base.remarriage = remarriage

    # Scrub candidate 8's order-split / rescale provenance; record the delta's.
    for key in (
        "remarriage_order_diagnostics",
        "remarriage_order_prerescale_cells",
        "remarriage_candidate6",
        "remarriage_rescale",
        "remarriage_mean_dissolved_weight_check",
        "delta_vs_candidate6",
    ):
        base.meta.pop(key, None)
    base.meta["remarriage_representation"] = (
        "weighted empirical hazard by ego age band "
        "(18-34/35-49/50-64/65-74/75+) x years-since-dissolution band x "
        "origin (divorced/widowed) x sex, add-one smoothed at wbar_diss -- "
        "candidate 10's pooled 50+ band split into 50-64/65-74/75+ (THE ONE "
        "DELTA vs candidate 10; comment 4919417729)"
    )
    base.meta["remarriage_age_banded"] = _remarriage_age_band_diag(
        panel, train_ids, remarriage
    )
    base.meta["remarriage_mean_dissolved_weight"] = float(wbar_diss)
    base.meta["delta_vs_candidate10"] = DELTA_VS_CANDIDATE10
    return base


# --------------------------------------------------------------------------
# Provable byte-identity: reuse candidate 10's EXACT code objects, rebound so
# their global lookups resolve against THIS module (candidate 11). The bytecode
# is candidate 10's; the only observable change is that ``REM_AGE_BANDS`` (and
# the helpers keyed on it) are candidate 11's 5-band split -- the one delta.
# --------------------------------------------------------------------------
def _rebind(fn: types.FunctionType) -> types.FunctionType:
    """Return a function sharing ``fn``'s code object but this module's globals.

    Candidate 10's band-dependent compute is band-count-agnostic (loops and
    array shapes derive from ``len(REM_AGE_BANDS)``). Reusing its code object
    verbatim -- only redirecting global name resolution to candidate 11's
    module -- makes ``everything else byte-identical`` provable at the bytecode
    level (``candidate11.f.__code__ is candidate10.f.__code__``) while the
    reused code calls candidate 11's 5-band age helpers.
    """
    return types.FunctionType(
        fn.__code__,
        globals(),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


#: THE DELTA's fit: candidate 10's exact ``fit_remarriage_age_banded`` code,
#: reading candidate 11's 5-band ``REM_AGE_BANDS`` -- so the pooled 50+ band is
#: estimated as three bands 50-64 / 65-74 / 75+ with the same add-one smoothing.
fit_remarriage_age_banded = _rebind(c10.fit_remarriage_age_banded)
#: Candidate 10's exact remarriage-probability lookup, indexing candidate 11's
#: 5-band ``rem_arr`` (``[age_band, ysd_band, origin, sex]``).
_remarriage_probs_age_banded = _rebind(c10._remarriage_probs_age_banded)
#: Candidate 10's exact lookup builder; ``rem_arr`` gains candidate 11's fifth
#: and fourth age rows automatically (shape from ``len(REM_AGE_BANDS)``).
_build_sim_lookups = _rebind(c10._build_sim_lookups)
#: Candidate 10's EXACT vectorised annual simulation. It calls
#: ``_remarriage_probs_age_banded`` / ``_build_sim_lookups`` (candidate 11's,
#: above) and ``observed_residual_counts`` (delta 1); every other statement --
#: first marriage, divorce, widowhood, fertility, the spawned gap-draw stream,
#: the per-year uniform blocks -- is candidate 10's, byte-for-byte.
simulate_holdout = _rebind(c10.simulate_holdout)
#: Candidate 10's exact single-draw moment builder.
_draw_moments = _rebind(c10._draw_moments)
#: Candidate 10's exact per-seed mean-over-K=20 scorer.
score_seed = _rebind(c10.score_seed)

# Fresh-run artifact-schema blocks are band-independent (they operate on the
# scored per-seed dicts); import-bound from candidate 10 (identical N_DRAWS /
# DRAW_SEED_BASE), so the [20, 46, 5] cube, the undefined-draw rule and the
# report-only dispersion are candidate 10's exact assembly.
_per_draw_per_cell_rates_block = c10._per_draw_per_cell_rates_block
_undefined_draw_block = c10._undefined_draw_block
_per_draw_dispersion_block = c10._per_draw_dispersion_block


# --------------------------------------------------------------------------
# Count-cell tilt vs candidate 10 (the elderly split's designed count effect)
# --------------------------------------------------------------------------
def _count_cell_tilt(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """The count-cell movement the elderly split is designed to produce.

    Candidate 10 left the female count at net +0.044 ln (3/5) and the male at
    +0.025 ln (4/5). The registration (comment 4919417729) designs the 50+
    split to remove the +0.043/person after-widowhood remarriage
    over-production, pulling the female net from +0.044 toward the residual-only
    ~0.00 the male side already enjoys. Reports candidate 11's realised signed
    ln-tilt ``ln(rbar / rate_a)`` per seed for each count cell against
    candidate 10's committed single-scored value, so the movement is auditable.
    """
    c10art = (
        json.loads(CANDIDATE10_ARTIFACT.read_text())
        if CANDIDATE10_ARTIFACT.exists()
        else None
    )
    by10 = {s["seed"]: s for s in c10art["per_seed"]} if c10art else {}

    out: dict[str, Any] = {
        "design_note": (
            "registration comment 4919417729: the 50+ split removes the "
            "+0.043/person after-widowhood remarriage over-production, "
            "designed to pull the female count net from candidate 10's +0.044 "
            "ln toward the residual-only ~0.00 the male side already enjoys; "
            "the male net (+0.025 ln, 4/5 under c10) sits near its boundary. "
            "The movement is reported here against candidate 10's committed "
            "count scores."
        ),
        "candidate10_female_net_ln": 0.044,
        "candidate10_male_net_ln": 0.025,
        "after_widowhood_over_production_removed_ln": 0.043,
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
            c10rec = by10[s["seed"]]["gated_cells"][cell] if by10 else None
            rows.append(
                {
                    "seed": s["seed"],
                    "rbar": rbar,
                    "rate_a": rate_a,
                    "signed_ln_tilt": tilt,
                    "score_abs_ln": rec["score"],
                    "tolerance": rec["tolerance"],
                    "pass": rec["pass"],
                    "candidate10_score": (c10rec["score"] if c10rec else None),
                    "candidate10_pass": (c10rec["pass"] if c10rec else None),
                    "delta_score_vs_c10": (
                        float(rec["score"] - c10rec["score"])
                        if c10rec
                        else None
                    ),
                }
            )
        n_pass = sum(r["pass"] for r in rows)
        c10_n_pass = (
            sum(1 for r in rows if r["candidate10_pass"]) if by10 else None
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
            "candidate10_n_seeds_pass": c10_n_pass,
        }
    both_pass = all(out["cells"][c]["n_seeds_pass"] >= 4 for c in COUNT_CELLS)
    out["count_cells_cleared"] = bool(both_pass)
    out["summary"] = (
        "both marriage-count cells clear >= 4/5 seeds"
        if both_pass
        else "at least one marriage-count cell holds below 4/5 seeds"
    )
    return out


# --------------------------------------------------------------------------
# Candidate-10 comparison: how the three target cells moved + the elderly-band
# remarriage rate table (candidate 10's pooled 50+ vs candidate 11's split).
# --------------------------------------------------------------------------
def _candidate10_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Movement of the delta's targets and the elderly-band rate table.

    Compares candidate 11's per-seed rbar-scores for the three registered
    target cells (share_widowed.75+|female + both marriage counts) and the four
    gated remarriage flows against candidate 10's committed scores, and
    tabulates the widowed remarriage rates candidate 10 pooled at 50+ against
    candidate 11's 50-64 / 65-74 / 75+ split (from the seed-0 fit) -- so the
    ~9x-too-high-for-75+ mechanism forensics 2 located is visible in the fit.
    """
    c10art = (
        json.loads(CANDIDATE10_ARTIFACT.read_text())
        if CANDIDATE10_ARTIFACT.exists()
        else None
    )
    by10 = {s["seed"]: s for s in c10art["per_seed"]} if c10art else {}

    def move(cell: str) -> dict[str, Any]:
        rows = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            c10rec = by10[s["seed"]]["gated_cells"][cell] if by10 else None
            rows.append(
                {
                    "seed": s["seed"],
                    "tolerance": rec["tolerance"],
                    "c10_score": c10rec["score"] if c10rec else None,
                    "c11_score": rec["score"],
                    "c10_rbar": c10rec["rbar"] if c10rec else None,
                    "c11_rbar": rec["rbar"],
                    "rate_a": rec["rate_a"],
                    "c10_pass": c10rec["pass"] if c10rec else None,
                    "c11_pass": rec["pass"],
                }
            )
        c10_np = sum(1 for r in rows if r["c10_pass"]) if by10 else None
        c11_np = sum(1 for r in rows if r["c11_pass"])
        return {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "c10_n_seeds_pass": c10_np,
            "c11_n_seeds_pass": c11_np,
            "improved": (
                bool(c11_np > c10_np) if c10_np is not None else None
            ),
        }

    # The elderly-band rate table: candidate 10's pooled 50+ widowed/divorced
    # remarriage rates (from its committed seed-0 fit) against candidate 11's
    # 50-64 / 65-74 / 75+ split (this run's seed-0 fit).
    c11_cells = per_seed[0]["component_meta"]["remarriage_age_banded"]["cells"]
    c10_cells = (
        c10art["per_seed"][0]["component_meta"]["remarriage_age_banded"][
            "cells"
        ]
        if c10art
        else {}
    )
    elderly: dict[str, Any] = {}
    for origin in ("widowed", "divorced"):
        for sex in ("female", "male"):
            for b in range(len(YSD_BANDS)):
                key = f"{origin}|{sex}|ysd{b}"
                pooled = c10_cells.get(f"age50+|ysd{b}|{origin}|{sex}")
                elderly[key] = {
                    "c10_pooled_50plus_hazard": (
                        pooled["hazard"] if pooled else None
                    ),
                    "c10_pooled_50plus_py": (
                        pooled["train_person_years"] if pooled else None
                    ),
                    "c11_50_64_hazard": _cell_hz(
                        c11_cells, f"age50-64|ysd{b}|{origin}|{sex}"
                    ),
                    "c11_65_74_hazard": _cell_hz(
                        c11_cells, f"age65-74|ysd{b}|{origin}|{sex}"
                    ),
                    "c11_75plus_hazard": _cell_hz(
                        c11_cells, f"age75+|ysd{b}|{origin}|{sex}"
                    ),
                    "c11_50_64_py": _cell_py(
                        c11_cells, f"age50-64|ysd{b}|{origin}|{sex}"
                    ),
                    "c11_65_74_py": _cell_py(
                        c11_cells, f"age65-74|ysd{b}|{origin}|{sex}"
                    ),
                    "c11_75plus_py": _cell_py(
                        c11_cells, f"age75+|ysd{b}|{origin}|{sex}"
                    ),
                }
    return {
        "note": (
            "candidate 11 = candidate 10 (comment 4917059482, #98) with "
            "exactly one delta: the pooled 50+ remarriage age band split into "
            "50-64 / 65-74 / 75+. Scores compared cell-by-cell against "
            "candidate 10's committed run (runs/gate2_hazard_v10.json)."
        ),
        "three_target_cells": {c: move(c) for c in THREE_TARGET_CELLS},
        "remarriage_gated_cells": {c: move(c) for c in REMARRIAGE_GATED_CELLS},
        "elderly_band_rate_table_seed0": {
            "note": (
                "widowed/divorced remarriage hazard at ysd band b: candidate "
                "10's single pooled 50+ rate vs candidate 11's 50-64/65-74/75+ "
                "split (seed-0 train fit). The 75+ widowed rates fall an order "
                "of magnitude below the pooled rate (the mechanism forensics 2 "
                "located)."
            ),
            "cells": elderly,
        },
    }


def _cell_hz(cells: dict[str, Any], label: str) -> float | None:
    rec = cells.get(label)
    return float(rec["hazard"]) if rec else None


def _cell_py(cells: dict[str, Any], label: str) -> int | None:
    rec = cells.get(label)
    return int(rec["train_person_years"]) if rec else None


# --------------------------------------------------------------------------
# Registered modal / targeted / decider (modal = share_widowed.75+|female)
# --------------------------------------------------------------------------
def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal (share_widowed.75+|female), targets, and the decider."""
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
    modal_failed = any(c in fails_by_cell for c in REGISTERED_MODAL_CELLS)
    modal_failed_seeds = sorted(fails_by_cell.get(MODAL_CELL, []))
    # The registered modal materialises if share_widowed.75+|female persists on
    # >= 2 seeds via the untouched inflow shortfall (comment 4919417729).
    modal_materialized = len(modal_failed_seeds) >= 2
    distinct_fail_cells = {
        f["cell"] for f in verdict["all_failing_gated_cells"]
    }

    if gate_pass:
        decider = "none (gate passed)"
    elif n_pass_no_modal >= 4:
        decider = (
            "the registered modal cell share_widowed.75+|female (forgiving it "
            "flips >= 4 seeds to pass)"
        )
    elif seeds_pass_if_forgiven(set(TARGETED_CELLS)) >= 4:
        decider = (
            "the delta-targeted cells (forgiving the widow-stock/count/"
            "remarriage targets flips >= 4 seeds to pass)"
        )
    else:
        decider = (
            "broader than the registered modal + delta-targeted cells "
            "(other gated cells also hold the gate below 4 passing seeds)"
        )

    return {
        "registered_modal": (
            "share_widowed.75+|female persisting on >= 2 seeds via the "
            "untouched inflow shortfall (the 50+ split fixes the outflow, not "
            "the ~10% aging-in inflow shortfall)"
        ),
        "modal_cells": list(REGISTERED_MODAL_CELLS),
        "modal_failed": modal_failed,
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


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins + the candidate-11 schema, c1-c10 and forensics shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for name in (1, 5, 6, 7, 8, 9, 10):
        pins[f"candidate{name}_runner"] = (
            f"scripts/run_gate2_candidate{name}.py"
        )
        pins[f"candidate{name}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{name}.py"
        )
    pins["candidate10_artifact"] = "runs/gate2_hazard_v10.json"
    pins["candidate10_artifact_sha256"] = c1._sha_of_file(CANDIDATE10_ARTIFACT)
    pins["candidate9_artifact"] = "runs/gate2_hazard_v9.json"
    pins["candidate9_artifact_sha256"] = c1._sha_of_file(CANDIDATE9_ARTIFACT)
    pins["forensics2_runner"] = "scripts/gate2_forensics2.py"
    pins["forensics2_runner_sha256"] = c1._sha_of_file(
        ROOT / "scripts" / "gate2_forensics2.py"
    )
    pins["forensics2_artifact"] = "runs/gate2_forensics2_v1.json"
    pins["forensics2_artifact_sha256"] = c1._sha_of_file(FORENSICS2_ARTIFACT)
    pins["estimator"] = "mean_over_K20_draws (5200 + k, k=0..19)"
    pins["delta"] = (
        "one delta vs candidate 10: remarriage current-age conditioning "
        "splits the pooled 50+ band into 50-64/65-74/75+ (full: "
        "{18-34,35-49,50-64,65-74,75+} x ysd x origin x sex, same add-one "
        "smoothing). Everything else byte-identical to candidate 10 (compute "
        "chain reuses candidate 10's exact code objects)"
    )
    pins["byte_identity_code_objects"] = {
        name: (getattr(c10, name).__code__ is globals()[name].__code__)
        for name in (
            "simulate_holdout",
            "_draw_moments",
            "score_seed",
            "fit_remarriage_age_banded",
            "_build_sim_lookups",
            "_remarriage_probs_age_banded",
        )
    }
    return pins


def _model_block() -> dict[str, Any]:
    """The model block, edited for the single candidate-11 delta."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a "
            "mortality-composed widowhood component (candidate 10 with one "
            "delta: the pooled 50+ remarriage current-age band split into "
            "50-64 / 65-74 / 75+), scored under the amended "
            "mean-over-K=20-draws estimator"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "delta_vs_candidate10": DELTA_VS_CANDIDATE10,
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic spline on age, "
                "knots 20/22/25/30/40, sex, birth-decade cohort -- "
                "BYTE-IDENTICAL to candidate 10 at a shared draw seed (the "
                "delta touches only the remarriage age bands)"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x order, "
                "add-one smoothed -- estimator BYTE-IDENTICAL to candidate 10 "
                "(events downstream of the remarriage trajectory move under "
                "the delta)"
            ),
            "widowhood": (
                "COMPOSED, parametric in period; surviving-spouse "
                "marriage-history widowhood level x candidate 5's committed "
                "NCHS betas -- estimator BYTE-IDENTICAL to candidate 10"
            ),
            "remarriage": (
                "THE DELTA: weighted empirical hazard by ego age band "
                "(18-34/35-49/50-64/65-74/75+) x years-since-dissolution band "
                "x origin (divorced/widowed) x sex, add-one smoothed at "
                "wbar_diss. Candidate 10's pooled 50+ band split into "
                "50-64/65-74/75+ so 75+ widows get their own near-zero rate "
                "instead of the ~9x-too-high pooled rate forensics 2 (#99) "
                "located"
            ),
            "fertility": (
                "single-year-of-age triangular-kernel rates within parity "
                "(0/1/2/3+) x birth-decade cohort -- BYTE-IDENTICAL to "
                "candidate 10 (no fertility delta)"
            ),
            "lifetime_marriage_count_initial_state": (
                "candidate 9's delta 1 (inherited via candidate 10, "
                "unchanged): each holdout person's simulated lifetime-marriage "
                "count initialises at their OBSERVED residual (n_marriages "
                "minus datable in-exposure marriage events) and accumulates "
                "the simulated datable transitions. An observed initial state "
                "per protocol; RNG-neutral; hazards untouched"
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
            "remarriage_age_bands": (
                "18-34 / 35-49 / 50-64 / 65-74 / 75+ on the ego's "
                "calendar-year age (year - birth_year) at the dissolved "
                "person-year and the remarriage event; the searchsorted band "
                "index clips ages below 18 into the youngest band (as "
                "candidate 10)"
            ),
            "remarriage_smoothing": (
                "the SAME add-one smoothing (wnum + wbar_diss)/(wden + "
                "2*wbar_diss) and the SAME wbar_diss (mean weight over ALL "
                "dissolved rows) as candidate 1 / candidate 10; only the 50+ "
                "band is split; NO aggregate-preservation rescale"
            ),
            "byte_identity": (
                "candidate 11 reuses candidate 10's EXACT code objects for "
                "simulate_holdout, _draw_moments, score_seed, "
                "fit_remarriage_age_banded, _build_sim_lookups and "
                "_remarriage_probs_age_banded (rebound to this module's "
                "globals), so everything but the age-band constants is "
                "candidate 10's bytecode -- byte-identity is machine-checkable "
                "(revision_pins.byte_identity_code_objects)"
            ),
            "everything_else": (
                "the observed undatable-marriage lifetime-count initial state "
                "(delta 1), the surviving-spouse widowhood level, the "
                "committed NCHS betas, the first-marriage spline, divorce, the "
                "spousal-gap draw, the single-year triangular-kernel "
                "fertility, the competing-risk step, the spawned gap-draw "
                "stream, one sequence per person per draw, and the locked "
                "protocol are byte-identical to candidate 10"
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

    # Preflight: candidate 10 (the base), its fit chain (9/8/7/6/5) and both
    # forensics artifacts must be present, plus the candidate-5 NCHS references.
    for name, path in (
        ("candidate-10", CANDIDATE10_ARTIFACT),
        ("candidate-9", CANDIDATE9_ARTIFACT),
        ("candidate-8", CANDIDATE8_ARTIFACT),
        ("candidate-7", CANDIDATE7_ARTIFACT),
        ("candidate-6", CANDIDATE6_ARTIFACT),
        ("candidate-5", CANDIDATE5_ARTIFACT),
        ("forensics", FORENSICS_ARTIFACT),
        ("forensics-2", FORENSICS2_ARTIFACT),
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

    # Byte-identity guard: the reused compute chain MUST share candidate 10's
    # exact code objects (the one-delta contract). Fail fast if not.
    for name in (
        "simulate_holdout",
        "_draw_moments",
        "score_seed",
        "fit_remarriage_age_banded",
        "_build_sim_lookups",
        "_remarriage_probs_age_banded",
    ):
        if globals()[name].__code__ is not getattr(c10, name).__code__:
            raise RuntimeError(
                f"{name} does not share candidate 10's code object; the "
                "one-delta byte-identity contract is violated."
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

    # Hard gate 2: the delta-1 reconciliation (candidate 9's, inherited via
    # candidate 10; train-side, licensed). Must reconcile to remainder 0.0.
    reconciliation = _delta1_reconciliation(panel, mh_records, GATE_SEEDS)
    if verbose:
        pid_max = reconciliation["per_person_identity_max_abs_residual"]
        agg_max = reconciliation["aggregate_reconciliation_max_abs_remainder"]
        print(
            "delta-1 reconciliation reconciled="
            f"{reconciliation['reconciled']} "
            f"(per-person identity max={pid_max:.2e}, "
            f"aggregate max remainder={agg_max:.2e}, "
            f"residual>=0: {reconciliation['residual_nonnegative']})"
        )
    if not reconciliation["reconciled"]:
        raise RuntimeError(
            "delta-1 reconciliation failed: the observed residual does not "
            "reproduce the forensics Q1 reconciliation to remainder 0.0 on "
            "train; refusing to proceed."
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

    # The undefined-draw rule is pre-specified: any undefined gated draw
    # invalidates the run (no verdict may be certified from an invalid run).
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
    comparison = _candidate10_comparison(per_seed)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 11",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate10_registration": CANDIDATE10_REGISTRATION,
        "candidate9_registration": CANDIDATE9_REGISTRATION,
        "candidate8_registration": CANDIDATE8_REGISTRATION,
        "candidate6_registration": CANDIDATE6_REGISTRATION,
        "candidate5_registration": CANDIDATE5_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "forensics_diagnostic": FORENSICS_REGISTRATION,
        "forensics2_diagnostic": FORENSICS2_DIAGNOSTIC,
        "delta_vs_candidate10": DELTA_VS_CANDIDATE10,
        "amended_estimator": (
            "gates.yaml gate_2 amendment 1 (ratified 2026-07-08, flip #97): "
            "per-cell score |ln(rbar / rate_a)| with rbar the mean over K=20 "
            "draws (default_rng(5200 + k), k=0..19) of the cell rate "
            "(inherited from candidate 10, unchanged)"
        ),
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79); amendment 1 flipped "
            "live (#97). Protocol/views/tolerances/schema read at runtime; no "
            "threshold moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.55-0.65",
            "conjunction_estimate": 0.60,
            "modal_failure": (
                "share_widowed.75+|female persisting on 2 seeds via the "
                "untouched inflow shortfall (the 50+ split fixes the outflow "
                "over-remarriage, not the ~10% aging-in inflow shortfall)"
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
                "the fresh candidate-11 one-shot run registered AFTER the "
                "2026-07-08 ratification (registration 4919417729)"
            ),
            "per_draw_per_cell_rates": per_draw_cube,
            "undefined_draw_rule": undefined_block,
            "per_draw_dispersion_disclosure": dispersion_block,
        },
        "data": data_meta,
        "precheck": precheck,
        "delta1_reconciliation": reconciliation,
        "per_seed": per_seed,
        "seed_conjunction": seed_conjunction,
        "report_only": report_block,
        "count_cell_tilt": count_tilt,
        "candidate10_comparison": comparison,
        "modal_failure_materialized": modal,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "registration_pointer": REGISTRATION_POINTER,
            "candidate10_registration": CANDIDATE10_REGISTRATION,
            "candidate10_artifact": "runs/gate2_hazard_v10.json",
            "forensics2_diagnostic": "runs/gate2_forensics2_v1.json (#99)",
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
        cm = comparison["three_target_cells"]
        for c in THREE_TARGET_CELLS:
            print(
                f"  {c}: c10 {cm[c]['c10_n_seeds_pass']}/5 -> "
                f"c11 {cm[c]['c11_n_seeds_pass']}/5"
            )
        print(
            "modal (share_widowed.75+|female) materialized="
            f"{modal['modal_materialized']} "
            f"(failed seeds {modal['modal_failed_seeds']}); "
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
