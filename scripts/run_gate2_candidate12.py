"""Gate-2 candidate 12 (run 1): candidate 11 + EXACTLY TWO deltas, scored under
the amended mean-over-K=20-draws estimator.

The TWELFTH pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42 comment
4925020986 (``SPEC_REGISTRATION``): candidate 11's frozen spec (comment
4919417729, ``scripts/run_gate2_candidate11.py``, merged #101) verbatim EXCEPT
two deltas, registered from gate-2 forensics 3 (#102,
``runs/gate2_forensics3_v1.json``). One-shot; no constant moves after the
registration comment; published REGARDLESS of verdict.

The two deltas vs candidate 11 (everything else byte-identical)
--------------------------------------------------------------
**Delta 1 -- entry-widowed observed initial state (forensics 3 Q6).** Forensics
3 found 12.1% of the reference 75+ female widowed stock is structurally
unreachable: it is marital status carried from spouse deaths PREDATING the
person's first observed PSID wave (onset year < first observed wave), with zero
non-derivable residual. No transition rate on the observed support produces it;
only an observed-initial-state injection can (the candidate-9 delta-1
precedent). Candidate 12 injects it: for every person observed already-widowed
at their first observed wave (all sexes, all ages), the reference-carried
widowed person-years are written onto the simulated panel post-assembly -- a
pure, RNG-neutral state injection symmetric with ``transitions``'s reference
construction, reusing forensics 3's carried-status classification
(:func:`gate2_forensics3.widowed_75plus_support_taxonomy`'s ``carried`` mask,
generalised to all ages/sexes).

**Delta 2 -- spousal-age-gap draw conditioned on the ego's age band (forensics
3 Q7).** Candidate 11 imputes each spouse's age from a single POOLED sex-
specific empirical gap distribution (``gap = self_birth - spouse_birth``),
ignoring the ego's age; forensics 3 traced the simulated young-widow surplus
(3.16x reference at 15-49, 1.41x at 50-64, while all 45+ incidence ratios are
<= 1) to that pooled draw assigning old spouses to young egos. Candidate 12
stratifies the train gap distribution on the ego's age band at marriage --
``{18-34, 35-49, 50-64, 65+}``, 1-year gap bins as before -- and imputes each
spouse's age at the marriage event from the ego's own band. A band with < 200
weighted train couples falls back to the adjacent (next-younger) pooled band,
documented per seed. The gap-draw stream is candidate 4's SPAWNED stream (it
does not advance the scored ``rng``), so first marriage, divorce, widowhood,
remarriage and fertility draws are byte-identical to candidate 11 at a shared
draw seed; only the imputed spouse ages (and thence the widowhood hazard) move.

Everything else is byte-identical to candidate 11 -- the 5-band remarriage
current-age table (candidate 11's delta over candidate 10), the observed
undatable-marriage lifetime-count initial state (candidate 9's delta 1), the
kernel fertility, the source-aligned NCHS-trended widowhood level, the RNG, the
K=20 mean-of-draws protocol, and ``fresh_run_artifact_schema`` conformance
(per-draw per-cell rates [20, 46, 5]; undefined draw invalidates; report-only
dispersion). Runner ``scripts/run_gate2_candidate12.py``, artifact
``runs/gate2_hazard_v12.json``.

Provable byte-identity (code-object reuse)
------------------------------------------
Candidate 11 rebinds candidate 10's exact code objects for the whole compute
chain. Candidate 12's two deltas touch ``simulate_holdout`` (the spouse-age-gap
draw and the post-assembly widowed injection) and ``fit_components`` (the
banded gap fit), so those two functions are RE-IMPLEMENTED. Every OTHER
band/estimator function is still candidate 11's (== candidate 10's) exact code
object, rebound (:func:`_rebind`) to THIS module's globals -- ``_draw_moments``
and ``score_seed`` (which therefore call candidate 12's ``simulate_holdout`` /
``fit_components`` through this module's globals),
``fit_remarriage_age_banded``, ``_build_sim_lookups`` and
``_remarriage_probs_age_banded``. Byte-identity of the reused chain is machine-
checkable (``revision_pins.byte_identity_code_objects``); the two delta'd
functions are pinned as DIVERGED from candidate 11.

Designed effect (registration)
------------------------------
Delta 1 lifts the 75+ widowed-stock ratio from 0.812 toward 0.93 (the modal
cell ``share_widowed.75+|female`` clears with room); delta 2 deflates the young
widowed pools toward reference, pulling the female marriage count back from
candidate 11's regression. ``P(pass) ~= 0.5-0.6``; modal failure if it fails:
the female count cells recovering only partially; secondary: an unforeseen
interaction of delta 2 with the widowhood-incidence cells.

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

    .venv/bin/python scripts/run_gate2_candidate12.py
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

# Candidate 11 supplies the machinery this build deltas TWICE: its compute chain
# (5-band remarriage over candidate 10 over candidate 8's fit and candidate 9's
# delta 1), the fresh-run artifact-schema blocks, and -- via its imports --
# candidate 1's precheck / verdict assembly and candidate 8's vectorised
# simulation helpers. Forensics 3 supplies the reused carried-status accounting.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import gate2_forensics3 as f3  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate4 as c4  # noqa: E402
import run_gate2_candidate5 as c5  # noqa: E402
import run_gate2_candidate6 as c6  # noqa: E402
import run_gate2_candidate8 as c8  # noqa: E402
import run_gate2_candidate9 as c9  # noqa: E402
import run_gate2_candidate10 as c10  # noqa: E402
import run_gate2_candidate11 as c11  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402

# ``hpanel`` is referenced by candidate 10/11's ``score_seed`` code object,
# which candidate 12 reuses rebound to THIS module's globals (see ``_rebind``);
# it must stay a module global even though candidate 12's own source never
# names it directly (hence the F401 suppression).
from populace_dynamics.harness import panel as hpanel  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v12.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
CANDIDATE11_ARTIFACT = ROOT / "runs" / "gate2_hazard_v11.json"
CANDIDATE10_ARTIFACT = ROOT / "runs" / "gate2_hazard_v10.json"
CANDIDATE9_ARTIFACT = ROOT / "runs" / "gate2_hazard_v9.json"
CANDIDATE8_ARTIFACT = ROOT / "runs" / "gate2_hazard_v8.json"
CANDIDATE7_ARTIFACT = ROOT / "runs" / "gate2_hazard_v7.json"
CANDIDATE6_ARTIFACT = ROOT / "runs" / "gate2_hazard_v6.json"
CANDIDATE5_ARTIFACT = ROOT / "runs" / "gate2_hazard_v5.json"
FORENSICS3_ARTIFACT = ROOT / "runs" / "gate2_forensics3_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v12"
RUN_NAME = "gate2_hazard_v12"

#: This run's frozen-spec registration (issue #42, comment 4925020986).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4925020986"
)
#: The bare registration comment id, pinned for the artifact pointer.
REGISTRATION_POINTER = "4925020986"
#: The candidate-11 spec this build deltas TWICE (comment 4919417729, #101).
CANDIDATE11_REGISTRATION = c11.SPEC_REGISTRATION
#: The registration chain threaded through for provenance.
CANDIDATE10_REGISTRATION = c11.CANDIDATE10_REGISTRATION
CANDIDATE9_REGISTRATION = c11.CANDIDATE9_REGISTRATION
CANDIDATE8_REGISTRATION = c11.CANDIDATE8_REGISTRATION
CANDIDATE6_REGISTRATION = c11.CANDIDATE6_REGISTRATION
CANDIDATE5_REGISTRATION = c11.CANDIDATE5_REGISTRATION
CANDIDATE1_REGISTRATION = c11.CANDIDATE1_REGISTRATION
#: Forensics 3 (#102) located both deltas' mechanisms.
FORENSICS3_DIAGNOSTIC = "runs/gate2_forensics3_v1.json (#102)"

# --- Amended-estimator draw stream (inherited from candidate 11, unchanged). -
#: The amended 20-draw stream base: draw k uses default_rng(5200 + k), the
#: committed forensics convention (gates.yaml gate_2, amendment 1).
DRAW_SEED_BASE = 5200
N_DRAWS = 20

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

# Candidate 8's simulation helpers (import-bound exactly as candidate 11 does).
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

# Candidate 9's observed residual + reconciliation (delta 1 of candidate 9),
# inherited via candidates 10/11, reused byte-for-byte (candidate 11's are the
# same objects).
observed_residual_counts = c9.observed_residual_counts
_delta1_reconciliation = c9._delta1_reconciliation

#: Candidate 10's simulation-lookup container (band-agnostic dataclass), reused.
_SimLookupsC10 = c11._SimLookupsC10

# --- The 5-band remarriage table (candidate 11's delta; UNCHANGED here). -----
REM_AGE_BANDS = c11.REM_AGE_BANDS
REM_AGE_LOWERS = c11.REM_AGE_LOWERS
_REM_AGE_LABEL = c11._REM_AGE_LABEL

# --------------------------------------------------------------------------
# DELTA 2 constants: spousal-age-gap ego-age bands (registration 4925020986)
# --------------------------------------------------------------------------
#: The ego's age band at marriage the gap draw conditions on. Distinct from the
#: 5-band remarriage current-age table (``REM_AGE_BANDS``); the gap uses the
#: registered {18-34, 35-49, 50-64, 65+}. searchsorted clips below-18 marriage
#: ages into the youngest band (rare underage records).
GAP_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (18, 34),
    (35, 49),
    (50, 64),
    (65, 120),
)
GAP_AGE_LOWERS = np.array([lo for lo, _ in GAP_AGE_BANDS], dtype=np.int64)
_GAP_AGE_LABEL = {0: "18-34", 1: "35-49", 2: "50-64", 3: "65+"}
#: A gap-band with fewer than this many WEIGHTED train couples (ego survey
#: weights normalised to unit mean within the sex -- an effective couple count)
#: falls back to the adjacent, next-younger pooled band (registration).
FALLBACK_MIN_WEIGHTED_COUPLES = 200.0

# --------------------------------------------------------------------------
# The two named deltas (registration comment 4925020986)
# --------------------------------------------------------------------------
DELTA1_ENTRY_WIDOWED = (
    "DELTA 1 (entry-widowed observed initial state; forensics 3 Q6): persons "
    "observed already-widowed at their first observed PSID wave (all sexes, "
    "all ages) enter the simulation widowed. The reference-carried widowed "
    "person-years (widowhood onset year < the person's first observed wave -- "
    "structurally unreachable by any transition rate on the observed support, "
    "12.1% of the reference 75+ female widowed stock, zero non-derivable "
    "residual) are injected onto the simulated panel post-assembly: a pure "
    "RNG-neutral state injection symmetric with the reference construction, "
    "reusing forensics 3's carried-status classification. The candidate-9 "
    "delta-1 (observed initial state) precedent"
)
DELTA2_GAP_BANDED = (
    "DELTA 2 (age-band-conditioned spousal-age-gap draw; forensics 3 Q7): the "
    "spouse's imputed age is drawn from the train sex-specific empirical gap "
    "distribution WITHIN the ego's age band at marriage ({18-34, 35-49, "
    "50-64, 65+}, 1-year gap bins as candidate 4), replacing candidate 11's "
    "single pooled distribution. A band with < 200 weighted train couples "
    "falls back to the adjacent next-younger pooled band (documented per "
    "seed). The gap draw is candidate 4's SPAWNED stream, so the scored rng is "
    "not advanced -- first marriage, divorce, widowhood, remarriage and "
    "fertility are byte-identical to candidate 11 at a shared draw seed"
)
DELTAS_VS_CANDIDATE11 = (
    "EXACTLY TWO deltas vs candidate 11 (comment 4919417729, merged #101), "
    "registered from forensics 3 (#102). "
    + DELTA1_ENTRY_WIDOWED
    + ". "
    + DELTA2_GAP_BANDED
    + ". Everything else -- the 5-band remarriage current-age table, the "
    "observed undatable-marriage lifetime-count initial state (candidate 9's "
    "delta 1), the surviving-spouse NCHS-trended widowhood level, the "
    "single-year triangular-kernel fertility, the RNG, the K=20 "
    "mean-of-draws protocol and the fresh-run artifact schema -- is "
    "byte-identical to candidate 11"
)

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate12_run1_cache.json"
)

#: The marriage-count cells (delta 2 reshapes the young-widow remarriage
#: exposure that feeds these; delta 1 leaves n_marriages untouched).
COUNT_CELLS = (
    "mean_lifetime_marriages|male",
    "mean_lifetime_marriages|female",
)
#: The gated remarriage cells the deltas reshape via the widowed trajectory.
REMARRIAGE_GATED_CELLS = (
    "remarriage.after_divorce",
    "remarriage.ysd0-4",
    "remarriage.ysd5-9",
    "remarriage.ysd10+",
)
#: The elderly-widow-stock cell delta 1 is designed to recover.
MODAL_CELL = "share_widowed.75+|female"
#: The registered modal failure (comment 4925020986): the female marriage count
#: cells recovering only partially.
REGISTERED_MODAL_CELLS = COUNT_CELLS
#: The gated widowhood-incidence cells delta 2 could perturb (secondary risk).
WIDOWHOOD_INCIDENCE_CELLS = (
    "widowhood.45-64|female",
    "widowhood.65-74|female",
    "widowhood.75+|female",
    "widowhood.45+|male",
)
#: The cells the two deltas most directly touch: the widow stock (delta 1),
#: both marriage counts (delta 2), the remarriage flows and the widowhood
#: incidence (delta 2).
TARGETED_CELLS = (
    (MODAL_CELL,)
    + COUNT_CELLS
    + REMARRIAGE_GATED_CELLS
    + WIDOWHOOD_INCIDENCE_CELLS
)
#: The young widowed-stock pools delta 2 is designed to deflate (forensics 3
#: Q7; diagnostic, NOT gated -- reference_moments produces share_widowed only
#: at 65-74 and 75+).
YOUNG_POOL_BANDS = ((15, 49), (50, 64))
WIDOWED_AGE_BANDS = f3.WIDOWED_AGE_BANDS


# ==========================================================================
# DELTA 2 fit: age-band-conditioned spousal-age-gap distribution (train)
# ==========================================================================
def spousal_gap_distribution_by_band(
    mh_records: pd.DataFrame,
    attrs: pd.DataFrame,
    train_ids: set[int],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    """Train sex-specific empirical spousal age-gap arrays by ego age band.

    The SAME record selection and 1-year-binned gap as
    :func:`candidate4.spousal_gap_distribution` (train self-persons whose
    marriage record joins a spouse with a known birth year;
    ``gap = self_birth - spouse_birth``), but stratified on the ego's AGE BAND
    AT MARRIAGE (``start_year - self_birth`` binned into
    ``{18-34, 35-49, 50-64, 65+}``). A band with fewer than
    :data:`FALLBACK_MIN_WEIGHTED_COUPLES` weighted train couples (ego survey
    weights normalised to unit mean within the sex, so the threshold reads as
    an effective couple count) is pooled with successively younger bands until
    the cumulative weighted count clears the threshold -- the registered
    adjacent-pooled-band fallback. Couples whose marriage start year is
    undatable (~0.7%; ego age at marriage not computable) enter only the
    sex-level pooled fallback, not the banded arrays. Returns the per-band
    sampling arrays and a per-(sex, band) diagnostic (raw and weighted counts,
    the fallback grouping, and the used array's moments).
    """
    person_birth = (
        mh_records.dropna(subset=["birth_year"])
        .groupby("person_id")["birth_year"]
        .first()
    )
    rec = mh_records[
        mh_records["is_marriage"]
        & mh_records["spouse_person_id"].notna()
        & mh_records["person_id"].isin(train_ids)
    ].copy()
    rec["self_birth"] = rec["person_id"].map(person_birth).astype("float64")
    rec["spouse_birth"] = (
        rec["spouse_person_id"].map(person_birth).astype("float64")
    )
    rec = rec[rec["self_birth"].notna() & rec["spouse_birth"].notna()].copy()
    rec["gap"] = np.rint(rec["self_birth"] - rec["spouse_birth"]).astype(
        np.int64
    )
    ego_age = rec["start_year"].astype("float64") - rec["self_birth"]
    rec["ego_age_at_marriage"] = ego_age
    wmap = attrs.set_index("person_id")["weight"]
    rec["couple_weight"] = (
        rec["person_id"].map(wmap).astype("float64").fillna(0.0)
    )

    n_bands = len(GAP_AGE_BANDS)
    dist: dict[str, dict[int, np.ndarray]] = {}
    diag: dict[str, Any] = {}
    for sex in ("female", "male"):
        sub = rec[rec["sex"] == sex]
        datable = sub[sub["ego_age_at_marriage"].notna()].copy()
        n_na_start = int(len(sub) - len(datable))
        band_idx = _bands_vec(
            np.rint(datable["ego_age_at_marriage"].to_numpy()).astype(
                np.int64
            ),
            GAP_AGE_LOWERS,
            n_bands,
        )
        datable = datable.assign(gap_band=band_idx)
        # Weighted (effective) couple count per band: normalise ego weights to
        # unit mean within the sex, so the 200 threshold reads as an effective
        # couple count rather than a raw population weight (in the millions).
        w = datable["couple_weight"].to_numpy(dtype=np.float64)
        wbar = float(w.mean()) if w.size and w.mean() > 0 else 1.0
        eff_w = w / wbar
        gaps_by_band: dict[int, np.ndarray] = {}
        raw_n = np.zeros(n_bands, dtype=np.int64)
        weighted_n = np.zeros(n_bands, dtype=np.float64)
        for b in range(n_bands):
            m = band_idx == b
            gaps_by_band[b] = datable.loc[m, "gap"].to_numpy(dtype=np.int64)
            raw_n[b] = int(m.sum())
            weighted_n[b] = float(eff_w[m].sum())

        pooled = c11_pooled_gap(mh_records, train_ids, sex)
        sex_dist: dict[int, np.ndarray] = {}
        sex_diag: dict[str, Any] = {}
        for b in range(n_bands):
            group = _fallback_group(b, weighted_n)
            arr = np.concatenate(
                [gaps_by_band[j] for j in group]
                + [np.empty(0, dtype=np.int64)]
            )
            fell_back = group != [b]
            if arr.size == 0:  # ultimate guard (never on the real file)
                arr = pooled
                group = [-1]
            sex_dist[b] = arr
            sex_diag[_GAP_AGE_LABEL[b]] = {
                "raw_couples": int(raw_n[b]),
                "weighted_couples": float(weighted_n[b]),
                "fell_back": bool(fell_back),
                "pooled_band_group": [
                    (_GAP_AGE_LABEL[j] if j >= 0 else "sex_pooled")
                    for j in group
                ],
                "used_n": int(arr.size),
                "used_gap_mean": float(arr.mean()),
                "used_gap_sd": float(arr.std()),
                "used_gap_min": int(arr.min()),
                "used_gap_max": int(arr.max()),
            }
        dist[sex] = sex_dist
        diag[sex] = {
            "n_na_start_year_pooled_out": n_na_start,
            "weight_normaliser_mean_ego_weight": wbar,
            "min_weighted_couples_threshold": FALLBACK_MIN_WEIGHTED_COUPLES,
            "bands": sex_diag,
        }
    return dist, diag


def c11_pooled_gap(
    mh_records: pd.DataFrame, train_ids: set[int], sex: str
) -> np.ndarray:
    """Candidate 11's pooled (un-banded) gap array for one sex.

    Byte-identical selection to :func:`candidate4.spousal_gap_distribution`
    (candidate 11 inherits it unchanged) -- the ultimate fallback and the
    old-vs-new comparison baseline.
    """
    return c4.spousal_gap_distribution(mh_records, train_ids)[sex]


def _fallback_group(b: int, weighted_n: np.ndarray) -> list[int]:
    """Bands pooled for band ``b``: extend toward younger until >= threshold.

    Deterministic, documented adjacent-pooled-band fallback. Band ``b`` uses
    the couples of ``[j..b]`` where ``j`` is the largest index (extending
    downward from ``b``) at which the cumulative weighted couple count first
    clears :data:`FALLBACK_MIN_WEIGHTED_COUPLES`, or ``0`` if even the whole
    younger run does not. A band already at or above the threshold uses only
    itself.
    """
    total = float(weighted_n[b])
    j = b
    while total < FALLBACK_MIN_WEIGHTED_COUPLES and j > 0:
        j -= 1
        total += float(weighted_n[j])
    return list(range(j, b + 1))


# ==========================================================================
# DELTA 1 accounting: entry-widowed carried-status classification (forensics 3)
# ==========================================================================
_CARRIED_CACHE: dict[int, dict[str, Any]] = {}


def observed_support(demo: pd.DataFrame) -> pd.DataFrame:
    """Per-person observed PSID support window (reused from forensics 3)."""
    return f3.observed_support(demo)


def entry_widowed_carried_cells(
    panel: transitions.MaritalPanel, demo: pd.DataFrame
) -> dict[str, Any]:
    """Reference-carried widowed person-years (delta 1's injection set).

    Every widowed reference person-year (all sexes, all ages) whose widowhood
    onset year (``year - years_since_dissolution``) predates the person's first
    observed PSID wave and whose support is observed -- forensics 3's
    ``carried`` mask (:func:`gate2_forensics3.widowed_75plus_support_taxonomy`),
    generalised beyond the 75+ female stock. These are structurally unreachable
    by any transition rate on the observed support, and the observed-initial-
    state injection (delta 1) writes them onto the simulated panel post-
    assembly. Split-independent (read once per reference panel); cached like the
    delta-1 marriage residual. Returns a fast (person_id * 10000 + year) integer
    key sorted with its per-cell ``years_since_dissolution``.
    """
    cache_key = id(panel)
    cached = _CARRIED_CACHE.get(cache_key)
    if cached is not None:
        return cached
    support = observed_support(demo)
    py = panel.person_years
    wid = py[py["marital_state"] == "widowed"].copy()
    ysd = wid["years_since_dissolution"].astype("float64")
    onset_year = wid["year"].to_numpy(dtype=np.float64) - ysd.to_numpy(
        dtype=np.float64
    )
    fw = wid["person_id"].map(support["first_wave"]).to_numpy(dtype=np.float64)
    defined = ~np.isnan(onset_year)
    has_support = ~np.isnan(fw)
    carried = defined & has_support & (onset_year < fw)

    cc = wid.loc[carried, ["person_id", "year", "weight"]].copy()
    cc["years_since_dissolution"] = ysd.to_numpy(dtype=np.float64)[
        carried
    ].astype(np.int64)
    pid = cc["person_id"].to_numpy(dtype=np.int64)
    yr = cc["year"].to_numpy(dtype=np.int64)
    key = pid * 10000 + yr
    order = np.argsort(key, kind="stable")
    out = {
        "key_sorted": key[order],
        "ysd_sorted": cc["years_since_dissolution"].to_numpy(dtype=np.int64)[
            order
        ],
        "person_id_sorted": pid[order],
        "weight_sorted": cc["weight"].to_numpy(dtype=np.float64)[order],
        "n_cells": int(carried.sum()),
        "n_persons": int(np.unique(pid).size),
        "weighted_py": float(cc["weight"].sum()),
    }
    _CARRIED_CACHE[cache_key] = out
    return out


def _entry_widowed_seed_counts(
    panel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Per-seed entry-widowed injection counts on the SCORED holdout (side A).

    For each gate seed, the persons and weighted widowed person-years the
    delta-1 injection writes onto side A (the simulated holdout) -- the
    seed-specific footprint of the observed-initial-state injection.
    """
    carried = entry_widowed_carried_cells(panel, demo)
    pid_sorted = carried["person_id_sorted"]
    w_sorted = carried["weight_sorted"]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        side_a, _side_b = hpanel.split_panel_by_person(
            panel.attrs, "person_id", fraction=0.5, seed=seed
        )
        ids_a = set(int(x) for x in side_a.person_id.unique())
        in_a = np.fromiter(
            (int(p) in ids_a for p in pid_sorted),
            dtype=bool,
            count=pid_sorted.size,
        )
        rows.append(
            {
                "seed": seed,
                "n_entry_widowed_persons": int(
                    np.unique(pid_sorted[in_a]).size
                ),
                "n_injected_widowed_person_years": int(in_a.sum()),
                "weighted_injected_widowed_py": float(w_sorted[in_a].sum()),
            }
        )
    return {
        "note": (
            "delta-1 entry-widowed observed-initial-state injection footprint "
            "on side A (the scored holdout) per gate seed; reference-carried "
            "widowed person-years (onset < first observed wave)"
        ),
        "panel_total_entry_widowed_persons": carried["n_persons"],
        "panel_total_carried_widowed_py_cells": carried["n_cells"],
        "panel_total_weighted_carried_py": carried["weighted_py"],
        "per_seed": rows,
    }


def _entry_widowed_reconciliation(
    panel: transitions.MaritalPanel,
    demo: pd.DataFrame,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Reconcile delta-1's carried classification to forensics 3's Q6 (hard).

    For each gate seed, forensics 3's ``initial_state_fixable_share`` (the
    carried 75+ female widowed-stock share, on side B, its holdout convention)
    is recomputed live from :func:`gate2_forensics3.widowed_75plus_support_
    taxonomy` and checked against the committed ``runs/gate2_forensics3_v1.json``
    to float precision, AND the same 75+ female carried stock share derived
    from candidate 12's own all-ages carried-cell set is checked to reproduce
    it -- proving the delta-1 injection reuses forensics 3's accounting
    exactly. ``max_abs_remainder`` closing to 0.0 is a hard gate.
    """
    support = observed_support(demo)
    committed = {
        s["seed"]: s["ref_support_taxonomy"]["initial_state_fixable_share"]
        for s in json.loads(FORENSICS3_ARTIFACT.read_text())["per_seed"]
    }
    carried = entry_widowed_carried_cells(panel, demo)
    key_sorted = carried["key_sorted"]

    rows: list[dict[str, Any]] = []
    max_abs_remainder = 0.0
    for seed in seeds:
        _side_a, side_b = hpanel.split_panel_by_person(
            panel.attrs, "person_id", fraction=0.5, seed=seed
        )
        ids_b = set(int(x) for x in side_b.person_id.unique())
        tax = f3.widowed_75plus_support_taxonomy(panel, support, ids_b)
        f3_live = float(tax["initial_state_fixable_share"])
        f3_commit = float(committed[seed])

        # The same 75+ female carried stock share from candidate 12's own
        # all-ages carried-cell set restricted to side B.
        py = panel.person_years
        fem75 = py[
            py["person_id"].isin(ids_b)
            & (py["sex"] == "female")
            & (py["age"] >= 75)
        ]
        den = float(fem75["weight"].sum())
        wid75 = fem75[fem75["marital_state"] == "widowed"]
        num_wid = float(wid75["weight"].sum())
        wkey = wid75["person_id"].to_numpy(dtype=np.int64) * 10000 + wid75[
            "year"
        ].to_numpy(dtype=np.int64)
        pos = np.searchsorted(key_sorted, wkey)
        pos = np.clip(pos, 0, max(key_sorted.size - 1, 0))
        is_carried = (key_sorted.size > 0) & (key_sorted[pos] == wkey)
        w75 = wid75["weight"].to_numpy(dtype=np.float64)
        own_carried_share = (
            float(w75[is_carried].sum()) / num_wid if num_wid > 0 else 0.0
        )
        own_of_all = float(w75[is_carried].sum()) / den if den > 0 else 0.0
        rem_commit = abs(f3_live - f3_commit)
        rem_own = abs(own_carried_share - f3_live)
        max_abs_remainder = max(max_abs_remainder, rem_commit, rem_own)
        rows.append(
            {
                "seed": seed,
                "forensics3_committed_fixable_share": f3_commit,
                "forensics3_live_fixable_share": f3_live,
                "candidate12_carried_cells_fixable_share": own_carried_share,
                "candidate12_carried_share_of_all_75plus_py": own_of_all,
                "remainder_vs_committed": rem_commit,
                "remainder_vs_own_classification": rem_own,
            }
        )
    return {
        "description": (
            "delta-1 carried-status classification reconciles to forensics 3's "
            "committed Q6 (initial_state_fixable_share) and candidate 12's own "
            "all-ages carried-cell set reproduces the 75+ female carried stock "
            "share, to float precision"
        ),
        "per_seed": rows,
        "max_abs_remainder": float(max_abs_remainder),
        "reconciled": bool(max_abs_remainder <= 1e-9),
    }


# ==========================================================================
# Fitted components (candidate 11's, with the two delta'd fields attached)
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
    """Candidate 11's components + candidate 12's two delta'd fields.

    Starts from :func:`candidate11.fit_components` -- so the first-marriage
    spline, divorce, the surviving-spouse NCHS-trended widowhood level, the
    5-band remarriage current-age table, the single-year triangular-kernel
    fertility, the observed marriage-count initial state (candidate 9's delta
    1) and the POOLED spousal-gap distribution are byte-identical to candidate
    11 by construction. Then candidate 12's two deltas are attached:

    * DELTA 2 -- the age-band-conditioned gap distribution
      (:func:`spousal_gap_distribution_by_band`) is stored in
      ``gap_dist_by_sex_band``; the simulation draws each spouse's age from the
      ego's own band. The pooled ``gap_dist_by_sex`` is retained (fallback and
      old-vs-new baseline).
    * DELTA 1 -- the reference-carried widowed cells
      (:func:`entry_widowed_carried_cells`) are stored in
      ``entry_widowed_cells``; the simulation injects them post-assembly. Read
      from the whole reference panel (split-independent), not fitted on train.
    """
    base = c11.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )

    # DELTA 2: age-band-conditioned spousal-age-gap distribution (train fit).
    gap_band, gap_band_diag = spousal_gap_distribution_by_band(
        mh_records, panel.attrs, train_ids
    )
    base.gap_dist_by_sex_band = gap_band

    # DELTA 1: reference-carried widowed cells (split-independent injection set).
    base.entry_widowed_cells = entry_widowed_carried_cells(panel, demo)

    base.meta["delta2_gap_age_bands"] = [list(b) for b in GAP_AGE_BANDS]
    base.meta["delta2_gap_band_distribution"] = gap_band_diag
    base.meta["delta1_entry_widowed"] = {
        "representation": (
            "reference-carried widowed person-years (onset year < first "
            "observed wave), all sexes/ages, injected as an observed initial "
            "state post-assembly (forensics 3 Q6 carried mask)"
        ),
        "panel_total_entry_widowed_persons": base.entry_widowed_cells[
            "n_persons"
        ],
        "panel_total_carried_widowed_py_cells": base.entry_widowed_cells[
            "n_cells"
        ],
        "panel_total_weighted_carried_py": base.entry_widowed_cells[
            "weighted_py"
        ],
    }
    base.meta["deltas_vs_candidate11"] = DELTAS_VS_CANDIDATE11
    return base


# ==========================================================================
# Provable byte-identity: reuse candidate 11's (== candidate 10's) EXACT code
# objects for the un-delta'd chain, rebound to THIS module's globals so
# ``_draw_moments`` / ``score_seed`` call candidate 12's ``simulate_holdout`` /
# ``fit_components``. Only ``simulate_holdout`` and ``fit_components`` diverge.
# ==========================================================================
def _rebind(fn: types.FunctionType) -> types.FunctionType:
    """Return a function sharing ``fn``'s code object but this module's globals.

    Sharing candidate 10/11's code object makes ``everything else
    byte-identical`` machine-checkable at the bytecode level
    (``candidate12.f.__code__ is candidate11.f.__code__``) while name
    resolution redirects to candidate 12's module (so the reused
    ``_draw_moments`` / ``score_seed`` call candidate 12's two delta'd
    functions).
    """
    return types.FunctionType(
        fn.__code__,
        globals(),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


#: The 5-band remarriage fit / lookup (candidate 11's == candidate 10's code),
#: rebound -- UNCHANGED by candidate 12's two deltas.
fit_remarriage_age_banded = _rebind(c11.fit_remarriage_age_banded)
_remarriage_probs_age_banded = _rebind(c11._remarriage_probs_age_banded)
_build_sim_lookups = _rebind(c11._build_sim_lookups)
#: Candidate 11's (== candidate 10's) exact single-draw moment builder and
#: per-seed mean-over-K=20 scorer. Rebound to THIS module's globals, they call
#: candidate 12's ``simulate_holdout`` (both deltas) and ``fit_components``.
_draw_moments = _rebind(c11._draw_moments)
score_seed = _rebind(c11.score_seed)

# Fresh-run artifact-schema blocks are band/estimator-independent; import-bound
# from candidate 10 (identical N_DRAWS / DRAW_SEED_BASE) exactly as candidate 11.
_per_draw_per_cell_rates_block = c10._per_draw_per_cell_rates_block
_undefined_draw_block = c10._undefined_draw_block
_per_draw_dispersion_block = c10._per_draw_dispersion_block

#: The reused-code-object contract (must share candidate 11's bytecode).
REUSED_CODE_OBJECT_NAMES = (
    "_draw_moments",
    "score_seed",
    "fit_remarriage_age_banded",
    "_build_sim_lookups",
    "_remarriage_probs_age_banded",
)
#: The two RE-IMPLEMENTED functions (the deltas): they must NOT share candidate
#: 11's code object.
DIVERGED_CODE_OBJECT_NAMES = ("simulate_holdout", "fit_components")


# ==========================================================================
# DELTA'd simulation: candidate 11's vectorised annual simulation with the
# age-band-conditioned spouse-age-gap draw (delta 2) and the post-assembly
# entry-widowed injection (delta 1).
# ==========================================================================
def _gap_band_arrays(
    components: Components,
) -> list[list[np.ndarray]]:
    """``[sex_idx][band_idx] -> gap array`` for the at-marriage banded draw."""
    band = components.gap_dist_by_sex_band
    return [
        [band[sex][b] for b in range(len(GAP_AGE_BANDS))]
        for sex in ("female", "male")
    ]


def _draw_banded_gaps(
    gap_rng: np.random.Generator,
    idx: np.ndarray,
    marriage_age: np.ndarray,
    is_male: np.ndarray,
    gap_band: list[list[np.ndarray]],
) -> np.ndarray:
    """Spousal-age gaps for a batch of marrying egos, conditioned on age band.

    ``idx`` are the marrying egos' row indices; ``marriage_age`` their age at
    the marriage; the gap of each is drawn (with replacement, 1-year bins) from
    its sex-and-age-band array. Draws are consumed in a fixed order (female
    bands 0..3 then male bands 0..3) so the spawned gap stream is deterministic.
    """
    out = np.empty(idx.size, dtype=np.float64)
    bands = _bands_vec(
        np.rint(marriage_age).astype(np.int64),
        GAP_AGE_LOWERS,
        len(GAP_AGE_BANDS),
    )
    male = is_male[idx] == 1.0
    for si, sex_mask in ((0, ~male), (1, male)):
        for b in range(len(GAP_AGE_BANDS)):
            m = sex_mask & (bands == b)
            cnt = int(m.sum())
            if cnt:
                out[m] = gap_rng.choice(gap_band[si][b], size=cnt)
    return out


def simulate_holdout(
    panel: transitions.MaritalPanel,
    holdout_ids: set[int],
    components: Components,
    sim_seed: int,
) -> tuple[transitions.MaritalPanel, pd.DataFrame]:
    """Candidate 11's simulation with candidate 12's two deltas.

    Byte-identical to :func:`candidate11.simulate_holdout` (== candidate 10's)
    EXCEPT:

    * DELTA 2 -- each spouse's age gap is drawn at the marriage event from the
      ego's age band (:func:`_draw_banded_gaps`) instead of once per person
      from the pooled sex distribution. The gaps use candidate 4's SPAWNED
      ``gap_rng`` (which never advances the scored ``rng``), so the per-year
      uniform blocks (``rng.random`` for competing risks then fertility) are
      byte-identical to candidate 11 -- only the imputed spouse ages move.
    * DELTA 1 -- after the panel is assembled, the reference-carried widowed
      person-years (persons observed already-widowed at their first observed
      wave) are injected onto the simulated ``marital_state`` (an observed
      initial state, RNG-neutral), reproducing the structurally unreachable
      widowed stock forensics 3 Q6 isolated.

    Candidate 9's observed marriage-count initial state (delta 1 of candidate 9)
    is applied unchanged (a post-assembly count add on ``n_marriages``).
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
    # delta 2, retained): the spawn does not advance rng's bit stream. Under
    # the amended estimator sim_seed = 5200 + k for draw k.
    rng = np.random.default_rng(sim_seed)
    gap_seed_seq = rng.bit_generator.seed_seq.spawn(1)[0]
    gap_rng = np.random.default_rng(gap_seed_seq)

    # DELTA 2: the spouse-age gap is drawn AT MARRIAGE from the ego's age band.
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
            # DELTA 2: impute each new spouse's age from the ego's age band.
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
            # DELTA 2: impute each remarried spouse's age from the ego's band.
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

    # DELTA 1: entry-widowed observed-initial-state injection. The simulated and
    # reference panels share the SAME (person_id, year) grid, so the reference-
    # carried widowed person-years are written onto the simulated marital_state
    # directly (RNG-neutral, no draw touched).
    _inject_entry_widowed(sim_panel, components.entry_widowed_cells)

    # DELTA 1 of candidate 9 (inherited, unchanged): observed undatable-marriage
    # lifetime-count initial state (a pure post-assembly count add on
    # n_marriages; RNG-neutral).
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


def _inject_entry_widowed(
    sim_panel: transitions.MaritalPanel, carried: dict[str, Any]
) -> None:
    """Write the reference-carried widowed person-years onto the sim panel.

    In-place override of ``marital_state`` (and ``years_since_dissolution`` /
    ``marriage_duration``) for the simulated person-years matching a reference-
    carried widowed cell. The grids align 1:1 on (person_id, year), so a sorted
    integer key + ``searchsorted`` matches without a merge.
    """
    key_sorted = carried["key_sorted"]
    if key_sorted.size == 0:
        return
    spy = sim_panel.person_years
    sim_key = spy["person_id"].to_numpy(dtype=np.int64) * 10000 + spy[
        "year"
    ].to_numpy(dtype=np.int64)
    pos = np.searchsorted(key_sorted, sim_key)
    pos = np.clip(pos, 0, key_sorted.size - 1)
    match = key_sorted[pos] == sim_key
    if not match.any():
        return
    ms = spy["marital_state"].to_numpy(dtype=object)
    ms[match] = "widowed"
    spy["marital_state"] = ms
    ysd_new = spy["years_since_dissolution"].to_numpy(dtype=object)
    ysd_new[match] = carried["ysd_sorted"][pos][match]
    spy["years_since_dissolution"] = pd.array(ysd_new, dtype="Int64")
    mdur = spy["marriage_duration"].to_numpy(dtype=object)
    mdur[match] = pd.NA
    spy["marriage_duration"] = pd.array(mdur, dtype="Int64")


# ==========================================================================
# Diagnostics: young-pool widowed shares (delta 2's designed deflation)
# ==========================================================================
def _widowed_share_by_age(
    panel: transitions.MaritalPanel, ids: set[int]
) -> dict[str, float]:
    """Female widowed-stock share by age band (15-49 / 50-64 / 65-74 / 75+)."""
    py = panel.person_years
    py = py[py["person_id"].isin(ids) & (py["sex"] == "female")]
    out: dict[str, float] = {}
    for lo, hi in WIDOWED_AGE_BANDS:
        grp = py[(py["age"] >= lo) & (py["age"] <= hi)]
        den = float(grp["weight"].sum())
        num = float(grp[grp["marital_state"] == "widowed"]["weight"].sum())
        out[transitions.band_label(lo, hi)] = num / den if den > 0 else 0.0
    return out


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
    """Candidate 12 female widowed-stock shares by age band vs candidate 11.

    Mirrors forensics 3's Q7 convention (fit + simulate side B, K=20 draws,
    draw-mean) so candidate 12's young widowed pools are directly comparable to
    the committed candidate-11 pools in ``runs/gate2_forensics3_v1.json``. The
    reference share is side B's own observed panel; the simulated share is the
    K=20 draw-mean under candidate 12 (both deltas). Reports the sim/ref ratio
    at 15-49 and 50-64 (the pools delta 2 is designed to deflate) alongside the
    committed candidate-11 ratios.
    """
    c11f3 = {
        s["seed"]: s
        for s in json.loads(FORENSICS3_ARTIFACT.read_text())["per_seed"]
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
        c11s = c11f3[seed]
        cells: dict[str, Any] = {}
        for lab in band_labels:
            key = f"{lab}|female"
            ref_share = ref[lab]
            c12_ratio = sim_mean[lab] / ref_share if ref_share > 0 else None
            c11_ref = c11s["ref_widowed_by_age"][key]["widowed_share"]
            c11_sim = c11s["sim_widowed_by_age_mean"][key]["widowed_share"]
            c11_ratio = c11_sim / c11_ref if c11_ref > 0 else None
            cells[lab] = {
                "ref_widowed_share": ref_share,
                "c12_sim_widowed_share_mean": sim_mean[lab],
                "c12_sim_over_ref": c12_ratio,
                "c11_sim_over_ref": c11_ratio,
            }
        rows.append({"seed": seed, "bands": cells})
        if verbose:
            y1 = cells["15-49"]
            y2 = cells["50-64"]
            print(
                f"  young-pool seed {seed}: 15-49 c11 "
                f"{y1['c11_sim_over_ref']:.3f} -> c12 "
                f"{y1['c12_sim_over_ref']:.3f}; 50-64 c11 "
                f"{y2['c11_sim_over_ref']:.3f} -> c12 "
                f"{y2['c12_sim_over_ref']:.3f}"
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
            "c11_sim_over_ref_mean": _mean_ratio(lab, "c11_sim_over_ref"),
            "c12_sim_over_ref_mean": _mean_ratio(lab, "c12_sim_over_ref"),
        }
        for lab in band_labels
    }
    return {
        "note": (
            "female widowed-stock share by age band; forensics 3 Q7 convention "
            "(fit + simulate side B, K=20 draw-mean). c11 ratios from "
            "runs/gate2_forensics3_v1.json; delta 2 is designed to deflate the "
            "15-49 (c11 ~3.16x) and 50-64 (c11 ~1.41x) pools toward reference"
        ),
        "bands": band_labels,
        "per_seed": rows,
        "seed_mean_sim_over_ref": summary,
    }


# ==========================================================================
# Candidate-11 comparison, count tilt, modal / decider
# ==========================================================================
def _count_cell_tilt(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """The count-cell movement the two deltas are designed to produce.

    Candidate 11 left the female count regressed to net +0.051 ln (2/5) and the
    male at +0.025 ln (4/5). Delta 2 removes the young-widow remarriage
    over-exposure the pooled gap draw created, designed to pull the female net
    back toward the residual-only ~0.00. Reports candidate 12's realised signed
    ln-tilt per seed against candidate 11's committed count scores.
    """
    c11art = (
        json.loads(CANDIDATE11_ARTIFACT.read_text())
        if CANDIDATE11_ARTIFACT.exists()
        else None
    )
    by11 = {s["seed"]: s for s in c11art["per_seed"]} if c11art else {}

    out: dict[str, Any] = {
        "design_note": (
            "registration comment 4925020986: delta 2 (age-band gap) removes "
            "the young-widow remarriage over-exposure the pooled gap draw "
            "created, designed to pull the female marriage count net back from "
            "candidate 11's regressed +0.051 ln (2/5) toward the residual-only "
            "~0.00 the male side enjoys; delta 1 leaves n_marriages untouched. "
            "Movement reported against candidate 11's committed count scores."
        ),
        "candidate11_female_net_ln": 0.051,
        "candidate11_male_net_ln": 0.025,
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
            c11rec = by11[s["seed"]]["gated_cells"][cell] if by11 else None
            rows.append(
                {
                    "seed": s["seed"],
                    "rbar": rbar,
                    "rate_a": rate_a,
                    "signed_ln_tilt": tilt,
                    "score_abs_ln": rec["score"],
                    "tolerance": rec["tolerance"],
                    "pass": rec["pass"],
                    "candidate11_score": (c11rec["score"] if c11rec else None),
                    "candidate11_pass": (c11rec["pass"] if c11rec else None),
                    "delta_score_vs_c11": (
                        float(rec["score"] - c11rec["score"])
                        if c11rec
                        else None
                    ),
                }
            )
        n_pass = sum(r["pass"] for r in rows)
        c11_n_pass = (
            sum(1 for r in rows if r["candidate11_pass"]) if by11 else None
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
            "candidate11_n_seeds_pass": c11_n_pass,
        }
    both_pass = all(out["cells"][c]["n_seeds_pass"] >= 4 for c in COUNT_CELLS)
    out["count_cells_cleared"] = bool(both_pass)
    out["summary"] = (
        "both marriage-count cells clear >= 4/5 seeds"
        if both_pass
        else "at least one marriage-count cell holds below 4/5 seeds"
    )
    return out


def _candidate11_comparison(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Movement of the delta targets vs candidate 11 + the gap-band table.

    Compares candidate 12's per-seed rbar-scores for the targeted cells against
    candidate 11's committed scores, and tabulates the OLD pooled spousal-gap
    distribution against candidate 12's NEW per-band distributions (from the
    seed-0 fit) so delta 2's stratification is visible.
    """
    c11art = (
        json.loads(CANDIDATE11_ARTIFACT.read_text())
        if CANDIDATE11_ARTIFACT.exists()
        else None
    )
    by11 = {s["seed"]: s for s in c11art["per_seed"]} if c11art else {}

    def move(cell: str) -> dict[str, Any]:
        rows = []
        for s in per_seed:
            rec = s["gated_cells"][cell]
            c11rec = by11[s["seed"]]["gated_cells"][cell] if by11 else None
            rows.append(
                {
                    "seed": s["seed"],
                    "tolerance": rec["tolerance"],
                    "c11_score": c11rec["score"] if c11rec else None,
                    "c12_score": rec["score"],
                    "c11_rbar": c11rec["rbar"] if c11rec else None,
                    "c12_rbar": rec["rbar"],
                    "rate_a": rec["rate_a"],
                    "c11_pass": c11rec["pass"] if c11rec else None,
                    "c12_pass": rec["pass"],
                }
            )
        c11_np = sum(1 for r in rows if r["c11_pass"]) if by11 else None
        c12_np = sum(1 for r in rows if r["c12_pass"])
        return {
            "tolerance": rows[0]["tolerance"],
            "per_seed": rows,
            "c11_n_seeds_pass": c11_np,
            "c12_n_seeds_pass": c12_np,
            "improved": (
                bool(c12_np > c11_np) if c11_np is not None else None
            ),
        }

    # Delta 2's gap-band table: the OLD pooled distribution (candidate 11) vs
    # candidate 12's per-band distributions (this run's seed-0 fit).
    band_diag = per_seed[0]["component_meta"]["delta2_gap_band_distribution"]
    gap_table: dict[str, Any] = {}
    for sex in ("female", "male"):
        sd = band_diag[sex]
        pooled_ns = [b["raw_couples"] for b in sd["bands"].values()]
        gap_table[sex] = {
            "old_pooled": {
                "n_couples_datable_banded": int(sum(pooled_ns)),
                "n_na_start_year_pooled_out": sd["n_na_start_year_pooled_out"],
            },
            "new_by_band": {
                label: {
                    "raw_couples": cell["raw_couples"],
                    "weighted_couples": cell["weighted_couples"],
                    "fell_back": cell["fell_back"],
                    "pooled_band_group": cell["pooled_band_group"],
                    "gap_mean": cell["used_gap_mean"],
                    "gap_sd": cell["used_gap_sd"],
                }
                for label, cell in sd["bands"].items()
            },
        }

    return {
        "note": (
            "candidate 12 = candidate 11 (comment 4919417729, #101) with "
            "exactly two deltas (entry-widowed initial state; age-band "
            "spousal-gap draw). Scores compared cell-by-cell against candidate "
            "11's committed run (runs/gate2_hazard_v11.json)."
        ),
        "modal_cell": {MODAL_CELL: move(MODAL_CELL)},
        "count_cells": {c: move(c) for c in COUNT_CELLS},
        "remarriage_gated_cells": {c: move(c) for c in REMARRIAGE_GATED_CELLS},
        "widowhood_incidence_cells": {
            c: move(c) for c in WIDOWHOOD_INCIDENCE_CELLS
        },
        "gap_band_table_seed0": {
            "note": (
                "old pooled spousal-age-gap distribution (candidate 11) vs "
                "candidate 12's per-ego-age-band distributions (seed-0 train "
                "fit). gap = self_birth - spouse_birth; positive => spouse "
                "older. The strong band gradient (older egos marry "
                "wider-gapped spouses) is what the pooled draw ignored"
            ),
            "cells": gap_table,
        },
    }


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Registered modal (the female marriage counts), targets, and decider."""
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
    female_count = "mean_lifetime_marriages|female"
    modal_failed_seeds = sorted(fails_by_cell.get(female_count, []))
    # The registered modal materialises if the female count cells recover only
    # partially -- i.e. the female count still fails on >= 2 seeds.
    modal_materialized = len(modal_failed_seeds) >= 2
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
    elif seeds_pass_if_forgiven(set(TARGETED_CELLS)) >= 4:
        decider = (
            "the delta-targeted cells (forgiving the widow-stock/count/"
            "remarriage/widowhood-incidence targets flips >= 4 seeds to pass)"
        )
    else:
        decider = (
            "broader than the registered modal + delta-targeted cells "
            "(other gated cells also hold the gate below 4 passing seeds)"
        )

    return {
        "registered_modal": (
            "the female marriage count cells recovering only partially (the "
            "count still failing on >= 2 seeds after delta 2 removes the "
            "young-widow remarriage over-exposure)"
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


# ==========================================================================
# Provenance
# ==========================================================================
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins + the candidate-12 schema, c1-c11 and forensics shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    for name in (1, 5, 6, 7, 8, 9, 10, 11):
        pins[f"candidate{name}_runner"] = (
            f"scripts/run_gate2_candidate{name}.py"
        )
        pins[f"candidate{name}_runner_sha256"] = c1._sha_of_file(
            ROOT / "scripts" / f"run_gate2_candidate{name}.py"
        )
    pins["candidate11_artifact"] = "runs/gate2_hazard_v11.json"
    pins["candidate11_artifact_sha256"] = c1._sha_of_file(CANDIDATE11_ARTIFACT)
    pins["forensics3_runner"] = "scripts/gate2_forensics3.py"
    pins["forensics3_runner_sha256"] = c1._sha_of_file(
        ROOT / "scripts" / "gate2_forensics3.py"
    )
    pins["forensics3_artifact"] = "runs/gate2_forensics3_v1.json"
    pins["forensics3_artifact_sha256"] = c1._sha_of_file(FORENSICS3_ARTIFACT)
    pins["estimator"] = "mean_over_K20_draws (5200 + k, k=0..19)"
    pins["deltas"] = (
        "two deltas vs candidate 11: (1) entry-widowed observed initial state "
        "(reference-carried widowed person-years injected post-assembly, "
        "RNG-neutral); (2) age-band-conditioned spousal-age-gap draw "
        "({18-34,35-49,50-64,65+} at marriage, <200-weighted-couple fallback "
        "to the adjacent pooled band). Everything else byte-identical to "
        "candidate 11 (the reused compute chain shares candidate 11's exact "
        "code objects)"
    )
    pins["byte_identity_code_objects"] = {
        name: (getattr(c11, name).__code__ is globals()[name].__code__)
        for name in REUSED_CODE_OBJECT_NAMES
    }
    pins["diverged_code_objects_vs_candidate11"] = {
        name: (getattr(c11, name).__code__ is not globals()[name].__code__)
        for name in DIVERGED_CODE_OBJECT_NAMES
    }
    return pins


def _model_block() -> dict[str, Any]:
    """The model block, edited for the two candidate-12 deltas."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a "
            "mortality-composed widowhood component (candidate 11 with two "
            "deltas: entry-widowed observed initial state; age-band-conditioned "
            "spousal-age-gap draw), scored under the amended "
            "mean-over-K=20-draws estimator"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "deltas_vs_candidate11": DELTAS_VS_CANDIDATE11,
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic spline on age, "
                "knots 20/22/25/30/40, sex, birth-decade cohort -- "
                "BYTE-IDENTICAL to candidate 11 at a shared draw seed (the "
                "deltas touch only the spouse-age-gap draw and a post-assembly "
                "widowed injection)"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x order, "
                "add-one smoothed -- estimator BYTE-IDENTICAL to candidate 11 "
                "(the shared-seed draw stream is unchanged)"
            ),
            "widowhood": (
                "COMPOSED, parametric in period; surviving-spouse "
                "marriage-history widowhood level x candidate 5's committed "
                "NCHS betas -- estimator BYTE-IDENTICAL to candidate 11; the "
                "spouse-age input moves under DELTA 2"
            ),
            "remarriage": (
                "weighted empirical hazard by ego age band "
                "(18-34/35-49/50-64/65-74/75+) x years-since-dissolution band "
                "x origin x sex, add-one smoothed -- the candidate-11 5-band "
                "table, BYTE-IDENTICAL"
            ),
            "fertility": (
                "single-year-of-age triangular-kernel rates within parity "
                "(0/1/2/3+) x birth-decade cohort -- BYTE-IDENTICAL to "
                "candidate 11 (no fertility delta)"
            ),
            "spousal_age_gap": (
                "DELTA 2: the spouse's age is imputed at marriage from the "
                "train sex-specific empirical gap distribution WITHIN the "
                "ego's age band (18-34/35-49/50-64/65+, 1-year bins), a band "
                "with < 200 weighted train couples pooling with the adjacent "
                "next-younger band. Replaces candidate 11's single pooled "
                "distribution; drawn on candidate 4's spawned gap stream so "
                "the scored rng is not advanced"
            ),
            "entry_widowed_initial_state": (
                "DELTA 1: persons observed already-widowed at their first "
                "observed PSID wave (all sexes/ages) enter widowed; the "
                "reference-carried widowed person-years (onset < first wave, "
                "structurally unreachable, forensics 3 Q6) are injected onto "
                "the simulated marital_state post-assembly. An observed initial "
                "state (candidate 9's delta-1 precedent); RNG-neutral"
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
            "entry_widowed_scope": (
                "all sexes, all ages; the injection set is the reference "
                "widowed person-years whose onset year (year - "
                "years_since_dissolution) is strictly less than the person's "
                "first observed PSID wave (forensics 3's carried mask), "
                "written onto the aligned simulated (person_id, year) grid"
            ),
            "gap_band_age": (
                "the ego's age at the marriage (start_year - self_birth on the "
                "train fit; the simulation year minus birth year at the "
                "simulated marriage), binned into 18-34/35-49/50-64/65+; "
                "searchsorted clips below-18 marriage ages into the youngest "
                "band"
            ),
            "gap_band_fallback": (
                "a band with < 200 weighted train couples (ego survey weights "
                "normalised to unit mean within the sex) pools with "
                "successively younger bands until the cumulative weighted count "
                "clears 200; the grouping is recorded per seed in "
                "component_meta.delta2_gap_band_distribution"
            ),
            "gap_stream": (
                "candidate 4's spawned gap_rng (spawn does not advance the "
                "scored rng), so first marriage, divorce, widowhood, "
                "remarriage and fertility draws are byte-identical to "
                "candidate 11 at a shared draw seed"
            ),
            "byte_identity": (
                "candidate 12 reuses candidate 11's (== candidate 10's) EXACT "
                "code objects for _draw_moments, score_seed, "
                "fit_remarriage_age_banded, _build_sim_lookups and "
                "_remarriage_probs_age_banded (rebound to this module's "
                "globals); only simulate_holdout and fit_components are "
                "re-implemented for the two deltas "
                "(revision_pins.byte_identity_code_objects / "
                "diverged_code_objects_vs_candidate11)"
            ),
            "everything_else": (
                "the 5-band remarriage table, the observed marriage-count "
                "initial state (candidate 9 delta 1), the NCHS-trended "
                "widowhood level, the first-marriage spline, divorce, the "
                "single-year triangular-kernel fertility, the competing-risk "
                "step, one sequence per person per draw, and the locked "
                "protocol are byte-identical to candidate 11"
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

    # Preflight: candidate 11 (the base), its fit chain and forensics 3 must be
    # present, plus the candidate-5 NCHS references.
    for name, path in (
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

    # Byte-identity guard: the reused compute chain MUST share candidate 11's
    # exact code objects; the two delta'd functions MUST NOT. Fail fast if not.
    for name in REUSED_CODE_OBJECT_NAMES:
        if globals()[name].__code__ is not getattr(c11, name).__code__:
            raise RuntimeError(
                f"{name} does not share candidate 11's code object; the "
                "reused-chain byte-identity contract is violated."
            )
    for name in DIVERGED_CODE_OBJECT_NAMES:
        if globals()[name].__code__ is getattr(c11, name).__code__:
            raise RuntimeError(
                f"{name} shares candidate 11's code object but must be "
                "re-implemented for a candidate-12 delta."
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
    # reproduce forensics 3's committed Q6 to float precision.
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
    comparison = _candidate11_comparison(per_seed)
    entry_counts = _entry_widowed_seed_counts(panel, demo, GATE_SEEDS)
    if verbose:
        print("young-pool diagnostic (delta 2's designed deflation):")
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
        "candidate": "candidate 12",
        "spec_registration": SPEC_REGISTRATION,
        "registration_pointer": REGISTRATION_POINTER,
        "candidate11_registration": CANDIDATE11_REGISTRATION,
        "candidate10_registration": CANDIDATE10_REGISTRATION,
        "candidate9_registration": CANDIDATE9_REGISTRATION,
        "candidate8_registration": CANDIDATE8_REGISTRATION,
        "candidate6_registration": CANDIDATE6_REGISTRATION,
        "candidate5_registration": CANDIDATE5_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "forensics3_diagnostic": FORENSICS3_DIAGNOSTIC,
        "deltas_vs_candidate11": DELTAS_VS_CANDIDATE11,
        "delta1_entry_widowed": DELTA1_ENTRY_WIDOWED,
        "delta2_gap_banded": DELTA2_GAP_BANDED,
        "amended_estimator": (
            "gates.yaml gate_2 amendment 1 (ratified 2026-07-08, flip #97): "
            "per-cell score |ln(rbar / rate_a)| with rbar the mean over K=20 "
            "draws (default_rng(5200 + k), k=0..19) of the cell rate "
            "(inherited from candidate 11, unchanged)"
        ),
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79); amendment 1 flipped "
            "live (#97). Protocol/views/tolerances/schema read at runtime; no "
            "threshold moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.5-0.6",
            "conjunction_estimate": 0.55,
            "pass_path_seeds": [0, 1, 3, 4],
            "modal_failure": (
                "the female marriage count cells recovering only partially; "
                "secondary: an unforeseen interaction of delta 2 with the "
                "widowhood-incidence cells (gap conditioning changes who gets "
                "widowed, not just when)"
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
                "the fresh candidate-12 one-shot run registered AFTER the "
                "2026-07-08 ratification (registration 4925020986)"
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
        "candidate11_comparison": comparison,
        "young_pool_diagnostic": young_pool,
        "modal_failure_materialized": modal,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "registration_pointer": REGISTRATION_POINTER,
            "candidate11_registration": CANDIDATE11_REGISTRATION,
            "candidate11_artifact": "runs/gate2_hazard_v11.json",
            "forensics3_diagnostic": "runs/gate2_forensics3_v1.json (#102)",
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
            f"  {MODAL_CELL}: c11 {mc['c11_n_seeds_pass']}/5 -> "
            f"c12 {mc['c12_n_seeds_pass']}/5"
        )
        for c in COUNT_CELLS:
            b = cm["count_cells"][c]
            print(
                f"  {c}: c11 {b['c11_n_seeds_pass']}/5 -> "
                f"c12 {b['c12_n_seeds_pass']}/5"
            )
        print(
            "modal (female count partial recovery) materialized="
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
