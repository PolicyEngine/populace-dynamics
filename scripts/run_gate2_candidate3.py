"""Gate-2 candidate 3 (run 1): candidate 2 + two named fixes.

The THIRD pre-registered gate-2 candidate run of
PolicyEngine/populace-dynamics. Its frozen specification is issue #42
comment 4911357564 (``SPEC_REGISTRATION``): candidate 2's frozen spec
(comment 4911167286) verbatim EXCEPT two named deltas. One-shot; no
constant moves after the registration comment.

The two deltas vs candidate 2 (everything else byte-identical)
--------------------------------------------------------------
1. **Mortality smoothing** -- the spouse-death hazard table's smoothing law
   changes everywhere the table is built. Candidate 2 add-one (Laplace)
   smoothed each decade-period x age band x sex cell at the train mean
   slice weight ``(Wd + w) / (We + 2w)``; on thinned early-decade cells the
   ~9,367-person-year pseudo-mass dominated and inflated spouse death rates
   up to ~900x, mass-widowing the 1970s-80s couples and cascading through
   the widowed pool. Candidate 3 replaces it with **exposure-weighted
   shrinkage toward the pooled-period age band x sex rate**:
   ``cell_rate = (weighted_deaths_cell + K * pooled_rate(age, sex)) /
   (weighted_exposure_cell + K)``, with **K = 500 weighted person-years,
   a-priori** (a round prior-strength constant chosen before any run;
   disclosed, not tuned). The shrinkage target ``pooled_rate(age, sex)`` is
   candidate 1's pooled (time-invariant) weighted central rate for that age
   band x sex -- ``build_mortality_floors.weighted_hazards`` on the same
   train slices, the exact table candidate 1 integrates -- so thin cells
   collapse to candidate 1's pooled behavior while thick cells keep the
   decade-period signal. The same decade x age band x sex structure and
   ``"period|band|sex"`` keys as candidate 2; the spousal-age-gap
   imputation is unchanged.
2. **First-marriage spline knots** -- the natural cubic (restricted) spline
   on age gains a knot at 22, becoming **20/22/25/30/40** (candidate 2 used
   20/25/30/40). This sharpens the young-age curvature both sexes clip on
   (``first_marriage.18-24``). The interactions are unchanged: the design
   stays age-spline x sex + age-spline x cohort + sex, the estimator,
   standardisation, weighting, and fitting rule are candidate 2's.

Everything else -- divorce, remarriage, fertility components, the
simulation loop, the RNG rule ``numpy.random.default_rng(4200 + seed)``,
one simulated sequence per person, and the LOCKED gate-2 protocol
(gates.yaml ``gate_2``, ratified PR #79 + flip #81) -- is byte-identical to
candidate 2. This runner IMPORTS candidate 2's machinery
(``run_gate2_candidate2``) and reuses every unchanged function: the
unchanged components come straight from ``candidate1.fit_components`` (so
they are provably identical), the simulation is candidate 2's
``simulate_holdout`` reused unchanged (no delta touches it -- delta 2 keeps
candidate 2's ``"period|band|sex"`` table structure, and delta 1 changes
only the fitted first-marriage model behind the unchanged ``predict``
interface), and only the two delta'd fitters are re-implemented. The
scoring path, precheck, and verdict assembly are candidate 1's, imported
unchanged.

Hard-stop precheck (identical to candidate 1): the scoring path must
reproduce, bit-for-bit, every committed full-panel reference moment, every
committed per-gate-seed ``rate_a``, and each gate seed's committed
holdout-id sha256, BEFORE any candidate is simulated. Any mismatch is a
hard stop. Run ONCE; publish REGARDLESS of verdict.

Environment: repo ``.venv`` (numpy/pandas/scipy/scikit-learn/pyyaml; no
populace-fit). Run from the repository root with the PSID history files
staged::

    .venv/bin/python scripts/run_gate2_candidate3.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Candidate 2 supplies the two-delta machinery this build minimally deltas
# again: its first-marriage age-spline x sex design class
# (``FirstMarriageModelC2``, reused unchanged -- delta 1 changes only the
# knots, not the interaction structure) and its period-aware
# ``simulate_holdout`` and simulation lookups (reused unchanged -- no delta
# touches the simulation). Candidate 1 in turn supplies the shared
# machinery both reuse (the divorce/remarriage/fertility/gap fitters via
# ``candidate1.fit_components``, the precheck, the verdict assembly, and the
# report-only summary). ``build_mortality_floors`` supplies the
# mortality-foundation construction and the pooled weighted-hazard table
# that is candidate 3's shrinkage target.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gate2_floors as g2f  # noqa: E402
import build_mortality_floors as mort  # noqa: E402
import run_gate2_candidate1 as c1  # noqa: E402
import run_gate2_candidate2 as c2  # noqa: E402

from populace_dynamics.data import marriage, transitions  # noqa: E402
from populace_dynamics.harness import panel as hpanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate2_hazard_v3.json"
FLOOR_RUN = ROOT / "runs" / "gate2_floors_v2.json"
ARTIFACT_SCHEMA_VERSION = "gate2_hazard_v3"
RUN_NAME = "gate2_hazard_v3"

#: This run's frozen-spec registration (issue #42, comment 4911357564).
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4911357564"
)
#: The candidate-2 spec this build minimally deltas (comment 4911167286).
CANDIDATE2_REGISTRATION = c2.SPEC_REGISTRATION
#: The candidate-1 spec candidate 2 minimally deltas (comment 4910914098).
CANDIDATE1_REGISTRATION = c1.SPEC_REGISTRATION

#: The two named deltas (registration comment 4911357564).
DELTAS_VS_CANDIDATE2 = (
    "mortality smoothing replaces candidate 2's add-one-at-mean-weight with "
    "exposure-weighted shrinkage toward the pooled-period age band x sex "
    "rate: cell_rate = (weighted_deaths_cell + K * pooled_rate(age, sex)) / "
    "(weighted_exposure_cell + K), K = 500 weighted person-years a-priori; "
    "same decade x age band x sex structure as candidate 2, thin cells "
    "collapse to candidate 1's pooled behavior",
    "first-marriage spline knots become 20/22/25/30/40 (adds a knot at 22); "
    "interactions unchanged (age-spline x sex + age-spline x cohort + sex), "
    "same weighted-MLE fitting rule",
)

# --- Frozen dials + band constants + pure helpers, reused from candidate 1
# (byte-identical; imported, never redefined). ---------------------------
GATE_SEEDS = c1.GATE_SEEDS
SIM_SEED_BASE = c1.SIM_SEED_BASE
EXACT_ATOL = c1.EXACT_ATOL

#: Candidate 2's first-marriage spline knots (kept for provenance/tests).
SPLINE_KNOTS_C2 = c1.SPLINE_KNOTS  # (20, 25, 30, 40)

DIV_BANDS = c1.DIV_BANDS
YSD_BANDS = c1.YSD_BANDS
ASFR_BANDS = c1.ASFR_BANDS
MORT_BANDS = c1.MORT_BANDS
DIV_LOWERS = c1.DIV_LOWERS
YSD_LOWERS = c1.YSD_LOWERS
ASFR_LOWERS = c1.ASFR_LOWERS
MORT_LOWERS = c1.MORT_LOWERS
_ASFR_LO = c1._ASFR_LO
_ASFR_HI = c1._ASFR_HI
_STATE = c1._STATE
_STATE_ABSORB = c1._STATE_ABSORB

_bands_vec = c1._bands_vec
_divorce_probs = c1._divorce_probs
_remarriage_probs = c1._remarriage_probs
_fertility_probs = c1._fertility_probs
_assemble_sim_panel = c1._assemble_sim_panel
Components = c1.Components

# DELTA 1 constant: first-marriage spline knots gain a knot at 22.
SPLINE_KNOTS_C3 = (20.0, 22.0, 25.0, 30.0, 40.0)
# DELTA 2 constant: shrinkage prior strength, in weighted person-years.
K_MORT_PRIOR = 500.0

# The candidate-3 simulation IS candidate 2's -- no delta touches the
# simulation loop. Delta 2 preserves candidate 2's ``"period|band|sex"``
# table structure (only the per-cell smoothing changes), and delta 1
# changes only the fitted first-marriage model behind the unchanged
# ``predict`` interface. Aliased (not redefined) so the identity is
# provable: ``run_gate2_candidate3.simulate_holdout is
# run_gate2_candidate2.simulate_holdout``. The first-marriage design class
# is candidate 2's, reused unchanged (delta 1 changes the knots passed to
# it, not its interaction structure).
FirstMarriageModelC3 = c2.FirstMarriageModelC2
simulate_holdout = c2.simulate_holdout
_build_sim_lookups = c2._build_sim_lookups
_widow_probs = c2._widow_probs
_period_index = c2._period_index

#: Per-seed cache OUTSIDE runs/ (never committed): a crash can be fixed and
#: relaunched without re-scoring an already-scored seed (one-shot rule).
DEFAULT_CACHE = (
    Path.home() / ".claude-worktrees" / "_gate2_candidate3_run1_cache.json"
)


# --------------------------------------------------------------------------
# DELTA 1: first-marriage logistic hazard, spline knots 20/22/25/30/40
# --------------------------------------------------------------------------
def fit_first_marriage(
    train_py: pd.DataFrame, event_years: set[tuple[int, int]]
) -> FirstMarriageModelC3:
    """Fit the candidate-3 first-marriage hazard.

    Byte-identical to ``candidate2.fit_first_marriage`` (candidate 2's
    age-spline x sex + age-spline x cohort design, via the reused
    :class:`FirstMarriageModelC3` = ``candidate2.FirstMarriageModelC2``)
    EXCEPT the spline knots are :data:`SPLINE_KNOTS_C3` (20/22/25/30/40)
    rather than candidate 2's 20/25/30/40. Same estimator, standardisation,
    weighting, and convergence bookkeeping; the interaction structure is
    unchanged (delta 1 is exactly the added knot at 22).
    """
    age = train_py["age"].to_numpy(dtype=np.float64)
    is_male = (train_py["sex"].to_numpy() == "male").astype(np.float64)
    decade = (train_py["birth_decade"].to_numpy()).astype(np.int64)
    weight = train_py["weight"].to_numpy(dtype=np.float64)
    pid = train_py["person_id"].to_numpy()
    yr = train_py["year"].to_numpy()
    y = np.fromiter(
        (
            (int(p), int(t)) in event_years
            for p, t in zip(pid, yr, strict=True)
        ),
        dtype=np.float64,
        count=len(train_py),
    )

    cohort_levels = sorted(int(d) for d in np.unique(decade))
    model = FirstMarriageModelC3(
        clf=LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-6),
        cohort_levels=cohort_levels,
        knots=SPLINE_KNOTS_C3,  # DELTA 1: adds the knot at 22.
        col_mean=np.zeros(1),
        col_sd=np.ones(1),
        n_train_rows=int(len(train_py)),
        n_train_events=int(y.sum()),
        n_iter=0,
        converged=False,
    )
    raw = model._raw_design(age, is_male, decade)
    col_mean = raw.mean(axis=0)
    col_sd = raw.std(axis=0)
    col_sd = np.where(col_sd > 0, col_sd, 1.0)
    model.col_mean = col_mean
    model.col_sd = col_sd
    x = (raw - col_mean) / col_sd
    model.clf.fit(x, y, sample_weight=weight)
    n_iter = int(np.max(model.clf.n_iter_))
    model.n_iter = n_iter
    model.converged = n_iter < model.clf.max_iter
    return model


# --------------------------------------------------------------------------
# DELTA 2: decade-period x age band x sex spouse-death table, shrunk toward
# the pooled band x sex rate (candidate 1's pooled behavior)
# --------------------------------------------------------------------------
def pooled_band_sex_rates(slices: pd.DataFrame) -> dict[str, float]:
    """Candidate 1's pooled (time-invariant) band x sex central death rate.

    Exactly ``build_mortality_floors.weighted_hazards``'s ``psid_m`` -- the
    weighted central rate ``sum(w * death) / sum(w * exposure)`` pooled over
    all periods, the table candidate 1 integrates for widowhood -- so the
    shrinkage target below is literally candidate 1's pooled behavior. Keyed
    ``"band|sex"``.
    """
    return {
        key: cell["psid_m"]
        for key, cell in mort.weighted_hazards(slices).items()
    }


def fit_period_mortality(
    demo: pd.DataFrame,
    death_records: pd.DataFrame,
    train_ids: set[int],
) -> dict[str, float]:
    """Train decade-period x age band x sex central death-rate table (DELTA 2).

    The same mortality-foundation construction as candidate 2
    (:func:`build_mortality_floors.build_exposure_slices`, the weighted
    numerator/denominator ``Wd = sum(w * death)``, ``We = sum(w * exposure)``,
    the wave-anchored ``start_wave`` decade-period, keyed ``"period|band|sex"``)
    but with candidate 3's smoothing law replacing candidate 2's add-one:
    **exposure-weighted shrinkage toward the pooled band x sex rate**

        cell_rate = (Wd_cell + K * pooled_rate(band, sex)) / (We_cell + K),

    with ``K =`` :data:`K_MORT_PRIOR` ``= 500`` weighted person-years,
    a-priori. ``pooled_rate(band, sex)`` is candidate 1's pooled
    (time-invariant) weighted central rate for that band x sex
    (:func:`pooled_band_sex_rates`). A thin period x band x sex cell
    (``We_cell`` small) collapses to ``pooled_rate`` -- candidate 1's pooled
    behavior; a thick cell keeps its own decade-period rate ``Wd/We``. This
    un-explodes candidate 2's add-one convention, which added ~9,367
    person-years of prior death mass at the train mean slice weight and blew
    up the thin early-decade cells. Keyed ``"period|band|sex"`` (e.g.
    ``"1990|75-84|female"``).
    """
    slices = mort.build_exposure_slices(demo, death_records)
    slices = slices[slices["person_id"].isin(train_ids)].copy()
    # Shrinkage target: candidate 1's pooled band x sex central rate.
    pooled = pooled_band_sex_rates(slices)
    slices["period"] = (slices["start_wave"] // 10 * 10).astype(np.int64)
    slices["we"] = slices["weight"] * slices["exposure"]
    slices["wd"] = slices["weight"] * slices["death"]
    grouped = slices.groupby(["period", "band", "sex"], observed=True).agg(
        we=("we", "sum"),
        wd=("wd", "sum"),
    )
    mortality: dict[str, float] = {}
    for (period, band, sex), row in grouped.iterrows():
        target = pooled[f"{band}|{sex}"]
        mortality[f"{int(period)}|{band}|{sex}"] = (
            float(row.wd) + K_MORT_PRIOR * target
        ) / (float(row.we) + K_MORT_PRIOR)
    return mortality


# --------------------------------------------------------------------------
# Fitted components (candidate 2's, with the two delta'd fields swapped)
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
    """Fit all five components on side B, deltas 1 and 2 applied.

    The unchanged components (divorce, remarriage, fertility, spousal gap)
    are taken directly from :func:`candidate1.fit_components` -- so they are
    byte-identical to candidates 1 and 2 by construction, not by
    re-implementation. Only the two delta'd fields are recomputed: the
    first-marriage model (knots 20/22/25/30/40, DELTA 1) and the spouse-death
    mortality table (shrinkage toward the pooled rate, DELTA 2). The never-
    married person-year selection and first-marriage event set are candidate
    1's / candidate 2's, unchanged.
    """
    base = c1.fit_components(
        panel,
        demo,
        death_records,
        mh_records,
        birth_records,
        order_map,
        train_ids,
    )

    py = panel.person_years
    ev = panel.events
    attrs = panel.attrs
    train_py = py[py["person_id"].isin(train_ids)]
    train_ev = ev[ev["person_id"].isin(train_ids)]
    attr_by = attrs.set_index("person_id")
    birth_decade = (attr_by["birth_year"] // 10 * 10).astype("int64")

    # DELTA 1: refit first marriage with the knot-at-22 spline on the same
    # train never-married person-years candidate 1 / candidate 2 used.
    nm = train_py[train_py["marital_state"] == "never_married"][
        ["person_id", "year", "age", "sex", "weight"]
    ].copy()
    nm["birth_decade"] = nm["person_id"].map(birth_decade).to_numpy()
    fm_ev = train_ev[train_ev["transition"] == "first_marriage"]
    event_years = {
        (int(p), int(t))
        for p, t in zip(
            fm_ev["person_id"].to_numpy(),
            fm_ev["year"].to_numpy(),
            strict=True,
        )
    }
    fm_model = fit_first_marriage(nm, event_years)

    # DELTA 2: period-varying spouse mortality, shrunk toward the pooled rate.
    mortality = fit_period_mortality(demo, death_records, train_ids)

    base.first_marriage = fm_model
    base.mortality = mortality
    base.meta["first_marriage_train_rows"] = fm_model.n_train_rows
    base.meta["first_marriage_train_events"] = fm_model.n_train_events
    base.meta["first_marriage_lbfgs_n_iter"] = fm_model.n_iter
    base.meta["first_marriage_converged"] = fm_model.converged
    base.meta["first_marriage_n_cohort_levels"] = len(fm_model.cohort_levels)
    base.meta["first_marriage_knots"] = list(SPLINE_KNOTS_C3)
    base.meta["first_marriage_design"] = (
        "age_spline + sex + age_spline:sex + cohort + age_spline:cohort"
    )
    base.meta["mortality_cells"] = len(mortality)
    base.meta["mortality_periods"] = sorted(
        {int(k.split("|")[0]) for k in mortality}
    )
    base.meta["mortality_stratification"] = "decade_period x band x sex"
    base.meta["mortality_smoothing"] = (
        "exposure_weighted_shrinkage_toward_pooled_band_sex_rate"
    )
    base.meta["mortality_prior_strength_K"] = K_MORT_PRIOR
    return base


# --------------------------------------------------------------------------
# Per-seed scoring (candidate 1's / candidate 2's, calling the candidate-3
# fit + candidate 2's reused simulate)
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

    Identical to :func:`candidate1.score_seed` / :func:`candidate2.score_seed`
    except it calls the candidate-3 :func:`fit_components` (the two deltas)
    and candidate 2's reused :func:`simulate_holdout`. The split, scoring
    statistic, gated/report partition, and per-seed record are candidate 1's.
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
# Modal-failure check (registered c3 modal + c2-killer movement)
# --------------------------------------------------------------------------
#: The registered modal failure (comment 4911357564): the widowed-stock
#: cell that failed even under candidate 1's time-invariant tables.
REGISTERED_MODAL_CELL = "share_widowed.75+|female"
#: The registered secondary: the tight-tolerance young-age clips the knot
#: targets.
SECONDARY_CELLS = (
    "first_marriage.18-24|female",
    "first_marriage.18-24|male",
)
#: The candidate-2 killers the delta-2 smoothing fix targets -- the
#: widowhood cascade that candidate 2's add-one convention exploded (the
#: remarriage bands, the divorced/widowed stocks, the male widowhood
#: hazard). The artifact records their movement whether they pass or fail.
C2_KILLER_CELLS = (
    "remarriage.ysd0-4",
    "remarriage.ysd5-9",
    "remarriage.ysd10+",
    "share_divorced.45-54|female",
    "share_divorced.45-54|male",
    "share_divorced.55-64|female",
    "share_divorced.55-64|male",
    "share_widowed.65-74|female",
    "share_widowed.75+|female",
    "widowhood.45+|male",
)


def _modal_failure_check(
    verdict: dict[str, Any], per_seed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Did the registered modal failure materialize, and how did c2's move?

    Registered modal failure (comment 4911357564):
    ``share_widowed.75+|female`` -- the widowed-stock cell that failed even
    under candidate 1's time-invariant tables, where the drift lives in the
    thin early-period cells shrinkage can only partly reach. Also tracks the
    registered secondary (first_marriage.18-24 tight-tolerance clips) and the
    candidate-2 killers the two deltas target, so the artifact records their
    movement whether they pass or fail.
    """
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

    modal_failed = REGISTERED_MODAL_CELL in fails_by_cell
    return {
        "registered_modal": (
            f"{REGISTERED_MODAL_CELL} (the widowed-stock cell that failed "
            "even under candidate 1's time-invariant tables; drift in the "
            "thin early-period cells shrinkage can only partly reach)"
        ),
        "modal_cell": REGISTERED_MODAL_CELL,
        "modal_failed": modal_failed,
        "modal_failed_seeds": sorted(
            fails_by_cell.get(REGISTERED_MODAL_CELL, [])
        ),
        "modal_track": track(REGISTERED_MODAL_CELL),
        "any_materialized": modal_failed,
        "registered_secondary": (
            "first_marriage.18-24 tight-tolerance clips surviving the knot"
        ),
        "secondary_track": {c: track(c) for c in SECONDARY_CELLS},
        "candidate2_killer_movement": {c: track(c) for c in C2_KILLER_CELLS},
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _revision_pins(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Candidate 1's pins, with the candidate-3 schema + c1/c2 runner shas."""
    pins = c1._revision_pins(thresholds)
    pins["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    pins["candidate1_runner"] = "scripts/run_gate2_candidate1.py"
    pins["candidate1_runner_sha256"] = c1._sha_of_file(
        ROOT / "scripts" / "run_gate2_candidate1.py"
    )
    pins["candidate2_runner"] = "scripts/run_gate2_candidate2.py"
    pins["candidate2_runner_sha256"] = c1._sha_of_file(
        ROOT / "scripts" / "run_gate2_candidate2.py"
    )
    return pins


def _model_block() -> dict[str, Any]:
    """Candidate 2's model block, edited for the two candidate-3 deltas."""
    return {
        "class": (
            "stratified empirical family-transition hazards with a "
            "mortality-composed widowhood component (candidate 2 + two named "
            "fixes: pooled-rate mortality shrinkage; a first-marriage spline "
            "knot at 22)"
        ),
        "stochastic": True,
        "populace_fit_used": False,
        "deltas_vs_candidate2": list(DELTAS_VS_CANDIDATE2),
        "components": {
            "first_marriage": (
                "discrete-time logistic hazard; natural cubic (restricted) "
                "spline on age, knots 20/22/25/30/40 (DELTA 1, adds 22; 4 "
                "basis columns), sex, birth-decade cohort; main effects + "
                "age-spline x sex interaction + age-spline x cohort "
                "interaction (interactions unchanged from candidate 2); "
                "sklearn LogisticRegression(penalty='l2', C=1.0, lbfgs) fit "
                "with sample_weight = person-year PSID weight (effectively "
                "the weighted MLE at train scale); design standardised for "
                "conditioning"
            ),
            "divorce": (
                "weighted empirical hazard by marriage-duration band x "
                "marriage order (1st vs 2+), add-one (Laplace) smoothed at "
                "the train mean married person-year weight (byte-identical "
                "to candidates 1 and 2)"
            ),
            "widowhood": (
                "COMPOSED: train DECADE-PERIOD x age-band x sex mortality "
                "hazard (build_mortality_floors construction, train persons) "
                "smoothed by DELTA 2 -- exposure-weighted shrinkage toward "
                "the pooled band x sex rate, cell_rate = (Wd + K * "
                "pooled_rate(band, sex)) / (We + K), K = 500 weighted "
                "person-years a-priori, pooled_rate = candidate 1's pooled "
                "weighted central rate -- applied to the spouse (opposite "
                "sex, age = self age + train mean spousal age gap by sex) at "
                "the simulated year's calendar decade; widowhood = induced "
                "transition"
            ),
            "remarriage": (
                "weighted empirical hazard by years-since-dissolution band x "
                "origin (divorced/widowed) x sex, add-one smoothed at the "
                "train mean dissolved person-year weight (byte-identical to "
                "candidates 1 and 2)"
            ),
            "fertility": (
                "weighted empirical age-band x parity (0/1/2/3+) rates by "
                "birth-decade cohort, train-estimated (no smoothing; "
                "byte-identical to candidates 1 and 2, and RNG-isolated from "
                "the marital process so its per-seed outcomes reproduce "
                "candidate 1 bit-for-bit)"
            ),
        },
        "registered_ambiguity_resolutions": {
            "mortality_shrinkage": (
                "exposure-weighted shrinkage: cell_rate = (weighted_deaths "
                "+ K * pooled_rate(age, sex)) / (weighted_exposure + K), K = "
                "500 weighted person-years a-priori (a round prior-strength "
                "constant chosen before any run; disclosed, not tuned); the "
                "shrinkage target pooled_rate(age, sex) is candidate 1's "
                "pooled band x sex weighted central rate "
                "(build_mortality_floors.weighted_hazards on the same train "
                "slices), so thin period x band x sex cells collapse to "
                "candidate 1's pooled behavior and thick cells keep the "
                "decade-period signal"
            ),
            "spline_knots": (
                "natural cubic = restricted cubic spline (Harrell), knots "
                "20/22/25/30/40 -> 4 basis columns; delta 1 is exactly the "
                "added knot at 22, sharpening the young-age curvature; the "
                "age-spline x sex and age-spline x cohort interactions are "
                "candidate 2's, unchanged"
            ),
            "period_definition": (
                "decade-period = the slice's wave-anchored start_wave decade "
                "(candidate 2's period concept, unchanged); the simulated "
                "spouse death in year y uses the calendar decade of y, "
                "nearest fitted period if outside the exposure span"
            ),
            "everything_else": (
                "divorce, remarriage, fertility, the spousal-age-gap "
                "imputation, the simulation loop (candidate 2's "
                "simulate_holdout, reused unchanged), the RNG rule "
                "default_rng(4200 + seed), one sequence per person, and the "
                "locked protocol are byte-identical to candidate 2"
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

    # Hard-stop precheck BEFORE any candidate is simulated (candidate 1's).
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
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": RUN_NAME,
        "gate": "gate_2",
        "candidate": "candidate 3",
        "spec_registration": SPEC_REGISTRATION,
        "candidate2_registration": CANDIDATE2_REGISTRATION,
        "candidate1_registration": CANDIDATE1_REGISTRATION,
        "deltas_vs_candidate2": list(DELTAS_VS_CANDIDATE2),
        "gate_lock": (
            "gates.yaml gate_2 locked (ratified PR #79 merge 82006877 + flip "
            "#81); protocol/views/tolerances read at runtime, no threshold "
            "moved."
        ),
        "pre_registered_forecast": {
            "p_pass": "0.40-0.50",
            "conjunction_estimate": 0.45,
            "component_probabilities": {
                "cascade_un_explodes": 0.85,
                "remarriage_share_divorced_return_to_c1": 0.75,
                "knot_clears_young_age_clips": 0.55,
                "widowed_stock_holds": 0.45,
            },
            "modal_failure": (
                "share_widowed.75+|female (the widowed stock failed even "
                "under candidate 1's time-invariant tables; shrinkage "
                "delivers period signal only where cells are thick, and the "
                "thin early-period cells are where the drift lives)"
            ),
            "secondary_failure": [
                "first_marriage.18-24 tight-tolerance clips surviving the knot"
            ],
            "deltas_vs_candidate2": list(DELTAS_VS_CANDIDATE2),
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
                "weight (populace_dynamics.data.panels.demographic_panel); "
                "every gated statistic weighted, none unweighted"
            ),
        },
        "data": data_meta,
        "precheck": precheck,
        "per_seed": per_seed,
        "seed_conjunction": seed_conjunction,
        "report_only": report_block,
        "modal_failure_materialized": modal,
        "verdict": verdict,
        "forecast_pointer": {
            "registration": SPEC_REGISTRATION,
            "candidate2_registration": CANDIDATE2_REGISTRATION,
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
            "registered modal failure (share_widowed.75+|female) "
            f"materialized: {modal['any_materialized']} "
            f"(seeds {modal['modal_failed_seeds']})"
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
