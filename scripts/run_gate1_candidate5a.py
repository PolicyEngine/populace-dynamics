"""Gate-1 candidate 5a: MINT-style donor splicing.

The FIFTH pre-registered model run of PolicyEngine/populace-dynamics, and
the first NON-generative candidate: deterministic donor matching plus
age-indexed whole-career splicing, the strategy SSA's MINT uses (donor
earnings records spliced onto targets). The candidate-5a spec is
registered, frozen before the run, in issue #42's candidate-5a comment
(https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4891949761);
every rule -- match, splice, scaling, fallbacks -- is pinned there and
implemented LITERALLY. No threshold is hardcoded, no rule is tuned
against holdout scores, and the run is one shot. The outcome publishes
whether it passes or fails.

Why donor splicing (from the candidate-5a registration). The four prior
runs exposed a symmetry: conditional-draw candidates (baseline, 2, 3)
cannot set person-level variance, and the structural candidate (4) broke
the marginal. Splicing reuses whole OBSERVED careers, inheriting the
marginal AND the person-level dependence from real data at once, so it
doubles as the field-standard benchmark. If it passes, the gate has its
first PASS and the generative program has the bar it must beat on
conditioning flexibility; its production limits (donor support, coarse
covariate conditioning) are why the generative track continues.

The candidate, per the frozen spec (each rule implemented literally):

* **Donor pool.** The seed's 80% train persons' observed trajectories in
  the locked filtered panel.
* **Match rule (deterministic).** For each holdout person, candidate
  donors are train persons observed in the SAME calendar PERIOD as the
  holdout's ANCHOR (their chronologically last observed period) with age
  within +/-2 years of the holdout's anchor age; among them select the
  donor with the smallest absolute difference in ANCHOR-PERIOD EARNINGS;
  ties broken by smaller absolute age difference, then smaller
  ``person_id``. If the cell is empty, widen the age window in +/-2 steps
  (+/-4, +/-6, ...) until non-empty (widening count reported).
* **Splice (age-indexed, whole-career, one donor per holdout person).**
  For each holdout observed period ``t`` OTHER than the anchor, the
  generated value is the donor's earnings at the HOLDOUT's age at ``t``;
  if the donor is not observed at that exact age, use the donor's
  earnings at the NEAREST observed age (ties -> younger); the nearest-age
  fallback usage rate is reported.
* **Level adjustment.** If both the holdout's anchor earnings AND the
  donor's earnings at (nearest to) the anchor age are positive, all
  spliced values are multiplied by their ratio, clipped to ``[0.2, 5]``;
  otherwise no scaling (donor values copy raw, zeros stay zeros). The
  clip rate is reported.
* **Anchor** keeps its REAL value, as in every prior candidate.

Determinism. No RNG is needed: the run is fully deterministic given the
seed's split. The gate seed enters ONLY through
``split_panel_by_person``. Because no model is fit, this candidate needs
NO populace-fit and no fitting of any kind; it runs under the repo's
``.venv`` (scikit-learn 1.9, pandas). The protocol machinery -- the
filter-first load, the person-disjoint 0.2 split per seed, the two locked
views, ``panel_scorecard`` scoring, the battery vs the committed
``battery_reference`` with locked definitions, the thresholds read from
``gates.yaml`` at runtime, the seed-level conjunction (>=4/5 both
blocks), and the battery-reference bit-exact precheck -- is IMPORTED from
the merged baseline runner (:mod:`run_gate1_baseline`, pull request 40),
byte-for-byte the prior runs'. Only the generation (matching + splicing)
is local.

Run from the repository root with the PSID family files staged, using the
repo ``.venv`` (no populace-fit needed for this deterministic candidate):

    .venv/bin/python scripts/run_gate1_candidate5a.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The protocol machinery is IMPORTED from the merged baseline runner so
# that the filtered-panel load, the person-disjoint split, the view
# construction, the battery definitions, the geometry / battery checks,
# the threshold loading, and the battery-reference reproduction are
# byte-for-byte identical to every prior gate-1 run. Only the generation
# (deterministic donor matching + age-indexed splicing) is local. The
# baseline module defers its populace-fit import to its fit path, so this
# import succeeds under the repo ``.venv``; this candidate fits no model
# and never triggers that path.
from run_gate1_baseline import (  # noqa: F401 (re-exported for tests)
    AGE_MAX,
    AGE_MIN,
    BATTERY_REFERENCE_RUN,
    PERIOD_MAX,
    PERIOD_MIN,
    PERIOD_STEP,
    SEEDS,
    build_panel_view,
    check_battery,
    check_geometry,
    compute_battery,
    load_filtered_panel,
    load_gate1_thresholds,
    reproduce_battery_reference,
    split_holdout_train,
)

from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate1_splice_v1.json"
ARTIFACT_SCHEMA_VERSION = "gate1_splice.v1"
SPEC_REGISTRATION = (
    "https://github.com/PolicyEngine/populace-dynamics/issues/42"
    "#issuecomment-4891949761"
)

#: The frozen level-adjustment clip bounds (reported, applied literally).
SCALE_CLIP_LO, SCALE_CLIP_HI = 0.2, 5.0


# --------------------------------------------------------------------------
# Anchors (chronologically last observed period per person)
# --------------------------------------------------------------------------
def anchor_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per person: their chronologically LAST observed period.

    Anchor = the person's maximum ``period`` in the filtered panel, with
    that row's ``earnings``, ``age``, ``weight``. This is the SAME anchor
    definition the baseline chain uses (its chronologically last observed
    period), stated once here as a person-level table. Deterministic
    (periods are unique per person-period, so the ``idxmax`` is unique).
    """
    idx = panel.groupby("person_id")["period"].idxmax()
    cols = ["person_id", "period", "earnings", "age", "weight"]
    return panel.loc[idx, cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Donor trajectories (age -> earnings, sorted by age, per donor person)
# --------------------------------------------------------------------------
def build_donor_trajectories(
    train: pd.DataFrame,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per train-person age->earnings arrays, sorted ascending by age.

    Ages are unique within a person on the biennial panel, so the sorted
    ``(ages, earnings)`` pair supports the nearest-observed-age lookup
    exactly. Returns ``{person_id: (ages, earnings)}``.
    """
    traj: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pid, g in train.groupby("person_id"):
        gg = g.sort_values("age")
        traj[int(pid)] = (
            gg["age"].to_numpy(dtype=np.float64),
            gg["earnings"].to_numpy(dtype=np.float64),
        )
    return traj


def donor_earnings_at_age(
    ages: np.ndarray, earnings: np.ndarray, target_age: float
) -> tuple[float, bool]:
    """Donor earnings at ``target_age`` via the nearest-observed-age rule.

    If the donor is observed at exactly ``target_age``, return that
    value with ``used_fallback = False``. Otherwise return the donor's
    earnings at the nearest observed age; on an age-distance tie the
    YOUNGER (smaller) age wins, and ``used_fallback = True``.
    """
    exact = np.nonzero(ages == target_age)[0]
    if exact.size:
        return float(earnings[exact[0]]), False
    diff = np.abs(ages - target_age)
    mind = diff.min()
    tied = np.nonzero(diff == mind)[0]
    # Ages are ascending, so the first tied index is the younger age.
    younger = tied[int(np.argmin(ages[tied]))]
    return float(earnings[younger]), True


# --------------------------------------------------------------------------
# Match rule (deterministic donor selection at the anchor)
# --------------------------------------------------------------------------
def match_donor(
    anchor_period: int,
    anchor_age: float,
    anchor_earnings: float,
    donor_by_period: dict[int, pd.DataFrame],
) -> tuple[int, int]:
    """Select the matched donor person for one holdout anchor.

    Candidate donors are train persons observed in the SAME calendar
    ``anchor_period`` with age within an age window of ``anchor_age``.
    The window starts at +/-2 and widens in +/-2 steps until the cell is
    non-empty. Among the non-empty cell, the donor with the smallest
    ``|donor anchor-period earnings - anchor_earnings|`` wins; ties break
    by smaller ``|donor age - anchor_age|``, then smaller ``person_id``.

    Returns ``(donor_person_id, widen_count)`` where ``widen_count`` is
    the number of +/-2 widenings applied beyond the initial +/-2 window
    (0 if the initial window already had a donor).
    """
    pool = donor_by_period[anchor_period]
    p_age = pool["age"].to_numpy(dtype=np.float64)
    p_earn = pool["earnings"].to_numpy(dtype=np.float64)
    p_pid = pool["person_id"].to_numpy()

    half = 2.0
    widen = 0
    while True:
        mask = np.abs(p_age - anchor_age) <= half
        if mask.any():
            break
        half += 2.0
        widen += 1

    c_age = p_age[mask]
    c_earn = p_earn[mask]
    c_pid = p_pid[mask]
    d_earn = np.abs(c_earn - anchor_earnings)
    d_age = np.abs(c_age - anchor_age)
    # lexsort: last key is primary. Primary = |earnings diff|, then
    # |age diff|, then person_id (all ascending).
    order = np.lexsort((c_pid, d_age, d_earn))
    return int(c_pid[order[0]]), widen


# --------------------------------------------------------------------------
# Generation (deterministic match + age-indexed splice + level scaling)
# --------------------------------------------------------------------------
def generate_candidate(
    holdout: pd.DataFrame,
    train: pd.DataFrame,
    all_anchor: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """MINT-style spliced candidate panel over the holdout persons.

    For each holdout person: match one donor at the anchor (the match
    rule), then splice the donor's whole career onto the holdout's
    observed periods by age index -- the generated value at each
    non-anchor period ``t`` is the donor's earnings at the holdout's age
    at ``t`` (nearest observed age fallback), scaled by the anchor
    earnings ratio when both endpoints are positive (clipped to
    ``[0.2, 5]``). The anchor keeps its REAL earnings.

    Returns ``(candidate, diagnostics)`` where ``candidate`` holds exactly
    the holdout persons on exactly their observed periods (the locked
    candidate-panel pin: only ``earnings`` is generated;
    ``person_id`` / ``period`` / ``age`` / ``weight`` copy from the
    holdout), and ``diagnostics`` carries the reported-not-gated
    distributions (age-window widening, nearest-age fallback rate,
    scaling-clip rate, donor reuse).
    """
    holdout_ids = holdout["person_id"].unique()
    holdout_anchor = (
        all_anchor[all_anchor.person_id.isin(holdout_ids)]
        .sort_values("person_id")
        .reset_index(drop=True)
    )

    donor = train[["person_id", "period", "earnings", "age", "weight"]]
    donor_by_period = {int(p): g for p, g in donor.groupby("period")}
    donor_traj = build_donor_trajectories(donor)

    hp = holdout.sort_values(["person_id", "period"]).reset_index(drop=True)
    gen_earn = hp["earnings"].to_numpy(dtype=np.float64).copy()
    hp_pid = hp["person_id"].to_numpy()
    hp_period = hp["period"].to_numpy()
    hp_age = hp["age"].to_numpy(dtype=np.float64)

    # Diagnostics accumulators.
    widen_counts: list[int] = []
    donor_of: dict[int, int] = {}
    n_spliced_obs = 0
    n_fallback_obs = 0
    n_scaled_persons = 0
    n_clipped_persons = 0
    scale_ratios: list[float] = []

    # Precompute holdout row positions per person (already grouped by the
    # stable sort). Iterate persons in ``person_id`` order for a
    # deterministic, reproducible pass (order is immaterial to the
    # result -- each person is spliced independently -- but pinned).
    row_positions: dict[int, list[int]] = {}
    for i in range(len(hp)):
        row_positions.setdefault(int(hp_pid[i]), []).append(i)

    for row in holdout_anchor.itertuples():
        pid = int(row.person_id)
        anc_period = int(row.period)
        anc_age = float(row.age)
        anc_earn = float(row.earnings)

        donor_pid, widen = match_donor(
            anc_period, anc_age, anc_earn, donor_by_period
        )
        widen_counts.append(widen)
        donor_of[pid] = donor_pid
        d_ages, d_earn = donor_traj[donor_pid]

        # Level-adjustment ratio: holdout anchor earnings over the
        # donor's earnings at (nearest to) the holdout's anchor age.
        donor_anchor_val, _ = donor_earnings_at_age(d_ages, d_earn, anc_age)
        if anc_earn > 0 and donor_anchor_val > 0:
            raw_ratio = anc_earn / donor_anchor_val
            ratio = float(np.clip(raw_ratio, SCALE_CLIP_LO, SCALE_CLIP_HI))
            scale = ratio
            n_scaled_persons += 1
            scale_ratios.append(raw_ratio)
            if raw_ratio < SCALE_CLIP_LO or raw_ratio > SCALE_CLIP_HI:
                n_clipped_persons += 1
        else:
            scale = 1.0  # no scaling: donor values copy raw.

        for i in row_positions[pid]:
            if int(hp_period[i]) == anc_period:
                continue  # anchor keeps its real value.
            val, used_fb = donor_earnings_at_age(
                d_ages, d_earn, float(hp_age[i])
            )
            n_spliced_obs += 1
            if used_fb:
                n_fallback_obs += 1
            gen_earn[i] = val * scale  # zeros stay zeros.

    out = hp.copy()
    out["earnings"] = gen_earn
    candidate = out[["person_id", "period", "earnings", "age", "weight"]]

    reuse = pd.Series(list(donor_of.values())).value_counts()
    reuse_dist = reuse.value_counts().sort_index()
    widen_arr = np.array(widen_counts, dtype=int)
    widen_unique, widen_counts_arr = np.unique(widen_arr, return_counts=True)
    widen_dist = {
        int(k): int(v)
        for k, v in zip(widen_unique, widen_counts_arr, strict=True)
    }
    diagnostics = {
        "n_holdout_persons": int(len(holdout_anchor)),
        "age_window_widening": {
            "distribution": widen_dist,
            "any_widened": int((widen_arr > 0).sum()),
            "max_widen_steps": int(widen_arr.max()) if len(widen_arr) else 0,
            "note": (
                "count of +/-2 age-window widenings applied beyond the "
                "initial +/-2 window per holdout person (0 = matched in "
                "the initial window)"
            ),
        },
        "nearest_age_fallback": {
            "n_spliced_observations": int(n_spliced_obs),
            "n_fallback_observations": int(n_fallback_obs),
            "rate": (
                float(n_fallback_obs / n_spliced_obs) if n_spliced_obs else 0.0
            ),
            "note": (
                "share of spliced (non-anchor) observations for which the "
                "donor had no observation at the holdout's exact age, so "
                "the nearest observed age (ties -> younger) was used"
            ),
        },
        "scaling_clip": {
            "n_scaled_persons": int(n_scaled_persons),
            "n_clipped_persons": int(n_clipped_persons),
            "rate_over_scaled": (
                float(n_clipped_persons / n_scaled_persons)
                if n_scaled_persons
                else 0.0
            ),
            "rate_over_holdout": (
                float(n_clipped_persons / len(holdout_anchor))
                if len(holdout_anchor)
                else 0.0
            ),
            "raw_ratio_min": (
                float(np.min(scale_ratios)) if scale_ratios else None
            ),
            "raw_ratio_max": (
                float(np.max(scale_ratios)) if scale_ratios else None
            ),
            "clip_bounds": [SCALE_CLIP_LO, SCALE_CLIP_HI],
            "note": (
                "a person is scaled when its anchor earnings and the "
                "donor's earnings at (nearest to) the anchor age are both "
                "positive; clipped when the raw ratio falls outside "
                "[0.2, 5]"
            ),
        },
        "donor_reuse": {
            "n_distinct_donors": int(len(reuse)),
            "max_reuse": int(reuse.max()) if len(reuse) else 0,
            "reuse_count_distribution": {
                int(k): int(v) for k, v in reuse_dist.items()
            },
            "note": (
                "reuse_count_distribution maps 'number of holdout persons "
                "sharing one donor' -> 'number of donors reused that many "
                "times'"
            ),
        },
    }
    return candidate, diagnostics


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_seed(
    seed: int,
    panel: pd.DataFrame,
    all_anchor: pd.DataFrame,
    view_specs: dict[str, Any],
    views_cfg: dict[str, Any],
    battery_reference: dict[str, float],
    battery_tol: dict[str, float],
    verbose: bool,
) -> dict[str, Any]:
    """Match, splice, and score candidate 5a for a single gate seed."""
    seed_t = time.time()
    holdout, train = split_holdout_train(panel, seed)
    candidate, diagnostics = generate_candidate(holdout, train, all_anchor)

    # --- geometry: score candidate vs holdout on both locked views ---
    geometry_by_view: dict[str, Any] = {}
    geometry_seed_pass = True
    n_windows: dict[str, int] = {}
    for vname, view in view_specs.items():
        scores = hpanel.panel_scorecard(candidate, holdout, view, seed=seed)
        checks = check_geometry(scores, views_cfg[vname]["geometry"])
        view_pass = all(c["pass"] for c in checks.values())
        geometry_seed_pass = geometry_seed_pass and view_pass
        cand_windows, _ = hpanel.project_panel(candidate, view)
        n_windows[vname] = int(len(cand_windows))
        geometry_by_view[vname] = {
            "scores": {k: float(v) for k, v in scores.items()},
            "thresholds": views_cfg[vname]["geometry"],
            "checks": checks,
            "view_pass": bool(view_pass),
        }

    # --- battery: on the CANDIDATE panel, vs committed reference ---
    battery_values = compute_battery(candidate)
    battery_checks = check_battery(
        battery_values, battery_reference, battery_tol
    )
    battery_seed_pass = all(c["pass"] for c in battery_checks.values())

    result = {
        "seed": seed,
        "n_persons": int(holdout.person_id.nunique()),
        "n_person_periods": int(len(holdout)),
        "n_train_persons": int(train.person_id.nunique()),
        "n_windows": n_windows,
        "splice_diagnostics": diagnostics,
        "geometry": geometry_by_view,
        "geometry_pass": bool(geometry_seed_pass),
        "battery_values": battery_values,
        "battery_checks": battery_checks,
        "battery_pass": bool(battery_seed_pass),
    }
    if verbose:
        d = diagnostics
        print(
            f"seed {seed}: geometry_pass={geometry_seed_pass} "
            f"battery_pass={battery_seed_pass} "
            f"fallback={d['nearest_age_fallback']['rate']:.3f} "
            f"clip={d['scaling_clip']['rate_over_holdout']:.3f} "
            f"max_reuse={d['donor_reuse']['max_reuse']} "
            f"({time.time() - seed_t:.0f}s)"
        )
    return result


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full pre-registered gate-1 candidate-5a run."""
    started = time.time()
    thresholds = load_gate1_thresholds()
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_1 thresholds are not locked; the pre-registered run may "
            "only execute against locked thresholds."
        )
    views_cfg = thresholds["views"]
    battery_tol = {
        k: v
        for k, v in thresholds["battery"].items()
        if k.endswith("_tolerance")
    }

    battery_ref_artifact = json.loads(
        (ROOT / BATTERY_REFERENCE_RUN).read_text()
    )
    battery_reference = battery_ref_artifact["battery_reference"]

    panel = load_filtered_panel()
    if verbose:
        print(
            f"filtered panel: {len(panel)} person-periods, "
            f"{panel.person_id.nunique()} persons"
        )

    # Identical battery-reference bit-exact precheck as every prior run:
    # the battery code path must reproduce every committed reference value
    # to float precision before any candidate is scored.
    repro = reproduce_battery_reference(panel)
    if verbose:
        print(
            "battery_reference reproduced exactly: "
            f"{repro['all_committed_values_reproduced_exactly']}"
        )
    if not repro["all_committed_values_reproduced_exactly"]:
        raise RuntimeError(
            "Battery code path does not reproduce the committed "
            "battery_reference to float precision; refusing to proceed "
            "with a divergent definition."
        )

    # Anchors on the FULL filtered panel (a person's last observed period
    # is a property of the panel, computed once and sliced per split).
    all_anchor = anchor_rows(panel)

    view_specs = {
        "psid_family_earnings_pairs": build_panel_view(
            "psid_family_earnings_pairs", window=2
        ),
        "psid_family_earnings_runs": build_panel_view(
            "psid_family_earnings_runs", window=3
        ),
    }

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        per_seed.append(
            run_seed(
                seed,
                panel,
                all_anchor,
                view_specs,
                views_cfg,
                battery_reference,
                battery_tol,
                verbose,
            )
        )

    n_geo = sum(1 for s in per_seed if s["geometry_pass"])
    n_bat = sum(1 for s in per_seed if s["battery_pass"])
    geometry_gate_pass = n_geo >= 4
    battery_gate_pass = n_bat >= 4
    gate_pass = geometry_gate_pass and battery_gate_pass

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "gate1_splice_v1",
        "gate": "gate_1",
        "spec_registration": SPEC_REGISTRATION,
        "description": (
            "Gate-1 candidate 5a: MINT-style donor splicing. For each "
            "held-out person a single train-person donor is matched "
            "deterministically at the anchor (same calendar period, age "
            "within a +/-2-year window widened as needed, smallest "
            "anchor-earnings gap; ties -> smaller age gap, then smaller "
            "person_id), and the donor's whole observed career is spliced "
            "onto the holdout's observed periods by age index (nearest "
            "observed age fallback, ties -> younger), scaled by the "
            "anchor-earnings ratio when both endpoints are positive "
            "(clipped to [0.2, 5]); the anchor keeps its real value. "
            "Fully deterministic given the split -- no RNG, no model fit, "
            "no populace-fit. Registered frozen before the run in issue "
            "#42 (see spec_registration). Candidate scored against the "
            "held-out PSID family earnings panel geometry (two locked "
            "views) and the locked moment battery, per the locked "
            "seed-level conjunction in gates.yaml (pull request 39). "
            "Protocol machinery imported byte-for-byte from the baseline "
            "runner (pull request 40)."
        ),
        "model": {
            "class": "deterministic donor matching + age-indexed splicing",
            "stochastic": False,
            "populace_fit_used": False,
            "runtime_environment": (
                "repo .venv (scikit-learn 1.9, pandas); no populace-fit"
            ),
            "donor_pool": (
                "the seed's 80% train persons' observed trajectories in "
                "the locked filtered panel"
            ),
            "match_rule": {
                "cell": (
                    "train persons observed in the same calendar period as "
                    "the holdout's anchor (chronologically last observed "
                    "period) with age within +/-2 years of the holdout's "
                    "anchor age"
                ),
                "selection": (
                    "smallest |donor anchor-period earnings - holdout "
                    "anchor earnings|"
                ),
                "tie_breaks": [
                    "smaller |donor age - holdout anchor age|",
                    "smaller donor person_id",
                ],
                "empty_cell": (
                    "widen the age window in +/-2 steps (+/-4, +/-6, ...) "
                    "until non-empty; widening count reported"
                ),
            },
            "splice_rule": {
                "scope": (
                    "age-indexed, whole-career, one donor per holdout "
                    "person"
                ),
                "value": (
                    "for each holdout observed period t other than the "
                    "anchor, the donor's earnings at the holdout's age at "
                    "t"
                ),
                "nearest_age_fallback": (
                    "if the donor is not observed at that exact age, the "
                    "donor's earnings at the nearest observed age (ties -> "
                    "younger); fallback usage rate reported"
                ),
            },
            "level_adjustment": {
                "condition": (
                    "both the holdout's anchor earnings and the donor's "
                    "earnings at (nearest to) the anchor age are positive"
                ),
                "factor": (
                    "holdout anchor earnings / donor earnings at (nearest "
                    "to) the anchor age, clipped to [0.2, 5]"
                ),
                "otherwise": "no scaling (donor values copy raw)",
                "zeros": "zeros stay zeros",
                "clip_bounds": [SCALE_CLIP_LO, SCALE_CLIP_HI],
            },
            "anchor": (
                "chronologically last observed period held at real " "earnings"
            ),
            "candidate_panel_pin": (
                "exactly the holdout persons on exactly their observed "
                "periods; only earnings generated; anchor keeps real "
                "value; person_id/period/age/weight copied from holdout"
            ),
            "determinism": (
                "fully deterministic given the split; the gate seed enters "
                "only through split_panel_by_person"
            ),
        },
        "protocol": {
            "filter": (
                f"age {AGE_MIN}-{AGE_MAX}, reference years "
                f"{PERIOD_MIN}-{PERIOD_MAX}, positive weights (applied "
                "before the split)"
            ),
            "split": (
                "populace_dynamics.harness.panel.split_panel_by_person("
                "panel, 'person_id', fraction=0.2, seed=s); the drawn 20% "
                "is the holdout, the complement is the training set / "
                "donor pool (imported from the baseline runner)"
            ),
            "seeds": list(SEEDS),
            "views": {
                "psid_family_earnings_pairs": {"window": 2, "period_step": 2},
                "psid_family_earnings_runs": {"window": 3, "period_step": 2},
            },
            "scoring": (
                "panel_scorecard(candidate, holdout, view, seed=s) per "
                "locked view; battery on the candidate panel vs committed "
                "battery_reference (imported from the baseline runner)"
            ),
            "pass_rule": (
                "seed passes geometry iff every locked threshold on every "
                "locked view holds; seed passes battery iff every locked "
                "tolerance holds; gate passes iff >=4/5 seeds pass geometry "
                "AND >=4/5 seeds pass battery"
            ),
        },
        "battery_reference_reproduction": repro,
        "battery_reference_run": BATTERY_REFERENCE_RUN,
        "per_seed": per_seed,
        "seed_conjunction": [
            {
                "seed": s["seed"],
                "geometry_pass": s["geometry_pass"],
                "battery_pass": s["battery_pass"],
            }
            for s in per_seed
        ],
        "splice_diagnostics_context": {
            "note": (
                "Reported-not-gated diagnostics per seed: age-window "
                "widening distribution, nearest-age fallback rate, "
                "scaling-clip rate, and donor reuse distribution. None of "
                "these enters the geometry or battery pass/fail; the gate "
                "rule names only those two families."
            ),
            "per_seed": [
                {
                    "seed": s["seed"],
                    "age_window_widening": s["splice_diagnostics"][
                        "age_window_widening"
                    ],
                    "nearest_age_fallback_rate": s["splice_diagnostics"][
                        "nearest_age_fallback"
                    ]["rate"],
                    "scaling_clip_rate_over_holdout": s["splice_diagnostics"][
                        "scaling_clip"
                    ]["rate_over_holdout"],
                    "donor_reuse": s["splice_diagnostics"]["donor_reuse"],
                }
                for s in per_seed
            ],
        },
        "verdict": {
            "n_seeds": len(SEEDS),
            "n_geometry_pass": n_geo,
            "n_battery_pass": n_bat,
            "geometry_gate_pass": bool(geometry_gate_pass),
            "battery_gate_pass": bool(battery_gate_pass),
            "gate_1_pass": bool(gate_pass),
            "rule": ">=4/5 seeds geometry AND >=4/5 seeds battery",
        },
        "revision_pins": _revision_pins(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        v = artifact["verdict"]
        print(
            f"\nVERDICT: gate_1_pass={v['gate_1_pass']} "
            f"(geometry {n_geo}/5, battery {n_bat}/5)"
        )
    return artifact


def _revision_pins() -> dict[str, Any]:
    """Repo/populace SHAs and schema version for provenance."""
    import subprocess

    def _sha(cwd: Path) -> str | None:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
                .decode()
                .strip()
            )
        except Exception:
            return None

    populace_root = Path("~/PolicyEngine/populace").expanduser()
    return {
        "populace_dynamics_sha": _sha(ROOT),
        "populace_repo_sha": _sha(populace_root),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "gates_yaml_locked": True,
    }


def main() -> None:
    artifact = run(verbose=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
