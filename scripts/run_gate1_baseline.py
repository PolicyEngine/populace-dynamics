"""Gate-1 baseline run: the chained weighted QRF backcast.

The first pre-registered model run of PolicyEngine/populace-dynamics.
Scores the locked baseline earnings process — populace-fit's regime-gated
chained weighted QRF, run as a one-step BACKWARD biennial backcast —
against the LOCKED gate-1 thresholds in ``gates.yaml`` (commit that
flipped ``locked: true``, pull request 39).

This is a one-shot pre-registered evaluation. The protocol is read from
``gates.yaml`` at runtime and implemented literally; no threshold is
hardcoded and no model choice is tuned against holdout scores. The
outcome publishes whether it passes or fails.

The baseline, exactly as fixed by the feature author:

* **Model.** ``populace.fit.qrf.RegimeGatedQRF`` at its DEFAULT
  hyperparameters (100 trees, ``zero_atol=1e-6``, ``max_samples_leaf=
  None``), seeded from the gate seed ``s``. The plain-table front door
  is used (explicit weight column), so no Frame is constructed.
* **Transition.** One-step BACKWARD biennial transition, mirroring
  production backcast use: target = earnings at period ``t-2``,
  predictors = (earnings at ``t``, age at ``t-2``), sample_weight = the
  pair's earlier-period (``t-2``) person-period weight. Fit on the 80%
  complement's adjacent-period pairs. ``RegimeGatedQRF`` handles the
  zero regime (the target is zero-inflated positive).
* **Generation.** Per holdout person: anchor the chronologically LAST
  observed period at its REAL earnings, then chain BACKWARD over that
  person's observed periods. Each earlier observed period's earnings is
  drawn from the fitted model conditioned on the next observed period's
  (generated or anchor) earnings and the person's age at the earlier
  period. Consecutive observed periods are 2 years apart; a gap (next
  observed period 4+ years later) gets the one-step model applied once
  across the gap (the naive baseline's documented gap handling). Only
  earnings is generated; person_id / period / age / weight copy from
  the holdout rows, and the anchor period keeps its real value. The
  candidate panel therefore holds exactly the holdout persons on
  exactly their observed periods (the locked candidate-panel pin).

Determinism. Generation batches by step-from-anchor; within a step,
rows are ordered by ``person_id``. With a freshly fitted model per seed
(its draw RNG seeded from the model seed) this fixes every draw, so the
run is reproducible from the seeds alone.

Run from the repository root with the PSID family files staged:

    .venv/bin/python scripts/run_gate1_baseline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Importing populace.fit runs its import-time compat gate.
from populace.fit.qrf import (
    DEFAULT_N_ESTIMATORS,
    DEFAULT_ZERO_ATOL,
    RegimeGatedQRF,
)

from populace_dynamics.data.family import family_earnings_panel
from populace_dynamics.harness import moments
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "gate1_qrf_baseline_v1.json"
BATTERY_REFERENCE_RUN = "runs/noise_floor_psid_family_9822.json"
ARTIFACT_SCHEMA_VERSION = "gate1_qrf_baseline.v1"

#: The locked filter and split constants (also read/echoed from gates.yaml).
AGE_MIN, AGE_MAX = 25, 59
PERIOD_MIN, PERIOD_MAX = 1998, 2022
PERIOD_STEP = 2
HOLDOUT_FRACTION = 0.2
SEEDS = (0, 1, 2, 3, 4)

#: Battery moment keys as computed here, aliased to the committed
#: ``battery_reference`` keys in the floor artifact.
_BATTERY_REF_ALIAS = {"mobility_diagonal": "mobility_diagonal_mean"}

# Battery column order used for reporting / the moments call.
_MOMENT_KW = dict(
    id_col="person_id",
    period_col="period",
    value_col="earnings",
    weight_col="weight",
)


# --------------------------------------------------------------------------
# Panel + split
# --------------------------------------------------------------------------
def load_filtered_panel() -> pd.DataFrame:
    """Load the PSID family panel and apply the locked view filter.

    Filter FIRST (age 25-59, reference years 1998-2022, positive
    weights), exactly as the locked protocol's ``split`` block states.
    """
    raw = family_earnings_panel()
    prime = raw[
        (raw.age >= AGE_MIN)
        & (raw.age <= AGE_MAX)
        & (raw.period >= PERIOD_MIN)
        & (raw.period <= PERIOD_MAX)
        & (raw.weight > 0)
    ].copy()
    return prime.reset_index(drop=True)


def split_holdout_train(
    panel: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The locked person-disjoint split: the DRAWN 20% is the holdout.

    ``split_panel_by_person`` returns ``(left, right)`` where ``left``
    is the drawn ``fraction``. Per the protocol the drawn 20% of
    persons is the holdout and the complement is the training set.
    """
    holdout, train = hpanel.split_panel_by_person(
        panel, "person_id", fraction=HOLDOUT_FRACTION, seed=seed
    )
    return holdout, train


# --------------------------------------------------------------------------
# Backward transition model
# --------------------------------------------------------------------------
def build_backward_pairs(train: pd.DataFrame) -> pd.DataFrame:
    """Adjacent-period BACKWARD pairs on the training panel.

    One row per (person, period ``t``) whose period ``t-2`` is also
    observed:

    * ``earnings`` = earnings at ``t``          (predictor),
    * ``age_tm2``  = age at ``t-2``             (predictor),
    * ``earnings_tm2`` = earnings at ``t-2``    (target),
    * ``weight_tm2``   = person-period weight at ``t-2`` (sample_weight).

    Only exact ``period_step`` (2-year) adjacencies form a pair; gaps
    do not, so the fitted transition is strictly the one-step biennial
    backward step.
    """
    base = train[["person_id", "period", "earnings", "age", "weight"]].copy()
    earlier = base.rename(
        columns={
            "earnings": "earnings_tm2",
            "age": "age_tm2",
            "weight": "weight_tm2",
        }
    )
    # The earlier row sits at period t-2; lift its period by the step so
    # it inner-joins the base row observed at period t.
    earlier = earlier.assign(period=earlier["period"] + PERIOD_STEP)
    pairs = base.merge(
        earlier[
            ["person_id", "period", "earnings_tm2", "age_tm2", "weight_tm2"]
        ],
        on=["person_id", "period"],
        how="inner",
    )
    return pairs


def fit_backward_model(pairs: pd.DataFrame, seed: int) -> Any:
    """Fit ``RegimeGatedQRF`` (defaults) on the backward pairs.

    Uses the plain-DataFrame front door with the explicit weight column
    ``weight_tm2`` (the earlier-period weight), so the fit is weighted
    without constructing a Frame.
    """
    model = RegimeGatedQRF(seed=seed)  # default hyperparameters
    return model.fit(
        pairs,
        predictors=["earnings", "age_tm2"],
        targets=["earnings_tm2"],
        weights="weight_tm2",
    )


# --------------------------------------------------------------------------
# Candidate generation (backward chain)
# --------------------------------------------------------------------------
def generate_candidate(fitted: Any, holdout: pd.DataFrame) -> pd.DataFrame:
    """Backward-chained candidate panel over the holdout persons/periods.

    Anchor each person's chronologically last observed period at its
    real earnings; chain backward over consecutive observed periods,
    drawing each earlier period's earnings from ``fitted`` conditioned
    on the next observed period's (generated/anchor) earnings and the
    earlier period's age. A gap (next observed period 4+ years later)
    applies the one-step model once across the gap. Batched by
    step-from-anchor; within a step rows are ordered by ``person_id``.

    Returns a panel with exactly the holdout persons on exactly their
    observed periods; only ``earnings`` changes (anchor rows keep the
    real value), and ``person_id`` / ``period`` / ``age`` / ``weight``
    copy from the holdout rows (the locked candidate-panel pin).
    """
    hp = holdout.sort_values(["person_id", "period"]).reset_index(drop=True)
    # Position within person counted from the LATEST period (0 = anchor).
    hp["rank_from_top"] = (
        hp.groupby("person_id")["period"].rank(ascending=False, method="first")
        - 1
    ).astype(int)
    hp["depth"] = (
        hp.groupby("person_id")["period"].transform("size").astype(int)
    )

    gen_earn = hp["earnings"].to_numpy(dtype=np.float64).copy()
    ages = hp["age"].to_numpy(dtype=np.float64)
    pids = hp["person_id"].to_numpy()
    ranks = hp["rank_from_top"].to_numpy()
    pos_by_key = {
        (pid, r): i for i, (pid, r) in enumerate(zip(pids, ranks, strict=True))
    }
    max_depth = int(hp["depth"].max()) if len(hp) else 0

    for j in range(1, max_depth):
        earlier_positions = np.nonzero(ranks == j)[0]
        if earlier_positions.size == 0:
            continue
        # Canonical, deterministic row order within the step.
        order = np.argsort(pids[earlier_positions], kind="stable")
        earlier_positions = earlier_positions[order]
        next_positions = np.array(
            [pos_by_key[(pids[p], j - 1)] for p in earlier_positions]
        )
        feat = pd.DataFrame(
            {
                "earnings": gen_earn[next_positions],
                "age_tm2": ages[earlier_positions],
            }
        )
        drawn = fitted.predict(feat)["earnings_tm2"].to_numpy(dtype=np.float64)
        gen_earn[earlier_positions] = drawn

    out = hp.copy()
    out["earnings"] = gen_earn
    return out[["person_id", "period", "earnings", "age", "weight"]]


# --------------------------------------------------------------------------
# Views (built from the locked gates.yaml view specs)
# --------------------------------------------------------------------------
def build_panel_view(name: str, window: int) -> hpanel.PanelView:
    """The locked earnings view: value=earnings, covariate=age, step 2."""
    return hpanel.PanelView(
        name=name,
        id_column="person_id",
        period_column="period",
        value_columns=("earnings",),
        covariate_columns=("age",),
        weight_column="weight",
        window=window,
        period_step=PERIOD_STEP,
    )


# --------------------------------------------------------------------------
# Battery (locked definitions; reproduces battery_reference to float)
# --------------------------------------------------------------------------
def compute_battery(panel: pd.DataFrame) -> dict[str, float]:
    """The locked gate-1 battery statistics on one panel.

    Definitions pinned at lock (gates.yaml battery block):

    * ``autocorr_log_{2,4,10}yr``: ``moments.autocorrelation`` on log
      earnings among positives, lags 1/2/5 of the biennial panel
      (= 2/4/10 years), weighted.
    * ``mobility_diagonal``: unweighted mean over all six
      origin-diagonal cells of the horizon-1 mobility matrix (five
      weighted quintile bins plus the zero bin).
    * ``zero_persistence`` = 1 - ``exit_rate``.
    * ``entry_rate`` / ``exit_rate`` / ``mean_spell_length`` from
      ``moments.zero_spells``.
    """
    ac = moments.autocorrelation(
        panel, lags=(1, 2, 5), period_step=PERIOD_STEP, log=True, **_MOMENT_KW
    )
    ac_by_lag = {int(r.lag): float(r.value) for r in ac.itertuples()}

    mob = moments.mobility_matrix(
        panel,
        horizon=1,
        period_step=PERIOD_STEP,
        n_bins=5,
        zero_bin=True,
        **_MOMENT_KW,
    )
    diag = mob[mob.origin == mob.destination]
    mobility_diagonal = float(diag["probability"].mean())

    zs = moments.zero_spells(panel, period_step=PERIOD_STEP, **_MOMENT_KW)
    zs_by = {r.statistic: float(r.value) for r in zs.itertuples()}
    exit_rate = zs_by["exit_rate"]

    return {
        "autocorr_log_2yr": ac_by_lag[1],
        "autocorr_log_4yr": ac_by_lag[2],
        "autocorr_log_10yr": ac_by_lag[5],
        "mobility_diagonal": mobility_diagonal,
        "zero_persistence": 1.0 - exit_rate,
        "entry_rate": zs_by["entry_rate"],
        "exit_rate": exit_rate,
        "mean_spell_length": zs_by["mean_spell_length"],
    }


# --------------------------------------------------------------------------
# Locked protocol (from gates.yaml)
# --------------------------------------------------------------------------
def load_gate1_thresholds() -> dict[str, Any]:
    """The gate-1 ``thresholds`` block, read from gates.yaml at runtime."""
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


def check_geometry(
    scores: dict[str, float], geometry: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Evaluate one view's locked geometry thresholds against scores.

    Threshold-name suffixes encode the comparison:

    * ``*_max``   -> ``score <= threshold``
    * ``*_min``   -> ``score >= threshold``
    * ``*_range`` -> ``lo <= score <= hi``

    The score key for each threshold strips the ``_max`` / ``_min`` /
    ``_range`` suffix (e.g. ``c2st_auc_max`` reads ``c2st_auc``, the
    tail ``q99_ratio_range`` reads ``q99_ratio.earnings_t1``).
    """
    results: dict[str, dict[str, Any]] = {}
    for tname, tvalue in geometry.items():
        if tname.endswith("_max"):
            metric = tname[: -len("_max")]
            score = _geometry_score(scores, metric)
            passed = score <= tvalue
            results[tname] = {
                "metric": metric,
                "score": score,
                "threshold": tvalue,
                "comparison": "<=",
                "pass": bool(passed),
            }
        elif tname.endswith("_min"):
            metric = tname[: -len("_min")]
            score = _geometry_score(scores, metric)
            passed = score >= tvalue
            results[tname] = {
                "metric": metric,
                "score": score,
                "threshold": tvalue,
                "comparison": ">=",
                "pass": bool(passed),
            }
        elif tname.endswith("_range"):
            metric = tname[: -len("_range")]
            score = _geometry_score(scores, metric)
            lo, hi = tvalue
            passed = (score >= lo) and (score <= hi)
            results[tname] = {
                "metric": metric,
                "score": score,
                "threshold": [lo, hi],
                "comparison": "in",
                "pass": bool(passed),
            }
        else:
            raise ValueError(
                f"Unrecognized threshold suffix in {tname!r}; expected "
                "_max / _min / _range."
            )
    return results


def _geometry_score(scores: dict[str, float], metric: str) -> float:
    """Map a locked geometry-metric name to the scorecard key.

    The scorecard emits ``prdc_coverage`` (not ``prdc_coverage``-free)
    and tail keys suffixed by the target dimension ``earnings_t1``.
    """
    direct = {
        "c2st_auc": "c2st_auc",
        "energy_distance": "energy_distance",
        "prdc_coverage": "prdc_coverage",
        "w1_over_sd": "w1_over_sd.earnings_t1",
        "q90_ratio": "q90_ratio.earnings_t1",
        "q99_ratio": "q99_ratio.earnings_t1",
    }
    key = direct.get(metric)
    if key is None:
        raise KeyError(
            f"No scorecard key mapped for geometry metric {metric!r}"
        )
    if key not in scores:
        raise KeyError(
            f"Scorecard is missing key {key!r} for metric {metric!r}; "
            f"have {sorted(scores)}."
        )
    return float(scores[key])


def check_battery(
    values: dict[str, float],
    reference: dict[str, float],
    tolerances: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Evaluate the locked battery tolerances against candidate values.

    Each ``*_tolerance`` names a battery statistic; the candidate value
    passes iff ``|value - reference| <= tolerance``. References come
    from the committed ``battery_reference`` block (aliased keys where
    the report name differs from the committed name).
    """
    results: dict[str, dict[str, Any]] = {}
    for tname, tol in tolerances.items():
        stat = tname[: -len("_tolerance")]
        ref_key = _BATTERY_REF_ALIAS.get(stat, stat)
        if stat not in values:
            raise KeyError(
                f"Battery value missing for {stat!r}; have {sorted(values)}."
            )
        if ref_key not in reference:
            raise KeyError(
                f"battery_reference missing {ref_key!r}; "
                f"have {sorted(reference)}."
            )
        value = float(values[stat])
        ref = float(reference[ref_key])
        deviation = abs(value - ref)
        results[stat] = {
            "value": value,
            "reference": ref,
            "tolerance": float(tol),
            "deviation": deviation,
            "pass": bool(deviation <= tol),
        }
    return results


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def reproduce_battery_reference(panel: pd.DataFrame) -> dict[str, Any]:
    """Recompute battery_reference on the full filtered panel.

    Proves the battery code path matches the committed definitions
    before any candidate is scored. Every committed value must
    reproduce to float precision.
    """
    committed = json.loads((ROOT / BATTERY_REFERENCE_RUN).read_text())
    reference = committed["battery_reference"]
    recomputed = compute_battery(panel)
    checks: dict[str, Any] = {}
    all_exact = True
    for ref_name, ref_val in reference.items():
        # invert the alias to find our local key
        local = next(
            (k for k, v in _BATTERY_REF_ALIAS.items() if v == ref_name),
            ref_name,
        )
        got = float(recomputed[local])
        exact = got == float(ref_val)
        all_exact = all_exact and exact
        checks[ref_name] = {
            "recomputed": got,
            "committed": float(ref_val),
            "exact_float_match": bool(exact),
        }
    return {
        "run": committed["run"],
        "all_committed_values_reproduced_exactly": bool(all_exact),
        "checks": checks,
    }


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full pre-registered gate-1 baseline run."""
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
        seed_t = time.time()
        holdout, train = split_holdout_train(panel, seed)
        pairs = build_backward_pairs(train)
        fitted = fit_backward_model(pairs, seed)
        candidate = generate_candidate(fitted, holdout)

        # --- geometry: score candidate vs holdout on both locked views ---
        geometry_by_view: dict[str, Any] = {}
        geometry_seed_pass = True
        n_windows: dict[str, int] = {}
        for vname, view in view_specs.items():
            scores = hpanel.panel_scorecard(
                candidate, holdout, view, seed=seed
            )
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

        per_seed.append(
            {
                "seed": seed,
                "n_persons": int(holdout.person_id.nunique()),
                "n_person_periods": int(len(holdout)),
                "n_train_persons": int(train.person_id.nunique()),
                "n_train_pairs": int(len(pairs)),
                "n_windows": n_windows,
                "regimes": fitted.regimes(),
                "geometry": geometry_by_view,
                "geometry_pass": bool(geometry_seed_pass),
                "battery_values": battery_values,
                "battery_checks": battery_checks,
                "battery_pass": bool(battery_seed_pass),
            }
        )
        if verbose:
            print(
                f"seed {seed}: geometry_pass={geometry_seed_pass} "
                f"battery_pass={battery_seed_pass} "
                f"({time.time()-seed_t:.0f}s)"
            )

    n_geo = sum(1 for s in per_seed if s["geometry_pass"])
    n_bat = sum(1 for s in per_seed if s["battery_pass"])
    geometry_gate_pass = n_geo >= 4
    battery_gate_pass = n_bat >= 4
    gate_pass = geometry_gate_pass and battery_gate_pass

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": "gate1_qrf_baseline_v1",
        "gate": "gate_1",
        "description": (
            "Gate-1 baseline: populace-fit regime-gated chained weighted "
            "QRF, one-step backward biennial backcast. Candidate scored "
            "against held-out PSID family earnings panel geometry (two "
            "locked views) and the locked moment battery, per the locked "
            "seed-level conjunction in gates.yaml (pull request 39)."
        ),
        "model": {
            "class": "populace.fit.qrf.RegimeGatedQRF",
            "front_door": "plain DataFrame (explicit weight column)",
            "hyperparameters": {
                "n_estimators": DEFAULT_N_ESTIMATORS,
                "zero_atol": DEFAULT_ZERO_ATOL,
                "max_samples_leaf": None,
                "note": "populace-fit defaults; seed = gate seed s",
            },
            "transition": {
                "direction": "backward (one-step biennial)",
                "target": "earnings at period t-2",
                "predictors": ["earnings at period t", "age at period t-2"],
                "sample_weight": "earlier-period (t-2) person-period weight",
                "fit_pairs": "80% complement's adjacent 2-year pairs",
            },
            "generation": {
                "anchor": (
                    "chronologically last observed period held at real "
                    "earnings"
                ),
                "chain": (
                    "backward over observed periods; each earlier period "
                    "drawn conditional on the next observed period's "
                    "(generated/anchor) earnings and the earlier period's "
                    "age"
                ),
                "gap_rule": (
                    "gap (next observed period 4+ years later) applies the "
                    "one-step model once across the gap"
                ),
                "row_order": (
                    "batched by step-from-anchor; within a step, ordered "
                    "by person_id (deterministic)"
                ),
                "candidate_panel_pin": (
                    "exactly the holdout persons on exactly their observed "
                    "periods; only earnings generated; anchor keeps real "
                    "value; person_id/period/age/weight copied from holdout"
                ),
            },
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
                "is the holdout, the complement is the training set"
            ),
            "seeds": list(SEEDS),
            "views": {
                "psid_family_earnings_pairs": {"window": 2, "period_step": 2},
                "psid_family_earnings_runs": {"window": 3, "period_step": 2},
            },
            "scoring": (
                "panel_scorecard(candidate, holdout, view, seed=s) per "
                "locked view; battery on the candidate panel vs committed "
                "battery_reference"
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
