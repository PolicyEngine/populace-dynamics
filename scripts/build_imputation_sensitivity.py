"""Reported (NOT gated) PSID income-assignment sensitivity of the battery.

Measures how far the LOCKED gate-1 battery references and the
deployment-scale geometry floors move when PSID income-assignment
(accuracy) observations are excluded from the held-out family panel.
Nothing here changes a gate, a threshold, or a committed floor/reference
artifact: it recomputes the same quantities the gate protocol locks and
reports the deltas, so a future referee round can see whether excluding
assigned observations would move the contested statistics -- above all
the 10-year log-earnings autocorrelation -- toward the candidate models.

The battery statistics and their definitions are the SAME code path the
locked ``battery_reference`` uses (``populace_dynamics.harness.moments``,
mirroring ``scripts/run_gate1_baseline.py``'s ``compute_battery``); this
script deliberately does NOT import ``populace.fit`` (which pins an older
scikit-learn), so it runs in the plain analysis environment. As every
gate runner does, it FIRST reproduces the committed ``battery_reference``
to float precision on the full filtered panel and hard-stops if that
fails, so a divergent battery definition can never silently reshape the
reported deltas.

Panels compared, all under the locked filter (age 25-59, reference years
1998-2022, positive weights):

* ``full_panel_recomputed`` -- the whole filtered panel (must reproduce
  the committed references exactly);
* ``excl_any_flag`` -- excluding person-periods with ``earnings_acc > 0``
  (any assigned/edited labor income);
* ``excl_code_1`` -- excluding only ``earnings_acc == 1`` (the single
  major-assignment code), a meaningful distinction because the accuracy
  codes span 0/1/2/3/4/5 in these waves.

It also recomputes the two deployment-scale geometry floors the gate-1
lock derives from (the ctx20 window-2 pairs and window-3 runs floors,
``split_panel_by_person`` fraction 0.4 seed 1000+s then fraction 0.5
seed s, seeds 0-4) on the ``excl_any_flag`` panel, so the artifact shows
whether the geometry targets move too.

Run from the repository root with the PSID family files staged:

    .venv/bin/python scripts/build_imputation_sensitivity.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from populace_dynamics.data.family import family_earnings_panel
from populace_dynamics.harness import moments
from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "runs" / "imputation_sensitivity_v1.json"
BATTERY_REFERENCE_RUN = "runs/noise_floor_psid_family_9822.json"
CTX20_FLOOR_RUN = "runs/noise_floor_psid_family_ctx20_9822.json"
RUNS_CTX20_FLOOR_RUN = "runs/noise_floor_psid_family_runs_ctx20_9822.json"

#: Committed candidate run artifacts whose per-seed battery values give
#: the contested-statistic context. ``latent_perm_v2`` (candidate 3)
#: lives on the candidate-3 branch, not on master; when it is absent the
#: artifact records the values measured from that branch as documented
#: constants rather than re-reading a file that is not in this tree.
CANDIDATE_RUNS = {
    "baseline": "runs/gate1_qrf_baseline_v1.json",
    "candidate2_latent_perm_v1": "runs/gate1_qrf_latent_perm_v1.json",
    "candidate3_latent_perm_v2": "runs/gate1_qrf_latent_perm_v2.json",
}

#: Candidate-3 per-seed 10yr autocorr, read from
#: runs/gate1_qrf_latent_perm_v2.json on branch gate1-candidate3-pa-decomp
#: (commit a6021de) on 2026-07-05; used only if that file is absent here.
_CANDIDATE3_AUTOCORR_10YR_FALLBACK = [
    0.6323197072287733,
    0.6336811468334078,
    0.6581682829049901,
    0.6433569252098327,
    0.666373460961008,
]

#: The locked filter constants (identical to the gate runners and the
#: floor builder).
AGE_MIN, AGE_MAX = 25, 59
PERIOD_MIN, PERIOD_MAX = 1998, 2022
PERIOD_STEP = 2
SEEDS = (0, 1, 2, 3, 4)

_MOMENT_KW = dict(
    id_col="person_id",
    period_col="period",
    value_col="earnings",
    weight_col="weight",
)

#: The three contested statistics whose candidate context we quote.
_CONTESTED = ("autocorr_log_2yr", "autocorr_log_4yr", "autocorr_log_10yr")


def load_filtered_panel():
    """The locked filter, applied before anything else."""
    raw = family_earnings_panel()
    prime = raw[
        (raw.age >= AGE_MIN)
        & (raw.age <= AGE_MAX)
        & (raw.period >= PERIOD_MIN)
        & (raw.period <= PERIOD_MAX)
        & (raw.weight > 0)
    ].copy()
    return prime.reset_index(drop=True)


def compute_battery(panel) -> dict[str, float]:
    """The locked gate-1 battery on one panel (moments code path).

    Byte-for-byte the definitions ``scripts/run_gate1_baseline.py``
    uses, keyed by the committed ``battery_reference`` names.
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
    mobility_diagonal_mean = float(diag["probability"].mean())
    zs = moments.zero_spells(panel, period_step=PERIOD_STEP, **_MOMENT_KW)
    zs_by = {r.statistic: float(r.value) for r in zs.itertuples()}
    exit_rate = zs_by["exit_rate"]
    return {
        "autocorr_log_2yr": ac_by_lag[1],
        "autocorr_log_4yr": ac_by_lag[2],
        "autocorr_log_10yr": ac_by_lag[5],
        "mobility_diagonal_mean": mobility_diagonal_mean,
        "zero_persistence": 1.0 - exit_rate,
        "entry_rate": zs_by["entry_rate"],
        "exit_rate": exit_rate,
        "mean_spell_length": zs_by["mean_spell_length"],
    }


def reproduce_reference(full_battery: dict[str, float]) -> dict[str, Any]:
    """Hard-stop precheck: full-panel battery == committed reference.

    The pattern every gate runner uses. Refuses to proceed on any
    non-exact float match, so the reported deltas can never rest on a
    battery definition that drifted from the lock.
    """
    committed = json.loads((ROOT / BATTERY_REFERENCE_RUN).read_text())[
        "battery_reference"
    ]
    checks = {}
    all_exact = True
    for name, ref in committed.items():
        got = float(full_battery[name])
        exact = got == float(ref)
        all_exact = all_exact and exact
        checks[name] = {
            "committed": float(ref),
            "recomputed": got,
            "exact_float_match": bool(exact),
        }
    if not all_exact:
        raise RuntimeError(
            "Full-panel battery does not reproduce the committed "
            "battery_reference to float precision; refusing to report "
            "deltas against a divergent definition."
        )
    return checks


def _floor_view(window: int) -> hpanel.PanelView:
    return hpanel.PanelView(
        name=f"psid_family_earnings_w{window}",
        id_column="person_id",
        period_column="period",
        value_columns=("earnings",),
        covariate_columns=("age",),
        weight_column="weight",
        window=window,
        period_step=PERIOD_STEP,
    )


def _seed_stats(rows: list[dict[str, float]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key in rows[0]:
        values = np.array([row[key] for row in rows], dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "values": [float(v) for v in values],
        }
    return out


def ctx20_floor(panel, window: int) -> dict[str, dict[str, Any]]:
    """The committed ctx20 derivation, recomputed on ``panel``.

    Per seed s: draw 40% of persons (fraction=0.4, seed=1000+s), halve
    it person-disjointly (fraction=0.5, seed=s), score one half against
    the other (seed=s). Identical construction to
    ``scripts/build_gate1_floor_artifacts.py``.
    """
    view = _floor_view(window)
    rows = []
    n_side = []
    for s in SEEDS:
        forty, _ = hpanel.split_panel_by_person(
            panel, "person_id", fraction=0.4, seed=1000 + s
        )
        a, b = hpanel.split_panel_by_person(
            forty, "person_id", fraction=0.5, seed=s
        )
        rows.append(hpanel.panel_scorecard(a, b, view, seed=s))
        n_side.append(int(a.person_id.nunique()))
    stats = _seed_stats(rows)
    stats["_n_persons_per_side_mean"] = float(np.mean(n_side))
    return stats


def flag_shares(panel) -> dict[str, Any]:
    """Assigned-observation shares, weighted and unweighted.

    Overall (among positive earnings) and per wave (by reference year),
    for ``earnings_acc > 0`` (any assignment) and ``earnings_acc == 1``.
    """
    pos = panel[panel.earnings > 0]

    def _share(frame, mask):
        if len(frame) == 0:
            return {"unweighted": None, "weighted": None, "n": 0}
        w = frame["weight"]
        return {
            "unweighted": float(mask.mean()),
            "weighted": float(w[mask].sum() / w.sum()),
            "n": int(mask.sum()),
        }

    overall = {
        "n_positive": int(len(pos)),
        "any_flag": _share(pos, pos.earnings_acc > 0),
        "code_1": _share(pos, pos.earnings_acc == 1),
    }
    by_wave = {}
    for period, grp in pos.groupby("period"):
        by_wave[int(period)] = {
            "n_positive": int(len(grp)),
            "any_flag": _share(grp, grp.earnings_acc > 0),
            "code_1": _share(grp, grp.earnings_acc == 1),
        }
    # code distribution among positive earnings (why excl_code_1 differs)
    code_counts = (
        pos.earnings_acc.value_counts().sort_index().astype(int).to_dict()
    )
    return {
        "overall_among_positive": overall,
        "by_wave_among_positive": by_wave,
        "code_distribution_among_positive": {
            int(k): int(v) for k, v in code_counts.items()
        },
    }


def candidate_context() -> dict[str, Any]:
    """Per-seed contested-statistic values for baseline and candidates.

    Reads the committed candidate run artifacts' per-seed
    ``battery_values``; records the min/max range on each contested
    statistic. Candidate 3 (latent_perm_v2) is read if present, else
    the documented fallback (measured from the candidate-3 branch).
    """
    out: dict[str, Any] = {}
    for name, rel in CANDIDATE_RUNS.items():
        path = ROOT / rel
        if path.is_file():
            artifact = json.loads(path.read_text())
            per_seed = artifact["per_seed"]
            stat_series = {
                stat: [float(s["battery_values"][stat]) for s in per_seed]
                for stat in _CONTESTED
            }
            out[name] = {
                "source": rel,
                "n_seeds": len(per_seed),
                "per_stat": {
                    stat: {
                        "values": vals,
                        "min": min(vals),
                        "max": max(vals),
                    }
                    for stat, vals in stat_series.items()
                },
            }
        elif name == "candidate3_latent_perm_v2":
            vals = _CANDIDATE3_AUTOCORR_10YR_FALLBACK
            out[name] = {
                "source": (
                    f"{rel} (absent on this branch; autocorr_log_10yr "
                    "read from branch gate1-candidate3-pa-decomp commit "
                    "a6021de on 2026-07-05)"
                ),
                "n_seeds": len(vals),
                "per_stat": {
                    "autocorr_log_10yr": {
                        "values": vals,
                        "min": min(vals),
                        "max": max(vals),
                    }
                },
            }
        else:
            out[name] = {"source": rel, "error": "artifact not found"}
    return out


def build() -> dict[str, Any]:
    started = time.time()
    prime = load_filtered_panel()
    n_full = int(len(prime))

    full = compute_battery(prime)
    repro = reproduce_reference(full)

    excl_any = prime[prime.earnings_acc == 0]
    excl_c1 = prime[prime.earnings_acc != 1]
    b_any = compute_battery(excl_any)
    b_c1 = compute_battery(excl_c1)

    committed = json.loads((ROOT / BATTERY_REFERENCE_RUN).read_text())[
        "battery_reference"
    ]
    battery = {}
    for name, ref in committed.items():
        battery[name] = {
            "committed_reference": float(ref),
            "full_panel_recomputed": float(full[name]),
            "excl_any_flag": float(b_any[name]),
            "excl_code_1": float(b_c1[name]),
            "deltas": {
                "excl_any_flag_minus_committed": float(b_any[name] - ref),
                "excl_code_1_minus_committed": float(b_c1[name] - ref),
                "excl_any_flag_minus_full": float(b_any[name] - full[name]),
                "excl_code_1_minus_full": float(b_c1[name] - full[name]),
            },
        }

    # Geometry floors recomputed on the excl-any-flag panel.
    ctx20_excl = ctx20_floor(excl_any, window=2)
    runs_ctx20_excl = ctx20_floor(excl_any, window=3)
    ctx20_committed = json.loads((ROOT / CTX20_FLOOR_RUN).read_text())
    runs_ctx20_committed = json.loads(
        (ROOT / RUNS_CTX20_FLOOR_RUN).read_text()
    )

    def _floor_block(committed_artifact, excl_stats):
        keys = ("c2st_auc", "prdc_coverage", "energy_distance")
        committed_floor = committed_artifact["noise_floor_seeds_0_4"]
        block = {
            "n_persons_per_side_mean_excl": excl_stats.pop(
                "_n_persons_per_side_mean"
            )
        }
        for key in keys:
            cm = committed_floor[key]["mean"]
            em = excl_stats[key]["mean"]
            block[key] = {
                "committed_mean": float(cm),
                "committed_sd": float(committed_floor[key]["sd"]),
                "excl_any_flag_mean": float(em),
                "excl_any_flag_sd": float(excl_stats[key]["sd"]),
                "excl_any_flag_values": excl_stats[key]["values"],
                "delta_mean": float(em - cm),
            }
        return block

    geometry_floors = {
        "note": (
            "committed floors are the full-panel ctx20 derivations; "
            "excl_any_flag recomputes the identical construction on the "
            "panel with earnings_acc>0 person-periods removed"
        ),
        "ctx20_pairs_window2": _floor_block(ctx20_committed, ctx20_excl),
        "ctx20_runs_window3": _floor_block(
            runs_ctx20_committed, runs_ctx20_excl
        ),
    }

    artifact = {
        "run": "imputation_sensitivity_v1",
        "reported_not_gated": True,
        "statement": (
            "Reported, not gated. This artifact changes NO gate, NO "
            "threshold, and NO committed floor or reference artifact. It "
            "recomputes the locked gate-1 battery references and the "
            "deployment-scale geometry floors on the held-out PSID family "
            "panel with PSID income-assignment (accuracy) observations "
            "excluded, and reports the movement. Any future amendment to "
            "a gate on the strength of these numbers requires a fresh "
            "referee round."
        ),
        "headline": (
            "Does excluding assigned observations move the 10-year "
            "log-earnings autocorrelation reference toward the "
            "candidates? Committed/full reference "
            f"{full['autocorr_log_10yr']:.4f}; excluding any assigned "
            f"observation raises it to {b_any['autocorr_log_10yr']:.4f} "
            f"(+{b_any['autocorr_log_10yr']-full['autocorr_log_10yr']:.4f}"
            "), excluding only code-1 to "
            f"{b_c1['autocorr_log_10yr']:.4f} "
            f"(+{b_c1['autocorr_log_10yr']-full['autocorr_log_10yr']:.4f}"
            "). Candidate 2 lands at 0.615-0.668 and candidate 3 at "
            "0.632-0.666, so the exclusion moves the reference toward "
            "the candidates by only a small fraction of the gap."
        ),
        "filter": (
            f"age {AGE_MIN}-{AGE_MAX}, reference years {PERIOD_MIN}-"
            f"{PERIOD_MAX}, positive weights (applied before anything)"
        ),
        "battery_code_path": (
            "populace_dynamics.harness.moments, identical definitions to "
            "scripts/run_gate1_baseline.py compute_battery; "
            "populace.fit is NOT imported"
        ),
        "battery_reference_run": BATTERY_REFERENCE_RUN,
        "battery_reference_reproduction": {
            "all_committed_values_reproduced_exactly": True,
            "checks": repro,
        },
        "n_person_periods": {
            "full_filtered": n_full,
            "excl_any_flag": int(len(excl_any)),
            "excl_code_1": int(len(excl_c1)),
            "dropped_excl_any_flag": n_full - int(len(excl_any)),
            "dropped_excl_code_1": n_full - int(len(excl_c1)),
        },
        "battery": battery,
        "flag_shares": flag_shares(prime),
        "geometry_floors_excl_any_flag": geometry_floors,
        "candidate_context": {
            "note": (
                "per-seed contested-statistic values from the committed "
                "candidate runs; shows where candidates landed on the "
                "statistics the exclusion perturbs, so the artifact can "
                "be read against whether the reference moves toward them"
            ),
            "runs": candidate_context(),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return artifact


def main() -> None:
    artifact = build()
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=1) + "\n")
    h = artifact["battery"]["autocorr_log_10yr"]
    print(f"wrote {ARTIFACT_PATH}")
    print(
        "autocorr_log_10yr: committed/full "
        f"{h['full_panel_recomputed']:.4f} -> excl_any "
        f"{h['excl_any_flag']:.4f} (delta "
        f"{h['deltas']['excl_any_flag_minus_full']:+.4f}), excl_code_1 "
        f"{h['excl_code_1']:.4f} (delta "
        f"{h['deltas']['excl_code_1_minus_full']:+.4f})"
    )


if __name__ == "__main__":
    main()
