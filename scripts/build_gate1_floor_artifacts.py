"""Build the two gate-1 floor artifacts the threshold lock cites.

Both floors come from the staged PSID family earnings panel under the
same filter the committed window-2 floor used (prime age 25-59,
reference years 1998-2022, positive weights):

* ``noise_floor_psid_family_runs_9822.json`` — half-vs-half floor for
  the window-3 runs view, the view registered to catch chained-model
  understatement of long-spell persistence.
* ``noise_floor_psid_family_ctx20_9822.json`` — the candidate-context
  floor: two disjoint 20%-of-persons real samples scored against each
  other, matching the scale at which the gate protocol scores a
  candidate against a 20% holdout. C2ST and the tail block run
  uncapped, so their sampling noise at this scale exceeds the
  full-panel half-vs-half floor's; thresholds derive from this
  artifact, not from the full-panel floor.

Run from the repository root with the PSID family files staged:

    .venv/bin/python scripts/build_gate1_floor_artifacts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from populace_dynamics.data.family import family_earnings_panel
from populace_dynamics.harness import panel as hpanel

RUNS = Path("runs")
SEEDS = (0, 1, 2, 3, 4)
FILTER = "age 25-59, reference years 1998-2022, positive weights"


def _view(window: int) -> hpanel.PanelView:
    return hpanel.PanelView(
        name=f"psid_family_earnings_w{window}",
        id_column="person_id",
        period_column="period",
        value_columns=("earnings",),
        covariate_columns=("age",),
        weight_column="weight",
        window=window,
        period_step=2,
    )


def _stats(rows: list[dict[str, float]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for key in rows[0]:
        values = np.array([row[key] for row in rows], dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "values": [float(v) for v in values],
        }
    return out


def main() -> None:
    raw = family_earnings_panel()
    prime = raw[
        (raw.age >= 25)
        & (raw.age <= 59)
        & (raw.period >= 1998)
        & (raw.period <= 2022)
        & (raw.weight > 0)
    ].copy()
    n_persons = int(prime.person_id.nunique())

    view3 = _view(3)
    windows3, _ = hpanel.project_panel(prime, view3)
    rows3 = [hpanel.noise_floor(prime, view3, seed=s) for s in SEEDS]
    artifact3 = {
        "run": "noise_floor_psid_family_runs_9822",
        "data": (
            "PSID family-file head/spouse labor income attached to "
            f"persons; {FILTER}; window-3 runs view (two consecutive "
            "biennial transitions per window)"
        ),
        "method": "half_vs_half person-disjoint noise floor, seeds 0-4",
        "n_person_periods": int(len(prime)),
        "n_persons": n_persons,
        "n_windows": int(len(windows3)),
        "view": {
            "window": 3,
            "period_step": 2,
            "value_columns": ["earnings"],
            "covariates": ["age"],
        },
        "noise_floor_seeds_0_4": _stats(rows3),
    }
    path3 = RUNS / "noise_floor_psid_family_runs_9822.json"
    path3.write_text(json.dumps(artifact3, indent=1) + "\n")
    print(f"wrote {path3}: n_windows={artifact3['n_windows']}")

    ctx_specs = (
        (2, "ctx20", "window-2 pairs view"),
        (3, "runs_ctx20", "window-3 runs view"),
    )
    ctx_artifacts = {}
    for window, tag, view_label in ctx_specs:
        view = _view(window)
        rows_ctx = []
        n_side = []
        for s in SEEDS:
            forty, _ = hpanel.split_panel_by_person(
                prime, "person_id", fraction=0.4, seed=1000 + s
            )
            a, b = hpanel.split_panel_by_person(
                forty, "person_id", fraction=0.5, seed=s
            )
            rows_ctx.append(hpanel.panel_scorecard(a, b, view, seed=s))
            n_side.append(int(a.person_id.nunique()))
        artifact_ctx = {
            "run": f"noise_floor_psid_family_{tag}_9822",
            "data": (
                "PSID family-file head/spouse labor income attached to "
                f"persons; {FILTER}; {view_label}"
            ),
            "method": (
                "candidate-context floor: per seed s, draw 40% of "
                "persons (split_panel_by_person, fraction=0.4, "
                "seed=1000+s), halve it person-disjointly "
                "(fraction=0.5, seed=s), and score one half against "
                "the other (panel_scorecard, seed=s) — real vs real "
                "at the ~20%-of-persons scale the gate protocol "
                "scores candidates at"
            ),
            "n_person_periods": int(len(prime)),
            "n_persons": n_persons,
            "n_persons_per_side_mean": float(np.mean(n_side)),
            "view": {
                "window": window,
                "period_step": 2,
                "value_columns": ["earnings"],
                "covariates": ["age"],
            },
            "noise_floor_seeds_0_4": _stats(rows_ctx),
        }
        path_ctx = RUNS / f"noise_floor_psid_family_{tag}_9822.json"
        path_ctx.write_text(json.dumps(artifact_ctx, indent=1) + "\n")
        ctx_artifacts[tag] = artifact_ctx
        print(
            f"wrote {path_ctx}: persons/side~"
            f"{artifact_ctx['n_persons_per_side_mean']:.0f}"
        )

    for name, artifact in (
        ("runs", artifact3),
        ("ctx20", ctx_artifacts["ctx20"]),
        ("runs_ctx20", ctx_artifacts["runs_ctx20"]),
    ):
        floor = artifact["noise_floor_seeds_0_4"]
        for key in ("c2st_auc", "prdc_coverage", "energy_distance"):
            print(
                f"  {name} {key}: {floor[key]['mean']:.4f} "
                f"+/- {floor[key]['sd']:.4f}"
            )


if __name__ == "__main__":
    main()
