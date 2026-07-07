"""Inner-validation harness: mirror the amended gate on INNER splits.

REPORTED, NOT GATED. This module reads no gate verdict and writes no gate
verdict. It never touches the OUTER holdout: for each outer seed ``s`` it
takes the seed's TRAIN complement -- the 80% the locked protocol leaves for
fitting -- and carves a person-disjoint INNER pair out of it, so model
development can iterate on inner evidence while the locked one-shot rule
still protects the outer holdout for the single registered outer run.

The locked gate's protocol (``gates.yaml`` gate_1, amended 2026-07-06) draws
the outer holdout as ``split_panel_by_person(panel, "person_id",
fraction=0.2, seed=s)`` (the drawn 20% is the outer holdout; the 80%
complement is the train set). This harness splits that TRAIN complement
again::

    inner_holdout, inner_train = split_panel_by_person(
        train, "person_id", fraction=0.25, seed=1000 + s
    )

``split_panel_by_person`` returns ``(left, right)`` with ``left`` the drawn
fraction, so ``inner_holdout`` is the drawn 25% of TRAIN persons and
``inner_train`` the 75% complement (the same convention the outer protocol
uses: the drawn side is the holdout). Both are person-disjoint and, being
subsets of the outer TRAIN complement, are disjoint from the outer holdout
by construction (:func:`inner_split` asserts all three).

Scoring mirrors the amended gate EXACTLY -- the same
:func:`run_gate1_baseline.panel_scorecard` on both locked views (the pairs
view's full geometry including its C2ST; the runs view's coverage, with its
C2ST demoted to reported per the amendment), the same
:func:`run_gate1_baseline.compute_battery` against the committed
``battery_reference`` under the locked tolerances, and the same
benefit-space block (the pinned PIA-proxy functional of
:mod:`build_downstream_relevance`, its per-seed metrics and the pooled Q0
gate) -- but on the INNER pair instead of the outer one. The thresholds are
read from ``gates.yaml`` at runtime; nothing is hardcoded.

Scale caveat, stated honestly and carried into every artifact and doc. The
inner holdout is ~25% of an ~13k-18k-person TRAIN complement (~3.3k-4.5k
persons, ~25k person-periods), and the inner donor pools / marginals are
built from the ~13k-person inner-train rather than the outer ~18k-person
train. Smaller samples on both sides run the geometry (C2ST, KS) and the
tail block slightly HOTTER than the outer ~20%-holdout scale the locked
thresholds were calibrated at. The harness is therefore a tool for RANKING
variants and CHECKING MARGINS against the amended thresholds, NOT for exact
outer prediction: a variant that clears a threshold comfortably on >=4/5
inner seeds is a strong outer candidate; a variant that clips it is a weak
one; a hair's-breadth inner pass or fail should be read as a margin, not a
verdict.

The benefit-space anchor. The outer gate weights each person by their
anchor-period weight and slices Q0 (zero anchor earnings) using
positive-anchor quartile edges fixed ONCE on the full filtered panel
(seed-stable). At inner scale the analogue is the anchor over the outer
TRAIN complement's persons (the population the inner pair is drawn from):
:func:`inner_anchor_cutpoints` computes the positive-anchor quartile edges
on that train anchor, and the inner benefit-space block uses the train
anchor for both weights and the Q0 slice. This keeps the inner Q0 slice a
fixed property of the inner population (not of the inner seed's draw),
exactly as the outer block keeps it a fixed property of the full panel.

Everything the harness scores is IMPORTED byte-for-byte from the merged
runners so the inner numbers are on the same measurement footing as the
outer ones: the load / split / view / battery / geometry-check /
battery-check / threshold-load from :mod:`run_gate1_baseline`; the anchor
machinery from :mod:`run_gate1_candidate5b`; the benefit-space functional
from :mod:`build_downstream_relevance`. The per-seed and pooled Q0 gate
logic is imported from the candidate-9 runner (:mod:`run_gate1_candidate9`)
so the inner verdict uses the identical amended-gate arithmetic.

Environment. Scoring the benefit-space block needs the SSA oracle
(``POPULACE_DYNAMICS_PE_US_DIR`` -> the pinned policyengine-us checkout);
the geometry and battery blocks are pure numpy/scipy. Candidate GENERATION
(the sweep in :mod:`run_inner_sweep`) needs populace-fit for the
participation gates. Run from the repository root with the PSID family
files staged, using the dedicated gate venv::

    export POPULACE_DYNAMICS_PE_US_DIR=$HOME/PolicyEngine/policyengine-us-main
    .venv-gate/bin/python scripts/run_inner_sweep.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The scoring machinery is IMPORTED from the merged baseline runner so the
# inner scorecard is byte-for-byte the outer one: the same view construction,
# the same battery definitions, the same geometry / battery checks, and the
# same threshold loading. Only the SPLIT differs (inner, not outer).
from run_gate1_baseline import (
    BATTERY_REFERENCE_RUN,
    SEEDS,
    build_panel_view,
    check_battery,
    check_geometry,
    compute_battery,
    load_gate1_thresholds,
    split_holdout_train,
)

# The amended-gate per-seed and pooled Q0 arithmetic is candidate 9's, so the
# inner verdict uses the identical amended-gate logic (only the pair differs).
from run_gate1_candidate9 import (
    check_benefit_space_per_seed,
    check_pooled_q0,
)

from populace_dynamics.harness import panel as hpanel

ROOT = Path(__file__).resolve().parents[1]

#: The INNER holdout fraction of each outer TRAIN complement. The drawn 25%
#: of TRAIN persons is the inner holdout; the 75% complement is inner-train
#: (the drawn side is the holdout, matching the outer protocol's convention).
INNER_HOLDOUT_FRACTION = 0.25
#: The inner split seed offset: inner seed = 1000 + outer seed, so the inner
#: draw is reproducible and independent of the outer draw.
INNER_SEED_OFFSET = 1000


def inner_seed(outer_seed: int) -> int:
    """The inner split seed for one outer seed (``1000 + outer_seed``)."""
    return INNER_SEED_OFFSET + int(outer_seed)


def inner_split(
    panel: pd.DataFrame, outer_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carve the inner pair from one outer seed's TRAIN complement.

    Applies the locked outer split first (the drawn 20% is the untouched
    outer holdout; the 80% complement is TRAIN), then splits TRAIN again
    person-disjointly at ``fraction=0.25, seed=1000 + outer_seed``. Returns
    ``(inner_holdout, inner_train, outer_holdout)``; the outer holdout is
    returned ONLY so the caller can assert non-contact -- it is never scored.

    Guarantees (all asserted): the inner pair is person-disjoint, covers
    TRAIN exactly, and neither side shares a person with the outer holdout.
    """
    outer_holdout, train = split_holdout_train(panel, outer_seed)
    inner_holdout, inner_train = hpanel.split_panel_by_person(
        train,
        "person_id",
        fraction=INNER_HOLDOUT_FRACTION,
        seed=inner_seed(outer_seed),
    )
    ih = set(inner_holdout["person_id"].to_numpy().tolist())
    it = set(inner_train["person_id"].to_numpy().tolist())
    tr = set(train["person_id"].to_numpy().tolist())
    oh = set(outer_holdout["person_id"].to_numpy().tolist())
    assert ih.isdisjoint(it), "inner holdout/train share a person"
    assert ih | it == tr, "inner split does not cover the outer train exactly"
    assert ih.isdisjoint(oh), "inner holdout touches the OUTER holdout"
    assert it.isdisjoint(oh), "inner train touches the OUTER holdout"
    return inner_holdout, inner_train, outer_holdout


def inner_anchor_cutpoints(train_anchor: pd.DataFrame) -> np.ndarray:
    """Positive-anchor quartile edges on the outer TRAIN complement's anchor.

    The inner analogue of the outer block's full-panel edges
    (:func:`build_downstream_relevance.anchor_quintile_cutpoints`): the
    weighted 25/50/75th percentiles of POSITIVE anchor earnings over the
    outer TRAIN complement's persons -- the population the inner pair is
    drawn from -- so the inner Q0 slice is a fixed property of the inner
    population, not of the inner seed's draw. Byte-for-byte the outer
    functional, evaluated on the train anchor instead of the full-panel
    anchor.
    """
    import build_downstream_relevance as ds

    return ds.anchor_quintile_cutpoints(train_anchor)


def load_amended_gate_config() -> dict[str, Any]:
    """The amended gate-1 blocks the inner harness mirrors, from gates.yaml.

    Returns the locked ``views`` config (per-view geometry thresholds and the
    runs-view ``reported_not_gated`` demotion), the battery tolerances, the
    committed ``battery_reference`` values, and the benefit-space metrics
    config -- everything the amended pass rule reads, loaded at runtime (no
    threshold hardcoded). Raises if the gate is not locked or the amendment's
    benefit-space block is absent (the amended gate cannot be mirrored
    without it).
    """
    import json

    thresholds = load_gate1_thresholds()
    if not thresholds.get("locked", False):
        raise RuntimeError(
            "gate_1 thresholds are not locked; the inner harness mirrors the "
            "locked amended gate."
        )
    benefit_cfg = thresholds.get("benefit_space")
    if benefit_cfg is None:
        raise RuntimeError(
            "gates.yaml gate_1 carries no benefit_space block; the inner "
            "harness mirrors the AMENDED gate (PR #57/#59), which requires "
            "it."
        )
    battery_tol = {
        k: v
        for k, v in thresholds["battery"].items()
        if k.endswith("_tolerance")
    }
    battery_reference = json.loads((ROOT / BATTERY_REFERENCE_RUN).read_text())[
        "battery_reference"
    ]
    return {
        "views_cfg": thresholds["views"],
        "battery_tol": battery_tol,
        "battery_reference": battery_reference,
        "benefit_metrics_cfg": benefit_cfg["metrics"],
    }


def build_inner_view_specs() -> dict[str, hpanel.PanelView]:
    """The two locked earnings views (pairs window-2, runs window-3)."""
    return {
        "psid_family_earnings_pairs": build_panel_view(
            "psid_family_earnings_pairs", window=2
        ),
        "psid_family_earnings_runs": build_panel_view(
            "psid_family_earnings_runs", window=3
        ),
    }


def score_geometry(
    candidate: pd.DataFrame,
    inner_holdout: pd.DataFrame,
    view_specs: dict[str, hpanel.PanelView],
    views_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Score both locked views' geometry on the inner pair (amended gate).

    Runs :func:`run_gate1_baseline.panel_scorecard` on each locked view
    (candidate vs inner holdout) and checks the locked geometry thresholds.
    The runs view's C2ST is demoted to reported-not-gated by the amendment,
    so ``check_geometry`` -- which reads only the thresholds present in each
    view's ``geometry`` block -- never gates it (the demoted metric is not in
    that block). Returns per-view scores, checks, and pass, plus the
    all-views geometry-thresholds pass.
    """
    geometry_by_view: dict[str, Any] = {}
    geometry_thresholds_pass = True
    n_windows: dict[str, int] = {}
    for vname, view in view_specs.items():
        scores = hpanel.panel_scorecard(
            candidate, inner_holdout, view, seed=seed
        )
        checks = check_geometry(scores, views_cfg[vname]["geometry"])
        view_pass = all(c["pass"] for c in checks.values())
        geometry_thresholds_pass = geometry_thresholds_pass and view_pass
        cand_windows, _ = hpanel.project_panel(candidate, view)
        n_windows[vname] = int(len(cand_windows))
        geometry_by_view[vname] = {
            "scores": {k: float(v) for k, v in scores.items()},
            "thresholds": views_cfg[vname]["geometry"],
            "checks": checks,
            "view_pass": bool(view_pass),
        }
    return {
        "by_view": geometry_by_view,
        "geometry_thresholds_pass": bool(geometry_thresholds_pass),
        "n_windows": n_windows,
    }


def score_battery(
    candidate: pd.DataFrame,
    battery_reference: dict[str, float],
    battery_tol: dict[str, float],
) -> dict[str, Any]:
    """Score the locked battery on the inner candidate panel (amended gate).

    Byte-for-byte the outer battery: :func:`run_gate1_baseline.compute_battery`
    on the candidate panel, checked against the committed ``battery_reference``
    under the locked tolerances. The reference is the SAME committed
    full-panel half-vs-half value the outer gate uses -- the inner harness
    does not re-derive a reference (that would move the target); it asks
    whether the inner candidate lands in the same locked bands.
    """
    values = compute_battery(candidate)
    checks = check_battery(values, battery_reference, battery_tol)
    seed_pass = all(c["pass"] for c in checks.values())
    return {
        "values": values,
        "checks": checks,
        "battery_pass": bool(seed_pass),
    }


def measure_benefit_space_inner(
    seed: int,
    inner_holdout: pd.DataFrame,
    candidate: pd.DataFrame,
    train_anchor: pd.DataFrame,
    params: Any,
    cutpoints: np.ndarray,
) -> dict[str, Any]:
    """PIA-proxy gap block on the inner pair (pinned PR-56 functional).

    Byte-for-byte :func:`run_gate1_candidate9.measure_benefit_space`, but with
    the inner anchor. Both the real inner-holdout histories and the candidate
    histories are pushed through the pinned
    :func:`build_downstream_relevance.panel_pia_proxy`, aligned on
    ``person_id`` (same persons, same rows, only earnings differ), and the
    weighted distribution-gap, person-level, and by-anchor-quintile blocks are
    reported with Q0 called out. The anchor supplying per-person weights and
    the Q0 slice is the outer TRAIN complement's anchor (``train_anchor``),
    and ``cutpoints`` are its positive-anchor quartile edges -- the inner
    population's fixed edges.
    """
    import build_downstream_relevance as ds

    real_px = ds.panel_pia_proxy(inner_holdout, train_anchor, params)
    cand_px = ds.panel_pia_proxy(candidate, train_anchor, params)
    merged = real_px.merge(
        cand_px, on="person_id", suffixes=("_real", "_cand")
    )
    assert np.allclose(
        merged["weight_real"].to_numpy(), merged["weight_cand"].to_numpy()
    ), "anchor weights diverged between sides"
    merged = merged.rename(columns={"weight_real": "weight"}).drop(
        columns=["weight_cand"]
    )
    assert (
        len(merged) == len(real_px) == len(cand_px)
    ), "candidate and real person sets differ"

    real = merged["pia_proxy_real"].to_numpy(dtype=np.float64)
    cand = merged["pia_proxy_cand"].to_numpy(dtype=np.float64)
    w = merged["weight"].to_numpy(dtype=np.float64)
    quintile_merged = merged.rename(
        columns={"pia_proxy_real": "pia_real", "pia_proxy_cand": "pia_cand"}
    )[["person_id", "pia_real", "pia_cand", "weight"]]

    return {
        "seed": seed,
        "n_persons": int(len(merged)),
        "n_real_zero_proxy": int(np.sum(real == 0.0)),
        "n_candidate_zero_proxy": int(np.sum(cand == 0.0)),
        "distribution": ds.distribution_gaps(cand, w, real, w),
        "person_level": ds.person_level_errors(cand, real, w),
        "by_anchor_quintile": ds.by_quintile(
            quintile_merged, train_anchor, cutpoints
        ),
    }


def score_inner_pair(
    seed: int,
    inner_holdout: pd.DataFrame,
    candidate: pd.DataFrame,
    train_anchor: pd.DataFrame,
    gate_cfg: dict[str, Any],
    view_specs: dict[str, hpanel.PanelView],
    benefit_params: Any,
    benefit_cutpoints: np.ndarray | None,
) -> dict[str, Any]:
    """Score one inner (candidate, holdout) pair as the amended gate would.

    Runs the three amended-gate blocks on the inner pair -- both views'
    geometry, the battery vs the committed reference, and the gated
    benefit-space block (its per-seed metrics folding into the seed's
    geometry verdict, exactly as the amended pass rule specifies) -- and
    returns a per-seed scorecard. The pooled Q0 gate is NOT decided here (it
    pools across inner seeds; :func:`inner_gate_verdict` decides it).

    When ``benefit_params`` is ``None`` the benefit-space block is skipped and
    the seed's geometry verdict is the locked geometry thresholds only,
    flagged as such -- but the amended gate is not fully mirrorable, so the
    sweep refuses to publish an inner verdict without the oracle.
    """
    geom = score_geometry(
        candidate,
        inner_holdout,
        view_specs,
        gate_cfg["views_cfg"],
        seed,
    )
    bat = score_battery(
        candidate,
        gate_cfg["battery_reference"],
        gate_cfg["battery_tol"],
    )

    benefit_space: dict[str, Any] | None = None
    benefit_checks: dict[str, Any] | None = None
    benefit_space_seed_pass: bool | None = None
    if benefit_params is not None and benefit_cutpoints is not None:
        benefit_space = measure_benefit_space_inner(
            seed,
            inner_holdout,
            candidate,
            train_anchor,
            benefit_params,
            benefit_cutpoints,
        )
        checked = check_benefit_space_per_seed(
            benefit_space, gate_cfg["benefit_metrics_cfg"]
        )
        benefit_checks = checked["checks"]
        benefit_space_seed_pass = checked["benefit_space_seed_pass"]

    if benefit_space_seed_pass is not None:
        geometry_seed_pass = bool(
            geom["geometry_thresholds_pass"] and benefit_space_seed_pass
        )
    else:
        geometry_seed_pass = bool(geom["geometry_thresholds_pass"])

    result: dict[str, Any] = {
        "seed": seed,
        "inner_seed": inner_seed(seed),
        "n_inner_holdout_persons": int(inner_holdout.person_id.nunique()),
        "n_inner_holdout_person_periods": int(len(inner_holdout)),
        "n_windows": geom["n_windows"],
        "geometry": geom["by_view"],
        "geometry_thresholds_pass": geom["geometry_thresholds_pass"],
        "geometry_pass": bool(geometry_seed_pass),
        "battery_values": bat["values"],
        "battery_checks": bat["checks"],
        "battery_pass": bat["battery_pass"],
    }
    if benefit_space is not None:
        result["benefit_space"] = benefit_space
        result["benefit_space_checks"] = benefit_checks
        result["benefit_space_seed_pass"] = bool(benefit_space_seed_pass)
    return result


def inner_gate_verdict(
    per_seed: list[dict[str, Any]],
    benefit_metrics_cfg: dict[str, Any],
) -> dict[str, Any]:
    """The amended-gate verdict on the inner seeds (reported, not gated).

    Byte-for-byte the amended pass rule (``gates.yaml``, ratified
    2026-07-06), applied to the INNER seed table: the gate passes iff >= 4/5
    seeds pass geometry (locked geometry thresholds AND per-seed
    benefit-space), >= 4/5 seeds pass battery, AND the pooled Q0 gate (abs
    pooled-mean Q0 % <= 5) holds. This is the RANKING verdict -- it says how a
    variant would fare under the amended gate at inner scale, which is
    evidence for the outer run, not a substitute for it.
    """
    n_geo = sum(1 for s in per_seed if s["geometry_pass"])
    n_bat = sum(1 for s in per_seed if s["battery_pass"])
    geometry_gate_pass = n_geo >= 4
    battery_gate_pass = n_bat >= 4
    pooled_q0 = check_pooled_q0(per_seed, benefit_metrics_cfg)
    pooled_q0_pass = pooled_q0["pooled_q0_pass"]
    gate_pass = bool(
        geometry_gate_pass and battery_gate_pass and pooled_q0_pass
    )
    return {
        "n_seeds": len(per_seed),
        "n_geometry_pass": n_geo,
        "n_battery_pass": n_bat,
        "geometry_gate_pass": bool(geometry_gate_pass),
        "battery_gate_pass": bool(battery_gate_pass),
        "pooled_q0_gate": pooled_q0,
        "pooled_q0_pass": bool(pooled_q0_pass),
        "pooled_q0_mean_pct_diff": pooled_q0["pooled_q0_mean_pct_diff"],
        "inner_gate_pass": gate_pass,
        "rule": (
            ">=4/5 inner seeds geometry (locked geometry thresholds AND "
            "per-seed benefit_space) AND >=4/5 inner seeds battery AND the "
            "pooled Q0 gate; REPORTED-NOT-GATED, inner scale"
        ),
    }


# --------------------------------------------------------------------------
# Margin reporting -- the distance from each amended threshold, per seed
# --------------------------------------------------------------------------
def geometry_margins(seed_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-gated-geometry-metric margin to its locked threshold, one seed.

    Margin sign convention: POSITIVE means inside the band (passing), with
    magnitude the distance to the nearest edge; NEGATIVE means over the edge
    (failing) by that magnitude. For ``*_max`` the margin is ``threshold -
    score``; for ``*_min`` it is ``score - threshold``; for a range it is the
    signed distance to the nearer edge (positive inside). Keyed
    ``<view>.<threshold_name>``.
    """
    out: dict[str, dict[str, Any]] = {}
    for vname, block in seed_result["geometry"].items():
        for tname, chk in block["checks"].items():
            comp = chk["comparison"]
            score = float(chk["score"])
            thr = chk["threshold"]
            if comp == "<=":
                margin = float(thr) - score
            elif comp == ">=":
                margin = score - float(thr)
            elif comp == "in":
                lo, hi = float(thr[0]), float(thr[1])
                margin = min(score - lo, hi - score)
            else:
                raise AssertionError(f"unknown comparison {comp!r}")
            out[f"{vname}.{tname}"] = {
                "score": score,
                "threshold": thr,
                "comparison": comp,
                "margin": margin,
                "pass": bool(chk["pass"]),
            }
    return out


def battery_margins(seed_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-battery-statistic margin to its locked tolerance, one seed.

    Margin = ``tolerance - |value - reference|`` (positive inside the band).
    """
    out: dict[str, dict[str, Any]] = {}
    for stat, chk in seed_result["battery_checks"].items():
        deviation = abs(float(chk["value"]) - float(chk["reference"]))
        out[stat] = {
            "value": float(chk["value"]),
            "reference": float(chk["reference"]),
            "tolerance": float(chk["tolerance"]),
            "deviation": deviation,
            "margin": float(chk["tolerance"]) - deviation,
            "pass": bool(chk["pass"]),
        }
    return out


def benefit_space_margins(
    seed_result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Per-gated-benefit-space-metric margin to its threshold, one seed.

    For the ``|.| <=`` metrics (mean %, median %, gated deciles) the margin is
    ``threshold - |value|``; for the KS gate it is ``threshold - value``
    (positive inside the band). An undefined value (real denominator zero) is
    a fail with ``margin`` ``None``.
    """
    out: dict[str, dict[str, Any]] = {}
    checks = seed_result.get("benefit_space_checks")
    if checks is None:
        return out
    for name, chk in checks.items():
        val = chk["value"]
        thr = float(chk["threshold"])
        if val is None:
            out[name] = {
                "value": None,
                "threshold": thr,
                "comparison": chk["comparison"],
                "margin": None,
                "pass": bool(chk["pass"]),
            }
            continue
        if chk["comparison"] == "|.| <=":
            margin = thr - abs(float(val))
        else:
            margin = thr - float(val)
        out[name] = {
            "value": float(val),
            "threshold": thr,
            "comparison": chk["comparison"],
            "margin": margin,
            "pass": bool(chk["pass"]),
        }
    return out


def summarize_margins(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Across-inner-seed margin summary for a variant (min/mean per metric).

    For each gated metric, the WORST (minimum) margin across the five inner
    seeds and the mean margin, plus how many seeds pass -- the compact
    evidence the ranking table reads. The minimum margin is the binding
    number: a variant with a comfortably positive worst-case margin on a
    metric clears it robustly at inner scale; a variant whose worst-case
    margin is barely positive (or negative) is fragile (or failing) there.
    """
    keys_seen: dict[str, list[dict[str, Any]]] = {}

    def _collect(block_fn, kind: str) -> None:
        for s in per_seed:
            for key, m in block_fn(s).items():
                keys_seen.setdefault(f"{kind}:{key}", []).append(m)

    _collect(geometry_margins, "geometry")
    _collect(battery_margins, "battery")
    _collect(benefit_space_margins, "benefit_space")

    summary: dict[str, Any] = {}
    for key, ms in keys_seen.items():
        margins = [m["margin"] for m in ms if m.get("margin") is not None]
        n_pass = sum(1 for m in ms if m["pass"])
        summary[key] = {
            "n_seeds": len(ms),
            "n_pass": n_pass,
            "min_margin": (float(np.min(margins)) if margins else None),
            "mean_margin": (float(np.mean(margins)) if margins else None),
            "worst_seed_value": _worst_seed_value(ms),
        }
    return summary


def _worst_seed_value(ms: list[dict[str, Any]]) -> Any:
    """The scored value at the worst-margin seed (for the ranking table)."""
    defined = [m for m in ms if m.get("margin") is not None]
    if not defined:
        return None
    worst = min(defined, key=lambda m: m["margin"])
    return worst.get("value", worst.get("score"))


__all__ = [
    "INNER_HOLDOUT_FRACTION",
    "INNER_SEED_OFFSET",
    "SEEDS",
    "inner_seed",
    "inner_split",
    "inner_anchor_cutpoints",
    "load_amended_gate_config",
    "build_inner_view_specs",
    "score_geometry",
    "score_battery",
    "measure_benefit_space_inner",
    "score_inner_pair",
    "inner_gate_verdict",
    "geometry_margins",
    "battery_margins",
    "benefit_space_margins",
    "summarize_margins",
]
