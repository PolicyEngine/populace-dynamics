"""Bind every locked gate-1 threshold to its stated derivation.

Round-2 review finding 5b: derivations lived only in YAML comments, so
an artifact rebuild that shifted a floor could silently break the
stated rationale. Each view's ``derivations`` block in gates.yaml is
machine-checkable data: threshold == round(floor mean + k * floor sd)
at the stated rounding. These tests run everywhere — they touch only
committed files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _gate_1() -> dict:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"]["thresholds"]


def _floor_stats(run_path: str, key: str) -> tuple[float, float]:
    artifact = json.loads((ROOT / run_path).read_text())
    stats = artifact["noise_floor_seeds_0_4"][key]
    return stats["mean"], stats["sd"]


def _derive(run_path: str, key: str, k: float, rounding: int) -> float:
    mean, sd = _floor_stats(run_path, key)
    return round(mean + k * sd, rounding)


def _view_cases():
    for view_name, view in _gate_1()["views"].items():
        derivations = view["derivations"]
        for threshold_name, rule in derivations["rules"].items():
            yield (
                view_name,
                threshold_name,
                derivations["floor_run"],
                rule,
                view["geometry"][threshold_name],
            )


@pytest.mark.parametrize(
    "view_name, threshold_name, floor_run, rule, locked",
    list(_view_cases()),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_locked_threshold_matches_stated_derivation(
    view_name, threshold_name, floor_run, rule, locked
):
    rounding = rule.get("rounding", 2)
    if isinstance(rule["k"], list):
        lower = _derive(floor_run, rule["key"], rule["k"][0], rounding)
        upper = _derive(floor_run, rule["key"], rule["k"][1], rounding)
        assert [lower, upper] == pytest.approx(locked), (
            f"{view_name}.{threshold_name}: derived [{lower}, {upper}] "
            f"!= locked {locked}"
        )
    else:
        derived = _derive(floor_run, rule["key"], rule["k"], rounding)
        assert derived == pytest.approx(locked), (
            f"{view_name}.{threshold_name}: derived {derived} "
            f"!= locked {locked}"
        )


def test_every_locked_geometry_threshold_has_a_derivation():
    for view in _gate_1()["views"].values():
        assert set(view["geometry"]) == set(view["derivations"]["rules"])


def test_battery_tolerances_have_committed_references():
    thresholds = _gate_1()
    reference = json.loads(
        (ROOT / "runs/noise_floor_psid_family_9822.json").read_text()
    )["battery_reference"]
    aliases = {"mobility_diagonal": "mobility_diagonal_mean"}
    for name in thresholds["battery"]:
        stat = name.removesuffix("_tolerance")
        stat = aliases.get(stat, stat)
        assert stat in reference, f"no committed reference for {stat}"


def test_zero_persistence_is_one_minus_exit_rate():
    """The lock annotates these as ONE constraint; pin the identity."""
    reference = json.loads(
        (ROOT / "runs/noise_floor_psid_family_9822.json").read_text()
    )["battery_reference"]
    assert math.isclose(
        reference["zero_persistence"],
        1.0 - reference["exit_rate"],
        rel_tol=0,
        abs_tol=1e-12,
    )


# --------------------------------------------------------------------------
# Amendment proposal binding (gate_1.amendment_proposed)
#
# The amendment proposal is an inert public OBJECT: it changes no locked
# value and no model reads it. These tests bind its arithmetic so a
# referee round (and any later ratification) inherits a proposal whose
# stated derivations already hold. They run everywhere -- they touch only
# committed files (gates.yaml plus the anchor artifact). All are no-ops
# (skip) until the amendment_proposed subsection exists.
# --------------------------------------------------------------------------
def _amendment() -> dict | None:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    return gates["gates"]["gate_1"].get("amendment_proposed")


def _benefit_space_change() -> dict | None:
    gates = yaml.safe_load((ROOT / "gates.yaml").read_text())
    ratified = gates["gates"]["gate_1"]["thresholds"].get("benefit_space")
    if ratified is not None:
        return ratified
    amendment = _amendment()
    if amendment is None:
        return None
    for change in amendment["changes"]:
        if change.get("type") == "add_gated_block":
            return change
    return None


def _anchor_stats(run_path: str, key: str) -> tuple[float, float]:
    artifact = json.loads((ROOT / run_path).read_text())
    stats = artifact["noise_floor_seeds_0_4"][key]
    return stats["mean"], stats["sd"]


def test_amendment_ks_threshold_matches_committed_anchor():
    """Proposed KS band == round(anchor mean + k * anchor sd) at rounding.

    The one floor-DERIVED band in the proposal (every other band is the
    a-priori paper criterion). Mirrors the locked-geometry derivation
    test: the stated ``k`` and ``rounding`` must reproduce the value.
    """
    change = _benefit_space_change()
    if change is None:
        pytest.skip("no amendment_proposed benefit_space block")
    derivations = change["derivations"]
    run_path = derivations["anchor_run"]
    rule = derivations["rules"]["weighted_ks_max"]
    rounding = rule.get("rounding", 4)
    mean, sd = _anchor_stats(run_path, rule["key"])
    derived = round(mean + rule["k"] * sd, rounding)
    locked = change["metrics"]["weighted_ks_max"]["value"]
    assert derived == pytest.approx(
        locked
    ), f"proposed KS band derived {derived} != stated {locked}"


def test_amendment_gated_metrics_carry_derivation_or_a_priori_source():
    """Every proposed GATED metric is justified.

    Each metric in the proposed benefit_space block either derives from
    the committed anchor (a ``derivations.rules`` entry) or cites an
    a-priori source (an ``a_priori_source`` field). No proposed gated
    number is left unexplained.
    """
    change = _benefit_space_change()
    if change is None:
        pytest.skip("no amendment_proposed benefit_space block")
    derived_metrics = set(change["derivations"]["rules"])
    for name, spec in change["metrics"].items():
        has_derivation = name in derived_metrics
        has_a_priori = bool(spec.get("a_priori_source"))
        assert has_derivation or has_a_priori, (
            f"proposed gated metric {name} carries neither a derivation "
            f"nor an a_priori_source citation"
        )


def test_amendment_derived_metrics_are_not_also_flat_a_priori():
    """The floor-derived KS band is not mislabeled as an a-priori band.

    Derived and a-priori are mutually exclusive justifications; a metric
    with a ``derivations.rules`` entry must NOT also claim an a-priori
    source (that would hide that its value tracks the floor, not a fixed
    criterion).
    """
    change = _benefit_space_change()
    if change is None:
        pytest.skip("no amendment_proposed benefit_space block")
    derived_metrics = set(change["derivations"]["rules"])
    for name in derived_metrics:
        assert not change["metrics"][name].get("a_priori_source"), (
            f"{name} is floor-derived; it must not also cite an "
            f"a_priori_source"
        )


def test_amendment_anchor_stats_recompute_from_stored_per_seed_values():
    """The anchor artifact's pooled stats recompute from its per-seed data.

    An artifact rebuild that shifted the floor must not silently keep
    stale pooled numbers: every headline the proposal leans on (the KS
    mean/sd it derives from, the abs mean/median/Q0 gap magnitudes, the
    pooled Q0 magnitude, and the per-decile 5%-clip counts) recomputes
    from ``per_seed``.
    """
    change = _benefit_space_change()
    if change is None:
        pytest.skip("no amendment_proposed benefit_space block")
    run_path = change["derivations"]["anchor_run"]
    art = json.loads((ROOT / run_path).read_text())
    per_seed = art["per_seed"]
    floor = art["floor_seeds_0_4"]

    def dist(row: dict, path: list[str]):
        node = row["distribution"]
        for key in path:
            node = node[key]
        return node

    # KS block (the derived one) recomputes, and the headline
    # noise_floor_seeds_0_4 mirror equals it.
    ks_vals = [dist(r, ["ks_distance"]) for r in per_seed]
    assert floor["ks_distance"]["values"] == pytest.approx(ks_vals)
    assert floor["ks_distance"]["mean"] == pytest.approx(
        sum(ks_vals) / len(ks_vals)
    )
    n = len(ks_vals)
    mean_ks = sum(ks_vals) / n
    var_ks = sum((v - mean_ks) ** 2 for v in ks_vals) / (n - 1)
    assert floor["ks_distance"]["sd"] == pytest.approx(var_ks**0.5)
    assert art["noise_floor_seeds_0_4"]["ks_distance"] == (
        floor["ks_distance"]
    )

    # Absolute percent-gap magnitudes recompute from the signed per-seed
    # distribution gaps.
    abs_mean = [abs(dist(r, ["mean", "pct_diff"])) for r in per_seed]
    abs_median = [abs(dist(r, ["median", "pct_diff"])) for r in per_seed]
    assert floor["abs_mean_pct_diff"]["values"] == pytest.approx(abs_mean)
    assert floor["abs_median_pct_diff"]["values"] == pytest.approx(abs_median)

    # Q0: the pooled magnitude the gate is scored on equals the absolute
    # across-seed mean of the signed per-seed Q0 gaps.
    q0_signed = [r["q0_zero_anchor"]["mean_pct_diff"] for r in per_seed]
    pooled_abs_q0 = abs(sum(q0_signed) / len(q0_signed))
    assert floor["pooled_abs_q0_mean_pct_diff"] == pytest.approx(pooled_abs_q0)

    # Per-decile 5%-clip counts (the reported-not-gated evidence for d1,
    # d2) recompute from the signed per-seed decile gaps.
    for dkey, count in floor["decile_seeds_clipping_5pct"].items():
        vals = floor["signed_decile_pct_diff"][dkey]["values"]
        assert count == sum(abs(v) > 5.0 for v in vals), dkey


def test_amendment_reported_not_gated_deciles_are_denominator_fragile():
    """d1/d2 are reported-not-gated because the FLOOR itself clips 5%.

    The amendment reports-not-gates exactly the deciles whose
    real-vs-real floor exceeds the +/-5% band on a majority of seeds
    (denominator fragility), and gates exactly the deciles whose floor
    clips no seed. This binds that the reported/gated partition matches
    the committed anchor, not a hand-picked list.
    """
    change = _benefit_space_change()
    if change is None:
        pytest.skip("no amendment_proposed benefit_space block")
    run_path = change["derivations"]["anchor_run"]
    art = json.loads((ROOT / run_path).read_text())
    clip = art["floor_seeds_0_4"]["decile_seeds_clipping_5pct"]
    n_seeds = art["n_persons"] and len(
        art["floor_seeds_0_4"]["ks_distance"]["values"]
    )
    majority = n_seeds // 2 + 1

    reported = set(change["reported_not_gated"])
    gated = set(change["metrics"]["decile_pct_diff_max"]["deciles_gated"])
    assert reported.isdisjoint(gated)

    # Every reported-not-gated decile clips 5% on a majority of seeds.
    for dkey in reported:
        assert clip[dkey] >= majority, (
            f"{dkey} is reported-not-gated but its real-vs-real floor "
            f"clips only {clip[dkey]}/{n_seeds} seeds"
        )
    # Every gated decile clips 5% on no seed.
    for dkey in gated:
        assert clip[dkey] == 0, (
            f"{dkey} is gated but its real-vs-real floor clips "
            f"{clip[dkey]} seeds"
        )
