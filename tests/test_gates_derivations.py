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
