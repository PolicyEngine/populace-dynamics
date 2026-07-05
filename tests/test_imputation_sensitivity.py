"""Tests for the reported (not gated) imputation-sensitivity artifact.

Two always-runnable consistency guarantees touch only committed files:
every stored delta recomputes from its stored endpoints, and the
committed references in the artifact match
``runs/noise_floor_psid_family_9822.json`` exactly. A third test
(skipped when the PSID family files are absent) pins the excl-flagged
battery values by recomputing them through the builder's own code path,
so the reported numbers are reproducible from the panel alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "imputation_sensitivity_v1.json"
BATTERY_REFERENCE_RUN = ROOT / "runs" / "noise_floor_psid_family_9822.json"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_artifact_is_reported_not_gated():
    art = _artifact()
    assert art["reported_not_gated"] is True
    assert "no gate" in art["statement"].lower()
    assert "referee" in art["statement"].lower()


def test_battery_deltas_recompute_from_stored_endpoints():
    """Every stored delta equals the difference of its stored values."""
    art = _artifact()
    for stat, block in art["battery"].items():
        ref = block["committed_reference"]
        full = block["full_panel_recomputed"]
        any_flag = block["excl_any_flag"]
        code_1 = block["excl_code_1"]
        d = block["deltas"]
        assert d["excl_any_flag_minus_committed"] == pytest.approx(
            any_flag - ref, abs=1e-12
        ), stat
        assert d["excl_code_1_minus_committed"] == pytest.approx(
            code_1 - ref, abs=1e-12
        ), stat
        assert d["excl_any_flag_minus_full"] == pytest.approx(
            any_flag - full, abs=1e-12
        ), stat
        assert d["excl_code_1_minus_full"] == pytest.approx(
            code_1 - full, abs=1e-12
        ), stat


def test_committed_references_match_noise_floor_artifact():
    """The artifact's committed references equal the locked battery."""
    art = _artifact()
    reference = json.loads(BATTERY_REFERENCE_RUN.read_text())[
        "battery_reference"
    ]
    for stat, block in art["battery"].items():
        assert block["committed_reference"] == pytest.approx(
            reference[stat], abs=0.0
        ), stat
    # The full-panel recomputation reproduced them exactly (the
    # builder hard-stops otherwise); the artifact records that.
    repro = art["battery_reference_reproduction"]
    assert repro["all_committed_values_reproduced_exactly"] is True
    for stat, check in repro["checks"].items():
        assert check["exact_float_match"] is True, stat
        assert check["committed"] == pytest.approx(reference[stat], abs=0.0)


def test_geometry_floor_deltas_recompute():
    """ctx20 floor deltas equal excl mean minus committed mean."""
    art = _artifact()
    floors = art["geometry_floors_excl_any_flag"]
    for view in ("ctx20_pairs_window2", "ctx20_runs_window3"):
        for key in ("c2st_auc", "prdc_coverage", "energy_distance"):
            block = floors[view][key]
            assert block["delta_mean"] == pytest.approx(
                block["excl_any_flag_mean"] - block["committed_mean"],
                abs=1e-12,
            ), f"{view}.{key}"


def test_dropped_counts_are_consistent():
    art = _artifact()
    n = art["n_person_periods"]
    assert (
        n["dropped_excl_any_flag"] == n["full_filtered"] - n["excl_any_flag"]
    )
    assert n["dropped_excl_code_1"] == n["full_filtered"] - n["excl_code_1"]
    # excluding any flag drops at least as many as excluding only code 1.
    assert n["dropped_excl_any_flag"] >= n["dropped_excl_code_1"]


@needs_real_family
def test_excl_flagged_battery_reproduces(tmp_path):
    """Seed-0-style pin: recompute the excl-flagged battery values.

    Reruns the builder's battery on the excl-any-flag and excl-code-1
    panels and pins them to the committed artifact to float precision,
    so the reported deltas are reproducible from the panel alone.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_imputation_sensitivity",
        ROOT / "scripts" / "build_imputation_sensitivity.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    prime = mod.load_filtered_panel()
    excl_any = prime[prime.earnings_acc == 0]
    excl_c1 = prime[prime.earnings_acc != 1]
    b_any = mod.compute_battery(excl_any)
    b_c1 = mod.compute_battery(excl_c1)

    art = _artifact()
    for stat, block in art["battery"].items():
        assert b_any[stat] == pytest.approx(
            block["excl_any_flag"], abs=1e-9
        ), stat
        assert b_c1[stat] == pytest.approx(
            block["excl_code_1"], abs=1e-9
        ), stat


@needs_real_family
def test_full_panel_reproduces_committed_reference_exactly():
    """The builder's full-panel battery == committed reference (float)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_imputation_sensitivity",
        ROOT / "scripts" / "build_imputation_sensitivity.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    prime = mod.load_filtered_panel()
    full = mod.compute_battery(prime)
    reference = json.loads(BATTERY_REFERENCE_RUN.read_text())[
        "battery_reference"
    ]
    for stat, ref in reference.items():
        assert full[stat] == float(ref), stat
