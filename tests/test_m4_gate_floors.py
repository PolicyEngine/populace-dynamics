"""PSID reproduction pins for the M4 disability gate floors
(runs/m4_gate_floors_v1.json).

Rebuilds the disability panel from the staged PSID individual file and reruns
the reference moments, the seed-0 half-split (cells + anchor half-statistics)
and the full-sample anchor statistics, matching the committed numbers to
float precision -- with ``populace.fit`` never imported. Skipped when the
PSID individual file (``~/PolicyEngine/psid-data/ind2023er``) is absent; the
always-runnable derivation bindings live in
``tests/test_m4_gate_derivations.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "m4_gate_floors_v1.json"
SCRIPTS = ROOT / "scripts"

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_ind = pytest.mark.skipif(
    not (REAL_DATA / "ind2023er").is_dir(),
    reason="PSID ind2023er not staged",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def _builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_m4_gate_floors as builder

    return builder


@needs_real_ind
def test_moments_and_seed0_and_anchors_reproduce_without_populace_fit():
    assert "populace.fit" not in sys.modules
    builder = _builder()
    assert (
        "populace.fit" not in sys.modules
    ), "importing the M4 gate builder pulled populace.fit"
    from populace_dynamics.data import deaths, disability

    status = disability.read_disability_status()
    panel = disability.build_disability_panel(
        status, deaths.read_death_records()
    )
    art = _artifact()

    # 1) reference moments reproduce to float precision.
    got = disability.reference_moments(panel)
    ref = art["reference_moments"]
    assert set(got) == set(ref)
    for key, cell in ref.items():
        assert got[key]["rate"] == pytest.approx(cell["rate"], abs=1e-12), key
        assert got[key]["n_events"] == cell["n_events"], key

    # 2) seed-0 half-split cells + anchor half-statistics reproduce.
    got0 = builder.measure_seed_halfsplit(0, panel)
    ref0 = next(
        s for s in art["internal_noise_floor"]["per_seed"] if s["seed"] == 0
    )
    assert got0["n_persons_side_a"] == ref0["n_persons_side_a"]
    for key, cell in ref0["cells"].items():
        assert got0["cells"][key]["rate_a"] == pytest.approx(
            cell["rate_a"], abs=1e-12
        ), key
        if cell["log_ratio_abs"] is None:
            assert got0["cells"][key]["log_ratio_abs"] is None, key
        else:
            assert got0["cells"][key]["log_ratio_abs"] == pytest.approx(
                cell["log_ratio_abs"], abs=1e-12
            ), key
    for sex in ("female", "male"):
        assert got0["anchor"][sex]["conv_exit_share_a"] == pytest.approx(
            ref0["anchor"][sex]["conv_exit_share_a"], abs=1e-12
        )
        assert got0["anchor"][sex]["prev_min_gap"] == pytest.approx(
            ref0["anchor"][sex]["prev_min_gap"], abs=1e-12
        )

    # 3) full-sample anchor statistics reproduce.
    ce = builder.conversion_exit_shares(panel)
    for sex in ("female", "male"):
        c = art["anchor_checks"][f"conversion_exit.retirement_dominant|{sex}"]
        assert ce[sex]["share"] == pytest.approx(
            c["psid_retirement_exit_share"], abs=1e-12
        )
        assert ce[sex]["n_exits"] == c["psid_n_exits"]

    prev = builder.prevalence_shares(got)
    for sex in ("female", "male"):
        c = art["anchor_checks"][f"prevalence_ageshape.comonotone|{sex}"]
        committed = list(c["psid_share_by_band"].values())
        assert list(prev[sex]) == pytest.approx(committed, abs=1e-12)

    assert "populace.fit" not in sys.modules


@needs_real_ind
def test_anchor_tables_extract_reproduces_from_committed_table():
    """The sha-pinned in-repo table extracts reproduce (a data-free check,
    kept here so the reproduction tier exercises the full builder path)."""
    builder = _builder()
    tables = json.loads(builder.ANCHOR_TABLES_PATH.read_text())
    art = _artifact()
    at = art["anchor_tables"]
    assert (
        builder._t19_within_sex_shares(tables)
        == at["t19_prevalence_age_shape_2023"]
    )
    assert (
        builder._t50_worker_terminations(tables)
        == at["t50_worker_terminations_2023"]
    )
