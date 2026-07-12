"""Committed-artifact bindings for W1 candidate 2 (the ``artifact`` tier).

Inspects the committed ``runs/gate_w1_candidate2_v1.json``: schema, the
55-cell amended surface (53 family-A + 2 family-C; family B report-only), the
regenerated-surface conformance, the byte-carry regression vs candidate 1, and
the per-delta fit-vs-raw record. Always-runnable -- needs neither the h5 nor
PSID.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate_w1_candidate2_v1.json"


@pytest.mark.skipif(
    not ARTIFACT.exists(), reason="candidate 2 artifact not built in this env"
)
def test_artifact_schema_and_amended_surface():
    a = json.loads(ARTIFACT.read_text())
    assert a["schema_version"] == "gate_w1_candidate2_v1"
    assert a["registration"]["comment_id"] == "4952253568"
    assert a["deployment_frame"]["artifact_sha256"].startswith("c2065b64")

    fa = a["family_a"]
    assert fa["cube_shape"] == [20, len(fa["gated_cells"]), 5]
    assert len(fa["gated_cells"]) == 53
    assert len(fa["per_seed"]) == 5
    # regenerated surface, not the prohibited identity map
    assert fa["conformance"]["identity_candidate"] is False
    assert fa["conformance"]["max_across_draw_sd"] > 0

    # family C: the two committed fingerprints, binary
    assert set(a["family_c"]["fingerprints"]) == {"c1", "c2"}

    # family B is report-only after amendment 1 (gates nothing)
    fb = a["family_b"]
    assert fb["reported_not_gated"] is True
    assert fb["contributes_to_gate"] is False
    assert fb["n_report_cells"] == 10


@pytest.mark.skipif(
    not ARTIFACT.exists(), reason="candidate 2 artifact not built in this env"
)
def test_verdict_uses_amended_gate_rule():
    a = json.loads(ARTIFACT.read_text())
    v = a["verdict"]
    # the gate is family A AND family C; family B contributes nothing
    assert v["family_b_gates"] is False
    assert v["family_b_pass"] is None
    for k in ("family_a_pass", "family_c_pass", "gate_pass"):
        assert isinstance(v[k], bool)
    assert v["gate_pass"] == bool(v["family_a_pass"] and v["family_c_pass"])
    assert isinstance(v["n_seed_pass_family_a"], int)
    assert 0 <= v["n_seed_pass_family_a"] <= 5
    # both fingerprints reported prominently
    for fp in ("c1_ppi_nra", "c2_elimination_plus2pp"):
        assert "reversed_to_anchor" in v["fingerprints"][fp]


@pytest.mark.skipif(
    not ARTIFACT.exists(), reason="candidate 2 artifact not built in this env"
)
def test_byte_carry_regression_and_fit_vs_raw_present():
    a = json.loads(ARTIFACT.read_text())
    reg = a["family_a"]["byte_carry_regression_vs_candidate1"]
    assert reg["n_carried_cells"] > 0
    # the non-boundary in-support earnings cells reproduce candidate 1
    assert reg["max_abs_rbar_deviation_vs_candidate1"] <= 1e-9
    assert reg["bit_identical"] is True

    # per-delta fit-vs-raw
    fvr = a["fit_vs_raw"]
    assert "q1_entry_state_model" in fvr
    assert "q2_boundary_support" in fvr
    # Q1: the 25-34 entry-band model is recorded per sex
    assert any("25-34" in k for k in fvr["q1_entry_state_model"])
    # Q2: the four boundary cells carry a fitted vs raw participation
    for key in ("18-24|female", "18-24|male", "60-69|female", "60-69|male"):
        rec = fvr["q2_boundary_support"][key]
        assert "fit_participation_1_minus_p0" in rec


@pytest.mark.skipif(
    not ARTIFACT.exists(), reason="candidate 2 artifact not built in this env"
)
def test_c1_to_c2_progression_recorded():
    a = json.loads(ARTIFACT.read_text())
    prog = a["decomposition"]["family_a"]["c1_to_c2_progression"]
    # every family-A view has a candidate1 vs candidate2 pass tally
    for fam in ("marital_share", "hh_size_share", "coresident_spouse"):
        assert fam in prog
        assert prog[fam]["candidate2_pass"] >= 0
