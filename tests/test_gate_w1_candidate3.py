"""Script-level tests for the W1 candidate-3 runner.

Covers the write_new one-shot guard (main refuses once the output artifact
exists) and the gates.yaml-bound partition (family-A gated = union(view
tolerances) - report_only; family-C gated = the mapped gate_partition), proving
the candidate auto-adapts to whatever surface is locked -- including the
amendment-2 flip -- with no code change. Every test uses synthetic gates.yaml
blocks or a monkeypatched tmp artifact path: no frame, no committed artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_gate_w1_candidate3 as m  # noqa: E402


# ==========================================================================
# The script imports and exposes its one-shot surface.
# ==========================================================================
def test__given_module__then_imports_with_main_and_artifact_path():
    assert callable(m.main)
    assert m.ARTIFACT_PATH.name == "gate_w1_candidate3_v1.json"
    # candidate 3 is not registered until the amendment-2 flip
    assert m.REGISTRATION_POINTER is None


# ==========================================================================
# write_new one-shot guard -- refuse once the artifact (or sidecar) exists.
# ==========================================================================
def test__given_no_artifact__when_refuse_guard__then_no_raise(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(m, "ARTIFACT_PATH", tmp_path / "cand3_out")
    m._refuse_if_artifact_exists()  # neither file exists -> no raise


def test__given_existing_artifact__when_refuse_guard__then_raises(
    monkeypatch, tmp_path
):
    art = tmp_path / "cand3_out"
    art.write_text("{}")
    monkeypatch.setattr(m, "ARTIFACT_PATH", art)
    with pytest.raises(FileExistsError):
        m._refuse_if_artifact_exists()


def test__given_existing_sidecar_only__when_refuse_guard__then_raises(
    monkeypatch, tmp_path
):
    art = tmp_path / "cand3_out"
    sidecar = tmp_path / "cand3_out.env.json"
    sidecar.write_text("{}")  # only the env sidecar exists
    monkeypatch.setattr(m, "ARTIFACT_PATH", art)
    with pytest.raises(FileExistsError):
        m._refuse_if_artifact_exists()


def test__given_existing_artifact__when_main__then_refuses_before_frame(
    monkeypatch, tmp_path
):
    # main() must refuse EVERY phase before any frame is touched, so an
    # accidental re-run of the completed one-shot cannot peek at the frame.
    art = tmp_path / "cand3_out"
    art.write_text("{}")
    monkeypatch.setattr(m, "ARTIFACT_PATH", art)
    # "seed" would otherwise need POPULACE_DYNAMICS_FRAME_PICKLE; the guard
    # fires first, so no frame is loaded.
    with pytest.raises(FileExistsError):
        m.main(["seed", "--seed", "0"])


# ==========================================================================
# gates.yaml-bound partition (pure; synthetic blocks mimic pre/post flip).
# ==========================================================================
def _family_a_block() -> dict:
    return {
        "views": {
            "earnings_participation": {
                "tolerances": {
                    "earnings_participation.18-24|female": 0.22,
                    "earnings_participation.25-34|female": 0.12,
                }
            },
            "marital_share": {
                "tolerances": {
                    "marital_share.married.25-34|female": 0.10,
                    "marital_share.married.65+|female": 0.10,
                }
            },
        },
        "report_only": ["earnings_p50p10.25-34|male"],
    }


def _family_c_block() -> dict:
    return {
        "fingerprints": {
            "c1": {"id": "ppi_nra"},
            "c2": {"id": "elimination_plus2pp"},
        },
        "gate_partition": {
            "gate_eligible": [
                "fingerprint.ppi_nra",
                "fingerprint.elimination_plus2pp",
            ]
        },
    }


def test__given_family_a_block__when_partition__then_union_minus_report_only():
    tol, gated = m.family_a_partition(_family_a_block())
    assert len(tol) == 4
    # report_only cell has no tolerance here (disjoint), so gated == all 4
    assert set(gated) == {
        "earnings_participation.18-24|female",
        "earnings_participation.25-34|female",
        "marital_share.married.25-34|female",
        "marital_share.married.65+|female",
    }
    # every gated cell carries a tolerance (drawn from the same union)
    assert all(cell in tol for cell in gated)


def test__given_amendment2_demotions__when_partition__then_demoted_drop_out():
    block = _family_a_block()
    # the amendment-2 flip adds the demoted cells to report_only
    block["report_only"] = list(block["report_only"]) + [
        "earnings_participation.18-24|female",
        "marital_share.married.65+|female",
    ]
    _tol, gated = m.family_a_partition(block)
    assert "earnings_participation.18-24|female" not in gated
    assert "marital_share.married.65+|female" not in gated
    # the surviving cells stay gated
    assert "earnings_participation.25-34|female" in gated
    assert "marital_share.married.25-34|female" in gated


def test__given_family_c_block__when_gated_fingerprints__then_maps_ids():
    assert m.family_c_gated_fingerprints(_family_c_block()) == ["c1", "c2"]


def test__given_c1_demoted__when_gated_fingerprints__then_only_c2():
    block = _family_c_block()
    # the amendment-2 flip demotes C1 (ppi_nra); only C2 remains gated
    block["gate_partition"]["gate_eligible"] = [
        "fingerprint.elimination_plus2pp"
    ]
    assert m.family_c_gated_fingerprints(block) == ["c2"]


def test__given_gated_c2_only__when_family_c_pass__then_ignores_c1():
    # C1 never reverses (robust non-reversal); post-flip it does not gate.
    fc = {
        "fingerprints": {
            "c1": {"reversed_to_anchor": False},
            "c2": {"reversed_to_anchor": True},
        }
    }
    assert m._family_c_pass(fc, ["c2"]) is True
    # pre-flip, C1's non-reversal still fails family C
    assert m._family_c_pass(fc, ["c1", "c2"]) is False


def test__given_no_gated_fingerprints__when_family_c_pass__then_false():
    fc = {"fingerprints": {"c1": {"reversed_to_anchor": True}}}
    assert m._family_c_pass(fc, []) is False
