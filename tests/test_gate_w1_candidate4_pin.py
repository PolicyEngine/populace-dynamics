"""Locked-hot artifact-tier pin for the gate_w1 candidate-4 record PASS.

Candidate 4 (`transport_deployment_v3`, the byte-identical candidate-3 model on
the ratified amendment-3 surface) is the first gate_w1 PASS in the record:
registration #42 comment 4964126356 -> run PR #184 (merged c130517) -> bit-exact
verification #42 comment 4964799327 -> record entry #42 comment 4964811349.

These bindings freeze the committed evidence and its mapping annotation:

* the git blob of ``runs/gate_w1_candidate4_v1.json`` (``d944ed22...``) and its
  schema / frame / cube shape;
* the PASS verdict (family A and C pass, gate passes, 5/5 seeds) and the two
  operative family-C statistics carried on the gated C2 fingerprint
  (``required_swap_realised`` true, ``reversed_to_anchor`` false published);
* bit-identity of the family-A cube to candidate 3 on the 43 shared cells,
  recomputed from both committed cubes;
* the live contract blob (``gates.yaml`` ``1efbf095...``), matching the sidecar
  the run was scored against;
* the ``runs/NOTES-gate_w1_candidate4.md`` mapping annotation of record.

A closing mutation check proves the bindings discriminate (>= 3 caught). This
module is ``artifact`` tier: always-runnable, needing neither the h5 nor PSID.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "gate_w1_candidate4_v1.json"
SIDECAR = ROOT / "runs" / "gate_w1_candidate4_v1.json.env.json"
C3_ARTIFACT = ROOT / "runs" / "gate_w1_candidate3_v1.json"
GATES = ROOT / "gates.yaml"
NOTES = ROOT / "runs" / "NOTES-gate_w1_candidate4.md"

# --- the pins (frozen record) -------------------------------------------
ARTIFACT_BLOB = "d944ed2270a2c46c36ed3d0cd7e3328f1c8dbe88"
CONTRACT_BLOB = "1efbf0958b722d8172697ac3f9a48c043de09bcf"
# The LIVE contract blob moves only at ratified flips; the scored-against pin
# above is frozen forever. Updated at the amendment-4 design_commit flip (#233).
CONTRACT_BLOB_LIVE = "269ff692f0e5a8d7985a3e52e72186aa2ee2fc21"
FRAME_SHA_PREFIX = "c2065b64"
GATED_CELLS = 43
CUBE_SHAPE = [20, GATED_CELLS, 5]
# the four cells amendment 3 demoted -> present in c3, absent from c4
C3_ONLY_CELLS = {
    "hh_size_share.1",
    "hh_size_share.3",
    "hh_size_share.4",
    "hh_size_share.5plus",
}


def git_blob_sha(data: bytes) -> str:
    """The git blob object id of ``data`` (sha1 of ``blob <len>\\0<data>``)."""
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- reusable invariants (shared by the pins and the mutation check) -----
def _verdict_pass_holds(artifact: dict) -> bool:
    v = artifact["verdict"]
    return (
        v["family_a_pass"] is True
        and v["family_c_pass"] is True
        and v["gate_pass"] is True
        and v["n_seed_pass_family_a"] == 5
        and all(s["pass"] is True for s in v["per_seed_family_a"])
    )


def _c2_operative_holds(artifact: dict) -> bool:
    c2 = artifact["verdict"]["fingerprints"]["c2"]
    return (
        c2["gated"] is True
        and c2["required_swap_realised"] is True
        and c2["reversed_to_anchor"] is False
    )


def _shared_cube(artifact: dict, cells3: list, cube3: np.ndarray):
    """The candidate cube restricted to cells shared with candidate 3."""
    cells4 = artifact["family_a"]["gated_cells"]
    cube4 = np.asarray(artifact["family_a"]["cube"], dtype=float)
    idx3 = {c: i for i, c in enumerate(cells3)}
    shared = [c for c in cells4 if c in idx3]
    sub4 = cube4[:, [cells4.index(c) for c in shared], :]
    sub3 = cube3[:, [idx3[c] for c in shared], :]
    return shared, sub4, sub3


def _bit_identity_holds(artifact: dict) -> bool:
    c3 = _load(C3_ARTIFACT)
    cells3 = c3["family_a"]["gated_cells"]
    cube3 = np.asarray(c3["family_a"]["cube"], dtype=float)
    shared, sub4, sub3 = _shared_cube(artifact, cells3, cube3)
    return len(shared) == GATED_CELLS and np.array_equal(sub4, sub3)


# ========================================================================
# 1. Artifact blob + schema pin.
# ========================================================================
def test_artifact_blob_and_schema_pinned():
    assert (
        ARTIFACT.exists()
    ), "the candidate-4 record artifact must be committed"
    assert git_blob_sha(ARTIFACT.read_bytes()) == ARTIFACT_BLOB
    a = _load(ARTIFACT)
    assert a["schema_version"] == "gate_w1_candidate4_v1"
    assert a["run"] == "gate_w1_candidate4_v1"
    assert a["deployment_frame"]["artifact_sha256"].startswith(
        FRAME_SHA_PREFIX
    )
    fa = a["family_a"]
    assert fa["cube_shape"] == CUBE_SHAPE
    assert len(fa["gated_cells"]) == GATED_CELLS
    assert len(fa["per_seed"]) == 5
    # a genuine regeneration, not the prohibited identity map
    assert fa["conformance"]["identity_candidate"] is False
    assert fa["conformance"]["max_across_draw_sd"] > 0


# ========================================================================
# 2. The PASS verdict (5/5 seeds, family A and C, gate).
# ========================================================================
def test_verdict_is_the_recorded_pass():
    a = _load(ARTIFACT)
    v = a["verdict"]
    assert _verdict_pass_holds(a)
    assert v["family_a_pass"] is True
    assert v["family_c_pass"] is True
    assert v["gate_pass"] is True
    assert v["n_seed_pass_family_a"] == 5
    # family B publishes but does not gate (amendment 1)
    assert v["family_b_gates"] is False
    assert v["family_b_pass"] is None
    # every seed cleared all 43 cells
    for seed in v["per_seed_family_a"]:
        assert seed["pass"] is True
        assert seed["cells_pass"] == GATED_CELLS
        assert seed["cells_fail"] == 0
    assert a["family_a"]["n_seed_pass"] == 5


# ========================================================================
# 3. Both operative family-C statistics, on the gated C2 fingerprint.
# ========================================================================
def test_family_c_operative_statistics_published():
    a = _load(ARTIFACT)
    fps = a["verdict"]["fingerprints"]
    # the amendment-3 pair scope: only C2 gates
    assert _c2_operative_holds(a)
    c2 = fps["c2"]
    assert c2["gated"] is True
    assert c2["required_swap_realised"] is True  # operative gate -> pass
    assert c2["reversed_to_anchor"] is False  # demoted rule, published only
    # C1 is report-only after the amendment-2 flip
    assert fps["c1"]["gated"] is False
    # the byte-copy label + prose describe the c3 model, not the c4 rule
    assert a["verdict"]["candidate"] == "w1_candidate3"
    assert a["registration"]["comment_id"] is None
    assert a["verdict"]["registration_pointer"] is None


# ========================================================================
# 4. Family-A cube bit-identical to candidate 3 on the 43 shared cells.
# ========================================================================
def test_family_a_cube_bit_identical_to_candidate3():
    a = _load(ARTIFACT)
    c3 = _load(C3_ARTIFACT)
    cells4 = set(a["family_a"]["gated_cells"])
    cells3 = set(c3["family_a"]["gated_cells"])
    # c4's gated set is a strict subset of c3's; the gap is the demoted quad
    assert cells4 < cells3
    assert cells3 - cells4 == C3_ONLY_CELLS
    # recompute the shared sub-cube from both committed cubes
    cube3 = np.asarray(c3["family_a"]["cube"], dtype=float)
    shared, sub4, sub3 = _shared_cube(a, c3["family_a"]["gated_cells"], cube3)
    assert len(shared) == GATED_CELLS
    assert sub4.shape == (20, GATED_CELLS, 5)
    assert sub4.size == 4300
    assert np.array_equal(sub4, sub3)  # 0 deviations across 4,300 values
    assert _bit_identity_holds(a)


# ========================================================================
# 5. The contract blob the run was scored against is the live gates.yaml.
# ========================================================================
def test_contract_blob_pinned_and_live():
    sidecar = _load(SIDECAR)
    # the frozen sidecar records the contract the run was scored against
    assert sidecar["contract"]["blob_sha"] == CONTRACT_BLOB
    assert sidecar["contract"]["path"] == "gates.yaml"
    # locked-hot: the live gates.yaml matches the current ratified contract
    # blob (moves only at ratified flips; unratified edits still fail here)
    assert git_blob_sha(GATES.read_bytes()) == CONTRACT_BLOB_LIVE


# ========================================================================
# 6. The runs/ mapping annotation of record exists and carries the map.
# ========================================================================
def test_runs_note_exists_and_carries_mapping():
    assert NOTES.exists(), "the c4 mapping annotation note must be committed"
    text = NOTES.read_text(encoding="utf-8")
    for needle in (
        ARTIFACT_BLOB[:8],  # d944ed22 (artifact blob)
        CONTRACT_BLOB[:8],  # 1efbf095 (contract blob)
        "4964126356",  # registration
        "4964799327",  # bit-exact verification
        "4964811349",  # record entry
        "transport_deployment_v3",  # the byte-copy model
        "w1_candidate3",  # the frozen byte-copy label
        "required_swap_realised",  # the operative family-C statistic
        "amendment-3",  # the scored surface
    ):
        assert needle in text, f"note is missing {needle!r}"


# ========================================================================
# 7. The bindings discriminate -- mutation check (>= 3 caught).
# ========================================================================
def test_pin_bindings_catch_mutations():
    a = _load(ARTIFACT)
    # sanity: the committed artifact satisfies every invariant
    assert _verdict_pass_holds(a)
    assert _c2_operative_holds(a)
    assert _bit_identity_holds(a)

    caught = 0

    # (1) a flipped gate verdict is rejected by the PASS binding
    m = copy.deepcopy(a)
    m["verdict"]["gate_pass"] = False
    if not _verdict_pass_holds(m):
        caught += 1

    # (2) a dropped passing seed is rejected by the 5/5 binding
    m = copy.deepcopy(a)
    m["verdict"]["n_seed_pass_family_a"] = 4
    if not _verdict_pass_holds(m):
        caught += 1

    # (3) flipping the operative pair statistic is rejected by family C
    m = copy.deepcopy(a)
    m["verdict"]["fingerprints"]["c2"]["required_swap_realised"] = False
    if not _c2_operative_holds(m):
        caught += 1

    # (4) perturbing one shared cube cell breaks bit-identity to c3
    m = copy.deepcopy(a)
    m["family_a"]["cube"][0][0][0] += 1.0
    if not _bit_identity_holds(m):
        caught += 1

    # (5) any byte edit changes the pinned git blob
    mutated = ARTIFACT.read_bytes().replace(
        b'"gate_pass": true', b'"gate_pass": false', 1
    )
    if git_blob_sha(mutated) != ARTIFACT_BLOB:
        caught += 1

    assert caught >= 3, f"pins under-discriminate: only {caught} caught"
    assert caught == 5
