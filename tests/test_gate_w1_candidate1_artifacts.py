"""Committed-artifact bindings for W1 candidate 1 (the ``artifact`` tier).

These inspect committed ``runs/*.json``: the floor's committed holdout
universe (the split binding recomputes every gate seed's household sha256
with no h5) and the candidate artifact's schema. Always-runnable -- neither
needs the h5 nor PSID.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from populace_dynamics.models import transport_deployment_v1 as td

ROOT = Path(__file__).resolve().parents[1]
FLOOR = json.loads((ROOT / "runs" / "gate_w1_floors_v1.json").read_text())
ARTIFACT = ROOT / "runs" / "gate_w1_candidate1_v1.json"


def test_holdout_split_reproduces_committed_shas():
    """The candidate scores on EXACTLY the floor's committed holdout
    households -- recomputed from the committed universe, no frame needed."""
    hi = FLOOR["holdout_ids"]
    universe = np.array(
        [int(x) for x in hi["household_id_universe_csv"].split(",")]
    )
    assert (
        hashlib.sha256(hi["household_id_universe_csv"].encode()).hexdigest()
        == hi["household_id_universe_sha256"]
    )
    for ps in hi["per_seed"]:
        hold = td.holdout_side_a_households(universe, ps["seed"])
        sha = hashlib.sha256(
            ",".join(str(i) for i in hold).encode()
        ).hexdigest()
        assert len(hold) == ps["n_holdout_households"]
        assert sha == ps["holdout_household_id_sha256"]


@pytest.mark.skipif(
    not ARTIFACT.exists(), reason="candidate artifact not built in this env"
)
def test_artifact_schema():
    a = json.loads(ARTIFACT.read_text())
    assert a["schema_version"] == "gate_w1_candidate1_v1"
    assert a["registration"]["comment_id"] == "4950931131"
    assert a["deployment_frame"]["artifact_sha256"].startswith("c2065b64")
    fa = a["family_a"]
    assert fa["cube_shape"] == [20, len(fa["gated_cells"]), 5]
    assert len(fa["gated_cells"]) == 53
    assert len(fa["per_seed"]) == 5
    # conformance: genuine regeneration (not the prohibited identity map)
    assert a["family_a"]["conformance"]["identity_candidate"] is False
    assert a["family_b"]["cube_shape"][0] == 20
    assert set(a["family_c"]["fingerprints"]) == {"c1", "c2"}
    for k in ("family_a_pass", "family_b_pass", "family_c_pass", "gate_pass"):
        assert isinstance(a["verdict"][k], bool)
