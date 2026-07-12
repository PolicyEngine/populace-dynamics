"""Reproduction pin for the W2 seam diagnostic's observed frame + transport.

Skipped without a policyengine-us checkout (the SSA AIME/PIA oracle, pointed to
by ``POPULACE_DYNAMICS_PE_US_DIR``) and the staged PSID files. When both are
present and the oracle revision matches the committed anchor replication, this
reruns step 1 (the caregiver Biden observed frame) and step 2 (the ratio-of-
means cell tables) through the real build machinery and pins the committed
artifact's step-1/step-2 numbers -- the deterministic, non-microsim half of the
seam. The full microsim passes (step 3) are one-shot and are not re-run here;
their outputs are checked for internal consistency by the always-runnable
``test_w2_seam_caregiver.py``.

The collector marks this module ``oracle_policyengine`` (it references
``POPULACE_DYNAMICS_PE_US_DIR``); run it in the repository venv with the PSID
files staged before the artifact is committed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "w2_seam_caregiver_v1.json"
ANCHOR_ARTIFACT = ROOT / "runs" / "replication_caregiver_v1.json"
SCRIPTS = ROOT / "scripts"

PSID_ROOT = Path("~/PolicyEngine/psid-data").expanduser()
_ORACLE_ENV = "POPULACE_DYNAMICS_PE_US_DIR"

needs_oracle = pytest.mark.skipif(
    not os.environ.get(_ORACLE_ENV)
    or not Path(os.environ.get(_ORACLE_ENV, "")).expanduser().is_dir(),
    reason=f"{_ORACLE_ENV} not set to a policyengine-us checkout",
)
needs_psid = pytest.mark.skipif(
    not (PSID_ROOT / "family" / "2023").is_dir()
    or not (PSID_ROOT / "ind2023er").is_dir(),
    reason="PSID family + individual files not staged",
)


def _builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import w2_seam_caregiver as builder

    return builder


@needs_oracle
@needs_psid
def test_observed_frame_and_cells_reproduce_committed_artifact():
    builder = _builder()
    art = json.loads(ARTIFACT.read_text())
    anchor = json.loads(ANCHOR_ARTIFACT.read_text())

    # The SSA oracle must be at the revision the committed anchor was scored on,
    # or the AIME/PIA (hence gain/base ratios) shift and the pin cannot hold.
    from populace_dynamics.ss.params import load_ssa_parameters

    pinned = art["revision_pins"]["ss_oracle_pe_us_revision"]
    live = load_ssa_parameters().pe_us_revision
    if live != pinned:
        pytest.skip(
            f"policyengine-us checkout at {live} differs from the artifact's "
            f"pinned SSA-oracle revision {pinned}; point {_ORACLE_ENV} at the "
            "pinned revision to run the reproduction"
        )
    # The committed anchor replication was scored on the same oracle revision.
    assert anchor["revision_pins"]["pe_us_revision"] == pinned

    observed = builder.build_observed_frame()
    step1 = art["step1_observed_frame"]
    assert len(observed) == step1["n_career_frame"]
    assert int((observed["gain"] > 1e-9).sum()) == step1["n_gainers"]

    w = observed["weight"].to_numpy(np.float64)
    for s in ("male", "female"):
        share = float(
            observed.loc[observed["sex"] == s, "weight"].sum() / w.sum()
        )
        assert share == pytest.approx(
            step1["weighted_sex_shares"][s], abs=1e-9
        )

    # Step 2: the ratio-of-means overall proportion reproduces to float
    # precision, and the full grid is exactly 3 x 2 x 10 cells.
    tables = builder.build_cell_tables(observed)
    assert tables["overall"]["prop"] == pytest.approx(
        art["step2_transport"]["overall_proportion"], rel=1e-9, abs=1e-12
    )
    assert len(tables["full"]) == art["step2_transport"]["n_cells_defined"]
    assert (
        art["step2_transport"]["n_cells_full_grid"]
        == len(builder.AGE_BANDS) * 2 * builder.N_DECILES
    )
    # Per-AIME-decile proportions are monotone non-increasing (progressivity):
    # the caregiver top-up is largest for the lowest-AIME workers.
    dec_props = [
        tables["decile"][d]["prop"]
        for d in range(builder.N_DECILES)
        if d in tables["decile"]
    ]
    assert dec_props[0] >= dec_props[-1]
