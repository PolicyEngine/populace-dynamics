"""Export the certified deployment-frame person table (policyengine.py .venv).

The DATA-BOUND step of W1 candidate 1: resolves the certified populace h5 by
its pinned revision + sha256 (``deployment_frame.CERTIFIED_PIN``) from the
Hugging Face hub, verifies the artifact sha256, builds the family-A person
frame, and writes a pickle + a provenance json. This is the only step that
needs huggingface_hub + pytables (the .venv-gate lacks them -- the floor's
certified_repro_env_note); the fit + score phases consume the pickle.

Run from the repo root with the policyengine.py .venv:
    PYTHONPATH=src <pe.py-venv>/python scripts/export_frame_persons.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ruff: noqa: E402 -- import follows the sys.path bootstrap above (script).
from populace_dynamics.data import deployment_frame as dfm


def main() -> None:
    from huggingface_hub import hf_hub_download

    pin = dfm.CERTIFIED_PIN
    h5 = hf_hub_download(
        repo_id=pin["hf_repo_id"],
        filename=pin["hf_filename"],
        repo_type=pin["hf_repo_type"],
        revision=pin["revision"],
    )
    sha = hashlib.sha256(Path(h5).read_bytes()).hexdigest()
    if sha != pin["artifact_sha256"]:
        raise SystemExit(
            f"certified frame sha mismatch: got {sha}, pinned "
            f"{pin['artifact_sha256']}"
        )
    persons, fractions = dfm.load_certified_persons(h5)

    scratch = ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    persons.to_pickle(scratch / "frame_persons.pkl")
    provenance = {
        "h5_path": h5,
        "artifact_sha256": sha,
        "sha256_verified": True,
        "n_persons": int(len(persons)),
        "n_households": int(persons["household_id"].nunique()),
        "total_weight": float(persons["weight"].sum()),
        "populated_source_fractions": {
            k: round(v, 4) for k, v in fractions.items()
        },
        "revision": pin["revision"],
    }
    json.dump(
        provenance, open(scratch / "frame_provenance.json", "w"), indent=1
    )
    print(json.dumps(provenance, indent=1))


if __name__ == "__main__":
    main()
