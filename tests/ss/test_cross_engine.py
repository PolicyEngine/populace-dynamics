"""Cross-engine PIA agreement: Axiom rules engine versus the Python oracle.

Two layers of test:

* Always-runnable (no engine needed): the committed artifact
  ``runs/pia_cross_engine_v1.json`` has the expected schema, its
  ``max_abs_diff_dollars`` is exactly 0 (the explained agreement bound), and
  every per-worker row agrees to the cent. Plus an exhaustive sweep proving the
  engine's dime floor (no 1e-9 epsilon) matches the oracle's (with epsilon) over
  every integer AIME in range — the documented reason the two floors coincide.

* Engine-backed (skipped when the maturin-built extension is unavailable):
  regenerate a small subset of workers through BOTH engines live and assert
  exact agreement, so the agreement is reproduced, not merely trusted from the
  committed file.

The engine extension lives in a separate venv (it is a Rust/PyO3 wheel not
installed in this repo's environment). The engine-backed test locates that
interpreter via ``AXIOM_ENGINE_PYTHON`` (default: the axiom-engine-67 worktree
venv) and skips if it cannot import ``axiom_rules_engine`` there.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

# The artifact builder lives in scripts/; the engine-backed tests import it
# lazily (see _load_builder) so the always-runnable committed-artifact tests
# below have NO import dependency on it or on populace_dynamics — they only read
# the JSON. This keeps schema/agreement checks collectable in a bare CI runner
# that has neither policyengine-us nor the engine wheel.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"

ARTIFACT_PATH = _REPO_ROOT / "runs" / "pia_cross_engine_v1.json"


def _load_builder():
    """Import the artifact builder module (adds scripts/ to the path)."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    import build_cross_engine_pia_artifact as builder

    return builder


PE_US = Path("~/PolicyEngine/policyengine-us").expanduser()
needs_pe_us = pytest.mark.skipif(
    not PE_US.is_dir()
    and "POPULACE_DYNAMICS_PE_US_DIR" not in __import__("os").environ,
    reason="policyengine-us not checked out and "
    "POPULACE_DYNAMICS_PE_US_DIR unset",
)


# ---------------------------------------------------------------------------
# Always-runnable: the committed artifact
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> dict:
    with ARTIFACT_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


class TestCommittedArtifact:
    """The checked-in artifact is exact and well-formed."""

    def test_artifact_exists(self) -> None:
        assert ARTIFACT_PATH.is_file(), (
            f"{ARTIFACT_PATH} missing; regenerate with "
            "scripts/build_cross_engine_pia_artifact.py"
        )

    def test_top_level_schema(self, artifact: dict) -> None:
        expected_keys = {
            "run",
            "description",
            "n_workers",
            "cohorts",
            "worker_design",
            "engine_revision",
            "pe_us_revision",
            "seed",
            "agreement_tolerance_dollars",
            "max_abs_diff_dollars",
            "n_exact_to_cent",
            "exercises",
            "notes",
            "per_cohort",
            "workers",
        }
        assert expected_keys <= set(artifact), (
            "artifact missing keys: "
            f"{sorted(expected_keys - set(artifact))}"
        )
        assert artifact["run"] == "pia_cross_engine_v1"
        assert isinstance(artifact["engine_revision"], str)
        assert artifact["engine_revision"] not in ("", "unknown")
        assert isinstance(artifact["pe_us_revision"], str)
        assert artifact["pe_us_revision"] not in ("", "unknown")

    def test_pins_both_cohorts(self, artifact: dict) -> None:
        cohorts = artifact["cohorts"]
        assert set(cohorts) == {"2020", "2026"}
        assert cohorts["2020"]["birth_year"] == 1958
        assert cohorts["2026"]["birth_year"] == 1964

    def test_exercises_the_full_chain(self, artifact: dict) -> None:
        joined = " ".join(artifact["exercises"]).lower()
        for needle in (
            "sum_top_n_over_periods",
            "person-varying n",
            "per-period",
            "invariant-bound",
            "bend point",
            "dime",
        ):
            assert (
                needle in joined
            ), f"artifact does not claim to exercise {needle}"

    def test_max_abs_diff_is_zero(self, artifact: dict) -> None:
        # The explained agreement bound: both sides are f64 driven from the same
        # parameter values, so PIAs agree exactly to the cent.
        assert artifact["max_abs_diff_dollars"] == 0.0

    def test_all_workers_present_and_exact(self, artifact: dict) -> None:
        rows = artifact["workers"]
        assert len(rows) == artifact["n_workers"]
        assert artifact["n_exact_to_cent"] == artifact["n_workers"]
        for r in rows:
            assert set(r) >= {
                "id",
                "cohort",
                "aime_oracle",
                "aime_engine",
                "pia_oracle",
                "pia_engine",
            }
            # PIA agreement to the cent.
            assert abs(r["pia_oracle"] - r["pia_engine"]) <= 0.005, r
            # AIME agreement (oracle floors to an int; engine returns the same).
            assert abs(r["aime_oracle"] - r["aime_engine"]) <= 1e-9, r

    def test_per_cohort_means_consistent(self, artifact: dict) -> None:
        for cohort, block in artifact["per_cohort"].items():
            crows = [
                r for r in artifact["workers"] if str(r["cohort"]) == cohort
            ]
            assert block["n"] == len(crows)
            mean_o = sum(r["pia_oracle"] for r in crows) / len(crows)
            mean_e = sum(r["pia_engine"] for r in crows) / len(crows)
            assert block["mean_pia_oracle"] == pytest.approx(mean_o, abs=1e-3)
            assert block["mean_pia_engine"] == pytest.approx(mean_e, abs=1e-3)
            # Engine and oracle cohort means coincide.
            assert block["mean_pia_oracle"] == pytest.approx(
                block["mean_pia_engine"], abs=1e-9
            )

    def test_all_three_pia_brackets_exercised(self, artifact: dict) -> None:
        """At least one worker lands in each of the 90/32/15 brackets, per
        cohort — otherwise the bend-point machinery is only partly tested."""
        bends = {"2020": (960.0, 5785.0), "2026": (1286.0, 7749.0)}
        for cohort, (b1, b2) in bends.items():
            aimes = [
                r["aime_oracle"]
                for r in artifact["workers"]
                if str(r["cohort"]) == cohort
            ]
            assert any(
                0 < a <= b1 for a in aimes
            ), f"cohort {cohort}: no worker in the 90%-only bracket"
            assert any(
                b1 < a <= b2 for a in aimes
            ), f"cohort {cohort}: no worker reaching the 32% bracket"
            assert any(
                a > b2 for a in aimes
            ), f"cohort {cohort}: no worker reaching the 15% bracket"


class TestDimeFloorEpsilonIsInert:
    """The engine uses ``floor(pia_raw * 10) / 10``; the oracle uses
    ``floor(amount * 10 + 1e-9) / 10``. For integer AIME and integer bend
    points the epsilon never changes the result, so the engine's plain floor
    agrees. Prove it exhaustively over the AIME range the artifact spans.
    """

    @needs_pe_us
    def test_epsilon_never_changes_result(self) -> None:
        params_mod = pytest.importorskip("populace_dynamics.ss.params")

        params = params_mod.load_ssa_parameters()
        f1, f2, f3 = params.pia_factors
        for elig_year in (2020, 2026):
            first, second = params.bend_points(elig_year)
            for aime in range(0, 15_001):
                amount = (
                    f1 * min(aime, first)
                    + f2 * max(0.0, min(aime, second) - first)
                    + f3 * max(0.0, aime - second)
                )
                with_eps = math.floor(amount * 10.0 + 1e-9) / 10.0
                without = math.floor(amount * 10.0) / 10.0
                assert with_eps == without, (
                    f"epsilon changes PIA at aime={aime}, "
                    f"elig={elig_year}: {without} vs {with_eps}"
                )


# ---------------------------------------------------------------------------
# Engine-backed: regenerate a small subset live and assert exact agreement
# ---------------------------------------------------------------------------


def _engine_python() -> str:
    import os

    return os.environ.get(
        "AXIOM_ENGINE_PYTHON",
        str(
            Path(
                "~/.claude-worktrees/axiom-engine-67/.venv/bin/python"
            ).expanduser()
        ),
    )


def _engine_available(engine_python: str) -> bool:
    if not Path(engine_python).exists():
        return False
    probe = (
        "import importlib.util as u; "
        "import sys; "
        "spec = u.find_spec('axiom_rules_engine'); "
        "from axiom_rules_engine.dense import NativeCompiledDenseProgram "
        "as N; "
        "sys.exit(0 if (spec and N is not None) else 1)"
    )
    try:
        return (
            subprocess.run(
                [engine_python, "-c", probe],
                capture_output=True,
                timeout=60,
            ).returncode
            == 0
        )
    except (subprocess.SubprocessError, OSError):
        return False


_ENGINE_PYTHON = _engine_python()
needs_engine = pytest.mark.skipif(
    not _engine_available(_ENGINE_PYTHON),
    reason=(
        "axiom_rules_engine extension unavailable at "
        f"{_ENGINE_PYTHON}; set AXIOM_ENGINE_PYTHON to a venv with the "
        "maturin-built wheel"
    ),
)


@needs_pe_us
@needs_engine
def test_subset_regenerates_and_agrees_exactly() -> None:
    """Take the first 10 workers of the artifact, recompute both sides live, and
    assert exact-to-cent agreement — a fresh proof, not the committed file."""
    params_mod = pytest.importorskip("populace_dynamics.ss.params")

    builder = _load_builder()
    params = params_mod.load_ssa_parameters()
    module_text = builder.build_engine_module(params)

    # Build the full deterministic worker sets, then take a 10-worker subset
    # spanning both cohorts (5 from each, including the edge shapes at index 0+).
    subset: dict[int, list] = {}
    for cohort, birth in builder.COHORTS.items():
        workers = builder._build_workers(cohort, birth, params)
        subset[cohort] = workers[:5]
    n_subset = sum(len(v) for v in subset.values())
    assert n_subset == 10

    engine_out = builder._engine_results(
        module_text, subset, params, _ENGINE_PYTHON
    )

    checked = 0
    for workers in subset.values():
        for w in workers:
            aime_o, pia_o = builder._oracle_results(w, params)
            eng = engine_out[w.worker_id]
            assert eng["aime"] == pytest.approx(aime_o, abs=1e-9), (
                w.worker_id,
                w.shape,
            )
            assert round(pia_o, 2) == round(eng["pia"], 2), (
                w.worker_id,
                w.shape,
                pia_o,
                eng["pia"],
            )
            checked += 1
    assert checked == 10


@needs_pe_us
@needs_engine
def test_committed_artifact_matches_live_engine_spotcheck() -> None:
    """The committed artifact's engine PIAs reproduce when the module is rebuilt
    and re-executed for a handful of workers — guards against a stale artifact.
    """
    params_mod = pytest.importorskip("populace_dynamics.ss.params")

    builder = _load_builder()
    with ARTIFACT_PATH.open(encoding="utf-8") as fh:
        artifact = json.load(fh)
    by_id = {r["id"]: r for r in artifact["workers"]}

    params = params_mod.load_ssa_parameters()
    if params.pe_us_revision != artifact["pe_us_revision"]:
        pytest.skip(
            f"policyengine-us checkout at {params.pe_us_revision} differs "
            f"from the artifact's pinned {artifact['pe_us_revision']}; "
            "point POPULACE_DYNAMICS_PE_US_DIR at the pinned revision "
            "to run the spotcheck"
        )
    module_text = builder.build_engine_module(params)
    subset: dict[int, list] = {}
    for cohort, birth in builder.COHORTS.items():
        workers = builder._build_workers(cohort, birth, params)
        # spot-check the two edge workers + one mid worker per cohort
        subset[cohort] = [workers[0], workers[1], workers[60]]

    engine_out = builder._engine_results(
        module_text, subset, params, _ENGINE_PYTHON
    )
    for workers in subset.values():
        for w in workers:
            assert w.worker_id in by_id, w.worker_id
            live = round(engine_out[w.worker_id]["pia"], 2)
            committed = by_id[w.worker_id]["pia_engine"]
            assert live == committed, (w.worker_id, live, committed)
