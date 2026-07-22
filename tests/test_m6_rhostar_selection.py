"""Synthetic proofs for the amendment-6 train-only rho selector."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reduce_m6_rhostar_selection as reducer  # noqa: E402
import select_m6_rhostar_train_only as selector  # noqa: E402


def _transition_chain() -> dict[str, object]:
    return {
        "even_calls": 10,
        "eligible_transitions": 6,
        "stationary_entries": {"empty_to_0": 1, "empty_to_1": 1},
        "consecutive_pairs": {
            "0_to_0": 1,
            "0_to_1": 1,
            "1_to_0": 1,
            "1_to_1": 1,
        },
        "resets": {
            "nonparticipation": 1,
            "zero_earnings": 1,
            "stream3_reentry": 1,
            "support_exit": 1,
        },
        "eligible_decomposition_conserves": True,
        "even_call_conservation_passed": True,
    }


def _boundary_record(
    rho: float, boundary: int, *, mode: str
) -> dict[str, object]:
    truth = {
        cell: {"value": 10.0, "metric": "abs_gap_corr", "n_obs": 100}
        for cell in selector.SELECTED_CELLS
    }
    support_hash = f"support-{boundary}"
    draws = []
    for offset, seed in enumerate(selector.SELECTION_DRAW_SEEDS):
        alternating = -0.01 if offset % 2 == 0 else 0.01
        if mode == "standard" and rho == -0.15:
            objective_error = (
                2.2 + alternating if offset < 10 else 1.2 + alternating
            )
            guard_error = 0.1 + alternating
        elif mode == "standard" and rho == -0.20:
            objective_error = 1.2 + alternating
            guard_error = 1.21 + alternating / 10.0
        elif mode == "standard" and rho == -0.10:
            objective_error = 0.0 if offset % 2 == 0 else 2.0
            guard_error = 0.1 + alternating
        elif mode == "standard" and rho == -0.05:
            objective_error = 1.2 + alternating
            guard_error = 0.1 + alternating
        elif mode == "tie" and rho in (-0.10, -0.05):
            objective_error = 1.2 + alternating
            guard_error = 0.1 + alternating
        else:
            objective_error = 2.0 + alternating
            guard_error = 0.2 + alternating
        values = {
            cell: 10.0
            + (
                objective_error
                if cell in selector.OBJECTIVE_CELLS
                else guard_error
            )
            for cell in selector.SELECTED_CELLS
        }
        draws.append(
            {
                "draw_seed": seed,
                "moment_values": values,
                "annual_level_sha256": f"level-{rho:.2f}-{boundary}-{seed}",
                "annual_participation_sha256": (
                    f"participation-{rho:.2f}-{boundary}-{seed}"
                ),
                "support_ids_sha256": support_hash,
                "transition_chain": _transition_chain(),
                "fresh_initial_state": True,
            }
        )
    return {
        "fit": {"signature": f"fit-{rho:.2f}-{boundary}"},
        "floor": {"signature": f"floor-{rho:.2f}-{boundary}"},
        "truth_moments": truth,
        "standardizers": {cell: 1.0 for cell in selector.SELECTED_CELLS},
        "support": {"truth_support_ids_sha256": support_hash},
        "rng_registry_sha256": f"rng-{boundary}",
        "per_draw": draws,
    }


def _selector_payload(mode: str = "standard") -> dict[str, object]:
    payload: dict[str, object] = {"rungs": {}}
    rungs = payload["rungs"]
    assert isinstance(rungs, dict)
    for rho in selector.RHO_GRID:
        rungs[f"{rho:.2f}"] = {
            "rho": rho,
            "fixed_q": selector.FIXED_Q,
            "boundaries": {
                str(boundary): _boundary_record(rho, boundary, mode=mode)
                for boundary in selector.PSEUDO_BOUNDARIES
            },
        }
    selector._finalize_selector(payload)
    return payload


def _passed_preflights() -> dict[str, object]:
    by_boundary = {}
    for boundary in selector.PSEUDO_BOUNDARIES:
        by_boundary[str(boundary)] = {
            "passed": True,
            "per_draw": [
                {
                    "draw_seed": seed,
                    "person_period_keys_equal": True,
                    "level_bytes_equal": True,
                    "participation_states_equal": True,
                    "all_six_moments_equal": True,
                    "streams_1_5_final_states_equal": True,
                    "truth_projection_support_equal": True,
                    "chain_count_conservation": True,
                    "passed": True,
                }
                for seed in selector.SELECTION_DRAW_SEEDS
            ],
        }
    return {
        "all_passed": True,
        "ladder_values_computed_before_pass": False,
        "failure_disposition": "STOP_AND_INVALIDATE_MECHANISM",
        "records": {
            "rho_zero_candidate2_equivalence": {
                "passed": True,
                "n_boundary_draw_equivalence_cells": 60,
                "boundaries": by_boundary,
            },
            "reset_law_discriminating_fixture": {"passed": True},
            "endogenous_participation_feedback": {"passed": True},
            "object_level_unchanged": {"passed": True},
        },
    }


def _raw_reducer_fixture() -> bytes:
    payload = _selector_payload()
    payload.update(
        {
            "schema": reducer.RAW_SCHEMA,
            "protocol": {
                "fixed_q": selector.FIXED_Q,
                "rho_grid": list(selector.RHO_GRID),
                "pseudo_boundaries": list(selector.PSEUDO_BOUNDARIES),
                "fit_seed": selector.FIT_SEED,
                "selection_draw_seeds": list(selector.SELECTION_DRAW_SEEDS),
                "fixed_halves": [
                    list(selector.FIRST_HALF_DRAW_SEEDS),
                    list(selector.SECOND_HALF_DRAW_SEEDS),
                ],
                "selected_cells": list(selector.SELECTED_CELLS),
                "objective_cells": list(selector.OBJECTIVE_CELLS),
                "feasibility_cells": list(selector.FEASIBILITY_CELLS),
                "substream_codes": selector.SUBSTREAM_CODES,
                "fresh_complete_qrf_refit_per_rho_boundary": True,
                "common_random_numbers_across_rungs_at_fixed_seed": True,
                "rho_zero_disposition": "DESIGNED_PAUSE",
                "no_candidate_1_or_candidate_2_artifact_read": True,
                "no_gate_score": True,
                "no_runs_write": True,
            },
            "preflights": _passed_preflights(),
            "fences": {
                "no_candidate_1_or_candidate_2_artifact_read": True,
                "no_gate_score": True,
                "no_runs_write": True,
            },
        }
    )
    return (json.dumps(payload, sort_keys=True) + "\n").encode()


def test_protocol_grid_seeds_cells_substreams_and_environment_are_exact():
    assert selector.FIXED_Q == 0.55
    assert selector.RHO_GRID == tuple(
        round(-0.80 + 0.05 * index, 2) for index in range(17)
    )
    assert selector.PSEUDO_BOUNDARIES == (2006, 2008, 2010)
    assert selector.FIT_SEED == 5200
    assert selector.SELECTION_DRAW_SEEDS == tuple(range(6200, 6220))
    assert selector.FIRST_HALF_DRAW_SEEDS == tuple(range(6200, 6210))
    assert selector.SECOND_HALF_DRAW_SEEDS == tuple(range(6210, 6220))
    assert selector.SUBSTREAM_CODES == {
        "gate": 1,
        "donor-draw": 2,
        "re-entry-draw": 3,
        "memory-refresh-gate": 4,
        "memory-refresh-rank": 5,
    }
    assert selector.EXPECTED_THREAD_ENVIRONMENT == {
        key: "8" for key in selector.THREAD_ENVIRONMENT_KEYS
    }
    assert len(selector.THREAD_ENVIRONMENT_KEYS) == 8
    assert selector.EXPECTED_RUNTIME_VERSIONS["python"] == "3.14.4"
    assert selector.EXPECTED_RUNTIME_VERSIONS["scikit_learn"] == "1.8.0"
    assert selector.EXPECTED_RUNTIME_VERSIONS["policyengine_us"] == "1.752.2"
    assert selector.EXPECTED_POPULACE_HEAD == (
        "ee8f7fc139271de5d4e448549c35e8c5eb992534"
    )
    assert selector.EXPECTED_POPULACE_FIT_TREE == (
        "5c866378fdf5906b7a61da9977b8d028d1d36e9f"
    )
    assert selector.EXPECTED_POPULACE_FRAME_TREE == (
        "7cfb9ee78beb74911963913f202a4471aae2f52b"
    )


def test_only_frozen_q_ledger_and_docs_analysis_outputs_are_named():
    assert selector.QSTAR_LEDGER_PATH.name == (
        "m6_qstar_train_only_selection_results" + "." + "json"
    )
    assert selector.QSTAR_LEDGER_SHA256 == (
        "d25b8e159384f8a84ed7f2218d863ca63d96fc9cb244536853b0a1f05c4025bb"
    )
    forbidden = "run" + "s/" + "gate_m6_candidate2_v1" + "." + "json"
    source = Path(selector.__file__).read_text(encoding="utf-8")
    assert forbidden not in source
    for path in (
        selector.PROGRESS_PATH,
        selector.FINDINGS_PATH,
        selector.FINDINGS_TMP_PATH,
    ):
        relative = path.relative_to(ROOT)
        assert relative.parts[:2] == ("docs", "analysis")


def test_executable_reset_and_feedback_preflights_discriminate():
    reset = selector._reset_law_preflight()
    assert reset["passed"]
    assert len(reset["histories"]) == 2
    for history in reset["histories"]:
        assert history["gap_level"] == 0.0
        assert history["gap_resets_to"] is None
        assert history["reentry_level"] > 0.0
        assert history["reentry_state"] is None
        assert history["post_gap_threshold"] == selector.FIXED_Q
        assert (
            history["post_gap_refresh"] != history["remembered_state_refresh"]
        )
    feedback = selector._participation_feedback_preflight()
    assert feedback["passed"]
    assert all(feedback["conditions"].values())


def test_failed_preflight_stops_before_any_ladder_fit(monkeypatch):
    called = False

    def forbidden_fit(*args, **kwargs):
        nonlocal called
        del args, kwargs
        called = True
        raise AssertionError("a ladder fit started")

    monkeypatch.setattr(selector, "_fit_boundary", forbidden_fit)
    with pytest.raises(RuntimeError, match="preflight pass gate failed"):
        selector._run_ladder(None, None, None, {"all_passed": False})
    assert not called


def test_selector_enforces_halves_guards_jackknife_and_closest_zero_one_se():
    payload = _selector_payload()
    result = payload["selector"]
    rungs = payload["rungs"]
    assert result["rho_min"] == -0.10
    assert result["selected_rho"] == -0.05
    assert rungs["-0.15"]["strict_improvement_vs_rho0"] == {
        "all_20": True,
        "first_10": False,
        "second_10": True,
    }
    assert not rungs["-0.15"]["retained_for_one_se"]
    assert not rungs["-0.20"]["feasible"]
    assert rungs["-0.10"]["retained_for_one_se"]
    assert rungs["-0.05"]["retained_for_one_se"]
    assert len(rungs["-0.10"]["objectives"]["delete_one"]) == 20
    deletes = np.asarray(
        [
            record["total"]
            for record in rungs["-0.10"]["objectives"]["delete_one"]
        ]
    )
    expected_se = np.sqrt(
        (19.0 / 20.0) * np.sum((deletes - deletes.mean()) ** 2)
    )
    assert result["rho_min_jackknife_standard_error"] == pytest.approx(
        expected_se
    )
    assert result["strict_vs_weak_improvement_outcome_invariant"]
    assert rungs["-0.05"]["train_f1_analog_disclosure"][
        "adds_no_selection_criterion"
    ]
    pair_counts = rungs["-0.05"]["boundaries"]["2006"][
        "transition_pair_counts"
    ]
    assert pair_counts["all_draws_conserve"]
    assert len(pair_counts["per_draw"]) == 20


def test_exact_argmin_tie_resolves_toward_zero():
    payload = _selector_payload("tie")
    assert (
        payload["rungs"]["-0.10"]["objectives"]["all_20"]["total"]
        == payload["rungs"]["-0.05"]["objectives"]["all_20"]["total"]
    )
    assert payload["selector"]["rho_min"] == -0.05
    assert payload["selector"]["selected_rho"] == -0.05


def test_zero_selection_is_published_as_designed_pause():
    payload = _selector_payload("pause")
    assert payload["selector"]["retained_rho"] == [0.0]
    assert payload["selector"]["selected_rho"] == 0.0
    assert payload["selector"]["disposition"] == "DESIGNED_PAUSE"


def test_reducer_hashes_stdout_and_retains_counts_f1_and_fences():
    raw = _raw_reducer_fixture()
    reduced = reducer.reduce(raw)
    assert reduced["schema"] == reducer.FINDINGS_SCHEMA
    assert reduced["full_stdout_sha256"] == hashlib.sha256(raw).hexdigest()
    record = reduced["rungs"]["-0.05"]["boundaries"]["2006"]
    assert "per_draw" not in record
    assert record["per_draw_summary"]["n_draws"] == 20
    assert record["per_draw_summary"][
        "truth_projection_support_equal_all_draws"
    ]
    assert len(record["transition_pair_counts"]["per_draw"]) == 20
    assert reduced["rungs"]["-0.05"]["train_f1_analog_disclosure"]
    assert reduced["fences"] == {
        "no_candidate_1_or_candidate_2_artifact_read": True,
        "no_gate_score": True,
        "no_runs_write": True,
    }


def test_reducer_rejects_preflight_support_and_transition_drift():
    failed = json.loads(_raw_reducer_fixture())
    failed["preflights"]["all_passed"] = False
    with pytest.raises(ValueError, match="passed preflight gate"):
        reducer.reduce(json.dumps(failed).encode())

    support = json.loads(_raw_reducer_fixture())
    support["rungs"]["-0.05"]["boundaries"]["2008"]["per_draw"][3][
        "support_ids_sha256"
    ] = "wrong-support"
    with pytest.raises(ValueError, match="projected support hashes"):
        reducer.reduce(json.dumps(support).encode())

    transition = json.loads(_raw_reducer_fixture())
    transition["rungs"]["-0.05"]["boundaries"]["2006"][
        "transition_pair_counts"
    ]["all_draws_conserve"] = False
    with pytest.raises(ValueError, match="does not conserve"):
        reducer.reduce(json.dumps(transition).encode())

    fence = json.loads(_raw_reducer_fixture())
    fence["fences"]["no_gate_score"] = False
    with pytest.raises(ValueError, match="fence fields"):
        reducer.reduce(json.dumps(fence).encode())
