"""Synthetic proofs for the amendment-6 train-only rho selector."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reduce_m6_rhostar_selection as reducer  # noqa: E402
import select_m6_rhostar_train_only as selector  # noqa: E402

FINDINGS_PATH = (
    ROOT / "docs/analysis/m6_rhostar_train_only_selection_results.json"
)
FINDINGS_SHA256 = (
    "db7fe83547cad6a4ea477bac9f71f11279e9f21c8399b60d8f77a5ca14d463ff"
)


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


def test_findings_whole_file_pin_reproduces_exact_selection_trace():
    raw = FINDINGS_PATH.read_bytes()
    findings = json.loads(raw)
    canonical = (
        json.dumps(
            findings,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()

    assert raw == canonical
    assert hashlib.sha256(raw).hexdigest() == FINDINGS_SHA256
    assert findings["schema"] == reducer.FINDINGS_SCHEMA

    rho_grid = tuple(round(-0.80 + 0.05 * index, 2) for index in range(17))
    boundaries = (2006, 2008, 2010)
    draw_seeds = tuple(range(6200, 6220))
    first_half = tuple(range(6200, 6210))
    second_half = tuple(range(6210, 6220))
    selected_cells = (
        "earn_p10.prime",
        "earn_dlog_mean.prime",
        "earn_dlog_sd.older",
        "earn_mob_h1_diag",
        "earn_autocorr_lag2",
        "earn_zero_rate.older",
    )
    objective_cells = (
        "earn_p10.prime",
        "earn_dlog_mean.prime",
        "earn_mob_h1_diag",
        "earn_autocorr_lag2",
    )
    feasibility_cells = (
        "earn_dlog_sd.older",
        "earn_zero_rate.older",
    )
    block_seeds = {
        "all_20": draw_seeds,
        "first_10": first_half,
        "second_10": second_half,
    }
    labels = tuple(f"{rho:.2f}" for rho in rho_grid)

    protocol = findings["protocol"]
    assert protocol["fixed_q"] == 0.55
    assert protocol["rho_grid"] == list(rho_grid)
    assert protocol["pseudo_boundaries"] == list(boundaries)
    assert protocol["selection_draw_seeds"] == list(draw_seeds)
    assert protocol["fixed_halves"] == [
        list(first_half),
        list(second_half),
    ]
    assert protocol["selected_cells"] == list(selected_cells)
    assert protocol["objective_cells"] == list(objective_cells)
    assert protocol["feasibility_cells"] == list(feasibility_cells)
    assert set(findings["rungs"]) == set(labels)

    required_fences = {
        "no_candidate_1_or_candidate_2_artifact_read": True,
        "no_gate_score": True,
        "no_runs_write": True,
    }
    assert findings["fences"] == required_fences
    assert {
        name: protocol[name] for name in required_fences
    } == required_fences

    preflights = findings["preflights"]
    assert preflights["all_passed"] is True
    assert preflights["ladder_values_computed_before_pass"] is False
    assert preflights["failure_disposition"] == "STOP_AND_INVALIDATE_MECHANISM"
    records = preflights["records"]
    assert set(records) == {
        "rho_zero_candidate2_equivalence",
        "reset_law_discriminating_fixture",
        "endogenous_participation_feedback",
        "object_level_unchanged",
    }
    assert all(record["passed"] is True for record in records.values())

    stationary_keys = ("empty_to_0", "empty_to_1")
    pair_keys = ("0_to_0", "0_to_1", "1_to_0", "1_to_1")
    reset_keys = (
        "nonparticipation",
        "zero_earnings",
        "stream3_reentry",
        "support_exit",
    )

    def exact_count(value):
        assert type(value) is int
        assert value >= 0
        return value

    def chain_conservation(chain):
        assert set(chain["stationary_entries"]) == set(stationary_keys)
        assert set(chain["consecutive_pairs"]) == set(pair_keys)
        assert set(chain["resets"]) == set(reset_keys)
        entries = sum(
            exact_count(chain["stationary_entries"][key])
            for key in stationary_keys
        )
        pairs = sum(
            exact_count(chain["consecutive_pairs"][key]) for key in pair_keys
        )
        resets = sum(exact_count(chain["resets"][key]) for key in reset_keys)
        eligible = exact_count(chain["eligible_transitions"])
        even_call_count = exact_count(chain["even_calls"])
        decomposition = entries + pairs == eligible
        even_calls = eligible + resets == even_call_count
        return decomposition, even_calls

    equivalence = records["rho_zero_candidate2_equivalence"]
    assert equivalence["n_boundary_draw_equivalence_cells"] == 60
    required_equivalence_checks = (
        "person_period_keys_equal",
        "level_bytes_equal",
        "participation_states_equal",
        "all_six_moments_equal",
        "streams_1_5_final_states_equal",
        "truth_projection_support_equal",
        "chain_count_conservation",
        "passed",
    )
    for boundary in boundaries:
        boundary_equivalence = equivalence["boundaries"][str(boundary)]
        assert boundary_equivalence["passed"] is True
        rows = boundary_equivalence["per_draw"]
        assert [row["draw_seed"] for row in rows] == list(draw_seeds)
        for row in rows:
            assert all(
                row[name] is True for name in required_equivalence_checks
            )
            decomposition, even_calls = chain_conservation(
                row["transition_chain"]
            )
            assert decomposition is True
            assert even_calls is True

    def truth_value(record):
        raw = record.get("rate", record.get("value"))
        if raw is None or isinstance(raw, bool):
            return None
        value = float(raw)
        if not math.isfinite(value):
            return None
        metric = str(record.get("metric", "log_ratio"))
        if (metric == "log_ratio" or "rate" in record) and value <= 0:
            return None
        return value

    def registered_score(projected, truth_record):
        expected = truth_value(truth_record)
        if (
            projected is None
            or expected is None
            or isinstance(projected, bool)
        ):
            return None
        projected_value = float(projected)
        if not math.isfinite(projected_value):
            return None
        metric = str(truth_record.get("metric", "log_ratio"))
        if metric == "log_ratio" or "rate" in truth_record:
            if projected_value <= 0:
                return None
            return abs(math.log(projected_value / expected))
        return abs(projected_value - expected)

    def assert_ulps_equal(actual, expected, *, max_ulps=8):
        """Compare a rebuilt float across supported Python/libm versions."""
        if actual is None or expected is None:
            assert actual is expected
            return
        assert not isinstance(actual, bool)
        assert not isinstance(expected, bool)
        actual_float = float(actual)
        expected_float = float(expected)
        assert math.isfinite(actual_float)
        assert math.isfinite(expected_float)
        tolerance = max_ulps * max(
            math.ulp(actual_float),
            math.ulp(expected_float),
        )
        assert abs(actual_float - expected_float) <= tolerance

    def assert_numeric_mapping(actual, expected):
        assert set(actual) == set(expected)
        for key in expected:
            assert_ulps_equal(actual[key], expected[key])

    numeric_objectives = {}
    delete_one_totals = {}
    standardized_by_label = {}
    valid_by_label = {}
    invalid_reasons_by_label = {}

    for rho, label in zip(rho_grid, labels, strict=True):
        rung = findings["rungs"][label]
        assert float(rung["rho"]) == rho
        assert float(rung["fixed_q"]) == 0.55
        invalid_reasons = []
        by_block = {name: {} for name in block_seeds}
        standardized_by_label[label] = {}

        for boundary in boundaries:
            boundary_label = str(boundary)
            record = rung["boundaries"][boundary_label]
            standardized_by_label[label][boundary_label] = {}
            summary = record["per_draw_summary"]
            undefined = summary["undefined_draw_seeds_by_cell"]
            ranges = summary["moment_range"]
            truth = record["truth_moments"]
            assert set(truth) == set(selected_cells)
            assert all(
                truth_value(truth[cell]) is not None for cell in selected_cells
            )
            standardizers = record["standardizers"]
            assert set(standardizers) == set(selected_cells)
            assert all(
                not isinstance(standardizers[cell], bool)
                and math.isfinite(float(standardizers[cell]))
                and float(standardizers[cell]) > 0
                for cell in selected_cells
            )
            fit = record["fit"]
            assert fit["empty_pool_check_passed"] is True
            stable_pools = fit["stable_pools"]
            assert stable_pools["empty_pool_check_passed"] is True
            stable_pool_counts = stable_pools["counts_by_bin"]
            assert set(stable_pool_counts) == {
                str(index) for index in range(8)
            }
            assert all(
                exact_count(count) > 0 for count in stable_pool_counts.values()
            )
            assert set(fit["donor_pools"]) == {
                "pairs",
                "triples",
                "reentry",
            }
            assert all(
                exact_count(pool["n_rows"]) > 0
                for pool in fit["donor_pools"].values()
            )
            all_cells_defined = all(
                not undefined[cell]
                and ranges[cell]["min"] is not None
                and ranges[cell]["max"] is not None
                and np.isfinite(float(ranges[cell]["min"]))
                and np.isfinite(float(ranges[cell]["max"]))
                for cell in selected_cells
            )
            regenerated = {
                cell: (
                    not undefined[cell]
                    and ranges[cell]["min"] is not None
                    and ranges[cell]["max"] is not None
                    and float(ranges[cell]["min"])
                    != float(ranges[cell]["max"])
                )
                for cell in selected_cells
            }
            assert summary["n_draws"] == 20
            assert summary["draw_seeds"] == list(draw_seeds)
            assert summary["all_cells_defined"] is all_cells_defined
            assert summary["all_fresh_initial_state"] is True
            assert summary["distinct_annual_level_surfaces"] == 20
            assert summary["distinct_annual_participation_surfaces"] == 20
            assert record["regeneration"]["by_cell"] == regenerated
            assert record["regeneration"]["all_six_cells_regenerated"] is all(
                regenerated.values()
            )
            for cell, did_regenerate in regenerated.items():
                if not did_regenerate:
                    invalid_reasons.append(
                        f"boundary {boundary} cell {cell} not regenerated"
                    )

            support_sha = record["support"]["truth_support_ids_sha256"]
            assert summary["projected_support_ids_sha256"] == support_sha
            assert summary["truth_projection_support_equal_all_draws"] is True

            transitions = record["transition_pair_counts"]
            transition_rows = transitions["per_draw"]
            assert [row["draw_seed"] for row in transition_rows] == list(
                draw_seeds
            )
            all_rows_conserve = True
            for row in transition_rows:
                decomposition, even_calls = chain_conservation(row)
                assert row["eligible_decomposition_conserves"] is decomposition
                assert row["even_call_conservation_passed"] is even_calls
                all_rows_conserve &= decomposition and even_calls

            aggregate_chain = {
                "even_calls": sum(
                    exact_count(row["even_calls"]) for row in transition_rows
                ),
                "eligible_transitions": sum(
                    exact_count(row["eligible_transitions"])
                    for row in transition_rows
                ),
                "stationary_entries": {
                    key: sum(
                        exact_count(row["stationary_entries"][key])
                        for row in transition_rows
                    )
                    for key in stationary_keys
                },
                "consecutive_pairs": {
                    key: sum(
                        exact_count(row["consecutive_pairs"][key])
                        for row in transition_rows
                    )
                    for key in pair_keys
                },
                "resets": {
                    key: sum(
                        exact_count(row["resets"][key])
                        for row in transition_rows
                    )
                    for key in reset_keys
                },
            }
            decomposition, even_calls = chain_conservation(aggregate_chain)
            aggregate_chain["eligible_decomposition_conserves"] = decomposition
            aggregate_chain["even_call_conservation_passed"] = even_calls
            assert transitions["all_20"] == aggregate_chain
            aggregate_conserves = (
                all_rows_conserve and decomposition and even_calls
            )
            assert transitions["all_draws_conserve"] is aggregate_conserves
            if not aggregate_conserves:
                invalid_reasons.append(
                    f"boundary {boundary} transition counts do not conserve"
                )

            chain_payload = [
                {
                    key: value
                    for key, value in row.items()
                    if key != "draw_seed"
                }
                for row in transition_rows
            ]
            chain_bytes = json.dumps(
                chain_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            assert summary["transition_chain_records_sha256"] == (
                hashlib.sha256(chain_bytes).hexdigest()
            )

            for block, expected_seeds in block_seeds.items():
                aggregate = record["aggregates"][block]
                assert aggregate["draw_seeds"] == list(expected_seeds)
                projected = aggregate["projected_moments"]
                assert set(projected) == set(selected_cells)
                scores = {
                    cell: registered_score(projected[cell], truth[cell])
                    for cell in selected_cells
                }
                standardized = {
                    cell: (
                        None
                        if scores[cell] is None
                        else float(scores[cell]) / float(standardizers[cell])
                    )
                    for cell in selected_cells
                }
                contributions = {
                    cell: (
                        None
                        if standardized[cell] is None
                        else float(standardized[cell]) ** 2
                    )
                    for cell in objective_cells
                }
                assert_numeric_mapping(aggregate["scores"], scores)
                assert_numeric_mapping(
                    aggregate["standardized_scores"], standardized
                )
                assert_numeric_mapping(
                    aggregate["objective_contributions"], contributions
                )
                assert all(
                    contributions[cell] is not None for cell in objective_cells
                )
                for cell in objective_cells:
                    assert contributions[cell] == standardized[cell] ** 2
                objective = float(
                    math.fsum(
                        float(contributions[cell]) for cell in objective_cells
                    )
                )
                assert_ulps_equal(aggregate["objective"], objective)
                by_block[block][boundary_label] = objective
                standardized_by_label[label][boundary_label][
                    block
                ] = standardized

        numeric_objectives[label] = {}
        for block, expected_seeds in block_seeds.items():
            objective = rung["objectives"][block]
            assert objective["draw_seeds"] == list(expected_seeds)
            assert_numeric_mapping(objective["by_boundary"], by_block[block])
            total = float(
                math.fsum(
                    by_block[block][str(boundary)] for boundary in boundaries
                )
            )
            assert_ulps_equal(objective["total"], total)
            numeric_objectives[label][block] = total

        if any(
            numeric_objectives[label][block] is None for block in block_seeds
        ):
            invalid_reasons.append(
                "one or more fixed-block objectives undefined"
            )
        valid_by_label[label] = not invalid_reasons
        invalid_reasons_by_label[label] = invalid_reasons

        delete_records = rung["objectives"]["delete_one"]
        assert [
            record["omitted_draw_seed"] for record in delete_records
        ] == list(draw_seeds)
        totals = []
        for omitted, record in zip(
            draw_seeds,
            delete_records,
            strict=True,
        ):
            expected_seeds = [seed for seed in draw_seeds if seed != omitted]
            assert record["draw_seeds"] == expected_seeds
            total = float(
                math.fsum(
                    float(record["by_boundary"][str(boundary)])
                    for boundary in boundaries
                )
            )
            assert_ulps_equal(record["total"], total)
            totals.append(total)
        delete_one_totals[label] = totals

    baseline_label = "0.00"
    assert valid_by_label[baseline_label] is True
    feasible_by_label = {}
    guards_by_label = {}
    improvements_by_label = {}
    retained_by_label = {}

    for rho, label in zip(rho_grid, labels, strict=True):
        feasible = valid_by_label[label]
        guards = {}
        for boundary in boundaries:
            boundary_label = str(boundary)
            candidate_scores = standardized_by_label[label][boundary_label][
                "all_20"
            ]
            baseline_scores = standardized_by_label[baseline_label][
                boundary_label
            ]["all_20"]
            cell_guards = {}
            for cell in feasibility_cells:
                candidate = float(candidate_scores[cell])
                baseline = float(baseline_scores[cell])
                limit = baseline + 1.0
                passed = candidate <= limit
                cell_guards[cell] = {
                    "candidate_standardized_score": candidate,
                    "rho0_standardized_score": baseline,
                    "limit": limit,
                    "passed": passed,
                }
                feasible &= passed
            guards[boundary_label] = cell_guards
        guards_by_label[label] = guards
        feasible_by_label[label] = feasible

        if rho == 0.0:
            improvements = {
                "all_20": True,
                "first_10": True,
                "second_10": True,
            }
            retained = True
        else:
            improvements = {
                block: (
                    numeric_objectives[label][block]
                    < numeric_objectives[baseline_label][block]
                )
                for block in block_seeds
            }
            retained = feasible and all(improvements.values())
        improvements_by_label[label] = improvements
        retained_by_label[label] = retained

    retained_labels = [label for label in labels if retained_by_label[label]]
    rho_min_label = min(
        retained_labels,
        key=lambda label: (
            numeric_objectives[label]["all_20"],
            abs(float(label)),
        ),
    )
    deletes = np.asarray(
        delete_one_totals[rho_min_label],
        dtype=np.float64,
    )
    delete_mean = float(deletes.mean())
    standard_error = float(
        np.sqrt((19.0 / 20.0) * np.sum((deletes - delete_mean) ** 2))
    )
    rho_min_objective = numeric_objectives[rho_min_label]["all_20"]
    cutoff = rho_min_objective + standard_error
    within_one_se = [
        label
        for label in retained_labels
        if numeric_objectives[label]["all_20"] <= cutoff
    ]
    selected_label = min(
        within_one_se,
        key=lambda label: abs(float(label)),
    )
    selected_rho = float(selected_label)
    disposition = (
        "DESIGNED_PAUSE" if selected_rho == 0.0 else "LOCK_ADDENDUM_ELIGIBLE"
    )

    weak_retained_labels = [baseline_label]
    for rho, label in zip(rho_grid, labels, strict=True):
        if rho == 0.0 or not feasible_by_label[label]:
            continue
        weak_improves = (
            numeric_objectives[label]["all_20"]
            <= numeric_objectives[baseline_label]["all_20"]
            and numeric_objectives[label]["first_10"]
            < numeric_objectives[baseline_label]["first_10"]
            and numeric_objectives[label]["second_10"]
            < numeric_objectives[baseline_label]["second_10"]
        )
        if weak_improves:
            weak_retained_labels.append(label)
    weak_min_label = min(
        weak_retained_labels,
        key=lambda label: (
            numeric_objectives[label]["all_20"],
            abs(float(label)),
        ),
    )
    weak_deletes = np.asarray(
        delete_one_totals[weak_min_label],
        dtype=np.float64,
    )
    weak_standard_error = float(
        np.sqrt(
            (19.0 / 20.0) * np.sum((weak_deletes - weak_deletes.mean()) ** 2)
        )
    )
    weak_cutoff = (
        numeric_objectives[weak_min_label]["all_20"] + weak_standard_error
    )
    weak_within_one_se = [
        label
        for label in weak_retained_labels
        if numeric_objectives[label]["all_20"] <= weak_cutoff
    ]
    weak_selected_label = min(
        weak_within_one_se,
        key=lambda label: abs(float(label)),
    )

    rung_trace = {
        label: {
            "objectives": numeric_objectives[label],
            "valid": valid_by_label[label],
            "feasible": feasible_by_label[label],
            "strict_improvement_vs_rho0": improvements_by_label[label],
            "retained": retained_by_label[label],
        }
        for label in labels
    }
    tie_break_trace = {
        "rungs": rung_trace,
        "retained_pool": [float(label) for label in retained_labels],
        "argmin_order": [
            float(label)
            for label in sorted(
                retained_labels,
                key=lambda label: (
                    numeric_objectives[label]["all_20"],
                    abs(float(label)),
                ),
            )
        ],
        "rho_min": float(rho_min_label),
        "rho_min_delete_one": delete_one_totals[rho_min_label],
        "rho_min_delete_one_mean": delete_mean,
        "rho_min_jackknife_standard_error": standard_error,
        "one_se_cutoff": cutoff,
        "within_one_se": [float(label) for label in within_one_se],
        "closest_to_zero_order": [
            float(label)
            for label in sorted(
                within_one_se,
                key=lambda label: abs(float(label)),
            )
        ],
        "selected_rho": selected_rho,
        "disposition": disposition,
        "weak_counterfactual": {
            "retained_pool": [float(label) for label in weak_retained_labels],
            "rho_min": float(weak_min_label),
            "one_se_cutoff": weak_cutoff,
            "within_one_se": [float(label) for label in weak_within_one_se],
            "selected_rho": float(weak_selected_label),
        },
    }
    trace_message = json.dumps(
        tie_break_trace,
        indent=2,
        sort_keys=True,
    )

    for label in labels:
        rung = findings["rungs"][label]
        assert (
            rung["invalid_reasons"] == invalid_reasons_by_label[label]
        ), trace_message
        assert rung["valid"] is valid_by_label[label], trace_message
        published_guards = rung["feasibility_guards"]
        expected_guards = guards_by_label[label]
        assert set(published_guards) == set(expected_guards), trace_message
        for boundary_label in expected_guards:
            assert set(published_guards[boundary_label]) == set(
                expected_guards[boundary_label]
            ), trace_message
            for cell in expected_guards[boundary_label]:
                published_guard = published_guards[boundary_label][cell]
                expected_guard = expected_guards[boundary_label][cell]
                assert set(published_guard) == set(
                    expected_guard
                ), trace_message
                for field in (
                    "candidate_standardized_score",
                    "rho0_standardized_score",
                    "limit",
                ):
                    assert_ulps_equal(
                        published_guard[field],
                        expected_guard[field],
                    )
                assert published_guard["passed"] is expected_guard["passed"]
        assert rung["feasible"] is feasible_by_label[label], trace_message
        assert (
            rung["strict_improvement_vs_rho0"] == improvements_by_label[label]
        ), trace_message
        assert (
            rung["retained_for_one_se"] is retained_by_label[label]
        ), trace_message
        assert rung["within_one_se_cutoff"] is (
            label in within_one_se
        ), trace_message
        assert rung["selected"] is (label == selected_label), trace_message
        assert (
            rung["train_f1_analog_disclosure"]["adds_no_selection_criterion"]
            is True
        )

    expected_selector = {
        "baseline_rho_retained": True,
        "effective_search_size": {
            "grid_rungs": len(labels),
            "valid_rungs": sum(valid_by_label.values()),
            "feasible_rungs_including_rho0": sum(feasible_by_label.values()),
            "retained_rungs_including_rho0": len(retained_labels),
            "retained_nonzero_rungs": sum(
                float(label) != 0.0 for label in retained_labels
            ),
        },
        "retained_rho": [float(label) for label in retained_labels],
        "rho_min": float(rho_min_label),
        "rho_min_objective": rho_min_objective,
        "rho_min_delete_one_mean": delete_mean,
        "rho_min_jackknife_standard_error": standard_error,
        "one_se_cutoff": cutoff,
        "rho_within_one_se": [float(label) for label in within_one_se],
        "selected_rho": selected_rho,
        "selected_rho_label": selected_label,
        "disposition": disposition,
        "closest_to_zero_tie_break_applied": True,
        "strict_vs_weak_improvement_outcome_invariant": (
            weak_selected_label == selected_label
        ),
        "weak_improvement_counterfactual": {
            "weakened_comparison": ("all_20 only; fixed halves remain strict"),
            "retained_rho": [float(label) for label in weak_retained_labels],
            "rho_min": float(weak_min_label),
            "jackknife_standard_error": weak_standard_error,
            "one_se_cutoff": weak_cutoff,
            "selected_rho": float(weak_selected_label),
        },
    }
    published_selector = findings["selector"]
    assert set(published_selector) == set(expected_selector), trace_message
    for field in (
        "baseline_rho_retained",
        "effective_search_size",
        "retained_rho",
        "rho_min",
        "rho_within_one_se",
        "selected_rho",
        "selected_rho_label",
        "disposition",
        "closest_to_zero_tie_break_applied",
        "strict_vs_weak_improvement_outcome_invariant",
    ):
        assert (
            published_selector[field] == expected_selector[field]
        ), trace_message
    for field in (
        "rho_min_objective",
        "rho_min_delete_one_mean",
        "rho_min_jackknife_standard_error",
        "one_se_cutoff",
    ):
        assert_ulps_equal(
            published_selector[field],
            expected_selector[field],
        )
    published_weak = published_selector["weak_improvement_counterfactual"]
    expected_weak = expected_selector["weak_improvement_counterfactual"]
    assert set(published_weak) == set(expected_weak), trace_message
    for field in (
        "weakened_comparison",
        "retained_rho",
        "rho_min",
        "selected_rho",
    ):
        assert published_weak[field] == expected_weak[field], trace_message
    for field in ("jackknife_standard_error", "one_se_cutoff"):
        assert_ulps_equal(published_weak[field], expected_weak[field])
