"""Hash-bound one-shot orchestration for the M6 candidate-2 sibling.

The candidate-1 runner is frozen evidence.  This module reuses its phase
implementations through explicit seams, but owns candidate identity, artifact
assembly, preregistration comparisons, and the exclusive artifact pair.
Importing it never reads repository data, invokes Git, loads PSID, or writes a
file.

The family registry exposes a separately ratified selected-C candidate-2
sibling alongside the immutable prefreeze record.  The run-start guard binds
only the production sibling and never mutates the prefreeze spec or substitutes
candidate 16.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from populace_dynamics.contract import ContractRef, environment_block
from populace_dynamics.engine import candidates as engine_candidates
from populace_dynamics.engine.candidates import (
    CandidateSpec as EngineCandidateSpec,
)
from populace_dynamics.engine.refit import (
    BOUNDARY_YEAR,
    M6RefitBundle,
    M6RefitInputs,
    refit_family_transitions,
    refit_first_marriage_modifier,
    refit_m6_components,
)
from populace_dynamics.harness import m6_runner
from populace_dynamics.harness.m6_inputs import M6HarnessInputs
from populace_dynamics.harness.m6_runner import (
    M6RefitPhase,
    M6ResolvedContract,
    M6SeedRun,
)
from populace_dynamics.models.family_transitions import (
    registry as family_candidates,
)
from populace_dynamics.models.family_transitions.components.first_marriage_support_aware import (
    FirstMarriagePreflightAbort,
    recompute_support_aware_first_marriage_checksums,
    validate_support_aware_first_marriage_fit,
)
from populace_dynamics.models.family_transitions.registry import (
    CandidateSpec as FamilyCandidateSpec,
)

SCHEMA_VERSION = "gate_m6_candidate2.v1"
DEFAULT_OUTPUT = Path("runs/gate_m6_candidate2_v1.json")
CANDIDATE1_OUTPUT = Path("runs/gate_m6_candidate1_v1.json")
CANDIDATE1_REGISTRATION_ID = "4990833378"
PROJECTION_END_YEAR = m6_runner.PROJECTION_END_YEAR
PHASE_ORDER = (
    "fit_only",
    "first_marriage_preflight",
    "fit_postrepair_incumbent_comparator",
    "load_full_inputs",
    "materialize_refit",
    "materialize_postrepair_incumbent_comparator",
    "first_marriage_conformance_reconciliation",
    "preflight_1",
    "preflight_2",
    "project_and_score_candidate2",
    "project_and_score_postrepair_incumbent_diagnostic",
    "first_marriage_estimator_decomposition",
    "gate_aggregate_and_acceptance",
    "report_only",
    "assemble_and_write",
)

MUST_NOT_REGRESS_TOLERANCES: Mapping[str, float] = MappingProxyType(
    {
        "divorce.18-44": 0.379,
        "incidence.20-66": 0.404,
        "recovery.20-66": 0.314,
        "earn_dlog_sd.older": 0.269,
        "earn_zero_rate.older": 0.163,
    }
)
MUST_NOT_REGRESS_REQUIRED_SEEDS = 4
MUST_NOT_REGRESS_SEEDS = (0, 1, 2, 3, 4)

DESIGN_BLOB_PATHS = (
    "docs/design/m6_projection_engine.md",
    "docs/design/m6_candidate2_program.md",
)
CONFORMANCE_EVIDENCE_PATH = (
    "docs/analysis/m6_candidate2_entry_dissolved_conformance.json"
)
CONFORMANCE_SOURCE_BINDING_KEYS = (
    "postrepair_marital_source_sha256",
    "transitions_source_sha256",
    "panel_builder_tests_sha256",
    "candidate2_program_sha256",
)
CONFORMANCE_PAIRED_INVARIANTS = (
    "formation_event_identity_exact_except_label_metadata",
    "married_person_years_exact",
    "nonformation_events_exact",
    "pooled_event_count_conserved",
    "pooled_event_f6_weight_conserved",
    "postrepair_carrier_state_ysd_survival",
    "relabel_transfer_count_zero_sum",
    "relabel_transfer_f6_weight_zero_sum",
    "truth_support_key_identity",
)
CONFORMANCE_AGGREGATE_METRICS = (
    "event_count",
    "f6_weight",
    "first_marriage_count_delta",
    "first_marriage_f6_weight_delta",
    "remarriage_count_delta",
    "remarriage_f6_weight_delta",
)
CONFORMANCE_EVALUATION_YEARS: Mapping[int, tuple[int, ...]] = MappingProxyType(
    {
        2006: (2007, 2008, 2009, 2010),
        2008: (2009, 2010, 2011, 2012),
        2010: (2011, 2012, 2013),
    }
)
FIRST_MARRIAGE_GATE_CELL = "first_marriage.18-29|female"
LANDED_REPAIR_COMMIT = "c16cb9d563bd573ce2b537b19e403fbddec3cba6"
SELECTION_EVIDENCE_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "qstar_q_grid": (
            "docs/analysis/m6_qstar_train_only_selection_results.json"
        ),
        "first_marriage_c": (
            "docs/analysis/m6_first_marriage_c_selection_findings.json"
        ),
        "remarriage_no_op": (
            "docs/analysis/m6_remarriage_round3_selection_results.json"
        ),
    }
)

# These are the only preregistered surfaces the runner hard-compares.  Source
# commit, design-document blobs, dependency/runtime identity, and environment
# sidecar are intentionally record-and-publish values.
PREREGISTERED_VALUES: Mapping[str, str] = MappingProxyType(
    {
        "design_commit": ("64ec6c04bf8f3e6a6f4fcaf71c086a128056a86f"),
        "floor_run_sha256": (
            "4cd2d01a9fd76064e701ae77a9226208cbae94d743f76f502d3d0a5f657d9523"
        ),
        "candidate_spec.family_transitions": (
            "734a5b04f347c5d4904bbc6d5ab9a1c2876272d35284eedd2f450518acf1cec5"
        ),
        "candidate_spec.engine_operations": (
            "8fbfcf4130fd9051aa063061bf7b2d8514773fc6a900c900caab18717ad8e14c"
        ),
        "selection_evidence.qstar_q_grid": (
            "d25b8e159384f8a84ed7f2218d863ca63d96fc9cb244536853b0a1f05c4025bb"
        ),
        "selection_evidence.first_marriage_c": (
            "4ff69bd87a5dc1580128ccc33844cf5c573a6d69437d626f622b9f1fe378b14d"
        ),
        "selection_evidence.remarriage_no_op": (
            "28e635fdd12d090e23066ea836b853af7c7f1760fc80fc4b214b9d529f93bfd0"
        ),
    }
)
PREREGISTERED_COMPARISON_KEYS = tuple(PREREGISTERED_VALUES)


@dataclass(frozen=True)
class M6Candidate2Identity:
    """The two candidate-2 registry specs and their registry-derived number."""

    number: int
    family_spec: FamilyCandidateSpec
    engine_spec: EngineCandidateSpec
    family_spec_sha256: str
    engine_spec_sha256: str
    first_marriage_params: Mapping[str, Any]


@dataclass(frozen=True)
class M6SourceIdentity:
    """Clean checkout identity captured before any repository data read."""

    commit_sha: str
    dirty: bool = False


@dataclass(frozen=True)
class M6Candidate2Bindings:
    """All compared and record-only identities published by one run."""

    source: M6SourceIdentity
    design_commit: str
    design_blob_sha256s: Mapping[str, str]
    floor_run: str
    floor_run_sha256: str
    candidate_number: int
    candidate_spec_ids: Mapping[str, str]
    candidate_spec_sha256s: Mapping[str, str]
    candidate_specs: Mapping[str, Mapping[str, Any]]
    selection_evidence_sha256s: Mapping[str, str]
    first_marriage_selection_ledger: Mapping[str, Any]
    entry_dissolved_conformance_sha256: str
    entry_dissolved_conformance: Mapping[str, Any]
    dependency_sha256: str
    sorted_pip_freeze: tuple[str, ...]
    runtime_identity: Mapping[str, str]
    environment_sidecar_path: str
    environment_sidecar_sha256: str
    contract_ref: Mapping[str, str]


@dataclass(frozen=True)
class M6Candidate2Preparation:
    """Pre-data guard result carried across the explicit input factory."""

    registration_id: str
    repository: Path
    destination: Path
    bindings: M6Candidate2Bindings
    resolved: M6ResolvedContract
    sidecar_payload: bytes


@dataclass(frozen=True)
class M6Candidate2InputPlan:
    """Fit-only inputs plus a callback that materializes holdout truth later."""

    fit_inputs: M6RefitInputs
    load_full_inputs: Callable[[], M6HarnessInputs]


class M6Candidate2DesignedAbort(RuntimeError):
    """Structured registered-run abort that must never become a gate FAIL."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        abort = self.report.get("abort", {})
        message = (
            abort.get("message")
            if isinstance(abort, Mapping)
            else "candidate-2 designed abort"
        )
        super().__init__(str(message))


@dataclass(frozen=True)
class M6Candidate2RunnerOperations:
    """Injectable phase seams for synthetic candidate-2 runner tests."""

    fit: Callable[[M6RefitInputs], M6RefitBundle | Any]
    first_marriage_preflight: Callable[
        [M6RefitBundle | Any], Mapping[str, Any]
    ]
    fit_postrepair_incumbent: Callable[
        [M6RefitInputs, M6RefitBundle | Any], M6RefitBundle | Any
    ]
    materialize: Callable[[M6HarnessInputs, M6RefitBundle | Any], M6RefitPhase]
    materialize_postrepair_incumbent: Callable[
        [M6RefitPhase, M6RefitBundle | Any], M6RefitPhase
    ]
    first_marriage_disclosure: Callable[
        [
            M6HarnessInputs,
            M6RefitPhase,
            M6ResolvedContract,
            Mapping[str, Any],
        ],
        Mapping[str, Any],
    ]
    preflight_1: Callable[[Any, M6RefitPhase, Any], Mapping[str, Any]]
    preflight_2: Callable[[Any, M6RefitPhase, Any], Mapping[str, Any]]
    score_seed: Callable[[Any, M6RefitPhase, Any, int], M6SeedRun]
    aggregate: Callable[[Any, Sequence[Any]], Any]
    domain_floor: Callable[
        [Any, M6RefitPhase, M6ResolvedContract], Mapping[str, Any]
    ]
    report_only: Callable[
        [
            Any,
            M6RefitPhase,
            M6ResolvedContract,
            Sequence[M6SeedRun],
            Mapping[str, Any],
        ],
        Mapping[str, Any],
    ]
    write: Callable[[Path, Mapping[str, Any], bytes], None]


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize one JSON value with the registered canonical convention."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_text(repository: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        command = " ".join(("git", *arguments))
        raise RuntimeError(
            f"candidate-2 runner failed to run `{command}`"
        ) from error


def _head_blob(repository: Path, commit_sha: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit_sha}:{path}"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"candidate-2 runner cannot read the HEAD blob {path!r}"
        ) from error


def capture_source_identity(repository: Path) -> M6SourceIdentity:
    """Capture HEAD and refuse a dirty tree before reading repository data."""
    commit_sha = _git_text(repository, "rev-parse", "HEAD")
    status = _git_text(repository, "status", "--porcelain")
    if status:
        raise RuntimeError(
            "candidate-2 run refuses a dirty worktree before any data read"
        )
    return M6SourceIdentity(commit_sha=commit_sha)


def _canonical_spec_sha256(spec: Any) -> tuple[dict[str, Any], str]:
    canonical = spec.canonical_dict()
    digest = _sha256_bytes(_canonical_json_bytes(canonical))
    if digest != spec.sha256:
        raise RuntimeError(
            f"candidate spec {spec.candidate_id!r} has an inconsistent sha256"
        )
    return canonical, digest


def resolve_candidate2_identity() -> M6Candidate2Identity:
    """Resolve candidate number 2 from the engine registry, never a default."""
    matches = [
        (int(number), spec)
        for number, spec in engine_candidates.REGISTRY.items()
        if spec is engine_candidates.CANDIDATE_2
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "engine registry must contain exactly one candidate-2 spec"
        )
    number, engine_spec = matches[0]
    if number != 2:
        raise RuntimeError(
            "candidate-2 engine spec is not registered as candidate number 2"
        )
    family_spec = family_candidates.M6_CANDIDATE_2
    candidate16 = family_candidates.CANDIDATE_16
    if (
        family_spec is candidate16
        or family_spec.candidate_id == candidate16.candidate_id
        or family_spec.sha256 == candidate16.sha256
    ):
        raise RuntimeError(
            "candidate-2 family binding resolved to forbidden candidate 16"
        )
    if not family_spec.candidate_id.startswith("m6_candidate2_"):
        raise RuntimeError(
            "family registry spec is not the candidate-2 sibling"
        )

    def carried(spec: FamilyCandidateSpec) -> list[dict[str, Any]]:
        return [
            {
                "kind": component.kind,
                "implementation_id": component.implementation_id,
                "params": dict(component.params),
            }
            for component in spec.components
            if component.kind != "first_marriage"
        ]

    if carried(family_spec) != carried(candidate16):
        raise RuntimeError(
            "candidate-2 family sibling changes a non-first-marriage component"
        )
    first_marriage = [
        component
        for component in family_spec.components
        if component.kind == "first_marriage"
    ]
    if len(first_marriage) != 1:
        raise RuntimeError(
            "candidate-2 family spec must contain one first-marriage component"
        )
    _, family_sha = _canonical_spec_sha256(family_spec)
    _, engine_sha = _canonical_spec_sha256(engine_spec)
    return M6Candidate2Identity(
        number=number,
        family_spec=family_spec,
        engine_spec=engine_spec,
        family_spec_sha256=family_sha,
        engine_spec_sha256=engine_sha,
        first_marriage_params=dict(first_marriage[0].params),
    )


def assert_candidate2_identity_is_frozen(
    identity: M6Candidate2Identity,
) -> None:
    """Refuse any unfrozen family sibling before data can be read."""
    selected_c = identity.first_marriage_params.get("selected_c")
    if "prefreeze" in identity.family_spec.candidate_id or selected_c is None:
        raise RuntimeError(
            "candidate-2 family registry is not frozen "
            "(selected_c=None); refusing all data reads until a ratified "
            "selected-C CandidateSpec is registered"
        )
    if isinstance(selected_c, bool) or not isinstance(
        selected_c, (int, float)
    ):
        raise RuntimeError(
            "candidate-2 family selected_c must be a finite positive number"
        )
    if not math.isfinite(float(selected_c)) or float(selected_c) <= 0.0:
        raise RuntimeError(
            "candidate-2 family selected_c must be a finite positive number"
        )
    ledger_sha = identity.first_marriage_params.get("selection_ledger_sha256")
    if (
        ledger_sha
        != PREREGISTERED_VALUES["selection_evidence.first_marriage_c"]
    ):
        raise RuntimeError(
            "candidate-2 family spec does not bind the preregistered "
            "first-marriage selection ledger"
        )
    checksums = identity.first_marriage_params.get("final_fit_checksums")
    if not isinstance(checksums, Mapping) or not checksums:
        raise RuntimeError(
            "candidate-2 family spec has no registered final-fit checksums"
        )


def _canonicalize_pip_freeze(output: str) -> tuple[tuple[str, ...], bytes]:
    lines = tuple(sorted(line for line in output.splitlines() if line))
    payload = "\n".join(lines)
    if lines:
        payload += "\n"
    return lines, payload.encode("utf-8")


def _dependency_snapshot() -> tuple[tuple[str, ...], str]:
    try:
        output = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "candidate-2 runner could not capture pip freeze"
        ) from error
    lines, canonical = _canonicalize_pip_freeze(output)
    return lines, _sha256_bytes(canonical)


def _live_gate_document(
    repository: Path,
    commit_sha: str,
) -> Mapping[str, Any]:
    document = yaml.safe_load(
        _head_blob(repository, commit_sha, "gates.yaml").decode("utf-8")
    )
    if not isinstance(document, Mapping):
        raise ValueError("gates.yaml must contain a mapping")
    return document


def _gate_design_commit(document: Mapping[str, Any]) -> str:
    gates = document.get("gates")
    block = gates.get("gate_m6") if isinstance(gates, Mapping) else None
    value = block.get("design_commit") if isinstance(block, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError("gates.yaml gate_m6 has no design_commit")
    return value


def _sidecar_snapshot(
    repository: Path,
    commit_sha: str,
) -> tuple[bytes, str, dict[str, Any], dict[str, str]]:
    environment = environment_block()
    contract_ref = asdict(
        ContractRef(
            blob_sha=_git_text(
                repository,
                "rev-parse",
                f"{commit_sha}:gates.yaml",
            ),
            head_sha=commit_sha,
            path="gates.yaml",
        )
    )
    sidecar = {
        "contract": contract_ref,
        "environment": environment,
    }
    payload = _canonical_json_bytes(sidecar)
    return payload, _sha256_bytes(payload), environment, contract_ref


def _resolve_pinned_contract(
    repository: Path,
    commit_sha: str,
    document: Mapping[str, Any],
) -> M6ResolvedContract:
    contract = m6_runner.contract_from_gate_document(document)
    floor_path = Path(contract.floor_run)
    if floor_path.is_absolute() or ".." in floor_path.parts:
        raise ValueError("candidate-2 floor path escapes the repository")
    floor_bytes = _head_blob(repository, commit_sha, floor_path.as_posix())
    floor_sha = _sha256_bytes(floor_bytes)
    if floor_sha != contract.floor_run_sha256:
        raise ValueError(
            "candidate-2 frozen floor sha256 mismatch: "
            f"{floor_sha} != {contract.floor_run_sha256}"
        )
    floor = json.loads(floor_bytes)
    if not isinstance(floor, Mapping):
        raise ValueError("candidate-2 floor artifact must be a mapping")
    return M6ResolvedContract(
        contract=contract,
        floor_artifact=floor,
        floor_path=floor_path.as_posix(),
        floor_sha256=floor_sha,
    )


def _assert_selection_semantics(
    identity: M6Candidate2Identity,
    evidence_blobs: Mapping[str, bytes],
) -> None:
    """Bind registry parameters to the values inside hash-verified ledgers."""
    first_marriage = json.loads(evidence_blobs["first_marriage_c"])
    selected_c = first_marriage.get("selection", {}).get("selected_c")
    checksums = (
        first_marriage.get("final_fit", {})
        .get("fit_audit", {})
        .get("checksums")
    )
    if identity.first_marriage_params.get("selected_c") != selected_c:
        raise RuntimeError(
            "candidate-2 family selected_c differs from its selection ledger"
        )
    if identity.first_marriage_params.get("final_fit_checksums") != checksums:
        raise RuntimeError(
            "candidate-2 family final-fit checksums differ from its ledger"
        )
    if identity.first_marriage_params.get("selection_ledger_sha256") != (
        _sha256_bytes(evidence_blobs["first_marriage_c"])
    ):
        raise RuntimeError(
            "candidate-2 family spec does not bind its exact selection ledger"
        )

    qstar = json.loads(evidence_blobs["qstar_q_grid"])
    selected_q = qstar.get("selector", {}).get("selected_q")
    refresh = identity.engine_spec.operation(
        engine_candidates.RANK_REFRESH_OPERATION_KIND
    )
    registered_q = None if refresh is None else refresh.params.get("q")
    if registered_q != selected_q:
        raise RuntimeError(
            "candidate-2 engine q differs from its selection ledger"
        )


def _validated_entry_dissolved_conformance(
    blob: bytes,
) -> Mapping[str, Any]:
    """Validate and retain the landed repair's exact pre-score ledger.

    This proof is source-pinned and record-and-publish, not an eighth
    preregistration comparison surface.  Its bytes are already covered by the
    clean source commit; the semantic checks prevent an unrelated JSON blob at
    the same path from masquerading as the program's section-7 proof.
    """
    document = json.loads(blob)
    if not isinstance(document, Mapping):
        raise RuntimeError("entry-dissolved conformance evidence is malformed")
    authority = document.get("authority")
    disposition = document.get("disposition")
    boundary = document.get("information_boundary")
    protocol = document.get("protocol")
    ledger = document.get("relabel_ledger")
    paired = document.get("paired_draw_invariants")
    source_bindings = document.get("source_bindings")
    if (
        document.get("schema")
        != ("m6.candidate2.entry_dissolved_conformance.v1")
        or document.get("status") != "PASS"
    ):
        raise RuntimeError("entry-dissolved conformance proof is not PASS")
    if not isinstance(authority, Mapping) or (
        authority.get("candidate_number") != 2
        or authority.get("repair_merge") != LANDED_REPAIR_COMMIT
    ):
        raise RuntimeError("entry-dissolved conformance authority changed")
    if not isinstance(disposition, Mapping) or any(
        disposition.get(key) != "PASS"
        for key in ("condition_4", "condition_5")
    ):
        raise RuntimeError(
            "entry-dissolved conformance conditions 4/5 are not PASS"
        )
    if (
        disposition.get("semantic_change_detected") is not False
        or disposition.get("gate_or_floor_byte_change_authorized") is not False
    ):
        raise RuntimeError(
            "entry-dissolved proof no longer describes landed conformance"
        )
    if not isinstance(boundary, Mapping) or any(
        boundary.get(key) is not False
        for key in (
            "candidate_output_contact",
            "floor_artifact_read",
            "gate_contract_read",
            "gate_scorer_called",
            "gate_tolerance_read",
            "post_2014_values_entered_computation",
            "runs_path_read_or_written",
        )
    ):
        raise RuntimeError(
            "entry-dissolved proof crossed its registered information boundary"
        )
    if not isinstance(protocol, Mapping) or (
        protocol.get("independent_of_candidate2_first_marriage_estimator")
        is not True
        or protocol.get("complete_zero_rows_published") is not True
    ):
        raise RuntimeError(
            "entry-dissolved proof does not isolate the landed repair"
        )
    paired_totals = _validate_conformance_paired_draws(document, paired)
    if not isinstance(source_bindings, Mapping) or any(
        not isinstance(source_bindings.get(name), str)
        or len(source_bindings[name]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_bindings[name]
        )
        for name in CONFORMANCE_SOURCE_BINDING_KEYS
    ):
        raise RuntimeError("entry-dissolved source bindings are incomplete")
    _validate_conformance_relabel_ledger(ledger, paired_totals)
    return document


def _validate_conformance_paired_draws(
    document: Mapping[str, Any],
    paired: Any,
) -> Mapping[tuple[int, int], tuple[float, float]]:
    rows = paired.get("rows") if isinstance(paired, Mapping) else None
    if (
        not isinstance(paired, Mapping)
        or paired.get("all_pass") is not True
        or paired.get("draw_count") != 120
        or not isinstance(rows, list)
        or len(rows) != 120
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise RuntimeError(
            "entry-dissolved paired-draw invariants are incomplete"
        )
    expected_pairs = {
        (boundary, seed)
        for boundary in CONFORMANCE_EVALUATION_YEARS
        for seed in range(7200, 7240)
    }
    observed_pairs = {
        (row.get("pseudo_boundary"), row.get("draw_seed")) for row in rows
    }
    if len(observed_pairs) != len(rows) or observed_pairs != expected_pairs:
        raise RuntimeError(
            "entry-dissolved paired-draw identities are incomplete"
        )
    boundary_rows = document.get("boundaries")
    if not isinstance(boundary_rows, list):
        raise RuntimeError("entry-dissolved carrier boundaries are absent")
    carrier_by_boundary = {
        row.get("pseudo_boundary"): row.get("entry_dissolved_carriers")
        for row in boundary_rows
        if isinstance(row, Mapping)
    }
    if set(carrier_by_boundary) != set(CONFORMANCE_EVALUATION_YEARS):
        raise RuntimeError("entry-dissolved carrier boundaries changed")
    totals: dict[tuple[int, int], tuple[float, float]] = {}
    for row in rows:
        if any(
            row.get(invariant) is not True
            for invariant in CONFORMANCE_PAIRED_INVARIANTS
        ):
            raise RuntimeError(
                "entry-dissolved paired-draw invariant row is not PASS"
            )
        boundary = row["pseudo_boundary"]
        if (
            row.get("postrepair_carriers_verified")
            != carrier_by_boundary[boundary]
        ):
            raise RuntimeError(
                "entry-dissolved paired-draw carrier count changed"
            )
        for name in ("relabel_event_count", "relabel_f6_weight"):
            value = _finite_number(row.get(name))
            if value is None or value < 0.0:
                raise RuntimeError(
                    "entry-dissolved paired-draw relabel value is invalid"
                )
        totals[(boundary, row["draw_seed"])] = (
            float(row["relabel_event_count"]),
            float(row["relabel_f6_weight"]),
        )
    return totals


def _conformance_cell_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("pseudo_boundary"),
        row.get("evaluation_year"),
        row.get("entry_origin"),
        row.get("sex"),
        row.get("age_band"),
    )


def _validate_conformance_relabel_ledger(
    ledger: Any,
    paired_totals: Mapping[tuple[int, int], tuple[float, float]],
) -> None:
    if not isinstance(ledger, Mapping):
        raise RuntimeError("entry-dissolved relabel ledger is absent")
    atomic_rows = ledger.get("atomic_rows")
    aggregates = ledger.get("aggregates_by_boundary_year_origin_sex_age")
    if (
        not isinstance(atomic_rows, list)
        or ledger.get("atomic_row_count") != 5280
        or len(atomic_rows) != 5280
        or not isinstance(aggregates, list)
        or len(aggregates) != 132
        or any(
            not isinstance(row, Mapping) for row in (*atomic_rows, *aggregates)
        )
    ):
        raise RuntimeError("entry-dissolved relabel ledger is incomplete")
    observed_atomic_sha = _sha256_bytes(_canonical_json_bytes(atomic_rows))
    if observed_atomic_sha != ledger.get("atomic_rows_sha256"):
        raise RuntimeError(
            "entry-dissolved atomic relabel ledger hash mismatch"
        )
    expected_cells = {
        (boundary, year, origin, sex, band)
        for boundary, years in CONFORMANCE_EVALUATION_YEARS.items()
        for year in years
        for origin in ("divorced", "widowed")
        for sex in ("female", "male")
        for band in ("18-29", "30-44", "45-64")
    }
    aggregate_by_cell = {_conformance_cell_key(row): row for row in aggregates}
    if (
        len(aggregate_by_cell) != 132
        or set(aggregate_by_cell) != expected_cells
    ):
        raise RuntimeError(
            "entry-dissolved aggregate relabel cells are incomplete"
        )
    atomic_by_cell: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    atomic_by_pair: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for row in atomic_rows:
        key = _conformance_cell_key(row)
        atomic_by_cell.setdefault(key, []).append(row)
    if set(atomic_by_cell) != expected_cells:
        raise RuntimeError(
            "entry-dissolved atomic relabel cells are incomplete"
        )
    expected_seeds = set(range(7200, 7240))
    for key in sorted(expected_cells):
        rows = atomic_by_cell[key]
        seeds = {row.get("draw_seed") for row in rows}
        if len(rows) != 40 or seeds != expected_seeds:
            raise RuntimeError(
                "entry-dissolved atomic relabel draw identities changed"
            )
        aggregate = aggregate_by_cell[key]
        evaluation_year = int(key[1])
        required_interview_year = (
            evaluation_year
            if evaluation_year % 2 == 1
            else evaluation_year + 1
        )
        if (
            aggregate.get("n_draws") != 40
            or aggregate.get("required_interview_year")
            != required_interview_year
            or any(
                row.get("required_interview_year") != required_interview_year
                or row.get("from_transition") != "first_marriage"
                or row.get("to_transition") != "remarriage"
                for row in rows
            )
        ):
            raise RuntimeError("entry-dissolved relabel cell protocol changed")
        for row in rows:
            count = _finite_number(row.get("event_count"))
            weight = _finite_number(row.get("f6_weight"))
            first_count = _finite_number(row.get("first_marriage_count_delta"))
            first_weight = _finite_number(
                row.get("first_marriage_f6_weight_delta")
            )
            remarriage_count = _finite_number(
                row.get("remarriage_count_delta")
            )
            remarriage_weight = _finite_number(
                row.get("remarriage_f6_weight_delta")
            )
            if (
                count is None
                or weight is None
                or count < 0.0
                or weight < 0.0
                or first_count != -count
                or first_weight != -weight
                or remarriage_count != count
                or remarriage_weight != weight
            ):
                raise RuntimeError(
                    "entry-dissolved atomic relabel transfer changed"
                )
            pair = (int(row["pseudo_boundary"]), int(row["draw_seed"]))
            atomic_by_pair.setdefault(pair, []).append((count, weight))
        for metric in CONFORMANCE_AGGREGATE_METRICS:
            values = [_finite_number(row.get(metric)) for row in rows]
            if any(value is None for value in values):
                raise RuntimeError(
                    "entry-dissolved atomic relabel metric is invalid"
                )
            numeric = [float(value) for value in values if value is not None]
            expected = {
                "max_per_draw": max(numeric),
                "mean_per_draw": math.fsum(numeric) / len(numeric),
                "min_per_draw": min(numeric),
                "sum_across_draws": math.fsum(numeric),
            }
            observed = aggregate.get(metric)
            if not isinstance(observed, Mapping) or any(
                _finite_number(observed.get(name)) != value
                for name, value in expected.items()
            ):
                raise RuntimeError(
                    "entry-dissolved aggregate relabel metric changed"
                )
    if set(atomic_by_pair) != set(paired_totals):
        raise RuntimeError(
            "entry-dissolved atomic/paired draw identities changed"
        )
    for pair, expected in paired_totals.items():
        atomic = atomic_by_pair[pair]
        observed = (
            math.fsum(value[0] for value in atomic),
            math.fsum(value[1] for value in atomic),
        )
        if observed != expected:
            raise RuntimeError(
                "entry-dissolved atomic/paired relabel totals changed"
            )


def collect_candidate2_bindings(
    repository: Path,
    *,
    source: M6SourceIdentity,
    identity: M6Candidate2Identity | None = None,
) -> tuple[M6Candidate2Bindings, M6ResolvedContract, bytes]:
    """Collect every compared and record-only binding after the clean check."""
    identity = identity or resolve_candidate2_identity()
    family_canonical, family_sha = _canonical_spec_sha256(identity.family_spec)
    engine_canonical, engine_sha = _canonical_spec_sha256(identity.engine_spec)
    document = _live_gate_document(repository, source.commit_sha)
    resolved = _resolve_pinned_contract(
        repository,
        source.commit_sha,
        document,
    )
    design_sha256s = {
        path: _sha256_bytes(_head_blob(repository, source.commit_sha, path))
        for path in DESIGN_BLOB_PATHS
    }
    evidence_blobs = {
        name: _head_blob(repository, source.commit_sha, path)
        for name, path in SELECTION_EVIDENCE_PATHS.items()
    }
    evidence_sha256s = {
        name: _sha256_bytes(blob) for name, blob in evidence_blobs.items()
    }
    _assert_selection_semantics(identity, evidence_blobs)
    conformance_blob = _head_blob(
        repository,
        source.commit_sha,
        CONFORMANCE_EVIDENCE_PATH,
    )
    conformance = _validated_entry_dissolved_conformance(conformance_blob)
    freeze_lines, dependency_sha = _dependency_snapshot()
    sidecar, sidecar_sha, environment, contract_ref = _sidecar_snapshot(
        repository,
        source.commit_sha,
    )
    runtime = {
        key: str(environment[key])
        for key in ("python", "numpy", "pandas", "scipy")
    }
    sidecar_path = f"{DEFAULT_OUTPUT.as_posix()}.env.json"
    bindings = M6Candidate2Bindings(
        source=source,
        design_commit=_gate_design_commit(document),
        design_blob_sha256s=design_sha256s,
        floor_run=resolved.floor_path,
        floor_run_sha256=resolved.floor_sha256,
        candidate_number=identity.number,
        candidate_spec_ids={
            "family_transitions": identity.family_spec.candidate_id,
            "engine_operations": identity.engine_spec.candidate_id,
        },
        candidate_spec_sha256s={
            "family_transitions": family_sha,
            "engine_operations": engine_sha,
        },
        candidate_specs={
            "family_transitions": family_canonical,
            "engine_operations": engine_canonical,
        },
        selection_evidence_sha256s=evidence_sha256s,
        first_marriage_selection_ledger=json.loads(
            evidence_blobs["first_marriage_c"]
        ),
        entry_dissolved_conformance_sha256=_sha256_bytes(conformance_blob),
        entry_dissolved_conformance=conformance,
        dependency_sha256=dependency_sha,
        sorted_pip_freeze=freeze_lines,
        runtime_identity=runtime,
        environment_sidecar_path=sidecar_path,
        environment_sidecar_sha256=sidecar_sha,
        contract_ref=contract_ref,
    )
    return bindings, resolved, sidecar


def _compared_values(bindings: M6Candidate2Bindings) -> dict[str, str]:
    return {
        "design_commit": bindings.design_commit,
        "floor_run_sha256": bindings.floor_run_sha256,
        "candidate_spec.family_transitions": (
            bindings.candidate_spec_sha256s["family_transitions"]
        ),
        "candidate_spec.engine_operations": (
            bindings.candidate_spec_sha256s["engine_operations"]
        ),
        "selection_evidence.qstar_q_grid": (
            bindings.selection_evidence_sha256s["qstar_q_grid"]
        ),
        "selection_evidence.first_marriage_c": (
            bindings.selection_evidence_sha256s["first_marriage_c"]
        ),
        "selection_evidence.remarriage_no_op": (
            bindings.selection_evidence_sha256s["remarriage_no_op"]
        ),
    }


def assert_preregistered_bindings(bindings: M6Candidate2Bindings) -> None:
    """Abort on drift in exactly the seven preregistered values."""
    observed = _compared_values(bindings)
    if tuple(observed) != PREREGISTERED_COMPARISON_KEYS:
        raise RuntimeError(
            "candidate-2 preregistration comparison scope drifted"
        )
    for surface, expected in PREREGISTERED_VALUES.items():
        actual = observed[surface]
        if actual != expected:
            raise RuntimeError(
                f"preregistered {surface} mismatch: {actual} != {expected}"
            )


def _repository(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[3]


def _assert_imported_source_tree(repository: Path) -> None:
    expected = {
        Path(__file__)
        .resolve(): (
            repository
            / "src/populace_dynamics/harness/m6_candidate2_runner.py"
        )
        .resolve(),
        Path(m6_runner.__file__)
        .resolve(): (repository / "src/populace_dynamics/harness/m6_runner.py")
        .resolve(),
        Path(engine_candidates.__file__)
        .resolve(): (repository / "src/populace_dynamics/engine/candidates.py")
        .resolve(),
        Path(family_candidates.__file__)
        .resolve(): (
            repository
            / "src/populace_dynamics/models/family_transitions/registry.py"
        )
        .resolve(),
    }
    for observed, required in expected.items():
        if observed != required:
            raise RuntimeError(
                "candidate-2 imported source is outside the guarded tree: "
                f"{observed} != {required}"
            )


def _resolve_destination(repository: Path, output: Path | str) -> Path:
    destination = Path(output)
    if not destination.is_absolute():
        destination = repository / destination
    destination = destination.resolve()
    candidate1 = (repository / CANDIDATE1_OUTPUT).resolve()
    candidate2 = (repository / DEFAULT_OUTPUT).resolve()
    if destination == candidate1:
        raise ValueError(
            "candidate-2 runner hard-refuses "
            f"{CANDIDATE1_OUTPUT.as_posix()}"
        )
    if destination != candidate2:
        raise ValueError(
            "candidate-2 runner writes exclusively to "
            f"{DEFAULT_OUTPUT.as_posix()}"
        )
    return destination


def _ensure_exclusive_targets(destination: Path) -> None:
    for target in (destination, Path(f"{destination}.env.json")):
        if os.path.lexists(target):
            raise FileExistsError(
                f"{target} already exists; candidate 2 is a one-shot run"
            )


def validate_candidate2_registration_id(value: str) -> str:
    """Require a fresh registration and reject candidate 1's consumed id."""
    registration = m6_runner.validate_registration_id(value)
    if registration == CANDIDATE1_REGISTRATION_ID or registration.endswith(
        f"#issuecomment-{CANDIDATE1_REGISTRATION_ID}"
    ):
        raise ValueError(
            "candidate-2 requires a fresh registration 8 and refuses the "
            "candidate-1 registration"
        )
    return registration


def guard_registered_m6_candidate2_run(
    *,
    registration_id: str,
    output: Path | str = DEFAULT_OUTPUT,
    root: Path | str | None = None,
) -> M6Candidate2Preparation:
    """Resolve and compare all guards before an input factory may be called."""
    registration = validate_candidate2_registration_id(registration_id)
    repository = _repository(root)
    destination = _resolve_destination(repository, output)
    _ensure_exclusive_targets(destination)
    source = capture_source_identity(repository)
    _assert_imported_source_tree(repository)
    identity = resolve_candidate2_identity()
    assert_candidate2_identity_is_frozen(identity)
    bindings, resolved, sidecar = collect_candidate2_bindings(
        repository,
        source=source,
        identity=identity,
    )
    assert_preregistered_bindings(bindings)
    return M6Candidate2Preparation(
        registration_id=registration,
        repository=repository,
        destination=destination,
        bindings=bindings,
        resolved=resolved,
        sidecar_payload=sidecar,
    )


def _fit_candidate2(fit_inputs: M6RefitInputs) -> M6RefitBundle:
    identity = resolve_candidate2_identity()
    assert_candidate2_identity_is_frozen(identity)
    return refit_m6_components(
        fit_inputs,
        boundary_year=BOUNDARY_YEAR,
        family_candidate_spec=identity.family_spec,
        earnings_candidate_spec=identity.engine_spec,
    )


def _fit_postrepair_incumbent_first_marriage(
    fit_inputs: M6RefitInputs,
    candidate_bundle: M6RefitBundle | Any,
) -> M6RefitBundle:
    """Build a post-repair C16 comparator, reusing every disjoint fit.

    The comparison changes the family first-marriage sibling and its dependent
    earnings modifier only.  Earnings candidate 2 and every other fitted
    component are the exact objects used by the scored candidate projection.
    """
    if not isinstance(candidate_bundle, M6RefitBundle):
        raise TypeError("candidate-2 comparator requires an M6RefitBundle")
    incumbent_fit = refit_family_transitions(
        fit_inputs.family_context,
        boundary_year=BOUNDARY_YEAR,
        candidate_spec=family_candidates.CANDIDATE_16,
    )
    candidate_family = candidate_bundle.family
    if candidate_family is None:
        raise RuntimeError("candidate-2 family fit is absent")
    implementation_ids = dict(candidate_family.fitted.implementation_ids)
    implementation_ids["first_marriage"] = (
        incumbent_fit.fitted.implementation_ids["first_marriage"]
    )
    comparator_fitted = replace(
        candidate_family.fitted,
        first_marriage=incumbent_fit.fitted.first_marriage,
        implementation_ids=implementation_ids,
    )
    for name in (
        "divorce",
        "widowhood",
        "remarriage",
        "fertility",
        "initial_states",
        "spousal_age_gaps",
    ):
        if getattr(comparator_fitted, name) is not getattr(
            candidate_family.fitted, name
        ):
            raise RuntimeError(
                "first-marriage comparator changed a carried family fit"
            )
    family = replace(
        candidate_family,
        candidate_id=family_candidates.CANDIDATE_16.candidate_id,
        spec_sha256=family_candidates.CANDIDATE_16.sha256,
        fitted=comparator_fitted,
    )
    fresh_modifier = refit_first_marriage_modifier(
        comparator_fitted,
        fit_inputs.modifier_marital_panel,
        fit_inputs.earnings_panel,
        fit_inputs.modifier_marriage_records,
        fit_inputs.modifier_person_weight,
        fit_inputs.ssa_params,
        params_vintage=fit_inputs.ssa_params_vintage,
        train_ids=fit_inputs.modifier_train_ids,
        interview_years=fit_inputs.modifier_interview_years,
        boundary_year=BOUNDARY_YEAR,
    )
    candidate_modifier = candidate_bundle.modifier
    if candidate_modifier is None:
        raise RuntimeError("candidate-2 first-marriage modifier is absent")
    _assert_committed_axis_exact(candidate_modifier.axis, fresh_modifier.axis)
    modifier = replace(
        candidate_modifier,
        modifier=fresh_modifier.modifier,
    )
    if modifier.axis is not candidate_modifier.axis:
        raise RuntimeError(
            "first-marriage comparator did not retain the permanent axis"
        )
    comparator = replace(
        candidate_bundle,
        family=family,
        modifier=modifier,
    )
    for name in (
        "household",
        "earnings",
        "disability",
        "claiming_pmfs",
        "mortality",
    ):
        if getattr(comparator, name) is not getattr(candidate_bundle, name):
            raise RuntimeError(
                "first-marriage comparator changed a disjoint fitted component"
            )
    return comparator


def _assert_committed_axis_exact(expected: Any, observed: Any) -> None:
    """Require the comparator's recomputed permanent axis to be bit-exact."""
    for name in ("earn", "sex", "birth_year", "person_weight"):
        if getattr(expected, name) != getattr(observed, name):
            raise RuntimeError(
                f"first-marriage comparator changed permanent axis {name}"
            )
    if tuple(expected.cuts) != tuple(observed.cuts) or not np.array_equal(
        expected.decile_edges,
        observed.decile_edges,
    ):
        raise RuntimeError("first-marriage comparator changed permanent cuts")
    if len(expected.supply_by_decile) != len(observed.supply_by_decile) or any(
        not np.array_equal(left, right)
        for left, right in zip(
            expected.supply_by_decile,
            observed.supply_by_decile,
            strict=True,
        )
    ):
        raise RuntimeError(
            "first-marriage comparator changed permanent decile supply"
        )


def _materialize_postrepair_incumbent_phase(
    candidate_phase: M6RefitPhase,
    comparator_bundle: M6RefitBundle | Any,
) -> M6RefitPhase:
    """Reuse the exact realized population for the isolated paired arm."""
    if not isinstance(comparator_bundle, M6RefitBundle):
        raise TypeError("postrepair comparator requires an M6RefitBundle")
    return replace(
        candidate_phase,
        bundle=comparator_bundle,
        lineage=m6_runner._refit_lineage(comparator_bundle),
    )


def _first_marriage_model(bundle: M6RefitBundle | Any) -> Any:
    family = getattr(bundle, "family", None)
    fitted = getattr(family, "fitted", None)
    model = getattr(fitted, "first_marriage", None)
    if model is None:
        raise FirstMarriagePreflightAbort(
            "NO_REGISTERABLE_FIRST_MARRIAGE_FIT: fitted model is absent"
        )
    return model


def _first_marriage_hazard_table(model: Any) -> list[dict[str, Any]]:
    cohorts = tuple(sorted(set(model.cohort_levels) | {1990}))
    targets = [
        (float(age), sex == "male", int(cohort))
        for sex in ("female", "male")
        for cohort in cohorts
        for age in range(18, 30)
    ]
    age = np.asarray([row[0] for row in targets], dtype=np.float64)
    is_male = np.asarray([row[1] for row in targets], dtype=bool)
    decade = np.asarray([row[2] for row in targets], dtype=np.int64)
    diagnostics = model.transport_diagnostics(age, is_male, decade)
    linear = model.linear_predictor(age, is_male, decade)
    probability = model.predict(age, is_male, decade)
    if not np.isfinite(linear).all() or not (
        np.isfinite(probability).all()
        and np.all(probability > 0.0)
        and np.all(probability < 1.0)
    ):
        raise FirstMarriagePreflightAbort(
            "NO_REGISTERABLE_FIRST_MARRIAGE_FIT: hazard table failed the "
            "strict numeric preflight"
        )
    return [
        {
            **record,
            "linear_predictor": float(linear_value),
            "probability": float(probability_value),
        }
        for record, linear_value, probability_value in zip(
            diagnostics.canonical_records(),
            linear,
            probability,
            strict=True,
        )
    ]


def _preflight_candidate2_first_marriage(
    bundle: M6RefitBundle | Any,
) -> Mapping[str, Any]:
    identity = resolve_candidate2_identity()
    assert_candidate2_identity_is_frozen(identity)
    expected = identity.first_marriage_params["final_fit_checksums"]
    if not isinstance(expected, Mapping):
        raise FirstMarriagePreflightAbort(
            "candidate-2 final-fit checksum binding is malformed"
        )
    model = _first_marriage_model(bundle)
    validate_support_aware_first_marriage_fit(
        model,
        expected_checksums=expected,
    )
    audit = model.audit_dict()
    selected_c = identity.first_marriage_params["selected_c"]
    if float(audit["c"]) != float(selected_c):
        raise FirstMarriagePreflightAbort(
            "candidate-2 fitted C differs from the registry spec"
        )
    recomputed = dict(recompute_support_aware_first_marriage_checksums(model))
    if recomputed != dict(expected):
        raise FirstMarriagePreflightAbort(
            "candidate-2 final-fit checksums differ from registration"
        )
    return {
        "passed": True,
        "selected_c": float(selected_c),
        "selection_grid_ledger": {
            "path": SELECTION_EVIDENCE_PATHS["first_marriage_c"],
            "sha256": PREREGISTERED_VALUES[
                "selection_evidence.first_marriage_c"
            ],
            "all_grid_results_retained": True,
        },
        "fit_audit": audit,
        "registered_checksums": dict(expected),
        "recomputed_checksums": recomputed,
        "hazard_table_ages_18_29": _first_marriage_hazard_table(model),
    }


def _weighted_classification(mask: np.ndarray, weight: np.ndarray) -> dict:
    return {
        "rows": int(mask.sum()),
        "f6_weight": float(weight[mask].sum()),
    }


def _first_marriage_transport_disclosure(
    inputs: M6HarnessInputs,
    phase: M6RefitPhase,
    resolved: M6ResolvedContract,
    fit_preflight: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Retain pre-score transport counts on the symmetric gated support."""
    model = _first_marriage_model(phase.bundle)
    frame = inputs.truth.marital_person_years
    required = {
        "person_id",
        "year",
        "age",
        "sex",
        "weight",
        "marital_state",
        "window",
    }
    if missing := required - set(frame):
        raise FirstMarriagePreflightAbort(
            "first-marriage diagnostic support lacks columns "
            f"{sorted(missing)}"
        )
    attrs = inputs.panel_builder_inputs.marital.attrs
    if not {"person_id", "birth_year"}.issubset(attrs):
        raise FirstMarriagePreflightAbort(
            "first-marriage diagnostic attributes lack birth_year"
        )
    birth_rows = attrs[["person_id", "birth_year"]].drop_duplicates()
    if birth_rows["person_id"].duplicated().any():
        raise FirstMarriagePreflightAbort(
            "first-marriage diagnostic attributes disagree on birth_year"
        )
    birth_decade_by_person = (
        birth_rows.set_index("person_id")["birth_year"] // 10 * 10
    ).astype("int64")
    disclosures: dict[str, Any] = {}
    rules = [
        rule
        for rule in resolved.contract.cells
        if rule.cell.startswith("first_marriage.")
    ]
    if not rules:
        raise FirstMarriagePreflightAbort(
            "locked gate has no first-marriage diagnostic cell"
        )
    for rule in rules:
        band_and_sex = rule.cell.removeprefix("first_marriage.")
        band, separator, sex = band_and_sex.partition("|")
        bounds = band.split("-", 1)
        if not separator or len(bounds) != 2:
            raise FirstMarriagePreflightAbort(
                f"cannot parse first-marriage cell {rule.cell!r}"
            )
        lower, upper = (int(value) for value in bounds)
        cell = frame[
            (frame["window"] == "gated")
            & (frame["marital_state"] == "never_married")
            & (frame["sex"] == sex)
            & frame["age"].between(lower, upper)
        ].copy()
        age = cell["age"].to_numpy(dtype=np.float64)
        is_male = np.full(len(cell), sex == "male", dtype=bool)
        mapped_birth_decade = cell["person_id"].map(birth_decade_by_person)
        if mapped_birth_decade.isna().any():
            raise FirstMarriagePreflightAbort(
                "first-marriage diagnostic support lacks canonical birth year"
            )
        birth_decade = mapped_birth_decade.to_numpy(dtype=np.int64)
        diagnostics = model.transport_diagnostics(
            age,
            is_male,
            birth_decade,
        )
        unseen = (
            diagnostics.target_birth_decade != diagnostics.mapped_birth_decade
        )
        age_out = (
            diagnostics.global_boundary_evaluated
            | diagnostics.cohort_boundary_evaluated
        )
        in_support = ~(unseen | age_out)
        weight = cell["weight"].to_numpy(dtype=np.float64)
        disclosures[rule.cell] = {
            "at_risk_rows": int(len(cell)),
            "at_risk_f6_weight": float(weight.sum()),
            "in_support": _weighted_classification(in_support, weight),
            "age_out_of_support": _weighted_classification(age_out, weight),
            "unseen_cohort": _weighted_classification(unseen, weight),
            "categories_may_overlap": True,
        }
    return {
        **dict(fit_preflight),
        "gated_cell_transport": disclosures,
        "completed_before_projection_or_score": True,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(payload)
    except FileExistsError:
        raise FileExistsError(
            f"{path} already exists; candidate 2 is a one-shot run"
        ) from None
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def write_new_candidate2_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    sidecar_payload: bytes,
) -> None:
    """Exclusively write the main artifact and its exact canonical sidecar."""
    destination = Path(path)
    if destination.name == CANDIDATE1_OUTPUT.name:
        raise ValueError(
            "candidate-2 writer hard-refuses "
            f"{CANDIDATE1_OUTPUT.as_posix()}"
        )
    if tuple(destination.parts[-2:]) != tuple(DEFAULT_OUTPUT.parts):
        raise ValueError(
            "candidate-2 writer writes exclusively to "
            f"{DEFAULT_OUTPUT.as_posix()}"
        )
    parsed_sidecar = json.loads(sidecar_payload)
    if _canonical_json_bytes(parsed_sidecar) != sidecar_payload:
        raise ValueError(
            "candidate-2 environment sidecar is not canonical JSON"
        )
    recorded = artifact.get("integrity")
    sidecar_record = (
        recorded.get("environment_sidecar")
        if isinstance(recorded, Mapping)
        else None
    )
    expected = (
        sidecar_record.get("sha256")
        if isinstance(sidecar_record, Mapping)
        else None
    )
    actual = _sha256_bytes(sidecar_payload)
    if expected != actual:
        raise ValueError(
            "candidate-2 artifact does not bind its exact environment sidecar"
        )
    sidecar = Path(f"{destination}.env.json")
    _ensure_exclusive_targets(destination)
    artifact_payload = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    primary_written = False
    try:
        _write_exclusive(destination, artifact_payload)
        primary_written = True
        _write_exclusive(sidecar, sidecar_payload)
    except BaseException:
        if primary_written:
            destination.unlink(missing_ok=True)
        raise


def default_operations() -> M6Candidate2RunnerOperations:
    incumbent = m6_runner.default_operations()
    return M6Candidate2RunnerOperations(
        fit=_fit_candidate2,
        first_marriage_preflight=_preflight_candidate2_first_marriage,
        fit_postrepair_incumbent=(_fit_postrepair_incumbent_first_marriage),
        materialize=m6_runner.materialize_m6_refit_phase,
        materialize_postrepair_incumbent=(
            _materialize_postrepair_incumbent_phase
        ),
        first_marriage_disclosure=_first_marriage_transport_disclosure,
        preflight_1=incumbent.preflight_1,
        preflight_2=incumbent.preflight_2,
        score_seed=incumbent.score_seed,
        aggregate=incumbent.aggregate,
        domain_floor=incumbent.domain_floor,
        report_only=incumbent.report_only,
        write=write_new_candidate2_artifact,
    )


def _plain_provenance(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_artifact"):
        return value.to_artifact()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return str(value)


def _assert_phase_candidate_binding(
    phase: M6RefitPhase,
    bindings: M6Candidate2Bindings,
) -> None:
    specs = phase.lineage.get("resolved_spec_sha256s")
    if not isinstance(specs, Mapping):
        raise RuntimeError("candidate-2 refit did not publish resolved specs")
    expected = bindings.candidate_spec_sha256s
    for name in ("family_transitions", "engine_operations"):
        lineage_name = (
            "engine_candidate" if name == "engine_operations" else name
        )
        if specs.get(lineage_name) != expected[name]:
            raise RuntimeError(
                f"candidate-2 refit lineage did not bind {name}"
            )
    engine_id = phase.lineage.get("engine_candidate_id")
    if engine_id != bindings.candidate_spec_ids["engine_operations"]:
        raise RuntimeError("candidate-2 refit lineage has the wrong engine id")


def _score_artifact(gate_score: Any, seed_runs: Sequence[M6SeedRun]) -> dict:
    payload = dict(gate_score.to_artifact())
    units = {run.seed: dict(run.side_a_units) for run in seed_runs}
    for record in payload.get("per_seed", []):
        record["n_side_a_units"] = units[int(record["seed"])]
    return payload


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_marriage_cell(run: M6SeedRun) -> Any:
    matches = [
        cell
        for cell in getattr(run.score, "cells", ())
        if str(getattr(cell, "cell", "")) == FIRST_MARRIAGE_GATE_CELL
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "paired first-marriage attribution requires exactly one gated cell"
        )
    return matches[0]


def _first_marriage_estimator_delta(
    candidate_runs: Sequence[M6SeedRun],
    incumbent_runs: Sequence[M6SeedRun],
    contract: Any,
) -> dict[str, Any]:
    """Isolate the new estimator on paired post-repair projections.

    Both arms use the landed marital assembler, candidate-2 earnings law, the
    same realized population, split seeds, draw addresses, and truth reducer.
    The comparator swaps only the family first-marriage fit and its dependent
    first-marriage earnings modifier back to C16.
    """
    candidate_by_seed = {int(run.seed): run for run in candidate_runs}
    incumbent_by_seed = {int(run.seed): run for run in incumbent_runs}
    expected_seeds = tuple(int(seed) for seed in contract.gate_seeds)
    if (
        len(candidate_by_seed) != len(candidate_runs)
        or len(incumbent_by_seed) != len(incumbent_runs)
        or set(candidate_by_seed) != set(expected_seeds)
        or set(incumbent_by_seed) != set(expected_seeds)
    ):
        raise RuntimeError(
            "paired first-marriage attribution seed identities differ"
        )
    per_seed: list[dict[str, Any]] = []
    for seed in expected_seeds:
        candidate_run = candidate_by_seed[seed]
        incumbent_run = incumbent_by_seed[seed]
        if dict(candidate_run.side_a_units) != dict(
            incumbent_run.side_a_units
        ):
            raise RuntimeError(
                "paired first-marriage attribution split populations differ"
            )
        candidate = _first_marriage_cell(candidate_run)
        incumbent = _first_marriage_cell(incumbent_run)
        candidate_truth = _finite_number(getattr(candidate, "rate_a", None))
        incumbent_truth = _finite_number(getattr(incumbent, "rate_a", None))
        if candidate_truth != incumbent_truth:
            raise RuntimeError(
                "paired first-marriage attribution truth reducers differ"
            )
        candidate_rates = tuple(getattr(candidate, "per_draw_rates", ()))
        incumbent_rates = tuple(getattr(incumbent, "per_draw_rates", ()))
        if (
            len(candidate_rates) != contract.n_draws
            or len(incumbent_rates) != contract.n_draws
        ):
            raise RuntimeError(
                "paired first-marriage attribution draw counts differ"
            )
        candidate_numbers = tuple(
            _finite_number(value) for value in candidate_rates
        )
        incumbent_numbers = tuple(
            _finite_number(value) for value in incumbent_rates
        )
        candidate_undefined = tuple(
            int(value)
            for value in getattr(candidate, "undefined_draw_indices", ())
        )
        incumbent_undefined = tuple(
            int(value)
            for value in getattr(incumbent, "undefined_draw_indices", ())
        )
        candidate_regenerated = bool(getattr(candidate, "regenerated", False))
        incumbent_regenerated = bool(getattr(incumbent, "regenerated", False))
        deltas = [
            (
                None
                if candidate_value is None or incumbent_value is None
                else candidate_value - incumbent_value
            )
            for candidate_value, incumbent_value in zip(
                candidate_numbers,
                incumbent_numbers,
                strict=True,
            )
        ]
        candidate_rbar = _finite_number(getattr(candidate, "rbar", None))
        incumbent_rbar = _finite_number(getattr(incumbent, "rbar", None))
        candidate_score = _finite_number(getattr(candidate, "score", None))
        incumbent_score = _finite_number(getattr(incumbent, "score", None))
        candidate_valid = (
            candidate_truth is not None
            and candidate_rbar is not None
            and candidate_score is not None
            and not candidate_undefined
            and candidate_regenerated
            and all(value is not None for value in candidate_numbers)
        )
        incumbent_valid = (
            incumbent_truth is not None
            and incumbent_rbar is not None
            and incumbent_score is not None
            and not incumbent_undefined
            and incumbent_regenerated
            and all(value is not None for value in incumbent_numbers)
        )
        valid = (
            candidate_valid
            and incumbent_valid
            and all(value is not None for value in deltas)
        )
        per_seed.append(
            {
                "seed": seed,
                "valid": valid,
                "n_side_a_units": dict(candidate_run.side_a_units),
                "truth_rate": candidate_truth,
                "postrepair_incumbent": {
                    "valid": incumbent_valid,
                    "undefined_draw_indices": list(incumbent_undefined),
                    "regenerated": incumbent_regenerated,
                    "per_draw_rate": list(incumbent_numbers),
                    "rbar": incumbent_rbar,
                    "score": incumbent_score,
                },
                "candidate2_estimator": {
                    "valid": candidate_valid,
                    "undefined_draw_indices": list(candidate_undefined),
                    "regenerated": candidate_regenerated,
                    "per_draw_rate": list(candidate_numbers),
                    "rbar": candidate_rbar,
                    "score": candidate_score,
                },
                "candidate2_minus_postrepair_incumbent": {
                    "per_draw_rate": deltas,
                    "rbar": (
                        None
                        if candidate_rbar is None or incumbent_rbar is None
                        else candidate_rbar - incumbent_rbar
                    ),
                    "score": (
                        None
                        if candidate_score is None or incumbent_score is None
                        else candidate_score - incumbent_score
                    ),
                },
            }
        )
    valid = all(record["valid"] for record in per_seed)
    return {
        "status": "COMPLETE" if valid else "INVALID",
        "valid": valid,
        "cell": FIRST_MARRIAGE_GATE_CELL,
        "comparison": (
            "postrepair incumbent C16 first-marriage estimator versus the "
            "candidate-2 support-aware estimator"
        ),
        "postrepair_incumbent_family_spec": {
            "candidate_id": family_candidates.CANDIDATE_16.candidate_id,
            "canonical_json_sha256": family_candidates.CANDIDATE_16.sha256,
        },
        "candidate2_disjoint_components_reused_by_object_identity": [
            "household",
            "earnings",
            "disability",
            "claiming_pmfs",
            "mortality",
            "first_marriage_modifier.permanent_axis",
            "family.divorce",
            "family.widowhood",
            "family.remarriage",
            "family.fertility",
            "family.initial_states",
            "family.spousal_age_gaps",
        ],
        "common_random_numbers": True,
        "same_truth_reducer": True,
        "gated": False,
        "acceptance_input": False,
        "diagnostic_only": True,
        "completed_before_gate_aggregation": True,
        "per_seed": per_seed,
    }


def _first_marriage_effect_decomposition(
    bindings: M6Candidate2Bindings,
    estimator_delta: Mapping[str, Any],
) -> dict[str, Any]:
    conformance = bindings.entry_dissolved_conformance
    relabel = conformance["relabel_ledger"]
    published_relabel_ledger = {
        "atomic_row_count": relabel["atomic_row_count"],
        "atomic_rows_sha256": relabel["atomic_rows_sha256"],
        "aggregates_by_boundary_year_origin_sex_age": list(
            relabel["aggregates_by_boundary_year_origin_sex_age"]
        ),
        "full_atomic_rows_embedded": False,
        "full_atomic_rows_bound_by_source_sha256": True,
    }
    return {
        "status": (
            "COMPLETE"
            if conformance.get("status") == "PASS"
            and estimator_delta.get("valid") is True
            else "INVALID"
        ),
        "landed_conformance_relabel": {
            "classification": "landed repair; not a candidate-2 model delta",
            "source": {
                "path": CONFORMANCE_EVIDENCE_PATH,
                "sha256": bindings.entry_dissolved_conformance_sha256,
                "repair_merge": LANDED_REPAIR_COMMIT,
            },
            "status": conformance["status"],
            "authority": dict(conformance["authority"]),
            "source_bindings": dict(conformance["source_bindings"]),
            "disposition": dict(conformance["disposition"]),
            "information_boundary": dict(conformance["information_boundary"]),
            "protocol": dict(conformance["protocol"]),
            "paired_draw_invariants": {
                "all_pass": conformance["paired_draw_invariants"]["all_pass"],
                "draw_count": conformance["paired_draw_invariants"][
                    "draw_count"
                ],
            },
            "relabel_ledger": published_relabel_ledger,
        },
        "support_aware_estimator_delta": dict(estimator_delta),
        "attribution_limits": {
            "mechanisms_are_separately_measured": True,
            "landed_relabel_is_not_attributed_to_candidate2_estimator": True,
            "arithmetic_sum_is_not_claimed": True,
            "reason": (
                "the landed relabel ledger uses leakage-safe <=2014 pseudo-"
                "boundaries, while the estimator arm is the paired registered "
                "postrepair gate projection"
            ),
        },
    }


def _must_not_regress_artifact(gate_score: Any) -> dict[str, Any]:
    seeds = tuple(getattr(gate_score, "seeds", ()))
    by_seed: dict[int, Any] = {}
    for seed_score in seeds:
        seed = int(seed_score.seed)
        if seed in by_seed:
            raise RuntimeError(
                "candidate-2 regression block received a duplicate seed"
            )
        by_seed[seed] = seed_score
    if set(by_seed) != set(MUST_NOT_REGRESS_SEEDS):
        raise RuntimeError(
            "candidate-2 regression seeds differ from the registered 0..4"
        )

    per_seed: list[dict[str, Any]] = []
    for seed in MUST_NOT_REGRESS_SEEDS:
        seed_score = by_seed[seed]
        cells: dict[str, Any] = {}
        for cell_score in getattr(seed_score, "cells", ()):
            name = str(cell_score.cell)
            if name in cells:
                raise RuntimeError(
                    "candidate-2 regression block received a duplicate cell"
                )
            cells[name] = cell_score
        missing = set(MUST_NOT_REGRESS_TOLERANCES) - set(cells)
        if missing:
            raise RuntimeError(
                "candidate-2 regression block is missing cells "
                f"{sorted(missing)}"
            )
        records: dict[str, Any] = {}
        for name, tolerance in MUST_NOT_REGRESS_TOLERANCES.items():
            cell_score = cells[name]
            raw_score = getattr(cell_score, "score", None)
            score = (
                float(raw_score)
                if raw_score is not None and math.isfinite(float(raw_score))
                else None
            )
            undefined = tuple(
                getattr(cell_score, "undefined_draw_indices", ())
            )
            regenerated = bool(getattr(cell_score, "regenerated", True))
            cell_valid = score is not None and not undefined and regenerated
            cell_pass = cell_valid and score <= tolerance
            records[name] = {
                "score": score,
                "tolerance": tolerance,
                "valid": cell_valid,
                "pass": cell_pass,
                "undefined_draw_indices": list(undefined),
                "regenerated": regenerated,
            }
        seed_valid = all(record["valid"] for record in records.values())
        n_cells_pass = sum(record["pass"] for record in records.values())
        per_seed.append(
            {
                "seed": seed,
                "valid": seed_valid,
                "seed_pass": (seed_valid and n_cells_pass == len(records)),
                "n_cells_pass": n_cells_pass,
                "cells": records,
            }
        )
    valid = all(record["valid"] for record in per_seed)
    n_seed_pass = sum(record["seed_pass"] for record in per_seed)
    passed = valid and n_seed_pass >= MUST_NOT_REGRESS_REQUIRED_SEEDS
    return {
        "valid": valid,
        "pass": passed,
        "required_seed_passes": MUST_NOT_REGRESS_REQUIRED_SEEDS,
        "n_seeds_pass": n_seed_pass,
        "seed_pass": {
            str(record["seed"]): record["seed_pass"] for record in per_seed
        },
        "tolerances": dict(MUST_NOT_REGRESS_TOLERANCES),
        "tolerance_source": (
            "docs/design/m6_candidate2_program.md section 8; original "
            "candidate-1 thresholds"
        ),
        "per_seed": per_seed,
    }


def _candidate2_acceptance(
    gate_contract_result: Mapping[str, Any],
    must_not_regress_result: Mapping[str, Any],
) -> dict[str, Any]:
    gate_valid = bool(gate_contract_result.get("valid"))
    regression_valid = bool(must_not_regress_result.get("valid"))
    valid = gate_valid and regression_valid
    gate_pass = bool(gate_contract_result.get("pass"))
    regression_pass = bool(must_not_regress_result.get("pass"))
    passed = valid and gate_pass and regression_pass
    if not valid:
        status = "INVALID"
    elif passed:
        status = "PASS"
    elif gate_pass and not regression_pass:
        status = "GATE_PASS_REGRESSION_FAIL"
    else:
        status = "FAIL"
    return {
        "gate_contract_result": dict(gate_contract_result),
        "must_not_regress_result": dict(must_not_regress_result),
        "conjunction": {
            "valid": valid,
            "pass": passed,
            "status": status,
            "requires_gate_contract_and_must_not_regress": True,
        },
    }


def _candidate2_run_verdict(
    acceptance_conjunction: Mapping[str, Any],
    first_marriage_decomposition: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep section-8 acceptance exact while failing closed on §7 integrity."""
    decomposition_valid = (
        first_marriage_decomposition.get("status") == "COMPLETE"
    )
    acceptance_valid = bool(acceptance_conjunction.get("valid"))
    valid = acceptance_valid and decomposition_valid
    passed = valid and bool(acceptance_conjunction.get("pass"))
    return {
        "valid": valid,
        "pass": passed,
        "status": (
            str(acceptance_conjunction.get("status"))
            if decomposition_valid
            else "INVALID"
        ),
        "section_8_acceptance_status": acceptance_conjunction.get("status"),
        "first_marriage_decomposition_valid": decomposition_valid,
    }


def assemble_m6_candidate2_artifact(
    *,
    registration_id: str,
    inputs: M6HarnessInputs,
    phase: M6RefitPhase,
    resolved: M6ResolvedContract,
    seed_runs: Sequence[M6SeedRun],
    gate_score: Any,
    first_marriage_diagnostics: Mapping[str, Any],
    first_marriage_estimator_delta: Mapping[str, Any],
    preflight_1: Mapping[str, Any],
    preflight_2: Mapping[str, Any],
    report_only: Mapping[str, Any],
    bindings: M6Candidate2Bindings,
) -> dict[str, Any]:
    """Build the candidate-2 payload without a candidate-1 artifact default."""
    _assert_phase_candidate_binding(phase, bindings)
    contract = resolved.contract
    family_a = _score_artifact(gate_score, seed_runs)
    must_not_regress = _must_not_regress_artifact(gate_score)
    first_marriage_decomposition = _first_marriage_effect_decomposition(
        bindings,
        first_marriage_estimator_delta,
    )
    acceptance = _candidate2_acceptance(family_a, must_not_regress)
    conjunction = acceptance["conjunction"]
    run_verdict = _candidate2_run_verdict(
        conjunction,
        first_marriage_decomposition,
    )
    passed = bool(run_verdict["pass"])
    valid = bool(run_verdict["valid"])
    provenance = _plain_provenance(getattr(inputs, "provenance", {}))
    family_spec = bindings.candidate_specs["family_transitions"]
    first_marriage = next(
        component
        for component in family_spec["components"]
        if component["kind"] == "first_marriage"
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "name": "gate_m6_candidate2",
            "phase_order": list(PHASE_ORDER),
            "boundary_year": m6_runner.BOUNDARY_YEAR,
            "projection_end_year": PROJECTION_END_YEAR,
            "build_only_guard": False,
        },
        "candidate": {
            "number": bindings.candidate_number,
            "number_source": "engine.candidates.REGISTRY",
            "description": "M6 candidate-2 sibling program",
            "family_spec_id": bindings.candidate_spec_ids[
                "family_transitions"
            ],
            "engine_spec_id": bindings.candidate_spec_ids["engine_operations"],
            "first_marriage_selected_c": first_marriage["params"].get(
                "selected_c"
            ),
        },
        "registration": {
            "issue": 42,
            "registration_id": registration_id,
            "fresh_registration_required": True,
        },
        "runtime_identity": dict(bindings.runtime_identity),
        "dependency_snapshot": {
            "command": f"{sys.executable} -m pip freeze",
            "canonicalization": "nonempty lines, lexical sort, LF terminator",
            "sorted_pip_freeze": list(bindings.sorted_pip_freeze),
            "sha256": bindings.dependency_sha256,
        },
        "integrity": {
            "source": {
                "git_commit": bindings.source.commit_sha,
                "worktree_clean": not bindings.source.dirty,
                "tree_pin": "git commit",
            },
            "design": {
                "gate_m6_design_commit": bindings.design_commit,
                "head_blob_sha256s": dict(bindings.design_blob_sha256s),
            },
            "floor": {
                "path": bindings.floor_run,
                "sha256": bindings.floor_run_sha256,
            },
            "candidate_specs": {
                name: {
                    "candidate_id": bindings.candidate_spec_ids[name],
                    "canonical_json_sha256": digest,
                    "canonical_spec": bindings.candidate_specs[name],
                }
                for name, digest in bindings.candidate_spec_sha256s.items()
            },
            "selection_evidence": {
                name: {
                    "path": SELECTION_EVIDENCE_PATHS[name],
                    "sha256": digest,
                }
                for name, digest in bindings.selection_evidence_sha256s.items()
            },
            "entry_dissolved_conformance": {
                "path": CONFORMANCE_EVIDENCE_PATH,
                "sha256": bindings.entry_dissolved_conformance_sha256,
                "record_and_publish": True,
                "covered_by_source_commit": True,
            },
            "environment_sidecar": {
                "path": bindings.environment_sidecar_path,
                "canonicalization": "sorted keys, compact JSON",
                "sha256": bindings.environment_sidecar_sha256,
            },
            "contract_ref": dict(bindings.contract_ref),
            "preregistered_comparisons": {
                surface: {
                    "expected": expected,
                    "observed": _compared_values(bindings)[surface],
                    "matched": True,
                }
                for surface, expected in PREREGISTERED_VALUES.items()
            },
        },
        "lineage": {
            **dict(phase.lineage),
            "floor_run": resolved.floor_path,
            "floor_sha256": resolved.floor_sha256,
        },
        "gate": {
            "name": "gate_m6",
            "status": "locked",
            "cells": [
                {
                    "cell": rule.cell,
                    "family": rule.family,
                    "split_unit": rule.split_unit,
                    "metric": rule.metric,
                    "tolerance": rule.tolerance,
                    "k": rule.k,
                    "rounding": rule.rounding,
                }
                for rule in contract.cells
            ],
            "tolerance_source": "locked gates.yaml gate_m6 views",
            "tolerances_recomputed": False,
        },
        "protocol": {
            "gate_seeds": list(contract.gate_seeds),
            "required_seed_passes": contract.required_seed_passes,
            "n_draws": contract.n_draws,
            "draw_index": list(range(contract.n_draws)),
            "draw_seeds": list(contract.draw_seeds),
            "split_fraction": 0.5,
            "split_units": {
                "marital": "household",
                "disability": "person",
                "earnings": "person",
            },
            "earnings_support": "realized_support_intersect_2014_domain",
            "earnings_support_symmetric_both_sides": True,
            "evaluation_support_adapter": (
                "engine.support.prepare_evaluation_support"
            ),
            "evaluation_mode": "GATED_REALIZED",
            "presence_basis": {
                "marital_disability": "START_OF_INTERVAL",
                "earnings": "EXACT_WAVE",
            },
            "undefined_draw_invalidates": True,
            "regenerated_surface_required": True,
        },
        "verdict": {
            "valid": valid,
            "pass": passed,
            "status": run_verdict["status"],
            "section_8_acceptance_status": conjunction["status"],
            "first_marriage_decomposition_valid": run_verdict[
                "first_marriage_decomposition_valid"
            ],
            "family_a_pass": bool(family_a["pass"]),
            "must_not_regress_pass": bool(must_not_regress["pass"]),
            "family_b_gated": False,
            "family_c_gated": False,
            "publishes_regardless": True,
            "certifies_nothing_about_mortality_drift": True,
            "earnings_certification": (
                "M6-first-certified forward earnings law; no gate_1 "
                "backward-law certificate transfers"
            ),
        },
        "candidate2_acceptance": acceptance,
        "gate_contract_result": family_a,
        "must_not_regress_result": must_not_regress,
        "acceptance_conjunction": conjunction,
        "run_conformance": run_verdict,
        "first_marriage_effect_decomposition": (first_marriage_decomposition),
        "family_a": family_a,
        "family_b": report_only["family_b"],
        "family_c": report_only["family_c"],
        "preflights": {
            "first_marriage_fit_and_transport": {
                **dict(first_marriage_diagnostics),
                "selection_ledger": dict(
                    bindings.first_marriage_selection_ledger
                ),
            },
            "candidate9_recertification": dict(preflight_1),
            "earnings_sign_path": dict(preflight_2),
        },
        "earnings_domain_floor_self_check": report_only[
            "domain_earnings_floor"
        ],
        "provenance": provenance,
        "fence": {
            "gates_yaml_read": (
                "gate_m6 protocol/cells plus design_commit identity"
            ),
            "post_boundary_macro_on_scored_path": False,
            "evaluation_mode": "GATED_REALIZED",
            "forward_realized_inputs": "rejected_by_contract",
            "certified_full_window_artifacts_read": False,
            "certified_full_window_artifacts_written": False,
        },
        "publishes_regardless": True,
    }
    return m6_runner._json_safe(artifact)


def _revalidate_source_identity(
    preparation: M6Candidate2Preparation,
) -> None:
    observed = capture_source_identity(preparation.repository)
    if observed != preparation.bindings.source:
        raise RuntimeError(
            "candidate-2 source commit changed after the run-start guard"
        )
    _assert_imported_source_tree(preparation.repository)


def _assert_preparation_unchanged(
    expected: M6Candidate2Preparation,
    observed: M6Candidate2Preparation,
    *,
    stage: str,
) -> None:
    if observed != expected:
        raise RuntimeError(
            f"candidate-2 guarded bindings changed {stage}; refusing run"
        )


def _designed_abort_report(
    preparation: M6Candidate2Preparation,
    error: FirstMarriagePreflightAbort,
    bundle: M6RefitBundle | Any | None,
    *,
    stage: str,
    full_inputs_loaded: bool,
    materialization_completed: bool,
) -> dict[str, Any]:
    family_spec = preparation.bindings.candidate_specs["family_transitions"]
    registered_first_marriage = next(
        component
        for component in family_spec["components"]
        if component["kind"] == "first_marriage"
    )
    diagnostics: dict[str, Any] = {
        "fit_audit": None,
        "recomputed_checksums": None,
        "registered_spec_params": dict(registered_first_marriage["params"]),
        "selection_ledger": dict(
            preparation.bindings.first_marriage_selection_ledger
        ),
        "hazard_table_ages_18_29": None,
    }
    if bundle is not None:
        try:
            model = _first_marriage_model(bundle)
            audit = getattr(model, "fit_audit", None)
            if audit is not None:
                diagnostics["fit_audit"] = audit.canonical_dict()
                try:
                    diagnostics["recomputed_checksums"] = dict(
                        recompute_support_aware_first_marriage_checksums(model)
                    )
                except (
                    FloatingPointError,
                    RuntimeError,
                    ValueError,
                ) as replay:
                    diagnostics["checksum_replay_error"] = {
                        "type": type(replay).__name__,
                        "message": str(replay),
                    }
            try:
                diagnostics["hazard_table_ages_18_29"] = (
                    _first_marriage_hazard_table(model)
                )
            except (
                AttributeError,
                FloatingPointError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as hazard:
                diagnostics["hazard_table_error"] = {
                    "type": type(hazard).__name__,
                    "message": str(hazard),
                }
        except FirstMarriagePreflightAbort:
            pass
    bindings = preparation.bindings
    report = {
        "schema_version": "gate_m6_candidate2.designed_abort.v1",
        "status": (
            "NO_REGISTERABLE_FIRST_MARRIAGE_FIT"
            if stage == "fit_preflight"
            else "FIRST_MARRIAGE_PRESCORE_ABORT"
        ),
        "candidate": {
            "number": bindings.candidate_number,
            "family_spec_id": bindings.candidate_spec_ids[
                "family_transitions"
            ],
            "engine_spec_id": bindings.candidate_spec_ids["engine_operations"],
        },
        "registration": {
            "issue": 42,
            "registration_id": preparation.registration_id,
        },
        "integrity": {
            "source": {
                "git_commit": bindings.source.commit_sha,
                "worktree_clean": not bindings.source.dirty,
            },
            "design": {
                "gate_m6_design_commit": bindings.design_commit,
                "head_blob_sha256s": dict(bindings.design_blob_sha256s),
            },
            "floor": {
                "path": bindings.floor_run,
                "sha256": bindings.floor_run_sha256,
            },
            "candidate_spec_sha256s": dict(bindings.candidate_spec_sha256s),
            "selection_evidence_sha256s": dict(
                bindings.selection_evidence_sha256s
            ),
            "entry_dissolved_conformance": {
                "path": CONFORMANCE_EVIDENCE_PATH,
                "sha256": bindings.entry_dissolved_conformance_sha256,
            },
            "dependency_snapshot": {
                "sha256": bindings.dependency_sha256,
                "sorted_pip_freeze": list(bindings.sorted_pip_freeze),
            },
            "runtime_identity": dict(bindings.runtime_identity),
            "environment_sidecar": {
                "path": bindings.environment_sidecar_path,
                "sha256": bindings.environment_sidecar_sha256,
            },
            "contract_ref": dict(bindings.contract_ref),
            "final_guard_revalidated": True,
        },
        "abort": {
            "type": type(error).__name__,
            "message": str(error),
            "stage": stage,
            "converted_to_gate_fail": False,
        },
        "first_marriage_diagnostics": diagnostics,
        "fences": {
            "full_inputs_loaded": full_inputs_loaded,
            "materialization_completed": materialization_completed,
            "projection_started": False,
            "score_started": False,
            "candidate_artifact_written": False,
        },
    }
    return m6_runner._json_safe(report)


def _verified_designed_abort(
    preparation: M6Candidate2Preparation,
    error: FirstMarriagePreflightAbort,
    bundle: M6RefitBundle | Any | None,
    *,
    stage: str,
    full_inputs_loaded: bool,
    materialization_completed: bool,
) -> M6Candidate2DesignedAbort:
    """Revalidate every binding before publishing a structured abort."""
    current = guard_registered_m6_candidate2_run(
        registration_id=preparation.registration_id,
        output=preparation.destination,
        root=preparation.repository,
    )
    _assert_preparation_unchanged(
        preparation,
        current,
        stage="before designed-abort publication",
    )
    _revalidate_source_identity(preparation)
    return M6Candidate2DesignedAbort(
        _designed_abort_report(
            preparation,
            error,
            bundle,
            stage=stage,
            full_inputs_loaded=full_inputs_loaded,
            materialization_completed=materialization_completed,
        )
    )


def execute_registered_m6_candidate2_run(
    input_plan: M6Candidate2InputPlan,
    *,
    registration_id: str,
    output: Path | str = DEFAULT_OUTPUT,
    root: Path | str | None = None,
    operations: M6Candidate2RunnerOperations | None = None,
    preparation: M6Candidate2Preparation,
) -> dict[str, Any]:
    """Fit/preflight before truth construction, then write the pair last."""
    if not isinstance(input_plan, M6Candidate2InputPlan):
        raise TypeError(
            "candidate-2 execution requires M6Candidate2InputPlan; eager "
            "M6HarnessInputs are prohibited"
        )
    prepared = preparation
    registration = validate_candidate2_registration_id(registration_id)
    if registration != prepared.registration_id:
        raise RuntimeError("candidate-2 registration changed after data load")
    repository = _repository(root)
    if repository != prepared.repository:
        raise RuntimeError("candidate-2 repository changed after data load")
    destination = _resolve_destination(repository, output)
    if destination != prepared.destination:
        raise RuntimeError("candidate-2 output changed after data load")
    current = guard_registered_m6_candidate2_run(
        registration_id=registration,
        output=destination,
        root=repository,
    )
    _assert_preparation_unchanged(
        prepared,
        current,
        stage="after the input-plan factory",
    )
    _revalidate_source_identity(prepared)

    ops = operations or default_operations()
    resolved = prepared.resolved
    bundle: M6RefitBundle | Any | None = None
    try:
        bundle = ops.fit(input_plan.fit_inputs)
        fit_preflight = ops.first_marriage_preflight(bundle)
    except FirstMarriagePreflightAbort as error:
        raise _verified_designed_abort(
            prepared,
            error,
            bundle,
            stage="fit_preflight",
            full_inputs_loaded=False,
            materialization_completed=False,
        ) from error
    incumbent_bundle = ops.fit_postrepair_incumbent(
        input_plan.fit_inputs,
        bundle,
    )
    inputs = input_plan.load_full_inputs()
    if not isinstance(inputs, M6HarnessInputs):
        raise TypeError("full-input loader must return M6HarnessInputs")
    if inputs.refit_inputs is not input_plan.fit_inputs:
        raise RuntimeError(
            "full inputs do not carry the exact preflighted fit-input object"
        )
    phase = ops.materialize(inputs, bundle)
    _assert_phase_candidate_binding(phase, prepared.bindings)
    incumbent_phase = ops.materialize_postrepair_incumbent(
        phase,
        incumbent_bundle,
    )
    try:
        first_marriage = ops.first_marriage_disclosure(
            inputs,
            phase,
            resolved,
            fit_preflight,
        )
    except FirstMarriagePreflightAbort as error:
        raise _verified_designed_abort(
            prepared,
            error,
            bundle,
            stage="transport_disclosure",
            full_inputs_loaded=True,
            materialization_completed=True,
        ) from error
    preflight_1 = ops.preflight_1(inputs, phase, resolved.contract)
    preflight_2 = ops.preflight_2(inputs, phase, resolved.contract)
    seed_runs = tuple(
        ops.score_seed(inputs, phase, resolved.contract, seed)
        for seed in resolved.contract.gate_seeds
    )
    if tuple(run.seed for run in seed_runs) != resolved.contract.gate_seeds:
        raise RuntimeError("seed phase returned results out of protocol order")
    incumbent_seed_runs = tuple(
        ops.score_seed(inputs, incumbent_phase, resolved.contract, seed)
        for seed in resolved.contract.gate_seeds
    )
    if (
        tuple(run.seed for run in incumbent_seed_runs)
        != resolved.contract.gate_seeds
    ):
        raise RuntimeError(
            "postrepair incumbent seed phase returned results out of order"
        )
    first_marriage_estimator_delta = _first_marriage_estimator_delta(
        seed_runs,
        incumbent_seed_runs,
        resolved.contract,
    )
    gate_score = ops.aggregate(
        resolved.contract, [run.score for run in seed_runs]
    )
    domain_floor = ops.domain_floor(inputs, phase, resolved)
    report = ops.report_only(
        inputs,
        phase,
        resolved,
        seed_runs,
        domain_floor,
    )
    artifact = assemble_m6_candidate2_artifact(
        registration_id=prepared.registration_id,
        inputs=inputs,
        phase=phase,
        resolved=resolved,
        seed_runs=seed_runs,
        gate_score=gate_score,
        first_marriage_diagnostics=first_marriage,
        first_marriage_estimator_delta=first_marriage_estimator_delta,
        preflight_1=preflight_1,
        preflight_2=preflight_2,
        report_only=report,
        bindings=prepared.bindings,
    )
    final = guard_registered_m6_candidate2_run(
        registration_id=registration,
        output=destination,
        root=repository,
    )
    _assert_preparation_unchanged(
        prepared,
        final,
        stage="before artifact publication",
    )
    _revalidate_source_identity(prepared)
    ops.write(prepared.destination, artifact, prepared.sidecar_payload)
    return artifact


__all__ = [
    "CANDIDATE1_OUTPUT",
    "CANDIDATE1_REGISTRATION_ID",
    "CONFORMANCE_EVIDENCE_PATH",
    "CONFORMANCE_SOURCE_BINDING_KEYS",
    "DEFAULT_OUTPUT",
    "DESIGN_BLOB_PATHS",
    "FIRST_MARRIAGE_GATE_CELL",
    "LANDED_REPAIR_COMMIT",
    "M6Candidate2Bindings",
    "M6Candidate2DesignedAbort",
    "M6Candidate2Identity",
    "M6Candidate2InputPlan",
    "M6Candidate2Preparation",
    "M6Candidate2RunnerOperations",
    "M6SourceIdentity",
    "MUST_NOT_REGRESS_REQUIRED_SEEDS",
    "MUST_NOT_REGRESS_SEEDS",
    "MUST_NOT_REGRESS_TOLERANCES",
    "PHASE_ORDER",
    "PREREGISTERED_COMPARISON_KEYS",
    "PREREGISTERED_VALUES",
    "SCHEMA_VERSION",
    "SELECTION_EVIDENCE_PATHS",
    "assemble_m6_candidate2_artifact",
    "assert_candidate2_identity_is_frozen",
    "assert_preregistered_bindings",
    "capture_source_identity",
    "collect_candidate2_bindings",
    "default_operations",
    "execute_registered_m6_candidate2_run",
    "guard_registered_m6_candidate2_run",
    "resolve_candidate2_identity",
    "validate_candidate2_registration_id",
    "write_new_candidate2_artifact",
]
