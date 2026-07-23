"""Additive candidate bindings for optional M6 engine operations.

The incumbent candidate registry under ``models.family_transitions`` is a
frozen family-model artifact.  Engine operations therefore live in this
sibling registry: a candidate with no entry continues to use the incumbent
engine law, while a registered entry declares only its additive operation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "CANDIDATE_2",
    "CANDIDATE_3",
    "CORRELATED_REFRESH_OPERATION_ID",
    "CORRELATED_REFRESH_OPERATION_KIND",
    "RANK_REFRESH_OPERATION_ID",
    "RANK_REFRESH_OPERATION_KIND",
    "REGISTRY",
    "CandidateSpec",
    "OperationSpec",
]

RANK_REFRESH_OPERATION_KIND = "forward_earnings.rank_refresh"
RANK_REFRESH_OPERATION_ID = "stable_coordinate_exact_age_weighted_knn.v1"
CORRELATED_REFRESH_OPERATION_KIND = "forward_earnings.correlated_refresh"
CORRELATED_REFRESH_OPERATION_ID = "stationary_markov_reset_on_gap.v1"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class OperationSpec:
    """One immutable engine operation selected by a candidate."""

    kind: str
    implementation_id: str
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", _freeze(self.params))

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "implementation_id": self.implementation_id,
            "params": _plain(self.params),
        }

    def __reduce__(self):
        return (
            type(self),
            (self.kind, self.implementation_id, _plain(self.params)),
        )


@dataclass(frozen=True)
class CandidateSpec:
    """A complete additive engine-operation declaration for one candidate."""

    candidate_id: str
    contract_revision: str
    operations: tuple[OperationSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        kinds = [operation.kind for operation in self.operations]
        if len(kinds) != len(set(kinds)):
            raise ValueError("candidate repeats an engine operation kind")

    def operation(self, kind: str) -> OperationSpec | None:
        """Return the declared operation of ``kind``, if present."""
        return next(
            (
                operation
                for operation in self.operations
                if operation.kind == kind
            ),
            None,
        )

    def canonical_dict(self) -> dict[str, Any]:
        """Return the deterministic resolved candidate payload."""
        return {
            "candidate_id": self.candidate_id,
            "contract_revision": self.contract_revision,
            "operations": [
                operation.canonical_dict() for operation in self.operations
            ],
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


CANDIDATE_2 = CandidateSpec(
    candidate_id="m6_candidate2_engine_v1",
    contract_revision="m6_amendment_4_qstar_lock_2026_07_18",
    operations=(
        OperationSpec(
            kind=RANK_REFRESH_OPERATION_KIND,
            implementation_id=RANK_REFRESH_OPERATION_ID,
            params={
                "q": 0.55,
                "substream_codes": {
                    "memory-refresh-gate": 4,
                    "memory-refresh-rank": 5,
                },
            },
        ),
    ),
)

CANDIDATE_3 = CandidateSpec(
    candidate_id="m6_candidate3_engine_v1",
    contract_revision="m6_amendment_6_rhostar_lock_2026_07_23",
    operations=(
        *CANDIDATE_2.operations,
        OperationSpec(
            kind=CORRELATED_REFRESH_OPERATION_KIND,
            implementation_id=CORRELATED_REFRESH_OPERATION_ID,
            params={"rho": -0.60},
        ),
    ),
)

# Candidate 1 has no additive engine operation and deliberately has no entry.
# Its existing family and earnings specs remain the only incumbent authority.
REGISTRY: Mapping[int, CandidateSpec] = MappingProxyType(
    {2: CANDIDATE_2, 3: CANDIDATE_3}
)
