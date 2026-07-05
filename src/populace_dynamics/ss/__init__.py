"""Title II benefit computation from observed earnings histories.

Role: the ORACLE, not the rules engine. Net-new statute encoding
lives in Axiom (rulespec-us ``us/statutes/42/415``, ``402``,
``416``); this module is a frozen-scope Python reference
implementation, parameterized from policyengine-us, that (a) runs
scoring on observed histories until the Axiom dense engine gains
cross-period reduction (axiom-rules-engine#67 — top-N over a
person's period axis, required for executable 415(b)), and
(b) thereafter serves as the cross-engine validation oracle:
engine-versus-oracle PIA agreement is a statutory-resolution
artifact. Do not extend this module's rule coverage; extend the
Axiom encodings.
"""

from __future__ import annotations

from populace_dynamics.ss.benefits import (
    age62_monthly_benefit,
    aime,
    early_reduction,
    pia,
)
from populace_dynamics.ss.params import SSAParameters, load_ssa_parameters

__all__ = [
    "SSAParameters",
    "load_ssa_parameters",
    "aime",
    "pia",
    "early_reduction",
    "age62_monthly_benefit",
]
