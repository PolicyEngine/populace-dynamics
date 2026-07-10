# ADR 0001: Populace/Axiom seam ownership

**Status:** Accepted

## Context

[Issue #100](https://github.com/PolicyEngine/populace-dynamics/issues/100)
records the integration design for carrying dynamics through the Populace panel
to full tax-benefit incidence. [External review issue
#106](https://github.com/PolicyEngine/populace-dynamics/issues/106), finding 8,
identified a contradiction between that decision, the technical specification,
and the implemented PolicyEngine-US boundary. This ADR accepts issue #100's
ownership decision as the authoritative record.

## Decision

We adopt the following three workstreams and sequencing from issue #100:

### W1 — Histories onto the Populace panel (the funded build)

Impute gate-certified earnings histories (gate-1-passing generator) and demographic histories (gate-2 transition models, pending that gate's first pass) onto the certified Populace CPS file: CPS↔PSID covariate transport, then calibration against SSA administrative targets (Statistical Supplement award/beneficiary tables, already staged with provenance in this program). Deliverable: Populace gains a longitudinal dimension with per-record benefit-relevant state (AIME/PIA, marital/claiming history). Each stage inherits the pre-registration discipline (transport validation gates before scored use).

### W2 — Benefits into policyengine-us (the one-variable seam)

Finding from the survivor-plumbing build (#80): pe-us takes `social_security` as an **uprated survey input** — no benefit formula exists upstream. Integration is therefore replacing a survey input with the modeled variable; all downstream incidence (taxation of benefits, program interactions, MTRs) flows through existing policyengine machinery unchanged. **Interim product available before W1 completes:** dynamics-computed benefit deltas for a reform, fed as inputs to a standard policyengine simulation on the current-year certified Populace file → full current-year tax-benefit incidence of an SS reform. (The taxation-of-benefits work already exercises half this pattern.)

### W3 — Axiom as the determinism layer (upstream the auxiliary rules)

The §415 chain here is cross-validated against the Axiom engine 240/240 exact-to-cent. Next: encode 402(b)/(c)/(e)/(f) (spousal, survivor incl. RIB-LIM, the remarriage-at-60 predicate — implemented here and validated against SSA worked examples in #80, but **absent from upstream pe-us**) as rulespec encodings; cross-validate against this repo's implementation; upstream real formulas to pe-us. Outcome: pe-us gains auxiliary-benefit computation, the encodings gain a validated Title II slice, and this program's statutory layer becomes independently verified per the estimation/determinism split.

Sequencing: W3 and W2-interim are startable now; W1 is the funded build and gates the panel-based projection products. Every scored output continues through the pre-registered evaluation protocol of this repository.

## Consequences

- W1 owns CPS↔PSID transport, SSA-target calibration, and the longitudinal
  Populace panel required for projection products.
- W2 owns the interim PolicyEngine-US seam: modeled `social_security` values or
  reform deltas enter the current-year Populace file so existing PolicyEngine
  machinery can calculate downstream tax-benefit incidence.
- W3 owns auxiliary-rule upstreaming through Axiom/rulespec, with the frozen
  oracle in this repository serving as the cross-validation target.
- W3 and the W2 interim product can proceed before W1 finishes; W1 gates
  panel-based projection products.
