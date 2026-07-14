# M7 trust-fund accounting — design and the M2-reproduction / balance-identity gate proposal (draft for referee)

- **Design id**: `2026-07-14-m7-trust-fund-accounting`
- **Roadmap**: [#113](https://github.com/PolicyEngine/populace-dynamics/issues/113)
  M7 (full revenue + trust-fund accounting), the milestone after #113 M6 (the
  projection engine) and before #113 M8 (integrated scoring). Program-design
  anchors: #74 (the anchor catalogue, the "Rosetta stone" anchor 2), #100
  (W1/W2/W3 seams), #106 (external review). Depends on M6 exposing the §6 levels
  (`docs/design/m6_projection_engine.md` §6) on the projected panel.
- **Status**: DESIGN (draft, revision 1) — **for the adversarial fable referee
  round, which follows the in-flight `gate_m6` verdict** (this document does not
  trigger it). **This document edits no `gates.yaml` cell, moves no threshold,
  builds no floor, writes no test, and runs no scored anything.** The proposed
  `gate_m7` in §6 is a *proposal for the referee*, not a lock.
- **What M7 is, in one sentence**: a **deterministic accounting layer** that
  takes M6's already-realized person-period panel plus vintage-pinned SSA
  parameters and emits, per projection year, taxable payroll, payroll-tax
  revenue, benefit outlays by beneficiary class, and a trust-fund balance
  ledger — reproducing M2's committed pseudo-projection numbers exactly on the
  M2 frame, and lifting the same arithmetic onto M6's open panel as report-only
  levels.
- **What M7 certifies (proposed)**: two **internal** identities — (1) the
  **M2-reproduction identity** (M7's accounting on the M2 setup reproduces
  `runs/m2_pseudo_projection_v1.json`'s committed numbers, or names every
  delta) and (2) **exact balance-identity closure** (`start + revenue +
  interest − outlays = end`, residual `== 0` to floating point, on every
  projection year). It **explicitly does not gate against external SSA/Trustees
  aggregates** — those enter as **report-only** anchors under a level-vs-
  composition caveat (§5). This is the M2 precedent ("levels are frame-relative;
  signs and orderings are the transportable content") carried into M7.
- **The one caveat the referee must weigh first**: the roadmap #113 M7 row asks
  to *"[r]eproduce the corresponding TR baseline deficit within a pre-registered
  tolerance; per-provision balance vs the Five-Approaches A-tables in LEVELS,
  not just ordinally."* That is an **external-level** gate. M2 established — and
  this design argues (§5.3, §8 decision 1) — that a **1,549-career PSID survey
  panel is not the covered-worker universe**, so external *levels* are not
  honestly gradable on this frame; only internal identities and normalized
  shapes are. §6 therefore proposes an internal gate and §8 decision 1 surfaces
  the contradiction with the roadmap wording for the referee to adjudicate,
  rather than silently resolving it.
- **Evidence base cited by path+field**: the roadmap (#113 M7 row, verbatim in
  §1); the committed M2 pseudo-projection (`runs/m2_pseudo_projection_v1.json`,
  `scripts/m2_pseudo_projection.py`, commit `747966cd`, `pe_us_revision
  bf71be3b`); the SSA parameter/benefit oracle (`ss/params.py`, `ss/benefits.py`);
  the M6 handoff (`docs/design/m6_projection_engine.md` §6, §2.8.10); the M4
  disability gate (`gates.gate_m4`, `disability_hazard_sim.py`,
  `disability_conversion.py`); the Mermin/Smith anchors
  (`scripts/replication_cost_ordering.py`, `scripts/replication_mermin_rows.py`);
  the external-vintage guards (`engine/refit.py:825-838`,
  `harness/m6_inputs.py:198-250`).

## Revision log (finding → section)

- Revision 1 seeds the document for the referee round. No findings yet; the
  referee's verdict and any ten-decision adjudication will be mapped here in
  revision 2, exactly as the M6 doc maps PR #170's round.

<!-- M7-CURSOR-DO-NOT-SHIP -->
