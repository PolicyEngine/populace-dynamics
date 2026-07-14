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

## 1. Summary

M7 is the **trust-fund accounting layer** of the roadmap. The #113 M7 row reads,
verbatim:

> **Full revenue + trust-fund accounting**: OASDI cost/income rates, trust-fund
> ratio path, 75-yr actuarial balance in % of taxable payroll | *gate:* Reproduce
> the corresponding TR baseline deficit within a pre-registered tolerance;
> per-provision balance vs the Five-Approaches A-tables in LEVELS, not just
> ordinally | *unlocks:* The Rosetta stone (#74 anchor 2) graded in its own units

M2 already built this accounting **once**, on a *closed* observed cohort: the
`scripts/m2_pseudo_projection.py` pseudo-projection added "taxable payroll,
combined-OASDI revenue, claiming × survival outlays on a calendar ledger, a PV
balance analogue, and a calibrated-reserve exhaustion year" to the #115 common
frame of 1,549 real PSID careers born 1943–1957 (`runs/m2_pseudo_projection_v1.json`).
M2 named its own limits precisely: *levels are frame-relative by construction and
are NOT graded; the transportable content is signs, orderings, and delta-
orderings.* M7 is the same accounting, generalized two ways: (a) it runs on M6's
**open** projected panel (immigration entrants, forward earnings, DI, forward
couple formation), and (b) it adds the beneficiary classes M2 excluded (DI
disabled-worker, 402(b)/(c)/(e)/(f) auxiliary), each **bounded by what its gate
actually certified** (§2.6).

**The layer is deterministic (§3).** All stochastic content — every earnings
draw, mortality draw, marital transition, claiming draw — is realized *upstream*
in M6's wave loop. M7 consumes a fixed panel and computes sums, caps, products,
and a recursion. Given the same panel and the same pinned parameters, M7's
accounts are **byte-identical**. There is no RNG at the accounting layer.

**M7 runs in two modes**, and the distinction is load-bearing for the gate:

1. **The M2-reproduction run** (the internal anchor, §5.1): feed M7's accounting
   engine the *exact M2 setup* — the closed #115 frame, OASI-only, wage-indexed
   real dollars, the calibrated reserve, no DI / no auxiliary / no immigration —
   and it must emit `runs/m2_pseudo_projection_v1.json`'s committed numbers, or
   name every delta. This isolates the **arithmetic** from the projection.
2. **The M6-panel run** (the deliverable levels): the same arithmetic on M6's
   open panel, producing the OASDI cost/income-rate and balance path — carried
   **report-only** against external Trustees corridors under the level-vs-
   composition caveat (§5.3).

**What M7 proposes to gate (§6)** is neither an external-level match nor an
ordering: it is **two internal identities** — the M2-reproduction and the
exact balance-identity closure (`residual == 0`). The choice not to gate the
external TR deficit *in levels* — despite the roadmap row's wording — is the
central design judgment, argued in §5.3 and surfaced for the referee in §8
decision 1. It follows M2's discipline directly: on a survey panel, the honest
certifiable content is arithmetic self-consistency and reproduction, not a
levels-match to a universe the panel does not represent.

## 2. Scope — the per-projection-year accounting on the projected panel

The accounting is defined per projection year `y` over M6's panel, aggregable by
cohort. Everything below is a pure function of (the panel, the pinned SSA
parameters); §3 states the determinism law that makes that precise.

### 2.1 The inputs M7 consumes from M6 (the §6 handoff)

M6 §6 ("M7 interface sketch — levels the engine must expose") already pins the
handoff. M7 **binds to exactly those exposed levels** and computes no dynamics of
its own. Quoting the M6 contract, M7 consumes, keyed by `(year)`:

- **covered / taxable earnings aggregate by year** — `Σ w · min(earnings_y,
  taxable_max_y)`, with **both capped and uncapped** person earnings exposed so
  cap-reform provisions (`cap_150k`, `elimination`) recompute without re-
  simulation (M6 §6; M2 `revenue_side.cap_150k`, `payroll_increments`);
- **the taxable-maximum interaction** — each person's year-`y` earnings and the
  year-`y` wage base;
- **benefit outlays by type by year** — OASI retired-worker, DI disabled-worker,
  402(b)/(c)/(e)/(f) auxiliary, survival-weighted on the calendar (M4 DI +
  auxiliaries; FRA conversions per `disability_conversion.py`);
- **immigration entries by year** — the synthetic entrant cohorts M2 structurally
  lacks (M2's frame is *closed*, `n_common_frame 1549`, `weight_sum 33696344.0`);
- **the per-year alignment-layer adjustments** — the versioned interventions M6
  applies each year (M6 §4.8), so M7 reports scheduled-vs-payable on an *audited*
  alignment (#113 named-hard-part 2);
- **the OASDI rate constants and Trustees vintage assumptions** — the pe-us rate
  nodes where they exist (M2 `revenue_side.oasdi_combined_rate.combined 0.124`)
  and the discount / interest / CPI assumptions M2 carries as TR-cited constants
  (M2 `balance_analogue.discount_rate 0.029`, `tr_vintage_cite`), which become
  M7's versioned alignment inputs (§4).

**Boundary discipline.** M6 "stops at exposing these levels"; M7 "lifts it onto
the open panel." M7 therefore introduces **no new stochastic law and reads no
gate M6 did not already certify** — it is arithmetic over M6's output plus §4's
parameter bindings. If a level M7 needs is *not* in M6's exposed surface, that is
an M6 gap to be raised against M6 §6, not silently synthesized here (§8 decision 7).

<!-- M7-CURSOR-DO-NOT-SHIP -->
