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

### 2.2 Taxable payroll and payroll-tax revenue (employer + employee, wage-base cap)

**Taxable payroll.** For each person `i` and covered year `y`,

```
taxable_earnings[i,y] = min( earnings[i,y] , wage_base(y) )
taxable_payroll[y]    = Σ_i  w[i,y] · taxable_earnings[i,y]
```

where `w[i,y]` is M6's per-year calibrated weight and `wage_base(y)` is the
contribution-and-benefit base. The cap is the committed `creditable_history`
convention (`ss/benefits.py:68-75`, "cap each year's earnings at that year's wage
base") applied over **all** working years — M2's `taxable_payroll_convention`
notes it reuses the AIME-convention per-year capping *"but over ALL working
years (every covered year pays tax), not the highest-35 selection."* `wage_base(y)`
is a step function read from pe-us (`ss/params.py:151-160`, `wage_base_for`).

**Revenue.** Payroll-tax income is the combined OASDI rate on taxable payroll:

```
revenue[y] = rate_combined · taxable_payroll[y]
           = ( rate_employee + rate_employer ) · taxable_payroll[y]
```

The employer and employee halves are **both** counted (the task's employer +
employee requirement): `rate_employee = rate_employer = 0.062`, summing to the
`rate_combined = 0.124` M2 loaded from `gov/irs/payroll/social_security/rate/
{employee,employer}` (26 USC 3101(a) employee + 3111(a) employer). This is the
**scheduled-rate** convention a Trustees projection uses, applied to all taxable
payroll — not the historical rate ramp (M2 `named_deltas`: "combined OASDI rate
from the pe-us statute series (12.4%) … as the current-law scheduled rate").

**Cap-reform recomputation.** Because M6 exposes **uncapped** person earnings
(§2.1), the two revenue-side provisions recompute without re-simulation:
`cap_150k` raises the base to $150k stated in 2016 dollars, wage-indexed by NAWI
to each earnings year (42 USC 430; M2 `revenue_side.cap_150k`); `elimination`
removes the cap (`min(earnings, ∞) = earnings`); `payroll_plus_{1,2}pp` add
`0.01`/`0.02` to `rate_combined`. Each is a pure recomputation on the exposed
levels.

**The OASI-vs-DI rate split is an open decision (§8 decision 4).** Statute splits
the 12.4% into OASI (10.6%) and DI (1.8%). M2 applied the *combined* 12.4% against
*OASI-only* outlays — a deliberate asymmetry that drives its positive baseline
balance (§2.4, §5.3). M7's "OASI(+DI)" scope must state, per run, whether revenue
is allocated to a combined OASDI fund or split across two funds; the M2-repro run
(§5.1) must reproduce M2's combined-rate / OASI-outlay convention exactly.

### 2.3 Benefit outlays by beneficiary class

The outlay convention is M2's, generalized. For a retired worker, M2 places at
calendar year `birth_year + a`:

```
outlay[i, a] = 12 · PIA[i] · Σ_{claim c ≤ a} pmf_c(i) · benefit_factor(c) · S(62→a | i)
```

where `pmf_c` is the B2 claim-age distribution, `benefit_factor(c)` the 402(q)/(w)
early-reduction / delayed-credit factor (`ss/benefits.py:134-164`), and `S` the
B1 NCHS-2023 × PSID-band survival weight (M2 `outlay_side.convention`). `PIA[i]`
is `pia(aime(history), eligibility_year)` (`ss/benefits.py:100-131`). M7 sums
`Σ_i w[i,y] · outlay[i,y]` by class:

- **OASI retired-worker** — the M2 class, unchanged. Own AIME→PIA→claim-factor,
  survival-weighted.
- **DI disabled-worker** — the M4 addition. **Bounded by `gate_m4`**: the M4 gate
  is *anchor-based* and certifies the work-limitation **incidence/recovery
  hazards**, the disabled-occupancy **prevalence stock** (`prevalence.50-59`),
  and the near-FRA **exit composition** (retirement-vs-return, Table 50
  dominance) — but its `covers` field states plainly *"no SSA DI LEVEL is
  gated."* So the disabled **headcount trajectory** feeding DI outlays rides
  certified shape/dominance, while the **DI benefit dollar level** is
  report-only (the disability PIA, like every PIA, is never gated; M6 §2.8.3a).
  The **DI→retirement conversion at FRA** (`disability_conversion.py`) moves the
  disabled worker onto the retired-worker rolls at FRA — the "conversion column"
  of Table 6.B5.1.
- **402(b)/(c) spouse and 402(e)/(f) survivor auxiliary** — the M3/household
  addition. `spousal_benefit` (`ss/benefits.py:205-236`) and `widow_benefit`
  (`ss/benefits.py:266-328`) encode the excess-spouse and widow(er) amounts
  including dual entitlement, the RIB-LIM, and the 71.5% survivor floor. These
  require M3 couple/household structure (who is married to whom, who survives
  whom) — **bounded by `gate_2` (2b household composition, 2c couple formation)
  and mortality**. Their rate constants have **no pe-us node** and are carried as
  statute-cited constants (§4.3, the LOUD gap).

**The unifying certification rule.** Across all three classes, **every benefit
dollar level is report-only** (AIME/PIA/claiming/auxiliary are never gated). What
*is* certified is the **population composition** that determines *who* draws
*which* class in *which* year — marital state (gate_2), disability state
(gate_m4), survival (mortality), earnings (gate_1, gate_m6). M7 outlays are thus
"certified-composition × report-only-level" products. This is the exact meaning
of the scope phrase "OASI(+DI, bounded by what M4/gate-2 actually certified)":
M7 does not manufacture certified benefit levels the underlying gates declined to
certify. §2.6 makes the boundary explicit per class.

### 2.4 The trust-fund balance identity

The per-year ledger is the OASDI fund recursion:

```
TF[y] = TF[y-1] + revenue[y] + interest[y] − outlays[y]              (I)
interest[y] = interest_rate(y) · TF[y-1]        (start-of-year-balance convention)
```

with `TF[y_0-1] = reserve_0` the opening reserve. From it derive the two headline
Trustees objects the #113 M7 row names:

- **the trust-fund ratio path** — `ratio[y] = TF[y-1] / outlays[y]`, start-of-year
  assets as a percent of that year's cost (the standard SSA definition);
- **the 75-year actuarial balance in % of taxable payroll** —
  `balance = [ PV(revenue) + reserve_0 − PV(outlays) − PV(target_ending_reserve) ]
  / PV(taxable_payroll)` in the summarized-rate form, or M2's flow-only analogue
  `[PV(revenue) − PV(outlays)] / PV(taxable_payroll)` (M2 `balance_analogue`,
  discount `0.029`).

**Two accounting views, which M7 must carry and reconcile.** M2 deliberately ran
two: a **discounted** PV balance analogue and an **undiscounted, reserve-
calibrated** exhaustion ledger (M2 `named_deltas`: "two accounting views, both
frame-relative"). The exhaustion year is the first `y` with `TF[y] < 0`
(fractional, linearly interpolated; M2 `exhaustion_analogue`). Identity (I) is an
**accounting identity** — it must close to floating-point zero by construction if
the components are materialized consistently. That closure is one half of the
proposed gate (§6): for every year `y`, the artifact carries `TF[y-1]`,
`revenue[y]`, `interest[y]`, `outlays[y]`, `TF[y]` independently, and a referee
can recompute (I) and confirm the residual is `0`. It is **non-vacuous** because a
dropped beneficiary class, a double-counted employer half, or a mis-discounted PV
breaks it — the same reconcile-to-`0.0` discipline M6 applied to its recombination
identity (M6 candidate-16 residual `1.7e-18`, "reconciled to 0.0").

**Interest and the opening reserve are LOUD input gaps (§4.3).** `interest_rate(y)`
(the trust-fund's special-issue yield) has **no pe-us node** — M2 confirms "no
policyengine-us rate node" for the TR rate and carried `discount_rate 0.029` as a
TR-cited constant. The opening reserve M2 did not source at all: it **calibrated**
`reserve_0 = 7,889,236,006,574.16` so the baseline exhausts in Smith's 2034 year
(M2 `calibration_disclosure`). Whether M7 keeps the calibrated-reserve convention
or binds an actual SSA Trustees opening-balance series is §8 decision 3.

### 2.5 COLA and awards indexation

Two distinct indexation channels feed the accounts, and only one is sourced today.

- **Awards indexation (present in pe-us).** Bend points and the wage base are
  NAWI-indexed at *eligibility* (415(a): `bend_points(year)` uses `NAWI(year−2) /
  NAWI(1977)`, `ss/params.py:138-149`; the wage base is its own NAWI-driven step
  series). This is fully sourced from pe-us and **vintage-pinned exactly as M6
  §2.8.10.2 pins it**: realized NAWI ≤2014, `I_proj` beyond — never realized
  post-`T*` NAWI on any scored path. For a forward projection to 2100, future
  eligibility years read projected NAWI; M7 inherits M6's wage-index surface
  rather than re-deriving it (§8 decision 6).
- **Benefits-in-payment indexation / COLA (the LOUD gap, §4.3).** After a worker
  claims, the benefit is uprated annually by the COLA (CPI-W, 215(i)). **pe-us's
  `ss/` machinery applies no COLA** and binds no CPI-W series — `ss/benefits.py`
  computes the claim-year benefit only. M2 sidestepped this by working in
  **wage-indexed real (2048 age-60-indexing) dollars**, in which a benefit is
  carried at constant real PIA across ages — COLA and NAWI-deflation net out of
  the *level* convention, so no explicit CPI-W series is needed. The Mermin
  `reduced_cola` provision (COLA − 0.4pp) is then a **benefit-side encoding**
  reused verbatim (M2 `outlay_side`), not a nominal-series perturbation.

**The consequence for M7 is a nominal-vs-real fork (§8 decision 2).** If M7 keeps
M2's **real** convention, no CPI-W series is required and interest is a *real*
rate (2.9%); the accounts stay comparable to M2. If M7 goes **nominal** (the units
a Trustees report actually prints), it must bind a CPI-W / COLA series *and* a
nominal special-issue interest series — **neither has a pe-us node** — and every
benefit-in-payment path grows by COLA. The two are a coherent pair (real flows ↔
real interest; nominal flows ↔ nominal interest + COLA); mixing them breaks
identity (I). M7 must declare the convention per run; the M2-repro run (§5.1) is
real by construction.

### 2.6 The DI / OASI certification boundary (what M4 and gate-2 actually certified)

Making the scope phrase "bounded by what M4/gate-2 actually certified" operational,
per class:

| Class | Composition certified by | Level status |
|---|---|---|
| OASI retired-worker | earnings (gate_1/gate_m6), claiming mix, survival | **report-only** (PIA never gated) |
| DI disabled-worker | gate_m4 hazards + `prevalence.50-59` stock + Table-50 exit **dominance** (all shape/dominance) | **report-only** (M4: "no SSA DI LEVEL is gated") |
| DI→retirement conversion at FRA | gate_m4 exit composition; `disability_conversion.py` | **report-only** level |
| 402(b)/(c) spouse | gate_2b/2c household + couple formation | **report-only** (aux rate = statute constant) |
| 402(e)/(f) survivor | gate_2b + mortality (who survives whom) | **report-only** (aux rate = statute constant) |

Every level is report-only; only composition is certified. M7 must **not** present
any beneficiary-class dollar aggregate as a certified level, and must **not** claim
a certified DI trust-fund balance — the DI fund is a report-only lift of a class
whose levels no gate certified. The auxiliary classes additionally depend on M3
tranches (2b/2c) whose own lock status bounds them; if a tranche M7 leans on is not
locked at run time, that class is disclosed report-only with its tranche gap named.

## 3. The determinism law

**Law.** *Given the same M6 panel and the same pinned parameter bundle (§4), the
M7 accounts are byte-identical across runs, processes, and platforms. There is no
stochastic element at the accounting layer.*

This is stronger than M6's discipline and simpler to guarantee, because M7
consumes no randomness. Every stochastic draw — earnings, mortality, marital
transitions, claiming age, disability incidence, immigrant entry — is realized
**upstream** in M6's wave loop and is already a fixed column of the panel M7
reads. M6 owns a projection RNG stream registry (M6 §3, the `(draw × module ×
period × person)` spawn tree); **M7 owns no RNG stream and spawns none.** The
accounting is a pure function

```
accounts = A( panel , params )        with no rng argument
```

built from caps (`min`), products, weighted sums (`Σ w·x`), a discount, and the
recursion (I). Determinism therefore reduces to two mechanical obligations:

1. **Pinned reduction order.** Floating-point addition is non-associative, so
   every aggregate `Σ_i w[i,y]·x[i,y]` fixes a canonical iteration order (sort by
   `(person_id, year)`) and a fixed dtype, so the summed bytes do not depend on
   row order, thread count, or grouping. This is the M7 analogue of M6's
   draw-by-draw byte-identity (M6 candidate-16 "RNG-neutral … byte-identical
   draw-by-draw"); here it is reduction-order neutrality.
2. **Pinned parameter bytes.** The parameter side is fully determined by the
   pe-us 1.752.2 pin plus the statute/TR constants (§4), each vintage-asserted and
   hash-gated (§4.4). No parameter is read from the environment beyond the pinned
   pe-us directory.

**Verification is trivial and belongs in the gate's determinism check:** run M7
twice on the same panel and byte-diff the artifact (modulo the wall-clock
`elapsed_seconds` field, as M2 already excludes). Any difference is a defect —
an unpinned reduction, a dict-ordering leak, or a stray RNG call — not noise. The
determinism law is what makes the M2-reproduction identity (§5.1) a *byte* claim
rather than an approximate one.

## 4. Input bindings — the SSA series, vintage-pinned with tamper gates

M7's parameter side must be pinned to the same bar M6 §2.8.10 set for its external
references: fixed source, fixed vintage, tamper-evident, and **loudly flagged**
where the series does not exist in policyengine-us and would need a 3d-style
amendment. The deployment pin is **policyengine-us 1.752.2** (the frame pin,
`data/deployment_frame.py`, shared with every `gate_w1` artifact and adopted by M6
§2.8.10.2). The vintage boundary `T* = 2014` is inherited: realized values ≤2014,
`I_proj` projections beyond, never realized post-`T*` NAWI on a scored path
(`engine/refit.py:825-838` `validate_external_vintage` raises on `vintage > 2014`).

### 4.1 The binding table

| Series | Source | pe-us node (1.752.2) | Status |
|---|---|---|---|
| Payroll tax rate (employee 6.2% / employer 6.2%) | 26 USC 3101(a)/3111(a) | `gov/irs/payroll/social_security/rate/{employee,employer}` | **present** |
| Wage base (taxable max) | 42 USC 430 | `gov/ssa/social_security/wage_base.yaml` | **present** (realized ≤2014, Trustees/`I_proj` beyond) |
| NAWI (average-wage index) | SSA determinations | `gov/ssa/nawi.yaml` | **present** (realized ≤2014, projections beyond) |
| PIA formula factors (90/32/15) | 42 USC 415(a) | `gov/ssa/social_security/pia/formula_factors.yaml` | **present** |
| FRA schedule | 42 USC 416(l) | `gov/ssa/.../full_retirement_age_by_birth_year.yaml` | **present** |
| Early-reduction & delayed-credit rates | 42 USC 402(q)/(w) | `gov/ssa/.../retirement_age_adjustment/...` | **present** |
| 1978-base bend constants ($180 / $1,085) | 42 USC 415(a)(1)(B) | — (statute constant, cross-checked to pe-us thresholds, `ss/params.py:60-62,340-354`) | **statute constant** |
| Auxiliary 402(b)/(c)/(e)/(f) rate constants | 42 USC 402(b)/(c)/(e)/(f)/(k)/(q) | **NONE** | **GAP → §4.3** |
| COLA / CPI-W (benefits-in-payment) | 42 USC 415(i) | **NONE in `ss/`** | **GAP → §4.3** (nominal runs only) |
| Trust-fund interest rate (special-issue yield) | TR intermediate | **NONE** | **GAP → §4.3** |
| TR ultimate assumptions (real int 2.9%, CPI 2.7%, real-wage diff 1.13%) | 2014 OASDI TR Table V.B1 | **NONE** | **GAP → §4.3** |
| Trust-fund opening reserve | SSA TR historical | **NONE** | **GAP → §4.3** (M2 *calibrated* it) |

### 4.2 What pe-us 1.752.2 exposes (the revenue and own-benefit side)

The entire **own-benefit** formula and the **revenue** rate/base are sourced from
the pinned pe-us tree with no hand-typing beyond the two statutory 1978-base bend
constants (`ss/params.py` docstring; `load_ssa_parameters` reads `nawi.yaml`,
`wage_base.yaml`, `pia/formula_factors.yaml`, `full_retirement_age_by_birth_year
.yaml`, the early/delayed adjustment rates, and cross-checks derived bend points
against pe-us's stored thresholds and `NAWI(1977) = 9,779.44`, `ss/params.py:334`).
These carry the M6 §2.8.10.2 vintage rule unchanged. So the **income side** and
the **OASI retired-worker level** are fully pe-us-sourced; the gaps are all on the
**auxiliary, DI-fund, indexation, and reserve/interest** perimeter.

### 4.3 What pe-us 1.752.2 LACKS — flagged loudly (each needs a 3d-style amendment)

**These are the series a faithful M7 needs that policyengine-us 1.752.2 does not
provide. Each is a design STOP, exactly like M6's second designed stop
(`gate_m6` §2.8.10): it must be pinned by a reviewed amendment before any scored
M7 run, not silently filled.**

1. **Auxiliary 402(b)/(c)/(e)/(f) rate constants — NO pe-us node anywhere.** Per
   `ss/params.py` (docstring lines 18-38): pe-us "carries `social_security_
   retirement`, `social_security_survivors` and `social_security_dependents` as
   uprated survey **inputs** (no formula)," computing only the worker's own PIA
   and 402(q)/(w) adjustment. The spousal share (0.5), survivor share (1.0),
   RIB-LIM (0.825), 71.5% survivor floor, and protected-remarriage ages are
   therefore carried as **statute-cited constants** validated against SSA worked
   examples (`tests/ss/test_aux_benefits.py`, `runs/aux_benefit_examples_v1.json`),
   not pe-us. Any auxiliary outlay M7 reports rides these constants — the same
   "constant of the statute itself" exception pe-us's own tree forces.
2. **Trust-fund interest rate (special-issue yield) — NO pe-us node.** M2 states
   it directly: "no policyengine-us rate node" for the TR rate. Required for
   `interest[y]` in identity (I). Bind as a TR-cited constant / versioned
   alignment input.
3. **TR ultimate assumptions (real interest 2.9%, CPI 2.7%, real-wage
   differential 1.13%) — NO pe-us node.** M2 carried the 2014 TR Table V.B1
   bundle as a "TR-cited constant (no TR PDF staged; no policyengine-us rate
   node)." These parameterize discounting and (in a nominal run) COLA/wage growth.
4. **COLA / CPI-W benefits-in-payment series — NO pe-us node in `ss/`.** Needed
   only for a **nominal** M7 (§2.5); a real-dollar M7 nets it out. If M7 goes
   nominal, this is a hard binding, not an option.
5. **Trust-fund opening reserve — NO pe-us node; M2 did not source it at all.**
   M2 **calibrated** `reserve_0` to hit Smith's 2034 exhaustion year
   (`calibration_disclosure`). A levels-honest M7 either keeps the calibrated
   convention (disclosed, frame-relative) or binds an SSA TR historical
   opening-balance series (§8 decision 3).

Gaps 2–5 share a root cause: **policyengine-us models the benefit/​tax *formula*,
not the *trust fund*.** The trust-fund perimeter (interest, reserve, the
macro/CPI assumption set) is Trustees-Report territory with no upstream node. M7's
amendment must pin these to staged TR sources at a fixed vintage with the §4.4
tamper gate — or, absent a staged TR artifact, carry them as explicitly TR-cited
frame-relative constants that feed only report-only levels, never a gated cell.

### 4.4 The tamper-gate factory (mirroring M6 §2.8.10.4)

M7's inputs are supplied by a **zero-argument factory** on the M6 pattern — a new
module `scripts/registered_m7_inputs.py` exposing `build_inputs() ->
M7AccountingInputs`, resolved the way `run_gate_m6_candidate1.py` resolves
`registered_m6_inputs:build_inputs`. Every binding is **hardcoded**; there is no
argument and no environment-derived vintage selection beyond
`POPULACE_DYNAMICS_PE_US_DIR` → the pinned 1.752.2 install. Deterministic steps,
in order:

1. **Version + directory assert (M6 §2.8.10.2/.5, F2).** Assert
   `importlib.metadata.version("policyengine-us") == CERTIFIED_PIN["model_version"]
   == "1.752.2"` **and** assert the resolved parameter directory is the
   metadata-versioned install — the metadata assert alone does not bind the
   *directory* `POPULACE_DYNAMICS_PE_US_DIR` loads from (the F2 finding M6 §2.8.10.5
   closes). Then `params_full = load_ssa_parameters()` runs its own load-time
   cross-check (derived bend points vs pe-us thresholds; `NAWI(1977)`).
2. **Content-hash tamper gate on every staged external artifact.** For each staged
   file under a repo-root-anchored `data/external/` (`DATA = Path(__file__)
   .resolve().parents[1] / "data" / "external"`, never CWD-relative, mirroring
   `claiming._ROOT`) — the TR-constants JSON (gaps 2–4), the auxiliary-constant
   provenance record (gap 1), and any staged reserve series (gap 5) — **assert the
   JSON file's own sha256 equals a constant hardcoded in the factory.** The raw
   source's hash (a TR PDF, an SSA table page) lives separately in
   `provenance.source_sha256` as **build-time** provenance, **not** the tamper
   gate (M6 §2.8.10.4's exact split: "the factory instead asserts the JSON file's
   own hash").
3. **Vintage refusal.** Route every staged series through
   `validate_external_vintage` (`engine/refit.py:825-838`) so any `vintage > T*`
   raises **before any accounting, artifact write, or gate evaluation** — the same
   fence M6's `_validate_external_inputs` (`harness/m6_inputs.py:198-250`) puts in
   front of fit/score/write.

The point of the hash gate on **report-only** frame-relative constants (gaps 2–5
feed no gated cell) is the same as M6's on its report-only claiming reference:
tamper-evidence. A constant that silently drifts would move the reported levels
and the M2-reproduction check without a diff anywhere; the sha256 gate makes the
one thing that could silently change — the staged TR/aux numbers — impossible to
change unnoticed. The factory is the single pinned entry point, candidate-blind
(no binding is tuned to any observable), exactly as M6 §2.8.10 requires of its own.

## 5. Anchors — all REPORT-ONLY

M7 carries three anchor families. **None gates** (§6 gates only internal
identities). They are reported alongside the levels so a reader can see agreement
and named disagreement, exactly as M2 reported its forecasts without grading them
("the orchestrator grades … this artifact reports outcomes vs forecasts and does
not grade").

### 5.1 The M2-reproduction anchor (the internal one, feeds the gate)

Unlike the two external families below, M2-reproduction is an **internal** anchor:
M7's accounting engine, restricted to the exact M2 setup, must reproduce
`runs/m2_pseudo_projection_v1.json` (commit `747966cd`, `pe_us_revision bf71be3b`)
**to floating point, or name every delta.** The committed targets:

| Quantity | Committed M2 value |
|---|---|
| common frame | `n_common_frame 1549`, `weight_sum 33696344.0`, born 1943–1957, earnings 1968–2018, benefits 2005–2057 |
| combined OASDI rate | `0.124` (employee `0.062` + employer `0.062`) |
| calibrated reserve | `7889236006574.162` |
| baseline exhaustion year | `2034.0` |
| baseline balance analogue | `0.04689166331354922` |
| cap_150k | balance `0.053154118080107754`, Δbalance `0.0062624547665585326`, exhaustion `2035.1188537982976`, Δyears `1.1188537982975504` |
| elimination | balance `0.06118610701057222`, Δbalance `0.014294443697022999`, exhaustion `2037.2048065597855`, Δyears `3.2048065597855384` |
| payroll +1pp | balance `0.05689166331354923`, Δbalance `0.010000000000000009`, exhaustion `2035.950472894025`, Δyears `1.950472894025097` |
| payroll +2pp | balance `0.06689166331354923`, Δbalance `0.02000000000000001`, exhaustion `2038.2313708228655`, Δyears `4.231370822865529` |
| FRA→72 | balance `0.072506643421113`, Δbalance `0.02561498010756378`, exhaustion `null` (never exhausts on-frame), Δyears `41.0` (censored) |
| PI outlay Δ | `-0.3326593963115843` (`-33.266%`) |
| PPI outlay Δ | `-0.11000744343246105` (`-11.001%`) |
| NRA→70 outlay Δ | `-0.20234306477534883` (`-20.234%`) |
| reduced-COLA outlay Δ | `-0.007987803755968265` (`-0.799%`) |
| forecast outcomes | F1 met (100%, 14/14); **F2 NOT met** (Kendall τ `0.667`, our order +2pp > elim > +1pp > cap vs Smith elim > +2pp > +1pp > cap); F3 met; F4 met |

**Why reproduction is the expectation, not a hope.** M7 generalizes M2 by *adding*
classes (DI, auxiliary) and *opening* the frame (immigration, forward dynamics).
On the M2 setup those additions are switched **off** (OASI-only, closed 1943–1957
cohort, real dollars, calibrated reserve), so a faithful generalization must
collapse to M2 exactly. A non-zero delta is therefore diagnostic: either a bug, or
a documented arithmetic change M7 must name and justify cell-by-cell (the M6
"reproduce M2's committed numbers or explain every delta" discipline). The F2
"NOT met" outcome is part of the committed record and must reproduce as-is —
reproduction means matching M2's *result*, including its registered forecast miss.

<!-- M7-CURSOR-DO-NOT-SHIP -->
