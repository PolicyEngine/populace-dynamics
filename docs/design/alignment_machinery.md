# Alignment machinery

**Status:** draft production-layer design for adversarial review. This document
specifies a future implementation; it changes no code, gate, floor, threshold, or
run artifact.

The alignment layer produces a labeled, Trustees-aligned `after` projection while
preserving the unaligned projection as the scientific and certified object. It
reconciles discrete outcomes by selecting whole panel units, reconciles continuous
covered wages by a disclosed scalar, and publishes the intervention needed for
every margin and year. It never independently reweights linked people.

## 1. Resolution boundary and inherited law

### 1.1 The one deferred object this document resolves

The M6 design names two report-only surfaces in one sentence:

> “**Report-only** — the shock-window diagnostics, the `not_certified` surface
> (mortality drift first, §4.10), the re-drawn-`T*`-seed comparison (decision 5b),
> entrants (family B), and the alignment displacement — all published, none gated.”

That is the binding phase-5 rule in
`docs/design/m6_projection_engine.md:1534-1536`. The same document later records
the current alignment state verbatim: the “runner records the alignment layer as
not collected (`build_alignment_displacement(None, None)`,
`m6_runner.py:1100`)” (`docs/design/m6_projection_engine.md:1782-1785`; the
current call site has moved to `src/populace_dynamics/harness/m6_runner.py:1120-1128`).

This document resolves only the missing producer of the aligned `after` frame and
the target, reconciliation, and disclosure contract around that producer. It does
**not** resolve the other null-with-reason surface. The re-drawn-`T*` comparison
continues to report
`reason = "successor_forward_seed_machinery_out_of_scope"`, and its pre-named
margin continues to report
`unresolved_reason = "pre_named_margin_absent_from_ratified_spec"`
(`src/populace_dynamics/harness/m6_reporting.py:418-459`). Nothing here supplies a
re-drawn state, chooses that margin, or creates the decision-5 successor gate.

### 1.2 Mode separation is law

Decision 9 is not reopened:

> “**Alignment layer report-only; gate the un-aligned drift — ACCEPTED,** plus the
> per-run **maximum alignment displacement** published as a reported (not gated)
> magnitude (§4.8).”

The quote is `docs/design/m6_projection_engine.md:2661-2663`. Family B likewise
places external alignment, interventions, and maximum displacement outside the
certifiable family-A surface (`docs/design/m6_projection_engine.md:2347-2362`).
Therefore:

1. A certification run always invokes the engine with alignment disabled. Supplying
   an alignment specification to a gate-scoring path is an error.
2. An aligned production run is a separate replay with a separate artifact. It
   references, and never overwrites, its immutable unaligned companion.
3. An aligned artifact always says `gate_scored = false`,
   `alignment_applied = true`, and `certification_transfer = false`. Matching an
   external target does not certify the transition law that missed it.
4. A report may display aligned and unaligned series together, but only the
   unaligned series may carry a gate verdict. Alignment cannot change an existing
   verdict or manufacture a new one.

“Unaligned forever” applies to the certification path, not to storage. A gate
artifact may link to a separately produced aligned companion as a family-B
disclosure. The gate runner must not construct that companion inside its scored
projection call.

### 1.3 Existing accounting remains the canonical surface

`build_alignment_displacement` already implements the report-only accounting
primitive (`src/populace_dynamics/harness/m6_reporting.py:321-415`). It is fully
value-column-parameterized. It:

- identifies `before` as unaligned and scored and `after` as aligned and
  report-only;
- requires unique, exactly equal keys, including `year`;
- rejects a one-sided or key-mismatched comparison;
- computes absolute displacement by named numeric value column;
- publishes per-year field maxima, the run maximum, and the number of intervened
  rows; and
- returns an explicit `not_computed` record, never a fabricated zero, when neither
  frame exists.

This design does not replace, fork, or weaken that function. The production layer
must construct lawful exact-key comparison views and call it. The current gate
runner's `build_alignment_displacement(None, None)` remains valid for a run with no
aligned companion. It is invalid for an artifact labeled as aligned: every aligned
artifact must carry a `status = "computed"` displacement record.

The function reports one `n_intervened_rows` value for the frame it receives, not a
per-margin weighted sum. The producer therefore calls it on each named
margin-year comparison slice and builds the required representation-weight sums
outside the function. Claiming that the existing function itself computes gross
displaced weight would be incorrect.

## 2. External evidence and the mechanism decision

The public record supports alignment, but it does not support treating an aligned
aggregate as model validation.

| model/source | documented method or margin | implication for this design |
|---|---|---|
| DYNASIM4 | The 2024 fact sheet names fertility, mortality, disability, immigration, and labor-force participation as aligned to SSA Office of the Chief Actuary targets ([Urban 2024, p. 2](https://www.urban.org/sites/default/files/2024-09/Urban%E2%80%99s%20Dynamic%20Simulation%20of%20Income%20Model%204.pdf)). A DYNASIM4 application says the employment selection criterion moves until sex-by-age employment meets Trustees targets and separately says wages follow Trustees real-wage growth and worker earnings targets ([Johnson et al. 2017, appendix p. 47](https://crr.bc.edu/wp-content/uploads/2017/11/wp_2017-17-1.pdf)). | DYNASIM4 supplies a selection-threshold precedent for discrete employment and a value adjustment precedent for wages. Its public demographic summary does not disclose the demographic reconciliation algorithm, so this design does not attribute one to DYNASIM4. |
| DYNASIM3 lineage | The predecessor multiplies fertility probabilities within five maternal-age groups by target/predicted ratios and applies analogous linear mortality adjustments in 12 age-sex groups ([Favreault and Smith 2004, printed pp. 7-8](https://www.urban.org/sites/default/files/publication/71226/410961-A-Primer-on-the-Dynamic-Simulation-of-Income-Model-DYNASIM-.PDF)). | Probability ratios can align expectations, but a realized weighted panel can still miss a count and can hit probability caps. This is evidence for the rejected discrete-event alternative, not evidence that DYNASIM4 still uses it. |
| MINT7 | MINT assigns multiple potential donors, projects, and swaps donors when mortality or DI prevalence misses an age-sex-year target. Unequal weights can cause overshoot or undershoot and require repeated swaps ([Smith and Favreault 2013, printed pp. 7-8 and footnote 24](https://www.urban.org/sites/default/files/publication/22116/413131%20-%20A-Primer-on-Modeling-Income-in-the-Near-Term-Version-MINT-.pdf)). The same source says employment assumptions inform MINT but are not directly calibrated (printed p. 2, footnote 7). | Donor swapping is a whole-record selection precedent. Its unequal-weight warning requires a closest-feasible rule and a resolution floor; it does not justify pretending an arbitrary real-valued target is exactly attainable. |
| POLISIM | SSA's public description says major submodules align annually. It reports fixed benefit take-up totals by age, sex, and year, with an equation choosing the recipients; it also warns that “the alignment step may hide the erroneous ‘raw’ projection” and calls for retaining intermediate output before and after alignment ([McKay 2003, pp. 2, 5-6](https://web.archive.org/web/20080807142659/https://guard.canberra.edu.au/natsem/conference2003/papers/pdf/mckay_steven-1.pdf)). MINT6 also uses POLISIM target records and statistical donor matching for later cohorts ([Smith et al. 2010, chapter VII, printed pp. VII-2 and VII-4](https://www.urban.org/sites/default/files/publication/24986/412479-Modeling-Income-in-the-Near-Term-Version-.PDF)). | POLISIM supports selection of people to satisfy fixed totals and preservation of raw output. The accessible public sources do not specify its exact general event-selection algorithm; this design does not invent one. |

The older comparative review makes the tradeoff explicit: DYNASIM2 scaled
probabilities or continuous predictions; CORSIM chose among linear, nonlinear,
and sorting algorithms. It also says large alignment factors may indicate a
specification, estimation, or data-quality problem, while some large wage-growth
adjustments simply impose a scenario assumption ([Toder et al. 2000, printed pp.
52-53](https://www.urban.org/sites/default/files/publication/62701/410242-Long-Term-Model-Development-for-Social-Security-Policy-Analysis.PDF)).
MINT6 provides a concrete example: substantial mortality and fertility alignment
led its authors to recommend re-estimating the underlying equations rather than
treating the adjusted match as success ([Smith et al. 2010, printed pp. I-5-I-6](https://www.urban.org/sites/default/files/publication/24986/412479-Modeling-Income-in-the-Near-Term-Version-.PDF)).

Recent methodological work reaches the same separation. Li and O'Donoghue say
alignment should not repair misspecification and evaluate both target deviation
and before/after distribution distortion ([2014, §§2.3 and 5.3-5.8](https://www.jasss.org/17/1/15.html#2.3)).
Cumpston records a DYNACAN mortality-coefficient error that alignment masked
([2010, §2](https://www.microsimulation.pub/articles/00037#s2)). These sources
support the campaign-honest rule: the aligned series is a disclosed production
scenario; the unaligned companion remains the scientific diagnostic.

### 2.1 Adjudication: selection for events, ratios for values

For binary or categorical events in a discrete weighted panel, **deterministic
stratified selection wins**. The reconciler changes whole event outcomes while
leaving each person's representation weight fixed. It publishes who changed, how
much weight those people represent, and the closest achievable weighted target.
This preserves person, spouse, parent-child, and household links.

Annual probability-ratio adjustment loses for discrete production margins. It
matches an expected rate before the draw, not necessarily the realized weighted
count; clipping at one breaks the claimed ratio; and it does not expose a unique
set of displaced people. Independently changing person weights also loses because
it can break the linked network and contaminate every downstream weighted outcome.

For a continuous value such as average covered wage, a uniform, disclosed ratio
remains admissible. It preserves positive-earner ranks, creates no fractional
person-state, and can match a correctly defined mean exactly apart from pinned
floating-point tolerance. This narrow exception does not authorize event-probability
ratios or annual person-weight repair.

## 3. Margins, granularity, and vintage

### 3.1 The first target bundle

The first proposed production bundle is
`ssa_trustees_2026_alternative_ii_v1`: the 2026 Trustees Report's intermediate
assumptions, released June 9, 2026 and retrieved for this design on July 15, 2026.
Alternative II is the Trustees' best-estimate set as of February 2026. The bundle
pins the [2026 report](https://www.ssa.gov/oact/TR/2026/), the
[single-year table index](https://www.ssa.gov/oact/TR/2026/lrIndex.html), and the
[single-year workbook](https://www.ssa.gov/oact/TR/2026/SingleYearTRTables_TR2026.xlsx).
It never resolves “current” or “latest” at runtime. A later report creates a new
bundle identifier and a new content hash; it does not mutate this one.

The direct public tables do not contain every micro-event target. In particular,
Table V.A1's total fertility rate and standardized death rates are summary rates,
not annual birth counts or age-sex death probabilities. A label such as “aligned
to V.A1” is therefore insufficient for person selection. The target bundle must
either pin an official granular OCACT extract or pin a documented, reviewable
bridge from the public series. If neither exists, that margin remains
`not_available` and the aligned run does not fabricate it.

### 3.2 Ordered margins

Demographic margins reconcile first because they determine the live roster on
which economic margins operate. Within each projection year, the order is net
entry at the existing pre-wave entry seam, mortality after step 1, fertility after
step 4, covered employment after step 6, and covered wage after employment.

| order | margin and target | production granularity | mechanism | binding and limitation |
|---:|---|---|---|---|
| 1 | net population entry | year, then age × sex × entrant-unit type when a granular bridge is pinned | whole donor/family-unit selection and scheduled entry; emigration selection where the target requires it | [Table V.A2, Immigration Assumptions](https://www.ssa.gov/oact/TR/2026/lr5a2.html), total net change in thousands. Adjustment-of-status flows cancel across categories and are not new people. The public table does not provide age-sex cells. |
| 2 | mortality | year × sex × single year of age, with a pre-registered pooling ladder for sparse cells | whole-person death selection | [Table V.A1, Fertility and Mortality Assumptions](https://www.ssa.gov/oact/TR/2026/lr5a1.html) supplies standardized summaries; [Table V.A4](https://www.ssa.gov/oact/TR/2026/lr5a4.html) and [V.A5](https://www.ssa.gov/oact/TR/2026/lr5a5.html) supply period and cohort life-expectancy checks. Selection requires a separately pinned official age-sex probability/count extract; summary life expectancy is validation-only. |
| 3 | fertility | year × mother's single year of age, pooled only through the registered ladder; plurality reported separately | whole maternal-birth-event selection; conditional plurality draw retained | V.A1 supplies annual total fertility rate. SSA's [2026 demographic assumptions memorandum](https://www.ssa.gov/oact/TR/2026/2026_Long-Range_Demographic_Assumptions.pdf), pp. 3 and 6, describes single-age birth-rate assumptions, but the public single-year table does not expose them. A granular extract or pinned age-pattern bridge is required. |
| 4 | OASDI-covered work | year total; age × sex only when a definition-matched granular extract is pinned | whole-person covered-worker selection | [Table IV.B4, Covered Workers and Beneficiaries](https://www.ssa.gov/oact/TR/2026/lr4b4.html), workers paid at any time in the year for OASDI-covered employment. [Table V.B2](https://www.ssa.gov/oact/TR/2026/lr5b2.html) reports CPS-concept unemployment and growth in total employment, not the same worker count. |
| 5 | average annual covered wage | year, positive OASDI-covered workers | one multiplicative scalar after employment selection | [Table V.B1, Principal Economic Assumptions](https://www.ssa.gov/oact/TR/2026/lr5b1.html), nominal and real annual change in average annual wage in covered employment. [Table VI.G1](https://www.ssa.gov/oact/TR/2026/lr6g1.html) provides AWI and taxable-payroll checks, not interchangeable targets. |

The reconciler does not align every published quantity. Table V.A3 population
stocks, V.A4/V.A5 life expectancy, V.B2 total employment and unemployment, and
VI.G1 taxable payroll remain validation checks. Aligning both components and their
derived totals would conceal accounting errors and overconstrain the panel.

### 3.3 Definitions that must not drift

Every target carries its universe and measurement definition:

- The SSA Social Security area includes more than residents of the 50 states and
  District of Columbia. The target bundle must publish the bridge to the
  production roster; it cannot silently call the two populations equal.
- V.A2's lawful-permanent-resident adjustment-of-status inflow is an offsetting
  outflow from the temporary/unlawfully present category. Counting both as entry
  would create people who do not exist in the SSA total.
- IV.B4 counts anyone with covered earnings at any time during a calendar year.
  V.B2's employment and unemployment concepts use average-week CPS measures.
  They cannot share a column or target.
- V.B1's average covered wage, the national AWI in VI.G1, and taxable payroll are
  distinct. The alignment scalar targets only the first; the others diagnose it.
- Estimated, preliminary, or revised historical values stay exactly as published
  in the pinned vintage. A revised SSA series requires a new bundle and a
  before/after vintage comparison.

## 4. Production mechanism

### 4.1 Two immutable projections

One production request creates two logically separate results from the same
initial slice, model commit, draw index, and module RNG registry:

1. `before`: the ordinary `ProjectionEngine.project` result, with no target bundle
   visible to the engine; and
2. `after`: a replay through an alignment-aware production wrapper that inserts
   reconciliation hooks at the named seams.

The ordinary engine remains the eight-step loop in
`src/populace_dynamics/engine/loop.py:268-321`. The wrapper does not add a ninth
certified module or change that order. It captures a step's pre-state, raw
post-state, candidate probabilities or values, and stable draw keys; it then
reconciles the raw result before the next existing step reads it. This timing is
load-bearing: an aligned death must leave before aging, marriage, disability, and
earnings; an aligned birth must enter before later waves; aligned employment must
exist before the wage scalar.

The wrapper writes a new `ProjectionResult`-shaped object plus an alignment ledger.
It never mutates `before`, shares a mutable frame with it, or returns `after` to a
gate scorer.

### 4.2 Discrete selection algorithm

Each binary margin-year-stratum supplies a candidate ledger with:

`alignment_unit_id`, `person_id` or linked-unit members, `year`, `stratum_id`,
positive representation `weight`, raw outcome `e_raw`, predicted probability
`p`, and the module uniform `u`. The registered selection score is `p - u`. The
raw Monte Carlo outcome is the zero-threshold selection (`p - u > 0`), so moving
one common threshold changes the least marginal candidates first while preserving
the model's risk ordering.

The reconciler:

1. sorts candidates by descending `p - u`, then by a dedicated alignment-stream
   tie key, then by canonical `alignment_unit_id`;
2. computes the cumulative representation weight for every prefix, including the
   empty prefix;
3. chooses the prefix whose cumulative weight is closest to the registered target;
4. breaks equal absolute residuals by fewer flips from `e_raw`, then no overshoot,
   then the smaller prefix; and
5. records every `0→1` and `1→0` flip before applying the selected whole-unit
   outcomes.

This is a deterministic stratified-selection rule, not an iterative search whose
stopping point can vary. With unequal positive weights it may not hit an arbitrary
real target exactly. The selected prefix and its residual are nevertheless unique,
auditable, and byte-reproducible. Section 6 requires the closest-prefix
feasibility floor to be frozen before the aligned run.

For a linked unit, `weight` is the registered unit weight and the outcome applies
atomically to every required member. Net-entry candidates use a pinned donor-unit
pool and donor-distance tier in place of a model event probability; within a tier,
the dedicated alignment key supplies the order. The bundle must name whether the
unit is a person, maternal event, couple, family, or household. It may not fall
back to person selection when that would orphan a relationship.

The implementation must expose the complete eligible candidate ledger, not only
raw positive events. Otherwise it could remove excess events but could not add a
shortfall without an unregistered redraw. A target outside the attainable range
is `structurally_infeasible`; the runner publishes the reason and does not issue an
aligned panel.

### 4.3 Continuous covered-wage ratio

After covered-worker selection, let `R_y` be the panel's definition-matched
weighted mean of positive annual covered wages and `T_y` the V.B1-derived target
level for year `y`. The scalar is

`a_y = T_y / R_y`, and `earnings_after_i = a_y × earnings_before_i`

for positive covered earnings only. The ledger publishes `R_y`, `T_y`, `a_y`, the
number and representation weight of changed workers, and weighted and unweighted
absolute dollar displacement. If `R_y <= 0 < T_y`, the margin is infeasible; the
runner never substitutes a zero, cap, or donor value. The bundle pins rounding,
numeric dtype, and whether self-employment is included before execution.

The wage scalar does not change representation weights, zero earnings, employment
status, or ranks among positive earners. Taxable payroll, AIME, PIA, taxes, and
benefits are recomputed downstream from the aligned values; they are not separately
forced to SSA totals.

### 4.4 RNG and byte reproducibility

Alignment consumes no existing module stream. A separate registry derives each
tie stream from canonical integer words encoding the root draw index,
`alignment_spec_sha256`, calendar year, margin identifier, stratum identifier, and
algorithm revision. It uses a cryptographic digest, never Python's process-random
`hash()`.

Candidate frames sort by canonical typed keys before scoring. The implementation
pins float64 arithmetic, stable sorting, target units, missing-value behavior, and
tie rules. It records the stream address and hashes of `before`, `after`, the target
bundle, and the displacement sidecar. A bit-exact replay check must reproduce all
four hashes before the first production release.

Adding or removing a production entrant must not shift the existing engine draws
for an existing person. The current loop pre-registers scheduled-entry IDs and
person ordinals before the wave loop (`src/populace_dynamics/engine/loop.py:192-237`)
and uses person-specific mortality generators (`src/populace_dynamics/engine/steps.py:113-134`).
The implementation review must prove the same isolation for every aligned event
hook and synthetic-child ID mapping.

### 4.5 Roster and scoring universes remain separate

Alignment operates on the lawful production roster; it does not redefine a gate
universe.

- The earnings-domain law marks people who are closed-panel members for flow
  modules but open additions for the 2014-anchored earnings law
  (`docs/design/m6_projection_engine.md:1221-1300`). Economic alignment may select
  only people for whom the production earnings adapter exposes a lawful potential
  earnings state. It cannot backfill a gated earnings anchor or turn a marked
  person into gated support.
- The marital entry law keeps anchor-present minors in the roster and seeds them
  only when they enter the certified marital risk set
  (`docs/design/m6_projection_engine.md:997-1140`). Alignment cannot remove them
  merely because they were outside a module's year-0 domain.
- Pending sibling amendment 3h, supplied as binding for this design in
  [PR #216](https://github.com/PolicyEngine/populace-dynamics/pull/216), separates
  the frame-independent fertility schedule used for scoring from the live
  post-mortality roster used for child materialization. Its law appears in
  [§2.8.2h at commit `b162f9c`](https://github.com/PolicyEngine/populace-dynamics/blob/b162f9c/docs/design/m6_projection_engine.md#L1183-L1264): only roster-present mothers materialize children; a scheduled birth to a removed mother enters report-only reconciliation. The alignment hook must use the live roster for materialization and must not resurrect a mother to satisfy a fertility target.

Every margin record therefore names `candidate_universe`, `materialization_universe`,
`measurement_universe`, and `gate_universe = none`. Counts from one universe may
not be presented as counts from another.

## 5. Displacement accounting and publication

### 5.1 Exact-key comparison views

The existing accounting function requires identical keys. Raw projection frames
cannot satisfy that contract for roster-changing events: an aligned death removes
a later person-wave, while an aligned birth or entrant adds one. The producer
therefore constructs an immutable **candidate audit view** at each reconciliation
seam, before it changes the roster. This is not a padded projection frame. It is
the complete, registered set of units that could lawfully receive the named event.

Each before/after audit-view pair has one row per
`(alignment_unit_id, year, margin_id, stratum_id)` and carries numeric fields with
fixed meanings:

| field | meaning |
|---|---|
| `event_selected` | zero or one for a discrete event; unchanged candidate rows remain present in both views |
| `target_contribution` | `event_selected * representation_weight` in the target's units |
| `materialized_people` | number of people added or removed by the selected linked-unit event |
| `covered_wage` | annual covered wage for the continuous margin; absent from discrete views |

The producer calls
[`build_alignment_displacement`](../../src/populace_dynamics/harness/m6_reporting.py)
with the four key columns and the applicable value columns. The implementation at
`src/populace_dynamics/harness/m6_reporting.py:321-415` remains authoritative.
For example, an employment call compares `event_selected` and
`target_contribution`; a wage call compares `covered_wage`. Calling by margin-year
keeps the function's `n_intervened_rows` and maximum unambiguous. The run-level
record also retains a call across all years of a margin so its native
`per_year_maximum` and `maximum_alignment_displacement` fields remain available.

An absent projection row never silently becomes a numeric zero. Zero is valid in
an audit view only when it means “this registered candidate was not selected for
this event.” A unit omitted from either side, a non-unique key, or an unregistered
candidate is a hard accounting error, as the existing function already requires.
The audit view and the materialized roster are linked by the recorded
`alignment_unit_id`; the producer checks that every selected addition/removal was
materialized exactly once.

### 5.2 Required per-margin, per-year ledger

Every aligned run publishes one record for every requested
`(margin_id, year, stratum_id)`, including records with no intervention. Each
record contains:

- target, raw, and aligned weighted totals in one declared unit;
- raw residual, aligned residual, and the precomputed feasibility floor;
- counts of `0→1` and `1→0` unit flips;
- distinct displaced people and displaced linked units;
- gross displaced representation weight, counted once per changed person or
  atomic linked unit, plus separate `0→1` and `1→0` represented weights;
- absolute target-contribution displacement and its maximum over a single unit;
- for wages, the scalar, changed-worker count and representation weight, and
  weighted and unweighted absolute dollar displacement;
- the canonical `build_alignment_displacement` result and hashes of its exact
  before/after audit views; and
- status, reason, candidate/materialization/measurement universes, target source,
  stream address, and algorithm revision.

“Displaced person count” means the number of distinct panel people whose event or
value changed, not the rounded sum of weights. “Gross displaced representation
weight” means the sum of the registered fixed weights of those distinct changed
units and is never netted between additions and removals. When a family event
changes several people atomically, the ledger reports both one displaced linked
unit and the distinct affected-person count; it uses the registered family-unit
weight exactly once in the gross unit-weight field. Person-weight and unit-weight
fields stay separate rather than presenting one as the other.

These sums supplement rather than reinterpret the canonical function. In
particular, `n_intervened_rows` is not relabeled as a person-weight count, and a
maximum change in `event_selected` is not relabeled as aggregate displacement.
The run summary publishes, for each margin and year, the canonical maximum and
the gross count/weight measures, then publishes the maximum across years without
discarding the underlying records.

### 5.3 Artifact labels and completeness

An aligned artifact is publishable only if all requested margins have either:

- `computed`, with a complete displacement ledger and fidelity result; or
- `not_available`, with a pre-run reason such as a missing definition-matched
  granular target.

`not_available` does not mean that the producer silently skipped an advertised
margin. The artifact title and machine label list the margins actually applied.
`structurally_infeasible`, audit-view key mismatch, materialization mismatch, or
hash mismatch withholds the aligned panel and publishes a failed production
attempt sidecar. It never converts the unaligned companion into a failure of the
M6 gate.

Every aligned publication carries this statement, without abbreviation:

> This production scenario was reconciled to pinned external targets. The M6 gate
> scored the separate unaligned projection. Alignment displacement is disclosed
> for every applied margin and year, and this aligned output does not transfer or
> strengthen certification.

The machine-readable companion fields are
`projection_mode = "aligned_production"`, `alignment_applied = true`,
`gate_scored = false`, `certification_transfer = false`, and
`unaligned_companion_sha256`. Removing the label or publishing the aligned panel
without its displacement sidecar is an artifact-schema failure.

## 6. Evaluation: floors first

### 6.1 Pre-run feasibility floors

The target bundle, candidate construction, weights, strata, pooling ladder, score,
tie rule, and numeric rules are frozen before an aligned run. From that frozen
candidate ledger, the producer enumerates the empty and every ordered prefix and
records the minimum attainable absolute residual in target units. That value is
the **feasibility floor** for the discrete margin-year-stratum. It also records
the target's distance outside the total attainable range, if any.

The floor comes first: the producer writes and hashes it before selecting the
aligned prefix, and the aligned evaluator reads rather than rewrites it. It may
not declare exact agreement when unequal panel weights make exact agreement
impossible. A continuous covered-wage margin has a zero arithmetic floor only
when its denominator is positive and all registered finite-value conditions hold;
otherwise it is infeasible.

Sparse cells follow only the target bundle's pre-registered pooling ladder. The
producer may move from single age to the next named age band when the exact cell
is infeasible, but it must publish both records and may not search for a favorable
pool after seeing the result. A target outside the candidate range remains
`structurally_infeasible`; pooling cannot manufacture eligible people.

### 6.2 Alignment-fidelity check

For every applied cell, the evaluator independently recomputes the aligned margin
from the materialized panel using the declared measurement universe. The check
passes only when:

`abs(aligned_total - target) <= feasibility_floor + numeric_tolerance`.

The bundle pins `numeric_tolerance` in target units; it cannot be a percentage
chosen after execution. The evaluator also requires raw and aligned ledger totals
to reproduce from their hashed audit views, selected units to reproduce from the
registered ordering, and materialized events to reconcile to selected events.

This is a production-integrity check, not a scientific gate. Failure withholds the
aligned output and leaves the unaligned gate artifact and verdict unchanged.
Passing says only that the disclosed reconciler produced the pinned scenario.

### 6.3 Displacement-magnitude corridors

Fidelity alone rewards a machinery that forces any target, however violently.
Every applied cell therefore receives a report-only displacement corridor for:

- displaced-person share of the eligible candidate universe;
- gross displaced representation-weight share;
- absolute target-contribution displacement relative to the target; and
- for wages, the absolute log scalar and weighted absolute dollar displacement.

Corridors are authored before release from pre-boundary historical backcasts,
definition-matched target revisions, and registered synthetic target shocks. Their
ceremony pins source vintages, years, horizons, strata, seeds, quantile rule,
pooling ladder, and artifact hash. The labels are `within_reference`,
`above_reference`, and `not_comparable`; they are not pass/fail labels. An
`above_reference` result triggers model review and must appear beside the aligned
series, but it neither invalidates nor rescues the unaligned M6 gate.

No numerical corridor is ratified by this design document. Until a reviewed
corridor artifact exists, every aligned run publishes the raw magnitude and
`corridor_status = "not_available"`; it must not substitute zero, an informal
percentage, or the fidelity floor. This preserves the floors-first order while
keeping a missing research benchmark visibly missing.

### 6.4 Validation panel

The release comparison contains the unaligned, target, and aligned series on the
same axes. It also compares untargeted distributions before and after alignment:
age, sex, family structure, earnings quantiles, covered-work transitions, life
expectancy, population stock, AWI, and taxable payroll where definition-matched.
The evaluator reports changes in these diagnostics even when every target margin
passes fidelity. This implements the target-fit and distribution-distortion split
recommended by [Li and O'Donoghue (2014), §§5.3-5.8](https://www.jasss.org/17/1/15.html#5.3).

A pre-boundary validation exercise turns alignment off for the module under test
and compares its raw projection with observed/reference data. That follows the
dynamic-microsimulation validation guidance to distinguish validation from a
baseline that already imposes external controls ([Harding et al. 2010, pp. 12-13](https://www.researchgate.net/profile/Ann-Harding/publication/228846890_Issues_in_the_validation_of_dynamic_microsimulation_models/links/02e7e51e884cf87db4000000/Issues-in-the-validation-of-dynamic-microsimulation-models.pdf)).
It is evidence about model behavior; the production fidelity check is evidence
about reconciliation behavior. The report never collapses the two.

## 7. External bindings and artifact identity

### 7.1 Target manifest

The target bundle is a content-addressed manifest. Each source series records:

- bundle identifier and schema/algorithm revision;
- publisher, report title, report date, Trustees alternative, table, row, column,
  year range, and source URL;
- retrieved-at timestamp, raw-file SHA-256, extraction-code commit, extracted-data
  SHA-256, and any signed official-extract identifier;
- unit, nominal/real basis, frequency, event timing, population universe, geography,
  status/coverage definition, age/sex categories, and revision status;
- transformation from source to annual target level, including the base level for
  a growth series, rounding, and reconciliation identities;
- bridge or pooling-ladder identifier and hash where the public table is not the
  micro-event target; and
- `available`, `not_available`, or `validation_only`, with a reason.

The manifest pins the 2026 Alternative II vintage described in §3.1. Network
access is forbidden during a production run; the runner accepts only the reviewed
local content hashes. Redirects, revised spreadsheets at the same URL, or a new
Trustees Report fail identity rather than updating a target in place.

### 7.2 Production manifest

The aligned sidecar binds the target manifest to the unaligned scientific object:
repository commit, environment lock hash, initial-slice hash, draw index, ordinary
module RNG registry hash, alignment RNG registry hash, algorithm revision,
candidate-ledger hashes, feasibility-floor hash, corridor hash or explicit null
reason, unaligned result hash, aligned result hash, and displacement-ledger hash.
It also records the exact ordered margins and the seam after which each ran.

Publishing a different target vintage, bridge, pooling ladder, candidate universe,
or algorithm creates a different production identity. Comparison tooling may show
two identities side by side, but it cannot call one a rerun of the other.

## 8. Open decisions

Implementation remains blocked on the following bounded choices. None authorizes
a gate-path change.

1. **Granular demographic target authority.** Obtain a reviewable official OCACT
   age-sex mortality series and single-age maternal fertility series, or ratify a
   public age-pattern bridge and sparse-cell pooling ladder. V.A1 summary rates
   alone are insufficient.
2. **Social Security area and migration units.** Ratify the bridge between the
   production roster and the Social Security area, the donor microdata and vintage,
   the atomic entrant unit, and whether the production target decomposes gross
   immigration/emigration or only V.A2 net change.
3. **Covered-work candidate state.** Define a lawful potential covered-earnings
   value for a raw nonworker selected `0→1`, settle wage-and-salary versus
   self-employment coverage, and bind the result to IV.B4's “paid at any time in
   year” concept without altering the certified earnings domain.
4. **Fertility measurement.** Choose maternal birth events, live births, or
   children as the target unit; specify plurality and infant-death timing; and
   bind the selected measure to the post-mortality live-roster materialization
   rule without changing the frame-independent scoring schedule.
5. **Displacement-corridor ceremony.** Pre-register the historical window,
   horizon bands, strata, target-revision vintages, synthetic shocks, seeds,
   quantiles, and response to `above_reference`. This design intentionally
   ratifies no numerical corridor.
6. **Policy-counterfactual semantics.** Decide whether an aligned reform run uses
   the baseline Trustees levels, aligns only common exogenous margins, or carries
   reform-specific external targets; specify common-random coupling and labels so
   alignment cannot erase a modeled policy effect.
7. **Release scope and economics phase.** Decide whether the first implementation
   ships demographic margins alone and keeps employment/wages unavailable until
   definition-matched inputs exist, or waits for the full ordered bundle. The
   ordering and disclosure requirements do not change either way.

## 9. What this does not change

This design does not:

- edit the eight-step engine loop, assembly law, current gate runner, gate
  configuration, threshold, holdout floor, one-shot rule, or certification claim;
- change the M6 decision that the gate scores the unaligned projection;
- replace or alter `build_alignment_displacement`; it supplies the missing lawful
  `after` producer and exact-key inputs around the existing function;
- ratify a re-drawn-`T*` seed margin, build successor forward-seed machinery, or
  convert any other null-with-reason surface to a numeric result;
- redefine the roster, resurrect dead people, create a gated state for family-B
  additions, or collapse the scoring/materialization universes in amendment 3h;
- recalibrate panel weights, use target matching to certify a module, or align
  validation-only derived totals; or
- claim that citing an SSA aggregate supplies a missing granular target, bridge,
  donor pool, corridor, or production implementation.

The next implementation proposal must resolve the relevant open decisions, add
code and tests in a separate change, and demonstrate the bit-exact replay,
alignment-fidelity, materialization-reconciliation, artifact-label, and
unaligned-gate-isolation checks. Until then, an alignment request remains explicit
`not_available`; the certified projection remains unaligned.
