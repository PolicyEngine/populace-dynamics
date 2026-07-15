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
(`src/populace_dynamics/harness/m6_reporting.py:37-38,418-459`). Nothing here
supplies a re-drawn state, chooses that margin, or creates the decision-5
successor gate.

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
aligned companion. Its inherited result is exactly `status = "not_computed"` and
`reason = "alignment_layer_output_not_collected"`. It is invalid for an artifact
labeled as aligned: every applied margin in an aligned artifact must carry a
`status = "computed"` displacement record.

The function reports one frame-global `n_intervened_rows`, not a per-year weighted
sum. It does not compute margin identifiers, target residuals, signed changes, or
summed representation-weight displacement. Section 5 specifies the caller-owned
group-bys around it. Claiming that the existing function itself computes gross
displaced weight would be incorrect.

## 2. External evidence and the mechanism decision

The public record supports alignment, but it does not support treating an aligned
aggregate as model validation.

| model/source | documented method or margin | implication for this design |
|---|---|---|
| DYNASIM4 | The 2024 fact sheet names fertility, mortality, disability, immigration, and labor-force participation as aligned to SSA Office of the Chief Actuary targets ([Urban 2024, p. 2](https://www.urban.org/sites/default/files/2024-09/Urban%E2%80%99s%20Dynamic%20Simulation%20of%20Income%20Model%204.pdf)). A DYNASIM4 application says the employment selection criterion moves until sex-by-age employment meets Trustees targets and separately says wages follow Trustees real-wage growth and worker earnings targets ([Johnson et al. 2017, appendix p. 47](https://crr.bc.edu/wp-content/uploads/2017/11/wp_2017-17-1.pdf)). | DYNASIM4 supplies a selection-threshold precedent for discrete employment and a value adjustment precedent for wages. Its public demographic summary does not disclose the demographic reconciliation algorithm, so this design does not attribute one to DYNASIM4. |
| DYNASIM3 lineage | The predecessor multiplies fertility probabilities within five maternal-age groups by target/predicted ratios and applies analogous linear mortality adjustments in 12 age-sex groups ([Favreault and Smith 2004, printed pp. 7-8](https://www.urban.org/sites/default/files/publication/71226/410961-A-Primer-on-the-Dynamic-Simulation-of-Income-Model-DYNASIM-.PDF)). | Probability ratios can align expectations, but a realized weighted panel can still miss a count and can hit probability caps. This is evidence for the rejected discrete-event alternative, not evidence that DYNASIM4 still uses it. |
| MINT7 | MINT assigns multiple potential donors, projects, and swaps donors when mortality or DI prevalence misses an age-sex-year target. Unequal weights can cause overshoot or undershoot and require repeated swaps ([Smith and Favreault 2013, printed pp. 7-8 and footnote 24](https://www.urban.org/sites/default/files/publication/22116/413131%20-%20A-Primer-on-Modeling-Income-in-the-Near-Term-Version-MINT-.pdf)). The same source says employment assumptions inform MINT but are not directly calibrated (printed p. 2, footnote 7). | Donor swapping is a whole-record selection precedent. Its unequal-weight warning requires an explicit resolution rule; it does not justify pretending an arbitrary real-valued target is exactly attainable. |
| POLISIM | SSA's public description says each major submodule is aligned to historical data. It reports fixed benefit take-up totals by age, sex, and year, with an equation choosing the recipients; it also warns that “the alignment step may hide the erroneous ‘raw’ projection” and calls for retaining intermediate output before and after alignment ([McKay 2003, pp. 2, 5-6](https://web.archive.org/web/20080807142659/https://guard.canberra.edu.au/natsem/conference2003/papers/pdf/mckay_steven-1.pdf)). MINT6 also uses POLISIM target records and statistical donor matching for later cohorts ([Smith et al. 2010, chapter VII, printed pp. VII-2 and VII-4](https://www.urban.org/sites/default/files/publication/24986/412479-Modeling-Income-in-the-Near-Term-Version-.PDF)). | POLISIM supports selection of people to satisfy fixed totals and preservation of raw output. The accessible public sources do not specify its exact general event-selection algorithm; this design does not invent one. |

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
much weight those people represent, and the closest registered threshold-prefix
target. Whole-unit selection preserves key and atomic-unit integrity; it does not
by itself repair spouse, parent-child, or household state after a roster change.

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
assumptions, [released June 9, 2026](https://www.ssa.gov/news/en/advocates/2026-06-09.html)
and retrieved for this design on July 15, 2026. [Alternative II is the Trustees'
best-estimate set as of February 2026](https://www.ssa.gov/oact/TR/2026/V_A_demo.html).
The bundle pins the [2026 report](https://www.ssa.gov/oact/TR/2026/), the
[single-year table index](https://www.ssa.gov/oact/TR/2026/lrIndex.html), and the
[single-year workbook](https://www.ssa.gov/oact/TR/2026/SingleYearTRTables_TR2026.xlsx).
It never resolves “current” or “latest” at runtime. A later report creates a new
bundle identifier and a new content hash; it does not mutate this one.

The manifest declares an applicable target-year interval. The current M6 panel
builders require the pinned 2022 horizon
(`src/populace_dynamics/engine/panel_builders.py:120-132`), so this document does
not make a 2026-forward Trustees run executable. Extending the production horizon
is a separate dependency and cannot be smuggled into the aligned-`after` producer.

The direct public tables do not contain every micro-event target. In particular,
Table V.A1's total fertility rate and standardized death rates are summary rates,
not annual birth counts or age-sex death probabilities. A label such as “aligned
to V.A1” is therefore insufficient for person selection. The target bundle must
either pin an official granular OCACT extract or pin a documented, reviewable
bridge from the public series. If neither exists, that margin remains
`not_available` and the aligned run does not fabricate it.

### 3.2 Ordered margins

Demographic margins reconcile first because they determine the live roster on
which economic margins operate. Entrant additions are selected and registered
before `ProjectionEngine.project`; the loop validates every scheduled entrant and
reserves IDs before its wave loop (`src/populace_dynamics/engine/loop.py:192-237`).
Within each year, those scheduled additions materialize at the existing pre-wave
seam; a separately registered emigration branch runs after addition and before
mortality; mortality reconciles after raw step 1; fertility after raw step 4;
covered employment after raw step 6; and covered wage after employment.

| order | margin and target | production granularity | mechanism | binding and limitation |
|---:|---|---|---|---|
| 1 | net population entry | year, then age × sex × entrant-unit type when a granular bridge is pinned | separate whole-unit addition and removal branches | [Table V.A2, Immigration Assumptions](https://www.ssa.gov/oact/TR/2026/lr5a2.html), total net change in thousands. Adjustment-of-status flows cancel across categories and are not new people. The public table supplies neither age-sex cells nor a unique gross addition/removal decomposition, so a net-only series remains unavailable until decision 2 is resolved. |
| 2 | mortality | year × sex × single year of age, with a pre-registered pooling ladder for sparse cells | whole-person death selection | [Table V.A1, Fertility and Mortality Assumptions](https://www.ssa.gov/oact/TR/2026/lr5a1.html) supplies standardized summaries; [Table V.A4](https://www.ssa.gov/oact/TR/2026/lr5a4.html) and [V.A5](https://www.ssa.gov/oact/TR/2026/lr5a5.html) supply period and cohort life-expectancy checks. Selection requires a separately pinned official age-sex probability/count extract; summary life expectancy is validation-only. |
| 3 | fertility | year × mother's single year of age, pooled only through the registered ladder; plurality reported separately | whole maternal-birth-event selection | V.A1 supplies annual total fertility rate. SSA's [2026 demographic assumptions memorandum](https://www.ssa.gov/oact/TR/2026/2026_Long-Range_Demographic_Assumptions.pdf), pp. 3 and 6, describes single-age birth-rate assumptions, but the public single-year table does not expose them. A granular extract or pinned age-pattern bridge is required. |
| 4 | OASDI-covered work | year total; age × sex only when a definition-matched granular extract is pinned | whole-person covered-worker selection | [Table IV.B4, Covered Workers and Beneficiaries](https://www.ssa.gov/oact/TR/2026/lr4b4.html), workers paid at any time in the year for OASDI-covered employment. [Table V.B2](https://www.ssa.gov/oact/TR/2026/lr5b2.html) reports unemployment and total-employment growth; [the covered-employment methods](https://www.ssa.gov/OACT/TR/2026/V_C_prog.html) explain why its CPS average-week concepts differ from an annual covered-worker count. |
| 5 | average annual covered-wage growth path | year, positive OASDI-covered workers | one multiplicative scalar after employment selection | [Table V.B1, Principal Economic Assumptions](https://www.ssa.gov/oact/TR/2026/lr5b1.html), nominal and real annual change in average annual wage in covered employment. It does not provide the base covered-wage level needed to construct `T_y`; that binding remains open. [Table VI.G1](https://www.ssa.gov/oact/TR/2026/lr6g1.html) provides AWI and taxable-payroll checks, not interchangeable targets. |

The reconciler does not align every published quantity. Table V.A3 population
stocks, V.A4/V.A5 life expectancy, V.B2 total employment and unemployment, and
VI.G1 taxable payroll remain validation checks. Aligning both components and their
derived totals would conceal accounting errors and overconstrain the panel.

### 3.3 Definitions that must not drift

Every target carries its universe and measurement definition:

- The SSA Social Security area includes more than residents of the 50 states and
  District of Columbia. The target bundle must publish the bridge to the
  production roster; it cannot silently call the two populations equal
  ([SSA 2026 long-range model documentation, Demography p. 2](https://www.ssa.gov/oact/TR/2026/2026_LR_Model_Documentation.pdf)).
- V.A2's lawful-permanent-resident adjustment-of-status inflow is an offsetting
  outflow from the temporary/unlawfully present category. Counting both as entry
  would create people who do not exist in the SSA total.
- IV.B4 counts anyone with covered earnings at any time during a calendar year.
  [V.B2's employment and unemployment concepts use average-week CPS measures](https://www.ssa.gov/OACT/TR/2026/V_C_prog.html).
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

“Same” means identical immutable inputs and stream addresses, not shared mutable
objects. Each replay receives a fresh `assemble_period_modules` closure, metadata
mapping, synthetic-ID allocator, and output collectors. The assembly closes over
mutable marital, fertility, disability, and household caches
(`src/populace_dynamics/engine/assembly.py:195-211,230-250`), while the loop places a
mutable allocator in metadata (`src/populace_dynamics/engine/loop.py:225-237`). A
shallow copy would allow one path to contaminate the other. The term “alignment
layer” in `src/populace_dynamics/engine/assembly.py:1-8` names the existing
native-panel adapter; the external target reconciler designed here is a distinct
production wrapper around freshly assembled modules.

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
positive `production_year_weight`, raw outcome `e_raw`, predicted probability
`p`, and the module uniform `u`. The weight is the pre-reconciliation production
year-`y` representation weight frozen for this margin call, not the M6 start-wave
gate weight. The registered selection score is `p - u`. The raw Monte Carlo
outcome is the zero-threshold selection (`p - u > 0`), so moving one common
threshold preserves the realized latent draw-distance ordering. It does not claim
to preserve probability-only risk ordering.

The reconciler:

1. sorts candidates by descending `p - u`, then by a dedicated alignment-stream
   tie key, then by canonical `alignment_unit_id`;
2. computes the cumulative production-year weight for every prefix, including the
   empty prefix;
3. chooses the prefix whose cumulative weight is closest to the registered target;
4. breaks equal absolute residuals by fewer flips from `e_raw`, then no overshoot,
   then the smaller prefix; and
5. records every `0→1` and `1→0` flip before applying the selected whole-unit
   outcomes.

This is a deterministic stratified-selection rule, not an iterative search whose
stopping point can vary. With unequal positive weights it may not hit an arbitrary
real target exactly. A non-prefix subset may be closer, but packing units by weight
would abandon the registered threshold rule and model-relative ordering. The
selected prefix and its residual are unique, auditable, and byte-reproducible.
Section 6 freezes the registered threshold-prefix resolution floor before the run.

For a linked unit, `production_year_weight` is the registered unit weight and the
outcome applies atomically to every required member. Net-entry candidates use a pinned donor-unit
pool and donor-distance tier in place of a model event probability; within a tier,
the dedicated alignment key supplies the order. The bundle must name whether the
unit is a person, maternal event, couple, family, or household. It may not fall
back to person selection when that would orphan a relationship.

The positive-prefix schema applies to one-sided events. Migration requires a
separately targeted addition prefix and removal prefix. A signed V.A2 net target
does not determine those two branches; the producer may not choose an arbitrary
decomposition or apply `event_selected * weight` as though all contributions had
one sign.

The implementation must expose the complete eligible candidate ledger, not only
raw positive events. Otherwise it could remove excess events but could not add a
shortfall without an unregistered redraw. A target outside the attainable range
is `structurally_infeasible`; the runner publishes the reason and does not issue an
aligned panel.

That complete ledger is an implementation precondition, not a claim about current
outputs. `simulate_maternal_births` currently evolves parity internally and returns
only positive events (`src/populace_dynamics/engine/marital.py:288-352`). A future
seam must expose each live-mother candidate's probability, uniform, parity, and
prospective birth order without a second draw; fertility remains `not_available`
until it does. The current earnings interface returns a continuous draw, not a
covered-work participation probability and uniform
(`src/populace_dynamics/engine/steps.py:169-287`); covered-work selection likewise
remains unavailable until decision 3 pins a lawful score.

### 4.3 Continuous covered-wage ratio

After covered-worker selection, let `R_y` be the panel's definition-matched
weighted mean of positive annual covered wages and `T_y` the V.B1-derived target
level for year `y`. Because V.B1 publishes growth, the bundle must construct `T_y`
from a separately pinned base covered-wage level and the named nominal or real
growth column. Without that base, the wage margin is `not_available`. The scalar is

`a_y = T_y / R_y`, and `earnings_after_i = a_y × earnings_before_i`

for positive covered earnings only. The ledger publishes `R_y`, `T_y`, `a_y`, the
number and representation weight of changed workers, and weighted and unweighted
absolute dollar displacement. If `R_y <= 0 < T_y`, the margin is infeasible; the
runner never substitutes a zero, cap, or donor value. The bundle pins rounding,
numeric dtype, and whether self-employment is included before execution.

The wage scalar does not change representation weights, zero earnings, employment
status, or ranks among positive earners. It does change the chained state read by
later waves. On an even projection year, the producer applies the same scalar or
employment override to both current `earnings` and the just-written
`gen_earn_w2`; it leaves `gen_earn_w4` as the already aligned older lag. On an odd
year it changes current `earnings` only, matching the no-shift behavior in
`apply_earnings` (`src/populace_dynamics/engine/steps.py:210-287`). The next even
draw reads all three fields
(`src/populace_dynamics/engine/forward_earnings.py:1046-1070`). Every changed chain
field enters the incremental and canonical audit views. Taxable payroll, AIME,
PIA, taxes, and benefits are recomputed downstream from aligned state; they are not
separately forced to SSA totals.

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

Adding or removing a production person must not shift any ordinary person's later
draws. The current loop pre-registers scheduled-entry IDs before the wave loop but
assigns newborn ordinals sequentially after each year
(`src/populace_dynamics/engine/loop.py:192-237,323-331`); person streams use those
ordinals, not person-ID values (`src/populace_dynamics/engine/loop.py:93-107`;
`src/populace_dynamics/engine/rng.py:158-169`). Merely choosing stable IDs is
therefore insufficient.

The producer completes `before` first and freezes a replay identity registry. Every
ordinary person, including each raw-positive synthetic child, retains the exact
`person_id`, person ordinal, and auxiliary values used by `before`. Suppressed raw
events leave their identities and ordinals reserved but inactive. The complete
candidate schedule assigns every counterfactual entrant or child a deterministic
ID and ordinal in disjoint ranges above the corresponding maxima in `before`,
ordered by stable event key; unused reservations remain holes. Common raw-positive
births reuse the ordinary path's child sex and other auxiliary draws, while a
counterfactual `0→1` birth uses per-event alignment streams.

The aligned loop must accept that frozen ordinal registry instead of appending
`len(person_ordinals)`, and child materialization must accept the reserved identity
and auxiliary record instead of the current length-dependent vector draw and
sequential allocator (`src/populace_dynamics/engine/steps.py:381-437`). Until this
production seam passes a zero-flip identity test and an add/suppress isolation
test, every roster-changing alignment margin is unavailable.

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
  [§2.8.2h at commit `b162f9c`](https://github.com/PolicyEngine/populace-dynamics/blob/b162f9c/docs/design/m6_projection_engine.md#L1183-L1264): only roster-present mothers materialize children; a scheduled birth to a removed mother enters report-only reconciliation. That lawful `roster_absent_births` drop stays in its upstream roster-reconciliation ledger. It is not an external-alignment flip. Fertility candidates are constructed only after the live-mother filter, and alignment must not resurrect or replace a mother to satisfy a target.

Every margin record therefore names `candidate_universe`, `materialization_universe`,
`measurement_universe`, its corresponding module scoring domain,
`gate_consumption = none`, and `aligned_values_consumed_by_gate = false`. Counts
from one universe may not be presented as counts from another.

Whole-unit selection does not automatically reconcile dependent relational state.
After mortality or emigration, the producer must validate spouse, parent-child,
household, cached marital, and cached composition references before the next
consumer. It may apply only a separately ratified dependent-state rule; it cannot
clear or invent relationships opportunistically. A margin whose roster change
would leave unresolved dependent state is unavailable pending decision 8.

## 5. Displacement accounting and publication

### 5.1 Two ledgers, one canonical before/after meaning

The existing function's labels make `before` the immutable unaligned scored path
and `after` the final aligned report-only path. An immediate raw snapshot at a
later hook is already conditional on earlier alignment, so passing it as `before`
would make the function's `scored_path = "unaligned"` false. The producer keeps two
separate ledgers:

1. a **canonical path-comparison ledger**, derived from the immutable unaligned
   companion and final aligned result and passed to the existing function; and
2. an **incremental intervention ledger**, comparing raw and reconciled outcomes
   inside the aligned replay and owned entirely by the production caller.

Raw projection panels cannot serve as the canonical pair for roster-changing
events: deaths remove rows, while births and entrants add them. The producer builds
a common audit keyspace over the union of stable
`(comparison_unit_id, year, margin_id, stratum_id)` keys. Initial people retain
their canonical IDs; entrants use donor-event IDs; synthetic children use the
event-derived identities required by §4.4. Each path receives every union key and
explicit numeric `eligible` and `present` indicators. This is a typed audit view,
not a padded live panel.

Each registered measure has one unit and a defined absence contribution:

| measure | unit and absence rule |
|---|---|
| `eligible` or `present` | binary; zero explicitly means the keyed unit is in the union but not eligible/present on that path |
| `represented_presence` | production-year represented people; `present * reference_weight`, accompanied by `present` |
| `event_selected` | binary event outcome; accompanied by path-specific `eligible` |
| `target_contribution` | the event's signed or unsigned contribution in the declared target unit; zero for an unselected event or explicit ineligibility, distinguished by `event_selected` and `eligible` |
| `covered_wage_contribution` | current covered-wage contribution; zero for explicit absence/noncoverage, with those indicators retained |

Undefined underlying state never becomes zero. The named contribution fields have
zero as part of their definition, and their presence/eligibility indicators make
the reason observable. `reference_weight` comes from the stable unit registry and
is identical in both union views: an existing person uses the frozen year-`y`
production weight, a prospective child uses the live mother's frozen year-`y`
weight that it will inherit, and an entrant uses the registered donor-member
weight. The producer rejects a cross-path mismatch, null keys, null or non-finite
numeric values, non-positive registered weights, duplicate keys, or an unmappable
live row before calling the existing function. The function's own exact-key check
then remains a second line of defense.

The producer invokes
[`build_alignment_displacement`](../../src/populace_dynamics/harness/m6_reporting.py)
separately for each margin and **homogeneous measure**, across all requested years.
The implementation at
`src/populace_dynamics/harness/m6_reporting.py:321-415` remains authoritative. A
binary event, represented-person contribution, materialized-person count, and
dollar value never share one call: its rowwise cross-field maximum would mix
units. Each single-unit call retains native `per_year_maximum`, run
`maximum_alignment_displacement`, and frame-global `n_intervened_rows`. The caller
computes per-year counts and sums. It publishes no scalar maximum across
incommensurate margins.

### 5.2 Incremental per-margin, per-year attribution

Every aligned run publishes one caller-owned record for every requested
`(margin_id, year, stratum_id)`, including records with no direct intervention.
Here `raw` means the raw module outcome conditional on all earlier aligned hooks;
it is never labeled as the gate-scored path. Each record contains:

- target, raw, and aligned weighted totals in one declared unit;
- raw residual, aligned residual, and the precomputed threshold-prefix resolution
  floor;
- counts of `0→1` and `1→0` unit flips;
- distinct affected decision people, added/removed materialized people, and
  displaced atomic units, with a margin-specific role code;
- `displaced_person_count`, `gross_displaced_person_weight`,
  `displaced_unit_count`, and `gross_displaced_unit_weight`, plus separate `0→1`
  and `1→0` fields;
- absolute target-contribution displacement and its maximum over a single unit;
- for wages, the scalar, changed-worker count and representation weight, and
  weighted and unweighted absolute dollar displacement;
- references to the canonical path-comparison results and hashes of the
  incremental pre/post hook views; and
- status, reason, candidate/materialization/measurement universes, target source,
  stream address, and algorithm revision.

The role code defines whose state changed. Mortality and covered work name the
person; a fertility event names the mother as decision person and the child as a
materialized addition; an entrant family names every entering member and one
atomic donor unit. `displaced_person_count` is the distinct union of these affected
people, never a rounded weight. `gross_displaced_person_weight` sums each affected
person's frozen pre-reconciliation production-year weight once. For a not-yet-live
addition, it uses the same pre-registered child or donor-member `reference_weight`
defined for the canonical views.
`gross_displaced_unit_weight` separately sums each changed atomic unit's registered
unit weight once. Adds and removals are never netted. The ledger also reports
decision-person and materialized-person subtotals so a maternal event does not
silently count a mother and child as interchangeable observations.

These sums supplement rather than reinterpret the canonical function. Its
`n_intervened_rows` is not relabeled as a per-year person count, a maximum change
in `event_selected` is not relabeled as aggregate displacement, and the canonical
path difference is not claimed to be direct causal attribution to one hook. The
run summary keeps both the path-comparison and incremental records for each margin
and year.

### 5.3 Artifact labels and completeness

An aligned artifact is publishable only if at least one requested margin is
`computed` and every requested margin has either:

- `computed`, with a complete displacement ledger and fidelity result; or
- `not_available`, with a pre-run reason such as a missing definition-matched
  granular target.

`not_available` does not mean that the producer silently skipped an advertised
margin. The artifact title and machine label list the margins actually applied.
If no margin can be applied, the producer publishes only an unavailable-attempt
sidecar; the inherited displacement surface remains
`status = "not_computed"`, `reason = "alignment_layer_output_not_collected"`, and no
panel is labeled `aligned_production`.
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

### 6.1 Pre-run threshold-prefix resolution floors

The target bundle, candidate construction, weights, strata, pooling ladder, score,
tie rule, and numeric rules are frozen before an aligned run. From that frozen
candidate ledger, the producer enumerates the empty and every ordered prefix and
records the minimum prefix residual in target units. That value is the
**selection-resolution floor** for the discrete margin-year-stratum. It is not the
minimum over every subset. The producer separately records whether the target lies
outside the empty-to-total candidate range.

The resolution floor comes first: the producer writes and hashes it before
selecting the aligned prefix, and the aligned evaluator reads rather than rewrites
it. It may not declare exact agreement when the registered threshold rule cannot
achieve it. A continuous covered-wage margin has a zero arithmetic floor only
when its denominator is positive and all registered finite-value conditions hold;
otherwise it is infeasible.

Sparse cells follow only the target bundle's pre-registered pooling ladder and
pre-run minimum-candidate rule. The producer decides pooling before inspecting
prefix residuals, publishes both cell definitions, and may not pool merely because
a non-prefix subset would fit better. A target outside the candidate range remains
`structurally_infeasible`; pooling cannot manufacture eligible people.

### 6.2 Alignment-fidelity check

For every applied cell, the evaluator independently recomputes the aligned margin
from the materialized panel using the declared measurement universe. The check
passes only when:

`abs(aligned_total - target) <= selection_resolution_floor + numeric_tolerance`.

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
percentage, or the selection-resolution floor. This preserves the floors-first order while
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
dynamic-microsimulation validation guidance to retain current, baseline, and
reference outputs and switch alignment off for the target module
([Harding et al. 2010, pp. 53-54](https://www.microsimulation.pub/articles/00038)).
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
candidate-ledger hashes, selection-resolution-floor hash, corridor hash or explicit null
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
   the atomic entrant unit, and the gross addition/removal targets or signed
   optimizer needed to implement V.A2 net change without an arbitrary split.
3. **Covered-work candidate state.** Define a lawful potential covered-earnings
   value, participation probability/uniform or alternate registered selection
   score for a raw nonworker selected `0→1`, and coherent chain-state updates.
   Settle wage-and-salary versus self-employment coverage and bind the result to
   IV.B4's “paid at any time in year” concept without altering the certified
   earnings domain.
4. **Fertility measurement.** Choose maternal birth events, live births, or
   children as the target unit; expose the complete probability/uniform schedule;
   specify parity evolution, birth order, plurality, and infant-death timing; and
   bind the measure to post-mortality live-roster materialization without changing
   the frame-independent scoring schedule.
5. **Displacement-corridor ceremony.** Pre-register the historical window,
   horizon bands, strata, target-revision vintages, synthetic shocks, seeds,
   quantiles, and response to `above_reference`. This design intentionally
   ratifies no numerical corridor.
6. **Policy-counterfactual semantics.** Decide whether an aligned reform run uses
   the baseline Trustees levels, aligns only common exogenous margins, or carries
   reform-specific external targets; specify common-random coupling and labels so
   alignment cannot erase a modeled policy effect.
7. **Release scope, horizon, and wage base.** Choose the first target-year interval
   and complete the separately reviewed horizon extension beyond the current 2022
   M6 panel builders. Pin the covered-wage base level/year and nominal-or-real V.B1
   growth column. Decide whether demographic margins ship before economics or the
   release waits for the full ordered bundle.
8. **Dependent relational state.** Ratify how mortality and emigration update
   spouse, parent-child, household, marital-core, and household-composition state,
   including the pending amendment-3h removed-spouse case, without rewriting a
   certified scoring schedule.

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
unaligned-gate-isolation checks. Until then, target margins may remain
`not_available`, while the inherited displacement surface remains
`status = "not_computed"` with
`reason = "alignment_layer_output_not_collected"`; the certified projection
remains unaligned.
