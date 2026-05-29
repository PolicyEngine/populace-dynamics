# Operationalizing mortality and projection drift

## Why this chapter exists

Mortality and projection control are easy to underspecify in a grant
proposal because they can sound like routine background machinery.

They are not.

In this project, they directly affect:

- survivor benefit incidence
- lifetime benefit totals
- distributional progressivity by lifetime earnings
- long-run beneficiary counts
- trust fund and fiscal projections
- whether reform results remain interpretable after the model is aged
  forward

The proposal therefore needs a more operational treatment of two linked
problems:

1. **mortality construction**
   who dies when, and how death risk varies with age, sex, cohort, and
   socioeconomic status
2. **projection drift**
   how the model is kept from wandering away from published demographic,
   labor-market, and program benchmarks over time

The first problem is about state evolution. The second is about
discipline and alignment.

## The main modeling distinction

The proposal should distinguish three separate objects.

1. **Micro mortality process**
   the person-level annual or monthly death hazard
2. **Mortality improvement assumptions**
   cohort or calendar-time improvements in survival used for forward
   projections
3. **Projection alignment and drift control**
   the mechanisms used to keep the aggregate projection near credible
   external benchmarks

These objects are related, but they are not interchangeable.

For example:

- a model can have a reasonable age-sex life table and still produce the
  wrong survivor counts if differential mortality by earnings is wrong
- a model can match baseline mortality rates and still drift badly in
  long-run beneficiary counts if cohort improvement assumptions are
  misaligned
- a model can hit aggregate population totals through heavy alignment
  while still hiding poor micro-level mortality dynamics

So the proposal should never talk as if "we use SSA life tables" fully
solves the mortality problem.

## Why mortality is policy-relevant

Mortality is not just a demographic background variable. For Social
Security, it changes the meaning of reform results.

Higher earners tend to live longer, which means they collect benefits
for more years even under a progressive benefit formula. Research using
tax and SSA-linked mortality data found a large life-expectancy gap
between the top and bottom of the income distribution and showed that
the gap increased over time [@chetty2016life].

That matters for:

- retirement-age increases
- lifetime progressivity analysis
- survivor-benefit reforms
- adequacy analysis for widows and lower-earning households
- racial and geographic equity discussions, to the extent those are
  modeled explicitly

If the project wants to claim any seriousness on lifetime incidence, it
needs a mortality layer that is more than an undifferentiated age-sex
table.

## What the public benchmark models tell us

The public record on benchmark models is useful here.

### DYNASIM

Public documentation says DYNASIM aligns major demographic and labor
outcomes, including mortality, to targets from SSA's Office of the Chief
Actuary [@favreault2015; @urban2024dynasim4].

That is important for two reasons:

- DYNASIM is not a purely free-running microsimulation
- mortality and projection alignment are treated as core operating
  machinery rather than an afterthought

This is the right benchmark for seriousness.

### MINT

MINT is useful because it reminds us that a Social Security model can be
institutionally credible while still leaning heavily on SSA assumptions
and administrative benchmarking [@ssa2024mint].

The lesson for this proposal is not "do everything endogenously." The
lesson is "be explicit about which parts are model-driven and which
parts are aligned."

### CBO

The public CBO record is especially relevant on the projection side.
CBOLT and the long-term Social Security outlook are designed around
official projection coherence, not public inspectability of every record
transition [@cbo2004; @cbo2018; @cbo2024longterm; @cbo2024finances].

That is a useful comparison because it clarifies our comparative
advantage:

- not official baseline authority
- but a public micro-level projection stack with explicit alignment and
  published failure modes

## Recommended state representation

The proposal should describe mortality as a state block, not just as a
reference table.

### Core mortality state

At minimum, the annual panel should distinguish:

1. `alive`
2. `dies_this_year`
3. `dead`

But that is not enough by itself. The state should also carry the main
mortality-risk conditioning variables:

- age
- sex
- birth cohort
- lifetime earnings rank or AIME proxy
- education
- disability status
- marital status or widowhood status where relevant
- current claiming or beneficiary status if it improves fit

The key point is that death should be modeled as a hazard conditional on
policy-relevant heterogeneity, not just age and sex.

### Event layer

Because survivor eligibility can depend on timing, the project should
also carry a death date or at least a death month bucket.

That does not require a fully monthly model for all domains. It does
require enough timing detail to support:

- widowhood timing
- survivor benefit eligibility in the year of death
- clean transitions from worker/spouse to survivor states

### Projection metadata

For forward runs, the panel should preserve additional metadata:

- raw mortality hazard before alignment
- applied mortality improvement factor
- aligned mortality hazard after calibration
- indicator that a record-level outcome was affected by aggregate
  alignment

That metadata is useful both for debugging and for making the
calibrated-versus-uncalibrated distinction visible to users.

## Phase 1 mortality design

The first funded version should aim for a credible mortality layer, not
the final word on longevity modeling.

### Core phase-1 mortality objects

The phase-1 mortality build should include:

- age-sex baseline hazards from SSA actuarial life tables
- differential mortality by lifetime earnings proxy and education
- mortality improvements over time using Trustees-style assumptions
- integration with disability and family states where feasible
- survivor-timing outputs good enough for spouse and widow analysis

This is already a meaningful mortality module.

### What phase 1 can simplify

The proposal should state plainly that phase 1 may simplify:

- cause-specific mortality
- fully endogenous health shocks and mortality feedback
- local-area mortality modeling beyond major stratifiers
- extremely fine race and ethnicity heterogeneity if data support is
  weak
- separate mortality regimes for every program pathway

Those simplifications are acceptable if they are disclosed and if the
remaining mortality layer is still validated on the main distributional
and program-facing outcomes.

### What phase 1 should not simplify away

Phase 1 should not collapse:

- mortality to age-sex only
- survivor timing into a generic death flag
- differential mortality by socioeconomic status into a vague narrative
  without quantitative implementation

Those simplifications would undermine lifetime-incidence claims too
directly.

## Mortality inputs and benchmark sources

The proposal should name the main inputs explicitly.

### SSA actuarial life tables

SSA's actuarial life tables provide the basic age-sex mortality schedule
for the Social Security area population. SSA currently publishes the
2022 period life table as used in the 2025 Trustees Report
[@ssa2025life; @ssa2025trustees].

This is the natural baseline for:

- single-year age hazards
- overall life expectancy validation
- consistency with the broader Social Security projection frame

### Trustees assumptions

The Trustees Reports provide the macro projection context:

- mortality improvement assumptions
- fertility assumptions
- wage and price growth
- labor-force and covered-worker trends
- trust-fund and benefit projections

The project should use those reports as the primary public benchmark for
forward demographic and Social Security alignment
[@ssa2025trustees].

### Differential mortality evidence

The project also needs a public source for socioeconomic mortality
differences. The strongest obvious benchmark is the Chetty et al.
evidence on income and life expectancy, which uses SSA death records
linked to tax data and documents large and growing mortality gaps by
income [@chetty2016life].

That is not a perfect one-to-one mapping to our synthetic panel. But it
is strong enough to justify a differential mortality layer and to anchor
validation targets.

## Recommended mortality construction strategy

The simplest credible design is a layered hazard.

### 1. Start with an SSA baseline hazard

For each person-year, begin with an age-sex baseline hazard from the SSA
actuarial life table.

### 2. Apply differential-risk adjustments

Adjust the baseline using a transparent risk model based on:

- lifetime earnings proxy
- education
- disability status
- possibly marital status or widowhood if it materially improves fit

The exact parameterization can vary, but the proposal should prefer
something inspectable over a black-box death model in phase 1.

### 3. Apply cohort or calendar improvement factors

Projecting forward requires more than freezing today's mortality table.
The model should incorporate mortality improvement assumptions that are
consistent with the Trustees projection environment.

### 4. Generate death events

Simulate death at the person-year level and record event timing for
survivor logic.

### 5. Align if needed

If the free-running mortality process drifts from published age-sex or
cohort survival expectations, apply explicit alignment factors and store
the before-and-after values.

This is the right place to be transparent: alignment is a feature, not a
sign of methodological failure.

## Why projection drift needs its own design

Once the model is projected forward, drift is unavoidable unless the
project plans for it explicitly.

Small mismatches accumulate through:

- mortality
- disability
- marriage and widowhood transitions
- covered-work participation
- wage growth
- cohort entry
- claiming behavior

So the proposal should describe projection drift as a central operating
problem, not a residual annoyance.

## Sources of projection drift

The main sources of drift are different and need different responses.

### 1. Jump-off error

If the base-year synthetic panel is slightly wrong, the projection does
not start from the correct population.

### 2. Transition mis-specification

Even if the base year is good, annual hazards can slowly compound into
bad aggregate outcomes.

### 3. Macro mismatch

The micro model may imply wage, mortality, beneficiary, or labor-force
paths that differ from published benchmark assumptions.

### 4. Interaction drift

A model can get each module roughly right in isolation and still drift
when modules interact. This is especially important for:

- disability and mortality
- widowhood and survivor claims
- claiming and continued work
- mortality and beneficiary duration

The proposal should say this explicitly. Not all drift comes from one
broken equation.

## Recommended drift-control stack

The proposal should describe drift control as a stack rather than a
single calibration step.

### Layer 1. Baseline calibration

Start from a base-year panel that already matches key cross-sectional
demographic, earnings, and beneficiary targets reasonably well.

### Layer 2. Process calibration

Tune transition parameters so the free-running model is close on:

- covered-worker shares
- mortality by age-sex
- disability incidence
- marital-status evolution
- claimant shares

This is preferable to relying only on heavy ex post reweighting.

### Layer 3. Periodic alignment

At defined intervals, align the model to external control totals such
as:

- age-sex population counts
- mortality schedules
- covered payroll
- beneficiary counts
- aggregate benefits

The proposal currently leans toward five-year alignment with annual
recalibration in the near term, which is a sensible starting point.

### Layer 4. Output-level monitoring

Even after alignment, monitor:

- raw versus aligned outcomes
- model-induced drift between alignment years
- whether alignment is compensating for a deeper process problem

This is the step that prevents alignment from becoming a black box.

## Near-term versus long-term projection strategy

The proposal should separate near-term and long-term projection goals.

### Near-term projection

For roughly the first 10 years, the model should emphasize:

- close tracking of published benchmarks
- annual or near-annual recalibration where needed
- diagnostic transparency over elegance

This is the horizon where users will most naturally compare outputs with
published SSA or CBO baselines.

### Long-term projection

For longer horizons, the model should emphasize:

- stability of cohort patterns
- interpretable alignment to Trustees assumptions
- explicit uncertainty bands
- clear disclosure of where confidence is lower

The project should not pretend that a public micro model will produce a
uniquely authoritative 75-year path without strong alignment and
uncertainty reporting.

## Evaluation metrics for mortality

The mortality layer should be judged on policy-facing metrics.

### Core mortality metrics

| Metric | Why it matters |
|---|---|
| Age-sex death rates | Basic mortality realism |
| Period life expectancy by sex | High-level survival fit |
| Survival to key ages such as 62, 67, and 85 | Directly relevant to claiming and benefit duration |
| Mortality differentials by earnings proxy or education | Lifetime incidence and progressivity |
| Widowhood prevalence by age and sex | Survivor-path realism |
| Survivor beneficiary counts where modeled | Benefit-facing mortality validation |

### Stretch metrics

If data support them, later phases should also evaluate:

- geographic mortality differentials
- race and ethnicity mortality differentials
- mortality by disability pathway
- cohort improvements in survival over time

## Evaluation metrics for projection quality

Projection quality should have its own scorecard.

### Core projection metrics

| Metric | Why it matters |
|---|---|
| Population counts by age and sex over time | Basic demographic stability |
| Covered-worker shares over time | Key to tax base and eligibility |
| Aggregate covered payroll | Core fiscal input |
| Beneficiary counts by major type | Core program output |
| Aggregate benefit payments | Fiscal realism |
| Trust-fund directional consistency with published baselines | High-level budget plausibility |
| Raw versus aligned gap by projection year | Detects hidden drift |

### Diagnostic drift metrics

The project should also track:

- cumulative drift between alignment dates
- sensitivity to mortality improvements
- sensitivity to wage growth and covered-worker assumptions
- whether reform deltas are stable across reasonable alignment choices

That last point matters. A reform result that changes sharply depending
on minor alignment choices is not ready for confident public use.

## Suggested stage-3 thresholds

The proposal should define rough projection gates rather than leaving
stage 3 qualitative.

| Metric | Draft stage-3 threshold |
|---|---|
| Age-sex population totals in near-term projection years | within 1 percent |
| Major beneficiary counts by type | within 2-3 percent |
| Aggregate benefit payments | within 2-3 percent |
| Covered payroll | within 2-3 percent |
| Period life expectancy by sex | within 0.2 years |
| Direction and rough magnitude of trust-fund trajectory | consistent with published benchmark range |

These should be refined during implementation, but the proposal should
still commit to the idea that stage 3 is falsifiable.

## What this should mean for the proposal

The proposal should make three points clearly.

### 1. Mortality is part of the incidence story

If the project wants to talk about lifetime fairness, widow outcomes, or
distributional reform effects, mortality cannot remain a one-line note
in the methods section.

### 2. Projection alignment is not optional

The benchmark models already teach this. The right comparison is not
"free-running purity" versus "aligned models." The real comparison is
"explicit alignment" versus "hidden drift."

### 3. Raw and aligned outputs should both be preserved

This is one of the main transparency advantages the public model can
offer. Users should be able to see:

- what the model would do on its own
- what changed because of alignment
- how sensitive the results are to that alignment

That is a stronger public contribution than pretending the calibration
layer does not exist.

## Recommended build sequence

The cleanest sequence is:

1. **Baseline mortality layer**
   age-sex hazard plus event timing
2. **Differential mortality adjustments**
   using earnings or education proxies
3. **Survivor-facing validation**
   widowhood and survivor counts
4. **Forward projection alignment**
   using Trustees-style benchmarks
5. **Projection drift scorecard**
   raw versus aligned monitoring and uncertainty reporting

This is realistic, fundable, and easier to validate than attempting a
fully endogenous mortality-and-macro system from the outset.

## Bottom line

The proposal should not describe mortality as "use SSA life tables" and
projection quality as "align to Trustees assumptions."

It should describe:

- an explicit mortality state layer
- differential survival by socioeconomic status
- survivor-relevant death timing
- a multi-layer drift-control stack
- evaluation metrics that make stage 3 falsifiable

That is the level of operational detail needed if the project is going
to claim serious long-run Social Security analysis.
