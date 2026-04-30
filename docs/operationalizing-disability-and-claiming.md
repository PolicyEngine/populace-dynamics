# Operationalizing Disability and Claiming

## Why This Chapter Exists

Earnings histories are only part of the Social Security problem. A model
can produce reasonable AIME distributions and still fail badly on the
actual pathways through which people enter benefits.

That is especially true for:

- SSDI entry and exit
- conversion from disabled-worker to retired-worker benefits
- early retirement claiming
- delayed claiming
- spousal and survivor benefit timing

These are not minor details. They affect beneficiary counts, average
benefits, distributional incidence, and the interpretation of reforms.
This chapter therefore does for disability and claiming what the earlier
operational chapter does for earnings: it turns a general aspiration
into a concrete build plan.

## The Main Modeling Distinction

The proposal should distinguish three different objects that often get
blurred together:

1. **Underlying work limitation or health decline**:
   a latent or observed deterioration in functional or work capacity
2. **Program pathway**:
   application, adjudication, award, denial, appeal, return-to-work, or
   conversion rules
3. **Benefit-claiming decision**:
   when an eligible person elects to claim retired-worker, spouse,
   survivor, or disability benefits

Those objects are related, but not identical.

For example:

- a person can have serious health limitations and never apply for SSDI
- a person can stop working before claiming retired-worker benefits
- a person can be fully insured and eligible at age 62 but delay
  claiming
- a spouse can receive a benefit driven more by the worker's record than
  by her own earnings history

The model should therefore carry separate states for impairment, program
status, and claiming status rather than collapsing them into one
"disabled/not disabled" or "retired/not retired" flag.

## What the Public Benchmark Models Tell Us

The public record on comparable models is revealing.

### DYNASIM

Public descriptions of DYNASIM indicate that it carries rich health,
disability, marriage, and program modules, with yearly transition
equations and Social Security rule calculators [@favreault2015; @urban2024dynasim4].
That is the right benchmark for seriousness. But the public record is
less explicit about the exact claiming and disability-administration
machinery than about the existence of those modules.

### MINT

MINT is more explicit about its limitations, which is useful for this
proposal. SSA's current public description says MINT includes most
current-law basic Social Security rules, but:

- it projects a **single claiming age**
- it does not allow sophisticated claiming strategies
- it does not model detailed disability adjudication outputs
- it pays disability benefits from onset until death or conversion to
  retired-worker benefits
- it does not model disability return-to-work rules such as the trial
  work period or extended period of eligibility [@ssa2024mint]

That is an important calibration point for us. A fundable phase 1 does
not need to promise the full monthly administrative machinery if even
MINT abstracts from some of it. But it does need to be explicit about
which abstractions it is making.

### CBO

The public CBO record remains thinner on record-level claiming and
disability construction. That itself is useful context. It means the
project should not overclaim comparative detail where public
documentation does not support it.

## Recommended State Representation

The proposal should specify disability and claiming as explicit state
machines layered on top of the longitudinal earnings panel.

### Disability state

At minimum, the annual disability state should distinguish:

1. `no_work_limitation`
2. `work_limited_not_applying`
3. `applying_for_disability`
4. `awarded_disabled_worker`
5. `denied_or_exited_application`
6. `terminated_or_recovered`
7. `converted_to_retired_worker`

The core point is that "disabled" and "receiving SSDI" are not the same
thing.

### Claiming state

At minimum, the benefit-claiming state should distinguish:

1. `fully_insured_not_claiming`
2. `claiming_exact_62`
3. `claiming_late_62`
4. `claiming_63_to_fra`
5. `claiming_at_fra`
6. `claiming_post_fra_to_70`
7. `disabled_worker_beneficiary`
8. `spouse_or_survivor_only`
9. `dually_entitled`

The first funded version does not need perfect sophistication on every
auxiliary path, but it should not flatten all retired-worker claiming
into a single age bucket or ignore dual-entitlement logic entirely.

### Event overlay

Because Social Security rules are sensitive to timing within a year, the
project should add a lightweight event layer on top of the annual panel.
That event layer should capture:

- month or month-bucket of retirement claim
- disability onset date or month bucket where available
- disability award date or lag bucket
- conversion from SSDI to retired-worker benefits at FRA
- death date for survivor eligibility

This does not require a full monthly microsimulation for every domain.
But it does require more than a single annual "claimed this year" flag.

## Phase 1 Disability Design

The first funded version should aim for a credible disability pathway,
not the full administrative process.

### Core phase-1 disability objects

The model should estimate:

- disability or work-limitation onset
- application or award probability conditional on onset and work
  history
- disability beneficiary status
- duration on the rolls
- conversion to retired-worker benefits at FRA

### What phase 1 should simplify

The proposal should state plainly that phase 1 may simplify:

- detailed adjudication stages
- appeal timing
- diagnosis-specific allowance pathways
- trial work period and extended period of eligibility
- continuing disability reviews
- multiple disability spells beyond a simple capped count

That would still leave a useful and fundable disability layer, provided
the simplifications are disclosed and bounded.

### Why pre-disability earnings decline matters

SSA research shows that DI applicants often experience a substantial
earnings decline in the years before application [@costa2017di].
That matters because a disability module that starts only at formal
award will miss an important part of the real path.

So the phase-1 disability layer should include:

- a pre-award or pre-application earnings-decline state
- reduced probability of strong covered earnings in the years just
  before award
- interaction between declining work attachment and retirement claiming

This is one of the main places where a public model can avoid an overly
mechanical interpretation of disability entry.

## Phase 1 Claiming Design

The first funded version should also implement a minimal but explicit
claiming model.

### The central empirical fact

Administrative studies from SSA show that working and claiming behavior
around age 62 is heterogeneous and cannot be summarized as a single
"claim at first eligibility" path [@waldron2020claiming; @waldron2020decile].

That means the model should not assume:

- everyone who claims at 62 stops work immediately
- everyone who stops work at 62 claims immediately
- claiming behavior is homogeneous across the earnings distribution

### Minimal claim-age structure

The claim-age model should at least separate:

- exact-62 claims
- later-at-62 claims
- claims after 62 but before FRA
- claims at FRA
- delayed claims after FRA

This is already materially better than a single claim age and is well
aligned with the kinds of administrative classifications SSA itself uses
in its own claiming research [@waldron2020claiming; @waldron2020decile].

### Predictors for claiming

A practical reduced-form claiming hazard should condition on:

- age
- current and lagged earnings
- lifetime rank or AIME proxy
- marital status
- spouse benefit relevance
- disability status or prior disability pathway
- wealth proxy where available
- health or work-limitation status
- calendar year and cohort
- projected benefit level under current law

The important thing is not to build a full structural retirement model.
It is to avoid a claiming model that is purely exogenous to the features
that obviously matter.

## Spousal and Survivor Logic

The model should acknowledge that auxiliary benefits are not just a rule
calculator problem. They are also a history-construction problem.

To support serious spouse and survivor analysis, the panel needs:

- marriage and divorce timing
- spouse links
- widowhood timing
- marriage duration
- claim timing for both members of the pair where relevant

That history-construction problem deserves its own treatment. See
[operationalizing-family-and-auxiliary-benefits.md](operationalizing-family-and-auxiliary-benefits.md)
for the proposed relationship-history layer and validation approach.

The first funded version may still choose to prioritize own-worker and
disabled-worker claiming over the full spouse and survivor timing model.
But if it does, that should be presented as a stage choice rather than
left implicit.

## Recommended Operational Sequence

The cleanest build sequence is:

1. **Own-worker retirement claiming**
   with explicit claim-age buckets
2. **Disability pathway**
   including onset, simplified award path, and conversion to retired
   worker at FRA
3. **Spousal and survivor timing**
   layered onto the family-history machinery
4. **More complex administrative features**
   such as adjudication details, return-to-work rules, and advanced
   claiming strategies

This sequence is realistic, fundable, and easier to validate than
trying to solve every benefit path at once.

## Estimation Inputs

The main data and benchmark inputs should include:

- **PSID and HRS** for longitudinal work, health, and retirement
  transitions
- **SSA published statistics** for disability incidence, prevalence,
  benefit counts, and claiming distributions
- **SSA administrative research papers** for working-and-claiming
  classifications and disability pre-application earnings patterns
- **MINT documentation** for a public benchmark on what a serious but
  simplified model includes and omits

The project should also be explicit where public data are weak. For
example:

- detailed adjudication timing is hard to reconstruct publicly
- diagnosis-specific disability pathways are harder than generic onset
  and award states
- sophisticated claiming strategies are not the right phase-1 promise

## Evaluation Metrics for Disability

The disability layer should be judged on policy-facing metrics, not only
internal fit.

### Core disability metrics

| Metric | Why it matters |
|---|---|
| Disability incidence by age and sex | Basic realism of onset path |
| Disabled-worker prevalence by age and sex | Roll-level fit |
| Average DI benefit | Benefit-level realism |
| Conversion from DI to retired-worker benefits | Correct treatment at FRA |
| Earnings decline before award or application | Captures pre-benefit deterioration rather than only formal program entry |

### Stretch metrics

If data support them, later stages should also examine:

- denial versus award patterns
- duration from onset to award
- duration on the rolls
- return-to-work transitions

## Evaluation Metrics for Claiming

The claiming layer should be judged on the patterns the policy community
actually argues about.

### Core claiming metrics

| Metric | Why it matters |
|---|---|
| Share claiming at exact 62, later 62, 63-FRA, FRA, and post-FRA | Central behavioral output |
| Claiming distribution by lifetime earnings proxy or AIME bucket | Distributional realism |
| Working-and-claiming categories around age 62 | Captures heterogeneity that a single claim-age model misses |
| Share stopping work before claiming | Important for interpretation of early claiming |
| Average benefit by claim-age bucket | Links behavior to benefit outcomes |

### Family-benefit metrics

Where spouse and survivor timing is in scope, add:

- spouse versus own benefit shares
- widow(er) beneficiary counts
- dual-entitlement shares

## Suggested Stage-1 Thresholds

The proposal should define at least rough stage-1 gates for this layer.

| Metric | Draft stage-1 threshold |
|---|---|
| Disabled-worker counts by age-sex | within 2-3 percent |
| Average DI benefit | within 2-3 percent |
| Retirement claim-age shares in major buckets | within 2 percentage points |
| Working-and-claiming category shares around age 62 | within 3 percentage points for major categories |
| Conversion from DI to retired-worker status at FRA | directionally correct and within published range |

As with the earnings chapter, these should be refined during
implementation. But the proposal should not dodge numeric commitments
entirely.

## How This Should Be Positioned in the Proposal

The proposal should make three points clearly.

### 1. We know what phase 1 can and cannot promise

The project can credibly promise:

- an explicit own-worker claiming model
- a simplified but real disability pathway
- conversion from SSDI to retired-worker benefits
- benchmarked spouse/survivor expansion if the family-history layer is
  ready

It should not prematurely promise:

- full disability adjudication
- diagnosis-specific award timing
- return-to-work program rules
- every sophisticated claiming strategy

### 2. Public transparency is still the differentiator

The proposal's advantage is not that it will instantly outperform SSA
internally. It is that the assumptions around disability and claiming
will be public, inspectable, and benchmarked.

### 3. This layer should have its own stage gate

Disability and claiming should not be treated as minor add-ons after
the earnings model "works." If the project cannot produce a credible
claiming and disability layer, it should narrow its policy claims
accordingly.

## Bottom Line

The proposal should not leave disability and claiming as vague hazard
models in the background.

It should describe them as explicit state machines, state what the
first funded version will simplify, benchmark those simplifications
against MINT and DYNASIM, and evaluate the resulting panel on the
patterns Social Security policy actually cares about.
