# Operationalizing family structure and auxiliary benefits

## Why this chapter exists

Family structure is not a side table in Social Security modeling. It is
one of the main reasons a dynamic model is needed at all.

That is especially true for:

- spousal benefits
- survivor and widow(er) benefits
- divorced spouse benefits
- dual entitlement
- dependent and child-in-care benefits
- reform proposals aimed at widows, caregivers, or households with
  interrupted work histories

A model can match own-worker benefit distributions and still be weak on
some of the most policy-relevant adequacy questions if it treats family
history as an afterthought. This chapter therefore does for family
structure what the earlier operational chapters do for earnings,
disability, and claiming: it turns a broad requirement into a concrete
build plan.

## The main modeling distinction

The proposal should keep three objects separate.

1. **Current family status**
   whether a person is married, divorced, widowed, never married, or
   living with dependent children in the current year
2. **Relationship history**
   the timing, duration, and ordering of marriages, divorces, deaths,
   remarriages, and parent-child links over the life course
3. **Benefit-facing auxiliary status**
   whether current law would make the person eligible for spouse,
   survivor, divorced spouse, or dependent benefits and whether those
   benefits top up or replace an own-worker entitlement

Those objects overlap, but they are not the same.

For example:

- two currently unmarried women of the same age can have very different
  benefit prospects if one is never married and the other is widowed
- two currently married couples can imply very different spouse and
  survivor outcomes depending on the earnings asymmetry within the pair
- divorced spouse benefits depend on marriage duration and remarriage
  history, not just current marital status
- child-in-care and dependent benefits depend on family links and child
  ages, not just on the worker's earnings record

The model should therefore carry explicit relationship history and
benefit-facing family states rather than relying on current marital
status alone.

## Why this matters for policy analysis

Some of the most visible Social Security adequacy debates run through
family pathways rather than through the retired-worker benefit formula
alone [@whitman2011; @tamborini2013].

That includes:

- widow benefit proposals
- divorced spouse eligibility and adequacy concerns
- gender disparities driven by interrupted work histories
- proposals that raise or lower own-worker benefits but have different
  effects on dually entitled beneficiaries
- caregiver credit proposals that interact with later spouse or survivor
  benefit receipt

This is exactly where a public model can otherwise overstate its
usefulness. If the family-history layer is weak, the model may still
look fine on aggregate OASDI payments while misrepresenting adequacy for
widows, divorced beneficiaries, and low-own-earnings spouses.

## What the public benchmark models tell us

The public record on existing models gives useful guidance about what
needs to be explicit.

### DYNASIM

Public descriptions of DYNASIM indicate that it carries family
structure, marriage, divorce, disability, and Social Security benefit
logic inside a larger retirement-income model
[@favreault2015; @urban2024dynasim4].

That is the right benchmark for seriousness. It implies that a model
meant to analyze spouse and survivor benefits cannot stop at person-only
earnings paths.

But DYNASIM does not solve the transparency problem for us:

- the full code is not open
- the public record is stronger on model scope than on exact
  relationship-history mechanics
- outside researchers cannot fully inspect how family-history
  construction and auxiliary-benefit logic interact

### MINT

MINT is a useful public benchmark because its documentation is clearer
about what is included and what is simplified
[@smith2021mint8; @ssa2024mint].

The main lesson here is that a credible phase 1 does not need to claim
the full combinatorial complexity of every family-benefit path. But it
does need to be explicit about:

- which auxiliary-benefit categories are in scope
- how marriage histories are carried
- whether claimant timing is simplified
- which exceptions and edge cases are deferred

### CBO

The public CBO record remains relatively thin on person-level family
construction and auxiliary-benefit pathways. That is itself useful. It
means the proposal should not pretend we know more about public CBO
micro-implementation than the documentation supports.

## Recommended state representation

The proposal should specify a family-history layer as a first-class
state block inside longitudinal `microplex`.

### Current marital-status state

At the annual level, the model should distinguish at least:

1. `never_married`
2. `currently_married_first`
3. `currently_married_remarried`
4. `divorced_not_remarried`
5. `widowed_not_remarried`
6. `separated_or_transitioning`

The exact categories can change, but the main point should remain:
current status needs to distinguish remarried and previously married
people from never-married people because their auxiliary-benefit
exposure differs materially.

### Relationship-history layer

The annual status is not enough. The panel should also carry a
relationship-history object with:

- current `spouse_id` where applicable
- prior spouse links
- marriage start date or year
- marriage end date or year
- end reason: divorce, death, separation, or unresolved
- marriage order
- marriage duration in months or month buckets
- age at marriage
- spouse age gap
- spouse education pairing
- spouse lifetime-earnings rank proxy

This is the layer that makes divorced spouse rules, widowhood rules, and
dual-entitlement logic possible.

### Parent-child and dependent layer

Auxiliary benefits are not only about spouses. The family-history layer
should also support:

- child links
- child year of birth or age
- child disability-before-22 flag where feasible
- child-in-care indicator
- co-residence where relevant to dependent pathways

Not all of those need to be in the first funded release, but the schema
should anticipate them.

### Benefit-facing auxiliary state

The proposal should also define an explicit benefit-facing family state.
At minimum, that state should distinguish:

1. `own_worker_only`
2. `spouse_eligible_topup`
3. `divorced_spouse_eligible`
4. `survivor_eligible`
5. `dually_entitled`
6. `family_maximum_relevant`
7. `child_or_dependent_pathway`

These are not all mutually exclusive in a legal sense, but they are
useful operational categories for simulation and validation.

### Event overlay

As with claiming, monthly timing can matter even if the main panel is
annual. The family-history layer should therefore include lightweight
event timing for:

- marriage
- divorce
- spouse death
- remarriage
- child birth where in scope

This does not require a fully monthly model for every process. It does
require more than a single annual marital-status flag.

## Phase 1 scope

A fundable phase 1 should be narrower than "all family pathways," but it
should still be strong enough to support the main auxiliary-benefit
questions.

### Core phase-1 family objects

The first funded version should aim to support:

- current and prior marriage histories
- marriage duration sufficient for the 10-year divorced spouse rule
- spouse links for current couples
- widowhood timing
- pair-level own-worker versus spouse-benefit relevance
- dual-entitlement logic for spouse top-ups
- survivor-benefit relevance for older adults

That is already a meaningful capability set. It covers many of the
questions policy organizations actually ask about widows, low earners,
and households with uneven earnings histories.

### What phase 1 can simplify

The proposal should state plainly that phase 1 may simplify:

- precise monthly sequencing for every remarriage interaction
- parent benefits
- some dependent-child and child-in-care pathways
- simultaneous management of many current and former spouse claim paths
- very detailed family-maximum interactions outside the main older-adult
  cases
- rare exception cases that matter little for aggregate incidence

These simplifications are acceptable if they are disclosed and bounded.
They are not acceptable if the proposal implies full household-history
completeness without actually funding it.

### What phase 1 should not simplify away

Phase 1 should not collapse:

- widowhood into generic non-married status
- divorced and never-married people into one category
- current couples into random pairings that ignore earnings asymmetry
- spouse and own-worker benefit receipt into a single "beneficiary"
  flag

Those shortcuts would erase exactly the parts of the program that make
family structure policy-relevant.

## Why older-adult history deserves priority

The Social Security-first version of the project does not need to model
every family process equally well at every age.

If resources force prioritization, the model should favor the histories
that matter most for near-retirement and beneficiary analysis:

- marriage duration for older adults
- widowhood timing
- remarriage after widowhood or divorce
- spouse earnings asymmetry
- current-law exposure to spouse and survivor benefits

That is a better phase-1 priority than trying to perfect general
fertility dynamics or every child-benefit path before the older-adult
auxiliary layer is credible.

## Recommended construction strategy

The family-history layer should be built in a way that respects both the
existing `microplex` cross section and the needs of Social Security
benefit logic.

### 1. Start from the base-year household network

`microplex` already provides a cross-sectional household and family
structure. That gives the project a real starting point for:

- current couples
- children in household
- household composition
- current relationship links

The longitudinal build should preserve this observed or synthesized
current network where possible rather than redrawing it from scratch.

### 2. Reconstruct missing histories backward

For current adults, the model needs plausible prior marriage and family
events that are not directly observed in the base-year cross section.
Backward reconstruction should infer:

- likely prior marriages
- timing of divorce or widowhood
- duration of prior marriages
- likely spouse characteristics where the former spouse is not present

This is a synthetic-history problem similar in spirit to earnings
reconstruction, but with stronger relational constraints.

### 3. Simulate forward transitions

After base-year history construction, the model should simulate:

- first marriage
- divorce
- widowhood
- remarriage
- dependent-child transitions where in scope

These can be implemented as hazard models or related reduced-form
transition models. The important thing is not sophistication for its own
sake. It is preservation of the joint relationship structure that
benefit rules depend on.

### 4. Match couples jointly, not independently

Person-level marriage hazards alone are not enough. A family-history
layer also needs a couple-formation mechanism.

That mechanism should preserve:

- age gaps
- educational homogamy
- race and ethnicity patterns where modeled
- geography where sample quality permits
- spousal earnings correlation
- dual-earner versus single-earner household patterns

This is one of the strongest reasons to think in terms of longitudinal
`microplex` rather than a loose collection of independent hazards.

### 5. Enforce relational consistency

The build needs an explicit repair or constraint pass so that:

- marriage links are two-sided
- no person has impossible overlapping marriages
- widowhood requires a spouse death event
- child ages are feasible relative to parent ages
- remarriage timing does not violate prior relationship end dates

Without this pass, a synthetic panel can look reasonable in marginal
statistics while still containing relational impossibilities that break
benefit calculations.

### 6. Derive benefit-facing family variables

Only after the relationship layer is coherent should the model derive:

- spouse PIA relevance
- ex-spouse PIA relevance
- deceased-spouse PIA relevance
- current spouse top-up eligibility
- divorced spouse eligibility
- widow(er) eligibility
- family-maximum relevance
- dual-entitlement categories

That ordering matters. Auxiliary-benefit variables should be outputs of
the relationship-history layer, not substitutes for it.

## Matching and household synthesis

The proposal should go one level deeper than "we will estimate marriage
hazards."

### Why hazards alone are not enough

A person-level hazard can say who is likely to marry or divorce, but it
does not by itself determine:

- whom they marry
- whether couples look realistic jointly
- whether spouse earnings asymmetry is preserved
- how remarriage affects later survivor exposure

Those are pair-level and household-level problems.

### Recommended practical approach

The simplest credible phase-1 approach is a hybrid:

1. estimate person-level entry and exit hazards from panel data
2. when a union is formed, perform constrained matching using a
   distance-based or score-based pairing algorithm
3. run a consistency repair pass on the resulting relationship network
4. benchmark the resulting couples against cross-sectional and panel
   targets

This is less elegant than a fully joint generative household model, but
it is likely more fundable and easier to validate in phase 1.

### Higher-upside extension

If longitudinal `microplex` advances enough, the project can later move
toward hierarchical or household-first generation that jointly models:

- household composition
- partner assignment
- child links
- earnings correlation within couples
- tax-unit and family-unit consistency

That is a plausible phase-2 or methodology-R&D direction. It should not
be the phase-1 dependency.

## Estimation inputs and benchmark sources

The family-history layer needs both panel data and benefit-facing
targets.

### Main empirical inputs

The main sources should include:

- **PSID** for marriage, divorce, remarriage, labor-force history, and
  pair-level earnings relationships
- **HRS** for widowhood, older-adult family status, retirement
  transitions, and benefit-facing older-cohort family histories
- **CPS and ACS** for current marital-status distributions, household
  structure, age gaps, educational pairing, and dual-earner patterns
- **SSA published statistics** for spouse, widow(er), and other
  auxiliary-beneficiary counts and average benefits
- **MINT and DYNASIM documentation** as public benchmarks for what a
  serious but still simplified family-history layer should contain

### Where public data are weak

The project should also be transparent about weak spots:

- exact histories for ex-spouses not present in the household
- rare auxiliary-benefit categories
- exact timing of sequential marriages and claims for every record
- parent-benefit pathways
- some child-in-care pathways outside the main older-adult use cases

These are reasons for scoped promises, not reasons to leave the family
layer vague.

## Evaluation metrics for family structure

The family-history layer should be judged on both demographic realism
and benefit relevance.

### Core demographic and relationship metrics

| Metric | Why it matters |
|---|---|
| Married, divorced, widowed, never-married shares by age and sex | Basic family-status realism |
| Share ever married by cohort | Tests life-course history, not only current status |
| Remarriage prevalence by age and sex | Important for divorced spouse and widow pathways |
| Widowhood prevalence by age and sex | Directly linked to survivor exposure |
| Mean spouse age gap | Couple realism |
| Educational homogamy within couples | Pairing realism |
| Dual-earner share among couples | Important for spouse-benefit exposure |
| Spousal earnings correlation or rank correlation | Central for adequacy and top-up patterns |

### Core auxiliary-benefit metrics

| Metric | Why it matters |
|---|---|
| Spouse beneficiaries by age and sex | Direct test of auxiliary exposure |
| Widow(er) beneficiaries by age and sex | Direct test of survivor pathways |
| Divorced spouse beneficiary counts where available | Tests duration and remarriage logic |
| Share of beneficiaries who are own-worker only versus dually entitled | Important for interpreting reforms |
| Average spouse and widow(er) benefit | Links history quality to actual benefit levels |
| Family-maximum incidence where in scope | Tests dependent-benefit logic |

### Distributional stress tests

If the project wants a stronger adequacy story, it should also examine:

- benefit-source decomposition for low-own-earnings women
- poverty and near-poverty among widow(er) households
- benefit outcomes for divorced versus never-married older adults
- reform incidence for households with uneven lifetime earnings

This is where the family-history layer becomes directly legible to
funders and policy partners.

## Suggested stage-1 thresholds

The proposal should not dodge stage-1 numeric expectations for this
layer either.

| Metric | Draft stage-1 threshold |
|---|---|
| Major marital-status shares by age-sex cells | within 2 percentage points |
| Widowhood and divorce prevalence at older ages | within 2 percentage points |
| Mean spouse age gap | within 0.5 years |
| Educational homogamy shares | within 2 percentage points |
| Spousal earnings rank correlation | within 0.05 |
| Spouse and widow(er) beneficiary counts | within 2-3 percent |
| Average spouse and widow(er) benefit | within 2-3 percent |
| Share dually entitled among older beneficiaries | within 3 percentage points |

The point is not that these exact cutoffs are sacred. The point is that
the proposal should make the family-history layer falsifiable.

## Recommended operational sequence

The build sequence should reflect both policy value and implementation
difficulty.

1. **Older-adult marriage histories**
   enough to distinguish never married, divorced, widowed, remarried,
   and currently married records in benefit-relevant ways
2. **Current spouse links and dual-entitlement logic**
   enough to support spouse top-ups and own-worker versus auxiliary
   decomposition
3. **Survivor and divorced spouse pathways**
   using widowhood timing, marriage duration, and remarriage history
4. **Dependent and child-in-care pathways**
   where they materially improve coverage of auxiliary categories
5. **Rare and complex edge cases**
   including parent benefits and more exact sequential claim
   interactions

This is realistic, fundable, and much easier to validate than promising
the whole family-benefit codebook at once.

## How this should be positioned in the proposal

The proposal should make three points clearly.

### 1. Family structure is a core state block, not downstream polish

Auxiliary benefits are not an optional enhancement to add after the
earnings model is "done." They are part of the minimum Social Security
story.

### 2. A narrower phase 1 can still be serious

The project can credibly promise:

- marriage histories good enough for spouse, survivor, and divorced
  spouse analysis
- couple matching good enough to preserve earnings asymmetry
- explicit dual-entitlement treatment
- bounded simplifications on rare auxiliary categories

That is already a strong and fundable claim.

### 3. This layer should have its own stage gate

If the model cannot produce directionally credible widow, spouse, and
dual-entitlement outcomes, the project should narrow its public policy
claims accordingly. Family structure should not be treated as something
that can silently underperform while the proposal still advertises full
adequacy analysis.

## Bottom line

The proposal should not describe family structure as a few marriage
hazards plus a spouse-benefit rule call.

It should describe an explicit relationship-history layer inside
longitudinal `microplex`, say what phase 1 will and will not include,
benchmark those choices against DYNASIM and MINT, and evaluate the
result against the auxiliary-benefit outcomes that policy users actually
care about.
