# Benchmark model component matrix

## Why this chapter exists

The [existing-models.md](existing-models.md) chapter is strategic. It
answers the question "where does this project fit in the landscape?"

This chapter is more operational. It answers a narrower question:

- how do the main benchmark models appear to construct the pieces we
  care about?
- which parts are documented publicly, and which are not?
- what should this proposal copy, avoid, or defer?

That distinction matters because the grant case is not just "we think a
public model would be useful." It is "we understand the current
benchmarks well enough to know what work the funding would actually pay
for."

## Scope and interpretation

This matrix is based on the **public record**, not on private access or
inference about internal codebases.

That means:

- if a capability is well documented publicly, the matrix says so
- if a capability likely exists but the public record is thin, the
  matrix says the public record is thin
- if a model is adjacent rather than directly comparable, the matrix
  says that too

This chapter covers five comparison objects:

1. **Our plan**
   longitudinal `populace` plus PolicyEngine-US plus this Social
   Security application layer
2. **DYNASIM**
   the main non-governmental dynamic benchmark
3. **MINT**
   the main administrative-data benchmark for Social Security analysis
4. **CBO / CBOLT**
   the official long-term fiscal benchmark in public discussion
5. **Morningstar**
   an adjacent retirement-adequacy benchmark that matters for wealth,
   retirement, and LTSS but is not a direct Social Security scoring
   model

Morningstar is included because Gopi's benchmark set included it and
because its recent public work is one of the clearest examples of a
modern retirement-outcomes model being used for LTSS policy analysis
[@morningstar2024modelpage; @look2025ltss; @look2025wish].

## High-level orientation

| Model | Primary objective | Public access | Main strength | Main limitation for our purposes |
|---|---|---|---|---|
| **Our plan** | Build a public longitudinal population asset and a transparent Social Security application layer | Intended to be open-source and publicly documented | Transparency, inspectability, integration with PolicyEngine | Must prove that public synthetic construction is good enough |
| **DYNASIM** | Broad retirement-income and aging microsimulation | Documentation and outputs are public; code is not | Breadth, maturity, family and LTSS scope [@favreault2015; @urban2024dynasim4] | Not independently reproducible |
| **MINT** | Social Security, SSI, and retirement-income analysis with administrative-data credibility | Public methodology; restricted data and code | Administrative earnings credibility [@smith2010mint; @smith2021mint8; @ssa2024mint] | Not a public model and simplified in some behavioral margins |
| **CBO / CBOLT** | Official long-term budget and Social Security outlook | Public reports, not public production microdata/code | Institutional authority and macro-fiscal integration [@cbo2004; @cbo2018; @cbo2024finances; @cbo2024longterm] | Public record is relatively thin on record-level construction details |
| **Morningstar** | Retirement-income adequacy and retirement-product or policy analysis | Public papers and technical notes; proprietary model | Household retirement adequacy, assets, and recent LTSS work [@look2024retirementoutcomes; @morningstar2024modelpage; @look2025ltss; @look2025wish] | Not a direct public Social Security microsimulation benchmark |

## Matrix 1: population and state construction

| Component | Our plan | DYNASIM | MINT | CBO / CBOLT | Morningstar |
|---|---|---|---|---|---|
| **Base population** | Public synthetic `populace`, PolicyEngine's ML-first microdata layer, extended longitudinally | SIPP-based starting sample with publicly documented 0.04% core and 0.4% expanded variants [@favreault2015; @urban2024dynasim4] | SIPP plus administrative earnings and program records for strong near-retirement credibility [@smith2010mint; @smith2021mint8; @ssa2024mint] | SSA Continuous Work History Sample foundation, with a 1-in-1,000 representative microsimulation sample and SIPP/CPS imputations for missing demographics and family structure [@cbo2018; @cbo2019replacementrates] | Household-oriented retirement simulation using current resources, projected longevity, healthcare, and retirement assets; public materials emphasize model outputs more than raw starting-file mechanics [@look2024retirementoutcomes; @morningstar2024modelpage] |
| **Historical earnings** | Synthetic reconstruction from panel data, calibrated against `populace`'s registry of administrative targets; benchmarked across candidate model families | Public record shows lifetime economic histories built from survey and linked administrative inputs, with annual updating and alignment [@favreault2015; @urban2024dynasim4] | Strongest public benchmark because administrative earnings are built in for many cohorts [@smith2010mint; @ssa2024mint] | Public record discusses lifetime earnings assumptions and fiscal outputs, but is much less explicit on the micro history-construction machinery [@cbo2004; @cbo2018; @cbo2024finances] | Public papers say the model estimates historical wages for each household member and then simulates accumulation and retirement adequacy; claim age is simplified in the inaugural analysis [@look2024retirementoutcomes] |
| **Family structure** | Explicit relationship-history layer with spouse links, widowhood, divorce duration, remarriage, and benefit-facing auxiliary states | Public documentation shows marriage, divorce, family structure, and spouse-related states are part of the annual simulation [@favreault2015; @urban2024dynasim4] | Public methodology supports spouse and survivor benefit analysis, but the public record is less explicit than DYNASIM on relationship-history mechanics [@smith2010mint; @ssa2024mint] | Public record is relatively thin on family-history construction at the record level [@cbo2004; @cbo2018] | Public outputs are household-based and broken out by family status, but the public record does not suggest a fully general spouse-former-spouse-child network like the one needed for detailed auxiliary-benefit analysis [@morningstar2024modelpage; @look2024retirementoutcomes] |
| **Disability and health** | Separate impairment, program-pathway, and claiming states, plus mortality and family interactions | Publicly documented health, disability, cognition, and work-limitation modules with yearly transitions [@favreault2015; @urban2024dynasim4] | Includes disability pathways but with publicly documented simplifications around adjudication and return-to-work rules [@ssa2024mint] | Public record is strong on aggregate Social Security finances and disability spending, weaker on record-level disability-state machinery [@cbo2024finances; @cbo2024longterm] | Public papers explicitly include healthcare costs, projected longevity, and LTSS states such as home healthcare and nursing home need, but not a public SSDI-style program pathway [@look2024retirementoutcomes; @look2025ltss] |
| **Wealth, assets, and LTSS** | Not phase-1 core, but preserved as a later extension track through longitudinal `populace` | Major documented strength: wealth, pensions, health spending, LTSS use, payer assignment, and Medicaid interaction [@favreault2015; @urban2024dynasim4; @favreault2020ltss] | Stronger than a Social Security-only model on pensions and SSI interactions, but not positioned publicly as a leading LTSS model [@smith2010mint; @ssa2024mint] | Public emphasis is fiscal outlook rather than household adequacy, wealth depletion, or LTSS risk pathways [@cbo2024finances; @cbo2024longterm] | Major strength: retirement assets, expenses, projected inadequacy, and recent LTSS and WISH analyses using the same model family [@look2024retirementoutcomes; @look2025ltss; @look2025wish] |

## Matrix 2: benefit logic, behavior, and policy use

| Component | Our plan | DYNASIM | MINT | CBO / CBOLT | Morningstar |
|---|---|---|---|---|---|
| **Social Security rule engine** | PolicyEngine-US provides an open rule engine, with this repository adding longitudinal inputs and validation | Public documentation says Social Security and SSI are strongly rule-based components [@favreault2015; @urban2024dynasim4] | Public documentation indicates current-law Social Security, SSI, and pension logic with SSA mission alignment [@smith2010mint; @ssa2024mint] | Public reports provide policy and fiscal results, but not a fully inspectable public rule engine comparable to PolicyEngine-US [@cbo2018; @cbo2024finances] | Public materials indicate Social Security benefits are estimated for each household member, but not a public, inspectable Social Security rules stack [@look2024retirementoutcomes] |
| **Claiming behavior** | Explicit claim-age buckets and reduced-form claiming hazards are in scope | Public record indicates a serious retirement-and-program timing model, but not a highly transparent public description of every claiming equation [@favreault2015; @urban2024dynasim4] | Public methodology explicitly notes simplification to a single claiming age and omission of sophisticated claiming strategies [@ssa2024mint] | Public CBOLT overview describes annual eligibility checks and claiming probabilities with spikes at age 62, Medicare eligibility age, FRA, and age 70 [@cbo2018] | In the inaugural Morningstar analysis, claim age is assumed to equal retirement age, which is analytically tractable but much simpler than the behavior we need for Social Security microsimulation [@look2024retirementoutcomes] |
| **Auxiliary benefits** | Spouse, survivor, divorced spouse, and dual-entitlement logic are treated as a dedicated family-history build problem | Public scope clearly includes family structure and Social Security benefit logic, making spouse and survivor analysis part of the model's natural domain [@favreault2015; @urban2024dynasim4] | Publicly useful for spouse and survivor outputs, but less explicit on the underlying family-history construction in current public docs [@smith2010mint; @ssa2024mint] | Publicly visible outputs matter here, but the construction details are not comparably open [@cbo2024finances] | Public model outputs are household-based, but the public record is not strong enough to treat Morningstar as an auxiliary-benefit benchmark in the MINT or DYNASIM sense [@look2024retirementoutcomes; @morningstar2024modelpage] |
| **Behavioral responses** | Reduced-form claiming, work, and transition responses; no phase-1 promise of full structural optimization | Public model supports retirement-income and policy analysis with rich state transitions and cross-domain behavior [@favreault2015; @urban2024dynasim4] | Stronger on administrative earnings credibility than on open behavioral detail; public docs disclose simplifications where relevant [@ssa2024mint] | Strongest on fiscal integration and official assumptions, not on public behavioral transparency [@cbo2018; @cbo2024longterm] | Strong on retirement timing, contribution behavior, asset accumulation and drawdown, and adequacy outcomes; weaker fit for detailed Social Security claimant-pathway modeling [@look2024retirementoutcomes; @morningstar2024modelpage] |
| **Policy use cases** | Transparent Social Security reform analysis first, with extension path to SSI, adequacy, and LTC | Broad retirement, Social Security, SSI, wealth, caregiving, and LTSS policy use [@favreault2015; @favreault2020ltss; @johnson2023caregiving] | Social Security, SSI, pensions, retirement-income distribution, and beneficiary analysis [@smith2010mint; @ssa2024mint] | Official long-term Social Security finances and budget effects [@cbo2024finances; @cbo2024longterm] | Retirement adequacy, plan design, savings policy, and now LTSS adequacy and WISH-style backstop analysis [@look2024retirementoutcomes; @look2025ltss; @look2025wish] |

## Matrix 3: alignment, validation, and transparency

| Component | Our plan | DYNASIM | MINT | CBO / CBOLT | Morningstar |
|---|---|---|---|---|---|
| **External alignment** | Explicit stage-gated calibration and validation against public targets | Publicly documented alignment to SSA OACT targets for major demographic and labor outcomes [@urban2024dynasim4] | Administrative-data benchmarking is built into the model's institutional setting [@ssa2024mint] | Deeply integrated with CBO's official long-term outlook assumptions [@cbo2018; @cbo2024longterm] | Public materials describe stochastic scenario modeling and household outcome projections, but not a public OACT-style alignment regime for Social Security outputs [@look2024retirementoutcomes; @morningstar2024modelpage] |
| **Validation emphasis** | Public validation artifacts are a core deliverable, not a footnote | Long publication trail and broad institutional credibility | Administrative-data credibility and SSA use case are the central strengths | Official scorekeeping credibility at the aggregate level | Public reports and technical appendices are useful, but validation is oriented toward adequacy findings rather than open benchmarking against SSA-style intermediate states |
| **Transparency** | Intended full transparency on methods, code, and validation | Documentation public; full codebase not public | Methodology public; data and code restricted | Reports public; production internals not public | Research papers and technical notes public; full model remains proprietary |
| **What we should learn** | N/A | Match the seriousness about state richness and alignment | Treat lifecycle validation as the hardest problem | Do not overclaim official-scoring parity | Remember that retirement adequacy and LTSS policy can be highly decision-relevant even when the Social Security micro-paths are simplified |

## What this means for each workstream

The matrix is only useful if it changes how we describe the build.

### 1. Earnings and history construction

The lesson is not that there is one canonical benchmark to copy.

- **MINT** is the benchmark for why historical earnings matter so much.
- **DYNASIM** is the benchmark for integrating those histories into a
  broader lifecycle model.
- **CBO** reminds us that long-run Social Security analysis lives in a
  larger fiscal conversation.
- **Morningstar** shows that modern retirement models can be influential
  even when they are organized around household adequacy rather than a
  classic Social Security microsimulation stack.

So the funding case should say: the hard work is building credible
public histories and validating them, not merely plugging current
earnings into a benefit formula.

### 2. Family, disability, and claiming

This is where the public comparison becomes especially clarifying.

- **DYNASIM** is the strongest public benchmark for saying these states
  need to be first-class objects.
- **MINT** is useful because it shows that a serious model can still
  disclose simplifications rather than pretending to do everything.
- **Morningstar** is useful because it shows that some very influential
  retirement analysis can proceed with much simpler Social Security
  timing assumptions.

That combination should make the proposal more disciplined, not less:
we should fund a real family-disability-claiming layer, but we should
also be explicit about what phase 1 simplifies.

### 3. Wealth, adequacy, and LTSS

This is the clearest place where the benchmark set splits.

- **DYNASIM** is far ahead on mature LTSS and payer-path modeling.
- **Morningstar** is increasingly relevant on retirement-adequacy and
  LTSS-risk analysis.
- **Our plan** should not claim near-term parity here.

Instead, the proposal should say that Social Security is the first
application and that the architecture preserves a path toward adequacy
and LTSS work once the longitudinal population is credible.

### 4. Transparency as a real differentiator

The matrix also clarifies what "open" should and should not mean.

It should not mean:

- we are automatically more credible than DYNASIM, MINT, or CBO
- public data are just as good as administrative records
- openness substitutes for validation

It should mean:

- methods, assumptions, and failures can be inspected publicly
- benchmark comparisons can be reproduced
- outside users can build on the same population platform

That is a real differentiator. It just is not the same thing as saying
we already have the best model.

## Why this supports the funding ask

This comparison helps make the actual work legible.

The funding is not for a thin wrapper around existing Social Security
rules. It is for:

- longitudinal population construction
- history reconstruction
- family and disability state construction
- validation against multiple benchmark traditions
- integration into an open, inspectable application layer

The benchmarks show that these are substantial model-building tasks, not
mere documentation tasks.

## Why this is a good candidate for interactivity

This is one of the few parts of the proposal where interactivity would
be genuinely useful rather than decorative.

An interactive companion could let readers switch between:

- model
- component
- level of public evidence
- implications for our build

That would make diligence faster for funders and collaborators.

But the written matrix should come first. Interactivity should compress
understanding of a settled comparison structure, not substitute for
doing the comparison.

## Bottom Line

The benchmark landscape is not just "DYNASIM is good, MINT is
administrative, CBO is official."

At the component level:

- **DYNASIM** is the main benchmark for breadth and state richness
- **MINT** is the main benchmark for the importance of earnings-history
  credibility
- **CBO** is the main benchmark for fiscal authority and macro
  integration
- **Morningstar** is the main adjacent benchmark for modern retirement
  adequacy and publicly visible LTSS analysis

That is the right context for this proposal. The project is strongest if
it presents itself as the attempt to build the most transparent public
stack in this space, while being explicit that credibility has to be
earned component by component.
