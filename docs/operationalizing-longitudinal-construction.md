# Operationalizing Longitudinal Construction

## Why this chapter exists

The hardest question in this project is not whether PolicyEngine-US can
compute Social Security benefits once the right variables exist. The
hard question is whether we can construct a public, person-level panel
with plausible lifetime earnings, family histories, disability spells,
and claiming-relevant states. That is the part that determines whether
longitudinal `microplex` is merely an interesting synthetic dataset or a
serious policy-analysis asset.

Throughout this chapter, `microplex` is PolicyEngine's ML-first
microdata layer, and `Ledger` is PolicyEngine's harness over dozens of
U.S. government survey and administrative datasets — the source
microdata and the calibration targets (CBO, IRS, SSA, Census, and
others) that `microplex` synthesizes from and calibrates against.

This chapter therefore goes one layer deeper than the methodology and
technical-specification chapters. It describes what the funded build
would actually do, in what order, and how we would know whether the
earnings machinery is good enough to support 1:1 record-level benefit
analysis.

The comparison points in this chapter come from public descriptions of
DYNASIM, MINT, and CBOLT. DYNASIM provides the clearest public account
of a rich dynamic microsimulation pipeline, including statistical
matching, yearly transition equations, and alignment procedures
[@favreault2004dynasim3; @favreault2015; @urban2024dynasim4; @smith2001historicalearnings].
MINT provides the benchmark for a Social Security-focused model with
matched administrative earnings histories and explicit program-rule
coverage [@smith2010mint; @butrica2006; @smith2021mint8; @ssa2024mint].
CBO's public materials provide a more limited view, but they still show
what official long-term analysis optimizes for: macro-fiscal coherence,
cohort and quintile projections, and official baseline authority
[@cbo2004; @cbo2018; @cbo2024longterm; @cbo2024finances].

## The specific standard we need to meet

For this project to justify a serious build, it needs to do more
than generate reasonable average earnings by age. It needs to support
the following chain end to end:

1. represent a public cross-sectional population in `microplex`
2. attach plausible lifetime earnings and family histories to each
   record
3. transform those histories into quarters of coverage, AIME, and PIA
4. apply current-law and reform rules through PolicyEngine-US
5. reproduce external benchmarks closely enough that outside reviewers
   believe the model is informative rather than decorative

That implies four distinct standards:

- **Record plausibility**: individual histories should look like
  possible careers, not pointwise draws from unrelated age-specific
  distributions
- **Distributional fit**: aggregates, percentiles, and transition
  matrices should match external targets
- **Program relevance**: errors should be small on variables that drive
  insured status, AIME, claiming, and beneficiary type
- **Transparency**: the construction process has to be inspectable and
  rerunnable with public ingredients

## How comparable models handle the problem

The public record suggests the following division of labor across the
main benchmark models:

| Component | DYNASIM public record | MINT public record | CBO public record | Implication for us |
|---|---|---|---|---|
| **Starting sample** | Starts from survey-based representative samples and augments them with multiple public and administrative data sources | Starts from SIPP matched to SSA administrative earnings and benefits | Uses CBOLT as the long-term baseline framework for fiscal and distributional analysis | `microplex`, synthesized from `Ledger`'s survey sources, can play the starting-sample role, but it does not inherit observed earnings histories |
| **Historical earnings** | Older DYNASIM work used statistical matching to attach historical earnings built from PSID and CPS/SER-style sources | Uses observed administrative earnings where available and projects the remainder | Public documentation is sparse on exact record construction | This is the central gap our project must close with public methods |
| **Annual labor-market process** | Relies on yearly transition equations, hazard-style modules, and Monte Carlo simulation | Projects labor force participation and earnings from an admin-linked base | Public emphasis is on cohort/quintile outputs and aggregate consistency | Our design should be annual and state-based, not only age-point imputation |
| **Alignment and calibration** | Explicitly aligns modules to observed history and future control totals | Uses Trustees assumptions and current-law rules for projections | Integrated to official long-term projections and budget baselines | We need explicit alignment layers, not a one-shot imputation |
| **Benefit rules** | Rich OASDI/SSI rule logic, with some other programs handled statistically | Includes most core current-law Social Security rules but omits some monthly and complex cases | Official distributional analyses, but less public rule-level detail | PolicyEngine-US is a real strength if the input panel is credible |
| **Claiming detail** | Public materials indicate retirement and benefit-take-up modules | MINT uses a single claiming age and omits sophisticated claiming strategies | Some public CBO analyses use stylized claiming assumptions for presentation | We should be explicit about what level of claiming realism phase 1 can support |
| **Transparency** | Methodology public, code closed | Methodology and outputs public, code/data restricted | Public summaries, little full-pipeline transparency | Public reproducibility remains the main differentiator |

This is also where a future interactive component explorer would make
sense. The comparison problem is inherently matrix-shaped: each major
component has a distinct "our plan versus DYNASIM versus MINT versus
CBO" story. But the first deliverable should still be written and
citable.

## Fundamental methodology review: what CBOLT and DYNASIM imply

The deeper benchmark review changes the shape of the build. The closest
public descriptions of CBOLT and DYNASIM do not describe a single
earnings imputation followed by a benefit calculator. They describe
annual microsimulation systems with:

- a representative starting population
- person-year state records
- annual demographic and economic transitions
- family links used for auxiliary benefit logic
- explicit alignment to external population, labor, earnings, and
  program controls
- benefit calculations derived from the simulated histories

That implies the project should be organized around a controlled annual
state engine. Machine learning should help estimate latent states,
transition probabilities, and conditional distributions, but the model
needs a conventional microsimulation backbone that makes the timing,
state changes, and alignment choices inspectable.

### CBOLT's useful lesson

CBOLT is valuable because CBO explains the architecture even though it
does not disclose the full production implementation. CBOLT has four
interacting components: a demographic model, a microsimulation model, a
long-term budget model, and a policy growth model [@cbo2018].
Its microsimulation starts from SSA's Continuous Work History Sample,
then imputes missing demographic and family characteristics from SIPP
and CPS. The microsimulation keeps annual records, uses past
characteristics to project current transitions, preserves links among
current spouses, former spouses, deceased spouses, parents, and
children, and uses those links for family-based Social Security
benefits [@cbo2018; @cbo2019replacementrates].

The sample scaling is important to state precisely. CBO describes a
sample in which each simulated person represents 1,000 people, but that
is a stable representation factor for the microsimulation sample, not a
license to recalibrate spouses, former spouses, parents, and children
independently year by year [@cbo2018].

The key operational lesson is not that this project can replicate
CBOLT's administrative-data advantage. It cannot. The lesson is that a
serious Social Security model must make the same categories of objects
explicit:

- annual demographic transitions
- annual labor-market transitions
- earnings above and below the taxable maximum
- spouse and former-spouse links
- claiming eligibility and claiming timing
- scheduled-benefit and payable-benefit scenarios
- aggregate fiscal outputs and cohort-distributional outputs

CBO's public overview also describes an important control-total
mechanism: for demographic transitions, the model estimates individual
probabilities, combines them with random numbers, ranks people within a
group, and selects the number of transitions implied by the aggregate
demographic model [@cbo2018]. That is a practical pattern this
project should copy in public form. It lets the model preserve
individual heterogeneity while still hitting aggregate controls.

### DYNASIM's useful lesson

DYNASIM is the better public benchmark for state richness. Public
DYNASIM4 materials describe a 2006 starting sample built from SIPP,
NLSY, PSID, Summary Earnings Records, and ACS; a 2007-2100 simulation
horizon; a basic 0.04 percent population sample with a 0.4 percent
expanded version; and annual simulation of demographics, employment,
income, wealth, health, disability, medical spending, and LTSS
[@urban2024dynasim4]. The 2015 overview adds the deeper
methodological point: DYNASIM projects annual work as a function of
demographics, family state, health, disability, spouse characteristics,
cohort, unemployment, and individual-specific error terms, then aligns
employment and earnings to Trustees targets [@favreault2015].

For this project, the implication is that the annual earnings engine
needs at least three layers:

1. a persistent person effect or latent rank
2. annual extensive-margin work and coverage states
3. conditional earnings, hours, and high-earner treatment

DYNASIM also shows why Social Security cannot be isolated too narrowly.
Even if the phase-1 product is Social Security, the state vector should
leave room for health, disability, pension, wealth, and LTSS extensions.
Those extensions are not garnish. They affect retirement timing,
claiming, SSI, Medicaid, and later adequacy analysis.

### What this means for our build

The minimum viable build is not "attach 35 years of earnings and call
PolicyEngine-US." It is a staged implementation of the following
systems:

| System | What it must do | Benchmark lesson |
|---|---|---|
| Starting-file adapter | Convert cross-sectional `microplex` into a person-year scaffold with stable IDs, household IDs, tax-unit IDs, and fixed representation factors or replicate counts | CBOLT and DYNASIM both begin with a representative population, not abstract cohorts |
| Historical earnings engine | Reconstruct covered earnings, uncovered earnings, self-employment, zero-earnings years, and taxable-maximum exposure | CBOLT's strength comes from CWHS; our public substitute must be validated hard |
| Annual transition engine | Simulate work, earnings, marital status, fertility, disability, mortality, claiming, and household changes year by year | Both benchmark systems are annual transition models |
| Family network engine | Maintain current, former, and deceased spouse links plus parent-child links needed for auxiliary benefits | CBOLT explicitly uses family links for benefits |
| Claiming engine | Check eligibility annually, assign claiming probabilities, and encode age spikes at 62, 65, FRA, and 70 | CBOLT documents this structure for retired-worker claiming |
| Alignment engine | Hit annual population, employment, earnings, beneficiary, revenue, and outlay controls without hiding model failure | DYNASIM aligns to Trustees targets; CBOLT uses aggregate controls |
| Validation ledger | Record which fields are observed, donated, imputed, projected, or alignment-adjusted, then publish component-level errors | Openness only matters if users can audit where model output came from |
| Scenario engine | Run current-law, scheduled-benefit, payable-benefit, and reform scenarios with comparable output tables | CBO separates scheduled and payable benefit concepts in public reporting |
| Output layer | Produce person-year, relationship, benefit, and aggregate tables, plus reproducible validation reports | The output needs to support both distributional and fiscal analysis |

The most important methodological risk is that we build a plausible
synthetic panel that fails at Social Security-specific margins. The
stage gates should therefore test the intermediate variables that drive
benefits, not only final benefit totals:

- years with zero covered earnings
- quarters of coverage
- AIME by cohort and lifetime earnings rank
- taxable-maximum exposure
- covered versus noncovered earnings
- claiming age distribution
- spouse and survivor eligibility
- disabled-worker conversion paths
- benefit type shares
- payable-versus-scheduled benefit scenarios

That is also the strongest case for the funding ask. The difficult work
is the controlled construction and validation of these intermediate
states, not the existence of the final benefit formula.

## Recommended architecture

The proposal should stop describing the earnings process as if one
standalone model family, such as age-specific QRF, is the architecture.
QRF can still be a tool inside the build. But the architecture should
be a factorized annual state process with explicit path constraints,
calibration, and benefit-facing outputs.

The expected production direction is:

1. build a person-year scaffold for each synthetic individual
2. estimate a latent earnings-capacity distribution for each person
3. model covered work, noncovered work, and self-employment as separate
   annual states
4. model earnings conditional on work using a distributional method
5. draw serially dependent shocks and residuals from panel-based
   empirical distributions
6. calibrate the base population and align process outputs to external
   targets
7. validate first at the earnings-process level and then at the Social
   Security-output level

That structure is closer to how serious microsimulation systems work in
practice [@favreault2004dynasim3; @favreault2015; @smith2010mint; @smith2021mint8]
while still leaving room for more ambitious joint generative methods
later.

The strongest version of that "more ambitious" path is probably a
zero-inflated all-at-once `microplex` trajectory model. But the
proposal should leave room for refreshed evaluation results to decide
whether the best implementation is ZI-QDNN, a flow-based model, or
another pathwise candidate.

## The annual data model we actually need

The build should treat annual person-year records as the canonical
intermediate object. For Social Security analysis, the key unit is not
simply the base-year person. It is the sequence:

```text
(person_id, calendar_year, age, work_state, covered_earnings, uncapped_earnings,
 self_employment_earnings, noncovered_indicator, disability_state,
 marital_state, children_present, claim_state, source_flag)
```

At minimum, the annual state vector should carry:

- `covered_work_indicator`
- `covered_earnings`
- `uncapped_labor_earnings`
- `self_employment_earnings`
- `noncovered_work_indicator`
- `quarters_of_coverage`
- `marital_state`
- `spouse_link` when applicable
- `child_under_18_indicator`
- `disability_state`
- `retired_or_claimed_indicator`
- `observed_vs_imputed_vs_projected_flag`

That `source_flag` matters. One of the recurring problems in
microsimulation review is that "model output" becomes a blur. We should
be able to say, for any field and year, whether it is:

- observed in the base cross section
- directly borrowed from a donor history
- estimated from a transition model
- adjusted by calibration or alignment

That level of provenance is part of the product.

## Decomposing the earnings problem

The annual earnings process should be modeled as a composition of
several pieces rather than a single black-box prediction.

Let `y_it` denote covered earnings for person `i` in year `t`. A useful
baseline decomposition is:

```text
covered_earnings_it = 1[covered_work_it = 1] *
                      min(exp(mu_it + eps_it), taxable_max_t)
```

where

```text
mu_it = f(age_it, cohort_i, education_i, sex_i, race_i,
          state_it, marital_state_it, children_it,
          disability_state_it, latent_rank_i, macro_t)
```

and `eps_it` is a serially dependent residual draw.

This immediately implies four separate modeling problems:

1. Who works in covered employment in a given year?
2. Conditional on working, what is their uncapped earnings potential?
3. How much of that labor income falls into covered versus noncovered
   buckets?
4. How persistent and how non-Gaussian are the residual shocks?

That decomposition is not just cleaner econometrically. It is more
useful for Social Security. The program is sensitive to:

- years with zero covered earnings
- whether earnings exceed the taxable maximum
- the timing of work interruptions
- the difference between covered and noncovered employment
- the long-run relation between current earnings and lifetime rank

Those are exactly the margins that get obscured if we only predict
earnings at age 30, 35, 40, and so on.

## Step 1: Estimate a latent earnings-capacity distribution

The project should not treat current earnings as a sufficient statistic
for lifetime earnings rank. The empirical literature is clear that
current earnings become a better proxy only around midcareer, and much
worse outside that window [@haider2006].

So the first operational task is to define and estimate a latent
earnings-capacity measure that can be attached to each `microplex`
person.

### Practical definition

A workable phase-1 target is:

- a percentile rank of wage-indexed average covered earnings over a
  stable midcareer window, such as ages 45-55 when available
- or, for younger workers, a posterior over that rank conditional on
  currently observed cross-sectional traits

The target should be estimated in PSID and similar panel sources, then
mapped back to `microplex` using supervised prediction. This is the
right use of tools like QRF or distributional regression: not as the
whole earnings engine, but as a way to recover a latent position in the
lifetime distribution.

### Why this layer matters

This layer gives the annual process a stable backbone:

- it explains why some people tend to remain high earners even after
  temporary setbacks
- it limits implausible path switching across the lifetime distribution
- it lets us calibrate by latent type, not only by observed current
  earnings
- it provides a natural bridge to later wealth, pension, and LTC
  extensions

### How comparable models differ

- **MINT** does not need to infer this nearly as aggressively for older
  cohorts because much of the earnings history is observed in
  administrative records [@smith2010mint; @smith2021mint8; @ssa2024mint]
- **DYNASIM** historically leaned on matched earnings histories and
  yearly processes rather than a purely cross-sectional latent-rank
  reconstruction [@favreault2004dynasim3; @smith2001historicalearnings]
- **CBO** public materials say much less about the record-level latent
  structure, which itself is informative: official users are not being
  invited to reproduce the pipeline [@cbo2004; @cbo2018]

## Step 2: Model the extensive margin explicitly

The next layer is the work-state process. Social Security benefits are
deeply sensitive to whether a year is counted as covered work, not just
to how large earnings are when positive.

The phase-1 annual state should distinguish at least:

1. no earnings
2. covered wage and salary work
3. self-employment with covered earnings
4. noncovered work
5. retired or otherwise out of the covered workforce

In practice, the estimation problem can be framed as a discrete-time
transition model:

- previous work state
- spell duration in current state
- age
- education
- sex
- race and ethnicity
- marital and child status
- disability state
- state of residence or broad labor market region
- macro year effects
- latent earnings rank

Hazard-style or multinomial transition models are the right baseline
here because they preserve state dependence and spell structure, which
are central to zero-earnings years and insured-status dynamics
[@favreault2015; @altonji2009earnings].

### Targets this layer has to hit

This module should be judged against:

- share with any covered earnings by age, sex, and education
- share at or above four quarters of coverage by age
- number of zero-covered-earnings years in the top 35 years for fully
  insured workers
- frequency and duration of work interruptions around childbirth,
  disability, and retirement

## Step 3: Model earnings conditional on work

Conditional on positive covered work, we then need a distributional
model for earnings.

The baseline should not try to predict only the conditional mean.
Benefit calculations are nonlinear in the distribution of annual
earnings because:

- earnings above the taxable maximum collapse to the cap for AIME
- low and sporadic earners are disproportionately affected by zero years
- spouse and survivor outcomes are sensitive to household earnings
  position, not just average wages

### Recommended phase-1 comparison set

Use a distributional model for conditional uncapped earnings, but do
not hard-code QRF as the default. The phase-1 comparison set should
include:

- QRF and ZI-QRF as interpretable age-point benchmarks
- ZI-QDNN as the most obvious zero-inflated neural candidate
- at least one pathwise `microplex` model that generates the earnings
  path all at once

In other words, the proposal should compare benchmark models against
the architecture we actually expect to want.

Key regressors should include:

- lagged earnings and lagged work state
- age and cohort
- education
- sex
- race and ethnicity
- occupation and industry where available
- state or region
- marital and parenting state
- disability status
- latent earnings rank
- macro year controls

The output should be:

- a full conditional distribution of uncapped earnings
- a separate probability of being above the taxable maximum
- residual bins that can be resampled with age- and rank-specific
  dependence

The likely production winner is a zero-inflated pathwise model, not a
plain age-point QRF. But that conclusion should be earned using
Social-Security-specific metrics rather than assumed from generic
imputation fidelity alone.

### Why the residual design matters

Recent administrative-data research shows that earnings shocks are not
well approximated by simple Gaussian noise. They are asymmetric,
fat-tailed, and vary over the life cycle and over the earnings
distribution [@guvenen2015earningsrisk]. Older structural panel
work likewise shows that employment shocks, job mobility, and
job-specific components have persistent effects on lifetime earnings
[@altonji2009earnings].

So the residual system should not simply be:

```text
eps_it ~ N(0, sigma^2)
```

Instead, a realistic phase-1 design is:

- estimate residual distributions by age group, latent-rank bucket, and
  previous work state
- resample residuals from empirical bins or donor pools
- impose serial dependence through lagged residual class or
  block-resampling over short windows

This is one of the places where a public model can be both serious and
transparent. The model can show not only the point estimates but the
shock library it is using.

## Step 4: Separate covered, uncovered, and taxable-max processes

Social Security modeling fails quickly if all labor earnings are treated
as interchangeable.

The first funded version should therefore distinguish:

- uncapped labor earnings
- OASDI-covered earnings
- Medicare-covered earnings
- noncovered earnings relevant for future WEP or GPO exposure
- self-employment income

This does not require a perfect public reconstruction of every pension
system in phase 1. But it does require explicit architecture.

### Practical phase-1 rule

For phase 1:

- model uncapped labor earnings first
- derive covered earnings using covered-work indicators and the taxable
  maximum
- track noncovered-work exposure as a separate annual flag
- accumulate enough noncovered history to support a later WEP/GPO
  module, even if that module is initially coarse

This is a cleaner and more defensible sequence than pretending we can
solve all pension interaction problems on day one.

## Step 5: Reconstruct the historical path, not just the current year

The key operational problem is backward construction.

For a person observed in `microplex` at age 52 in 2025, we need a path
from age 18 to age 52 that is consistent with:

- their current observed earnings or benefit state
- their demographic profile
- cohort-specific macro conditions
- lifetime rank and work-interruption patterns
- external aggregate targets

The baseline procedure should be:

1. draw or infer the person's latent earnings-rank posterior
2. construct a candidate path from age 18 forward using the annual
   work-state and earnings modules
3. score candidate paths against the base-year anchor
4. retain, resample, or retune candidate paths until the accepted set
   matches the person's observed current state and the external
   calibration targets

That is more like constrained simulation than plain imputation.

### Base-year anchoring rules

The acceptance score for a candidate path should include:

- closeness to observed current earnings
- closeness to current work state
- consistency with observed marital and child states
- consistency with observed disability or beneficiary status where
  present
- consistency with plausible quarters-of-coverage accumulation

For currently retired or disabled beneficiaries, the anchor should also
use benefit-facing information. Where the base record includes current
benefit receipt or amount, the historical path should be filtered toward
paths that imply a plausible AIME and current benefit after COLAs.

This is one of the strongest reasons not to treat the task as simple
cross-sectional imputation. The person-year path has to be judged in the
space where the policy rules operate.

### Why older cohorts are the hardest case

PSID begins in 1968. That means older cohorts' early careers are only
partially observed, or not observed at all, in the main public panel.

DYNASIM historically handled this by matching historical earnings files
to SIPP using PSID and CPS/SER-related inputs [@favreault2004dynasim3; @smith2001historicalearnings].
MINT handles it much more directly because the underlying SSA-linked
file already contains administrative earnings history for large parts of
the sample [@smith2010mint; @smith2021mint8].

We do not have either advantage in public form. So the proposal should
say this plainly:

- for older cohorts, early-career years will require stronger
  dependence on public aggregate alignment and donor-based simulation
- this is a core research problem, not a footnote
- a significant share of the project's effort goes toward buying down
  exactly this risk

## Step 6: Forward projection uses the same process, with explicit alignment

Once the historical panel is credible, forward projection should use the
same annual state machinery. But the project should distinguish clearly
between:

- the stochastic model-driven evolution of individuals
- the alignment factors used to match SSA Trustees or other control
  totals

That distinction matters because the model will otherwise look more
confident than it is.

The forward process should therefore include:

- cohort entry for new adult cohorts
- annual aging, mortality, marriage, fertility, disability, and labor
  market transitions
- macro alignment for wage growth, covered-worker shares, mortality, and
  disability incidence
- explicit storage of calibrated versus uncalibrated outputs

This mirrors the public record on DYNASIM and MINT more than a purely
static extension does [@favreault2015; @urban2024dynasim4; @ssa2024mint].

## Alignment has to operate on events, not just weights

The current proposal talks too much about weight calibration. That is
useful for building the base cross-section, but it is not sufficient
for a dynamic panel and can become actively misleading if used
carelessly after relationships and histories are simulated.

Weight calibration can help with:

- age-sex structure
- current-year earnings distributions
- beneficiary counts
- some cross-sectional household structure

But it cannot, by itself, fix:

- incoherent spouse, former-spouse, parent, or child links
- one member of a couple effectively representing a different population
  than the other member
- too much or too little path persistence
- the wrong number of zero-earnings years
- the wrong mass at the taxable maximum
- implausible shock distributions
- the wrong relationship between current earnings and lifetime AIME

So the operational plan should treat alignment as a stack:

### Layer 1: Base-population calibration

Calibrate the cross-sectional `microplex` population before the dynamic
simulation begins. After longitudinalization, carry fixed representation
factors or replicate counts through the relationship network rather than
freely changing individual weights every year.
This matches the broader dynamic microsimulation warning that weights
become a representation and household-consistency issue once events such
as union formation, divorce, migration, and household splitting are
simulated [@dekkers2012weights].

### Layer 2: Controlled event selection

Estimate individual transition probabilities, combine them with random
draws, and select events within cells so the model hits aggregate
controls for:

- deaths
- births
- marriages
- divorces
- disability incidence
- benefit claiming
- immigration and emigration

This is the CBOLT-style lesson that matters most for a public build:
individual heterogeneity determines who is selected, while aggregate
controls determine how many events occur.

### Layer 3: Process calibration

Adjust model parameters, intercepts, donor probabilities, or residual
draws so that the annual work-state and earnings process hits:

- covered-worker shares
- age-earnings profiles
- taxable-maximum incidence
- mobility matrices
- zero-earnings-year distributions

### Layer 4: Network-preserving resampling

If sparse selection is needed for performance, select coherent
households, relationship networks, or full histories. Do not
independently reweight spouses, former spouses, parents, and children
after the network exists.

### Layer 5: Policy-output validation

Check whether the resulting panel, after benefit calculation, matches:

- AIME distributions
- beneficiary counts
- average benefit levels
- replacement-rate distributions

Only after all five layers should we say the model is benefit-ready.

## Validation should be tiered and numeric

The proposal will be much stronger if it states ex ante what counts as a
passing build. A draft acceptance table could look like this:

| Validation object | Example benchmark | Phase-1 tolerance |
|---|---|---|
| Share with covered earnings by age-sex | SSA annual earnings statistics | within 2 percentage points |
| Mean and median earnings by age-sex-education | SSA and CPS distributions | within 2-3 percent |
| Share at taxable maximum | SSA public tabulations | within 1 percentage point |
| Zero-earnings years in top 35 | SSA and MINT comparison studies | within 0.5 years on average |
| Five-year earnings-quintile transitions | PSID panel estimates | within 3 percentage points per major cell |
| Cross-age earnings correlation profile | PSID | within 0.05 correlation points |
| AIME distribution for retired workers | SSA or published benchmarks | within 5 percent on key percentiles |
| Beneficiary counts by type | SSA Annual Statistical Supplement | within 1-2 percent |
| Average benefits by type | SSA | within 2-3 percent |
| Replacement rates by lifetime earnings quintile | CBO or MINT public analyses | directionally correct and within published range |

The point is not that every number above is final. The point is that the
proposal should commit to the discipline of numeric gates. The next
chapter, [evaluation-and-model-selection.md](evaluation-and-model-selection.md),
turns that idea into an explicit model-selection framework.

## What we should benchmark against, component by component

Reviewers will reasonably ask not only "does your model fit the data?"
but also "how does your design compare with the institutions that
already do this?"

The proposal should explicitly benchmark the following pieces:

### Starting file

- **Our plan**: `microplex` cross section, synthesized from `Ledger`'s
  survey sources
- **DYNASIM**: representative survey base, augmented by multiple surveys
  and matched historical earnings work
- **MINT**: SIPP linked to administrative earnings and benefits
- **CBO**: official long-term system, less public record-level detail

### Historical earnings

- **Our plan**: public reconstruction from panel data plus alignment
- **DYNASIM**: statistical matching to historical earnings inputs
- **MINT**: observed administrative earnings for much of the relevant
  history
- **CBO**: public record sparse

### Annual work and earnings transitions

- **Our plan**: factorized annual state process with latent rank,
  explicit work states, and distributional residuals
- **DYNASIM**: yearly equations and Monte Carlo simulation
- **MINT**: projected labor force participation and earnings from linked
  base records
- **CBO**: not enough public detail for full replication

### Alignment

- **Our plan**: documented multi-layer calibration
- **DYNASIM**: alignment to historical patterns and future control
  totals
- **MINT**: Trustees assumptions and current-law program rules
- **CBO**: integration with official long-term baseline

### Claiming and benefit rules

- **Our plan**: PolicyEngine-US rule layer, with phase-1 limitations
  stated explicitly
- **DYNASIM**: rich rule modules
- **MINT**: broad current-law coverage, but some annualized and
  simplified claiming logic
- **CBO**: public outputs often abstract from some microbehavioral
  detail for presentation

This could later become an interactive comparison surface, but the
written version should come first because funders and reviewers will
want something they can annotate and cite.

## The main research questions the funded build should answer

The first phase should not merely "implement the model." It should
answer a bounded set of research questions.

### 1. Which earnings architecture survives validation?

The project should compare at least three candidate families:

1. **Age-point benchmark family**:
   separate age-specific distributional models, including QRF and
   ZI-QRF
2. **Factorized annual process**:
   latent rank plus work-state plus conditional earnings plus calibrated
   residuals
3. **Joint trajectory generator**:
   a pathwise zero-inflated generator inside `microplex`, with
   ZI-QDNN, ZI-MAF, or related sequence models as candidates

The project should precommit that the simplest architecture that clears
the validation gates wins. Today, that probably means the simplest
zero-inflated pathwise model that beats the age-point benchmarks on
Social-Security-relevant targets.

### 2. How much do current-year anchors improve path quality?

Specifically:

- do candidate histories conditioned on current earnings materially
  improve AIME fit?
- do current benefit anchors materially improve older-beneficiary fit?
- how much path diversity is lost when the anchor is made stricter?

### 3. What cannot be identified from public data without alignment?

This is especially important for:

- older cohorts' early careers
- noncovered employment histories
- very high earnings near or above the taxable maximum
- complex disability and claiming interactions

The answer should be explicit, because it will shape how conservative
the public claims need to be.

### 4. Which targets belong in calibration versus validation?

A credible public model should resist the temptation to calibrate away
every weakness. The funded build should produce a principled target map:

- targets we directly align to
- targets we use only for validation
- targets we cannot support in phase 1

## What a fundable year-one work package should deliver

A fundable year-one earnings build should produce concrete artifacts,
not just a promise of future microsimulation.

At minimum, year one should deliver:

1. a harmonized PSID person-year training file with cohort, family, and
   earnings-state features
2. a benchmark note comparing the public record on DYNASIM, MINT, and
   CBO component by component
3. a longitudinal `microplex` alpha with historical earnings paths and
   source provenance flags
4. a validation report covering earnings distributions, mobility,
   taxable-maximum incidence, zero-earnings years, and AIME-sensitive
   outputs
5. a decision memo recommending the production architecture for stage 2

That is a real work package. It is also the right package to fund first,
because if it fails, the project should narrow or stop before investing
heavily in a broader public interface.

## Bottom line

The proposal should frame lifetime earnings construction as the central
technical object of the project.

The right architecture is not "QRF everywhere." It is a transparent
annual state process for covered work and earnings, anchored to current
records and disciplined by external calibration, with a likely
production path toward zero-inflated all-at-once `microplex`
trajectory models. QRF can still play an important role as a benchmark
and diagnostic tool, but it should no longer be described as the
default destination.

The public differentiator remains real:

- `microplex` as PolicyEngine's reusable public population layer
- explicit benchmark comparison to DYNASIM, MINT, and CBO
- record-level provenance
- published validation gates

That is the package that can plausibly justify serious funding.
