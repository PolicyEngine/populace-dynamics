# Policy applications and users

## Overview

Funders and collaborators need to know what this model is for before
they care how it works. The point of this project is not to create a
generic simulator in the abstract. It is to answer concrete policy
questions that require lifecycle information and distributional detail.
More specifically, this repository should be understood as the first
serious proving ground for longitudinal `populace`: if the population
platform can support Social Security well, it earns the right to support
adjacent domains later.

## Core policy questions

### Solvency packages

The model should evaluate packages that change the long-run finances of
Social Security, including:

- payroll tax rate changes
- taxable maximum reforms
- retirement age changes
- benefit formula changes
- progressive indexing variants

The output should include both fiscal outcomes and distributional
outcomes, since a package that improves actuarial balance may still
shift losses toward vulnerable groups.

### Benefit adequacy and poverty

The model should support analysis of:

- minimum benefit proposals
- widow and survivor benefit reforms
- caregiver credit proposals
- impacts on SSI-reliant households
- poverty and near-poverty among older adults and disabled workers

This is one of the clearest gaps in many public discussions of Social
Security reform: solvency is visible, but adequacy is often treated as a
secondary appendix.

### Claiming and retirement timing

Some of the most important policy questions run through claiming
behavior rather than pure benefit arithmetic. The model should be able
to explore:

- how claiming ages vary across the earnings distribution
- whether reforms shift early claiming or delayed claiming
- how disability status and weak labor-force attachment affect take-up
- how household wealth changes the option value of delaying benefits

### Family structure and auxiliary benefits

Social Security is not an individual-only program. The model should be
able to analyze:

- spousal and survivor benefits
- divorced spouse provisions
- dependent and child-in-care benefits
- family-level effects of labor-force interruptions and remarriage

These are core questions for women, widows, divorced beneficiaries, and
households with uneven work histories.

### Distributional and equity analysis

The project should support analysis by:

- lifetime earnings
- age and cohort
- race and ethnicity
- sex
- marital history
- disability history
- geography where sample quality permits

The important design principle is that these should not be afterthought
cuts added at the end of the pipeline. The model should be built so that
distributional analysis is native to the workflow.

## Early products before full dynamic completion

Not every useful output requires a complete public launch. Before the
full model is available, the project can still deliver:

- validation notes and technical papers on public lifecycle modeling
- baseline distributions for earnings, AIME, claiming, and benefits
- replications of published reform analyses
- targeted tools or briefs on especially decision-relevant questions

This matters for both funders and users: value should accrue before the
very end of the roadmap.

## A quantifiable demand proxy

One useful proxy for demand is the asymmetry between policy interest and
public modeling supply.

On the tax side, an open modeling ecosystem already exists. Several
stacks are openly callable on public data:

- **Tax-Calculator** — open-source federal income and payroll tax
  microsimulation [@taxcalc2026]
- **PolicyEngine** — open-source federal and state tax-benefit
  microsimulation with a calibrated public population, a REST API,
  and a web interface [@policyengine2026]
- **FiscalSim-US** — open-source federal and state tax-benefit
  microsimulation [@fiscalsim2026]

Behind that open layer, several organizations run proprietary
models to produce published analysis — the Tax Policy Center
[@tpcmodelfaq2025], ITEP [@itepmodel2025], the Tax Foundation
[@taxfoundationtag2025], and the Penn Wharton Budget Model
[@pwbm2025]. The point is that tax policy has both an open,
self-service modeling layer and a proprietary one.

On the Social Security side, the benchmark model families are
real but the open layer is thin:

- **DYNASIM** is the main non-governmental breadth benchmark, but
  not an open public model [@favreault2015; @urban2024dynasim4]
- **MINT** is the administrative-data benchmark, but not a public
  model [@smith2021mint8; @ssa2024mint]
- **CBO / CBOLT** provide official long-term baselines, but not a
  public self-service microsimulation environment
  [@cbo2018; @cbo2024longterm]
- **Morningstar** has a proprietary retirement-outcomes model with
  recent LTSS analysis, but not a public Social Security modeling
  tool [@look2024retirementoutcomes; @look2025ltss; @look2025wish]
- the **Cato Social Security model** is the one open analogue, but
  it is narrower in scope, lacks a calibrated public population and
  an API, and is not integrated with a tax-benefit platform
  [@catossmodel2026]

So a fair summary is that tax policy has multiple openly callable,
self-service public models, while Social Security has effectively
none in the same class. That does not prove that DYNASIM alone is
the bottleneck. It does make the broader story plausible: policy
interest is real, but the open modeling supply is unusually thin.

## Who the model should serve

### Researchers

Researchers need a replicable platform for testing assumptions,
reproducing standard analyses, and extending methods publicly.

### Advocacy and policy organizations

Smaller organizations often cannot buy access to proprietary models or
maintain specialized teams. A public model lowers that barrier.

### Journalists and public communicators

A transparent interface can make Social Security distributional effects
easier to explain and harder to obscure.

### Students and instructors

Social Security modeling is currently too inaccessible for most
classrooms. A public model changes that.

## Likely early adopters

The proposal should not treat all user groups as equally important at
the start. The earliest real demand is likely to come from groups that
already want model-backed Social Security analysis but lack affordable
and transparent tools.

### 1. Policy organizations working on reform packages

These users are likely to want:

- baseline solvency and adequacy tables
- widow and survivor reform analysis
- minimum-benefit and caregiver-credit analysis
- distributional cuts by lifetime earnings and family status

Plausible examples include organizations such as the National Academy of
Social Insurance (NASI), Brookings, and similar retirement-policy or
aging-policy groups that produce substantial Social Security analysis
but do not operate a broad public dynamic microsimulation platform of
their own. The proposal should not present these organizations as
committed partners unless that commitment is explicit.

Specific plausible early users or validators, based on fit with the
proposal's use cases, could include:

- expert users and validators with wealth, retirement, and Social
  Security methodology expertise
- **Wendell Primus** or similarly positioned Social Security policy
  experts focused on reform design and distributional consequences
- **Brookings retirement-security researchers**, especially where
  solvency, adequacy, and later LTC intersections matter
- **Bipartisan Policy Center (BPC)** for externally legible reform and
  adequacy analysis
- **Committee for a Responsible Federal Budget (CRFB)** for public-facing
  baseline and reform framing

If the architecture later supports a credible LTC extension, adjacent
interest could broaden further to Brookings or similar teams working at
the Social Security-retirement-LTSS boundary, including teams around
Gopi Shah Goda's current work. Again, these should be framed as
plausible early users or design partners, not as committed collaborators
unless that has been confirmed.

CRFB is an especially useful example of the type of demand this project
is trying to meet. It can already use PolicyEngine for narrower Social
Security-adjacent analyses such as taxation of benefits, but broader
dynamic Social Security work still pushes users toward closed benchmark
models such as DYNASIM. That is exactly the gap this project is trying
to narrow.

### 2. Researchers who care about transparent validation

These users are likely to want:

- public benchmark comparisons
- reusable synthetic longitudinal population files
- replication assets for published baseline tables
- open methods for history construction and calibration

### 3. Journalists and translators of policy debates

These users are likely to want:

- simple public interfaces
- headline distributional outputs
- transparent caveats and methodology notes

### 4. Instructors and students

These users are likely to want:

- reproducible notebooks
- smaller teaching datasets
- a usable public interface rather than access to a closed institutional
  model

## What external demand should look like in practice

From a funder's perspective, the demand story is stronger if it is tied
to concrete adoption tests rather than left as a general expectation.

The proposal should therefore aim to show:

- **design partners in stage 0**:
  a small set of external organizations or researchers willing to help
  prioritize early outputs
- **pilot analyses by stage 2**:
  a small number of externally legible baseline or reform analyses that
  real users have asked for
- **external pilot use by stage 3**:
  outside users or organizations testing the validated baseline and
  reform workflows
- **public uptake by stage 4**:
  evidence that outside users can reproduce headline examples and cite
  the model in their own work

In practice, the most credible early design-partner set would mix:

- a retirement-policy or Social Security-focused research organization
  such as NASI
- a policy research institution such as Brookings
- a public-facing reform organization such as CRFB or BPC
- an expert validator with Social Security and wealth-distribution
  expertise
- outside researchers who care about transparent validation
- at least one downstream communicator or policy-translator user
  category

Those are much stronger demand signals than abstract statements about
future openness.

## Why open source matters here

Open source is not a branding preference. It changes the structure of
the policy ecosystem:

- assumptions can be debated in public
- validation can be replicated
- outside researchers can inspect edge cases
- distributional claims are less dependent on institutional trust alone

For a program as consequential as Social Security, that shift is a
substantive contribution.

## Adjacent applications

The same dynamic infrastructure could later support:

- retirement adequacy analysis that combines Social Security with wealth
  and pensions
- SSI interaction analysis beyond purely static scoring
- long-term care and caregiving policy, where disability, wealth,
  Medicaid, and family supports interact over time

If the project later extends into long-term care, the demand-side case
may be even stronger. There is substantial think tank and policy
activity around LTSS financing, Medicare home care, caregiving, and
retirement adequacy risks:

- Brookings has recent LTSS and home-care policy analysis
  [@brookings2024homecare]
- NASI has active LTSS and caregiving work
  [@nasi2025ltssdemand; @nasi2022caregiving]
- BPC has recent LTSS finance work [@bpc2025ltss]
- Urban has one of the few serious LTSS modeling backbones through
  DYNASIM [@favreault2020ltss; @urban2024dynasim4]
- Morningstar is now publishing LTSS and WISH-style policy modeling
  using its retirement outcomes model [@look2025ltss; @look2025wish]

Accessible public modeling supply appears even thinner than in Social
Security. The fair claim is not that any one closed model is the sole
bottleneck. It is that LTC policy interest is substantial and visible,
while LTSS modeling capacity is concentrated in a small number of closed
or proprietary systems. If longitudinal `populace` becomes credible,
LTC may become one of the highest-value adjacent domains for expansion.

The most plausible first adjacent LTC product is not a national dynamic
LTSS scorekeeper. It is a state-specific pilot that proves the rules and
workflow layer can handle real cases. A Colorado-style pilot, for
example, could answer:

- is this household eligible now or likely to become eligible soon?
- what spend-down, trust, or spousal-protection pathway would be needed?
- what patient liability would remain after approval?
- how do institutional, HCBS, and PACE pathways differ?
- how much out-of-pocket cost or home equity might be preserved under
  different strategies?

That kind of pilot is concrete enough to create product pull and partner
feedback before a full national dynamic LTC model exists.

[`appendix-colorado-ltc-rules-packet.md`](appendix-colorado-ltc-rules-packet.md)
translates that idea into a source-of-truth packet built from official
state and federal materials, which is the right standard for deciding
whether a Colorado pilot is real enough to fund.

Those are not phase-1 commitments. They are reasons to design the core
architecture well.
