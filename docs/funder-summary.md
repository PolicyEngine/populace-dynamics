# Concept Note: An Open Dynamic Model for Social Security

## What this is

This concept note describes the design of an open, dynamic
microsimulation model of the U.S. Social Security system, alongside a
published validation record that lets outsiders inspect and reproduce
its behavior.

The proposed contribution is methodological. The right standard is not
matching closed institutional models in their first release; it is
building the most transparent public alternative in this space and
validating it rigorously enough that it becomes useful for research,
advocacy, exploratory policy design, and public education.

## Why this matters

Closed institutional models dominate Social Security policy analysis.
They live inside government, depend on restricted administrative
records, or sit behind institutional relationships. Tax
microsimulation has matured into real public infrastructure over the
past decade. Social Security modeling has not.

For a program at this scale, the gap is unusually large. Researchers,
smaller policy organizations, journalists, and advocates can discuss
reform packages, but they often cannot inspect or reproduce the
assumptions behind the models that shape those debates. The public
ecosystem depends more on institutional trust than it should.

## Design

The proposed model has four components. This note describes the
design first as an architecture, then a concrete implementation in
the PolicyEngine open-source stack.

### Component 1: Synthetic longitudinal population

The population layer must support:

- lifetime earnings histories (the highest-35 earnings record that
  drives benefit calculation)
- marriage, divorce, widowhood, and remarriage histories needed for
  spousal, survivor, and divorced-spouse benefits
- disability onset, recovery, and termination dynamics
- mortality with differential rates by lifetime earnings, education,
  race, and sex
- forward projection that does not drift off course
- coherent household and couple structure preserved across years

The population must also be public, reproducible, and validatable
against administrative targets. That rules out dependence on
restricted matched data, which is how MINT operates. It rules in
favor of synthesis methods that combine survey panel data,
administrative aggregate targets, and modern imputation.

The methodological structure has three layers:

1. A calibrated cross-sectional baseline grounded in public survey
   data and adjusted against administrative aggregates
2. Longitudinal extension via imputation from panel data and/or
   generative trajectory synthesis with explicit handling of zero
   earnings, mobility, and cohort effects
3. Calibration to administrative targets at both cross-sectional and
   longitudinal levels, with explicit handling of network constraints
   created by marriage, divorce, fertility, and death

The hard part of this project lives almost entirely in component 1.

### Component 2: Open tax-benefit rules engine

The rules engine must calculate:

- retirement benefits (OASI), including AIME, PIA, bend points, COLA,
  and early/delayed retirement adjustments
- disability benefits (SSDI), including the waiting period and
  Medicare interaction
- dependent, spousal, and survivor benefits
- benefit taxation under federal income tax rules
- interactions with means-tested programs (SSI, Medicare, Medicaid)
  for adequacy analysis
- WEP, GPO, and other adjustments
- reform variants via parameter and formula modification

An open rules engine matters not only because users should be able
to inspect the rules themselves, but because analysts must be able
to encode and audit the reforms they want to test. Closed rules
engines force users to trust the implementation.

The rules engine must also support vectorized operations on large
synthetic panels so the model can run reform analysis at scale.

### Component 3: Validation and benchmark layer

The validation premise distinguishes this project from a model
description. The project must validate at multiple levels and
publish each level.

The validation stack:

- **Input data quality**: panel earnings vs. SSA cohort earnings;
  benefit receipt in survey data vs. SSA administrative totals;
  demographic distributions vs. Census
- **Imputation quality**: held-out panel validation, quantile
  coverage, autocorrelation structure of imputed histories
- **Cross-sectional calibration**: age-earnings profiles, beneficiary
  counts by type, average benefit amounts, benefit distributions
- **Longitudinal validation**: earnings mobility matrices, variance
  decomposition between- and within-person, cohort-specific
  age-earnings profiles
- **Fiscal aggregates**: total OASDI benefit payments, total covered
  earnings, near-term and long-term trust fund projections
- **Policy-relevant outcomes**: replacement rates by lifetime
  earnings quintile, lifetime benefit distribution by cohort,
  distributional progressivity, poverty rates among elderly by
  demographic group

The stage-gate logic is risk management. The project should not
advance from one stage to the next without published validation
passing pre-specified thresholds. False confidence — building a model
that looks plausible but does not survive external scrutiny — is the
principal risk this project faces, not coding speed.

### Component 4: Public delivery surface

The model should expose three surfaces:

- a Python library and CLI that researchers can use directly
- a REST API that programmatic users can call
- an MCP server so AI agents can run baseline distributions, score
  reform packages, and generate cohort-specific outputs through
  natural language

The MCP surface matters because it dramatically lowers the cost of
substantive Social Security analysis for users without direct
modeling expertise — journalists, congressional staff, advocacy
organizations, and researchers in adjacent fields. We are not aware
of an existing dynamic Social Security model that exposes an
AI-callable interface of this kind.

A public web interface is part of the long-term delivery design but
should follow validation rather than precede it.

## What is methodologically new

The design described above is not unprecedented in any single
component. The contribution is the combination of:

- a public synthetic longitudinal population, calibrated to
  administrative targets
- an open rules engine that integrates benefit calculation natively
- explicit benchmark comparison to DYNASIM, MINT, CBOLT, and
  Morningstar
- a public validation record at the intermediate-state level, not
  only at the headline output level
- AI-callable analysis interfaces from day one
- a path to productization that follows credibility rather than
  precedes it

No equivalent bundle currently exists for U.S. Social Security
analysis.

## Implementation: the PolicyEngine stack

The most natural implementation of this design today is the
PolicyEngine open-source stack. PolicyEngine maintains:

- **Arch**: a harness over dozens of U.S. government survey and
  administrative datasets — the source microdata and the calibration
  targets (from CBO, IRS, SSA, Census, and others) — assembled and
  maintained as a common data layer.
- **microplex**: PolicyEngine's ML-first microdata layer. Built from
  the ground up around modern machine-learning synthesis and
  calibration methods rather than retrofitting them onto older
  microsim infrastructure, it synthesizes populations from Arch's
  sources and calibrates them against Arch's administrative targets,
  with explicit support for longitudinal sources. The synthesis
  methods include quantile regression forests, quantile deep neural
  networks, and masked autoregressive flows; calibration uses
  gradient descent with optional L0-regularized record selection;
  authenticity and privacy evaluation uses precision, recall,
  density, and coverage (PRDC) metrics.
- **PolicyEngine-US**: open Python rules engine for U.S. federal and
  state tax-benefit policy. Calculates OASDI benefits, benefit
  taxation, and means-tested program interactions.
- **PolicyEngine-API** and **frontend**: production REST API and
  interactive interface that think tanks, researchers, and
  congressional staff use.
- **microimpute, microcalibrate, L0**: tooling for imputation,
  gradient-descent calibration, and sparsification.

The Social Security build adds the longitudinal extension on top of
microplex, the Social Security application and validation layer on
top of PolicyEngine-US, and an MCP server on top of the API.

The design itself is architecture-agnostic — other open-source
stacks could realize it — but the PolicyEngine stack is the path
with the most production-ready foundation, existing Social Security
calculation logic in the rules engine, and existing API and delivery
infrastructure.

## Why this is suddenly achievable

A serious open dynamic Social Security model would have been
impractical at small-organization scale a decade ago. It is plausible
now for four reasons:

- PolicyEngine's microplex stack demonstrates that public-data
  reconstruction of calibrated cross-sectional populations works at
  production scale
- synthetic population methodology has progressed enough to make
  longitudinal extension plausible without restricted matched data
- open rules engines for U.S. tax-benefit policy now implement core
  Social Security calculation logic
- AI-assisted development substantially reduces the engineering cost
  of infrastructure, validation, replication, and documentation work
  relative to a decade ago

## The open-modeling landscape

The contrast between U.S. tax microsimulation and U.S. Social
Security microsimulation is a useful proxy for the public
infrastructure gap.

On the tax side, the ecosystem spans a spectrum of openness.

**Openly callable, public-data models**:

- **Tax-Calculator** (Policy Simulation Library): open-source federal
  tax microsimulation model [@taxcalc2026]
- **PolicyEngine**: open-source federal and state tax-benefit
  microsimulation with a calibrated public microdata foundation, a
  REST API, and an interactive web interface [@policyengine2026]
- **FiscalSim-US** (Center for Growth and Opportunity, Utah State):
  open-source federal and state tax-benefit microsimulation
  [@fiscalsim2026]

**Source-available, restricted-data models**:

- **Yale Budget Lab Tax-Simulator**: the Budget Lab publishes its
  code on GitHub but the model depends on the IRS Public Use File,
  which the IRS does not release publicly, so outside users cannot
  reproduce production runs [@yaletaxsimulator2026]

**Proprietary models** used for outside-facing analysis:

- **Tax Policy Center** microsimulation model [@tpcmodelfaq2025]
- **ITEP** microsimulation tax model [@itepmodel2025]
- **Tax Foundation** Taxes and Growth model [@taxfoundationtag2025]
- **Penn Wharton Budget Model** [@pwbm2025]

The Social Security side has a much thinner open-modeling layer. The
institutional benchmark models are real and important, but outside
users can reach them only through institutional relationships
[@favreault2015; @urban2024dynasim4; @ssa2024mint; @cbo2018; @cbo2024longterm; @look2024retirementoutcomes]:

- **DYNASIM** (Urban Institute) — the benchmark for breadth,
  maturity, and state richness. The open model will not match its
  institutional continuity, but offers full inspectability and
  reproducibility.
- **MINT** (SSA) — the benchmark for earnings-history credibility
  through administrative data. The open model trades administrative
  earnings for fully reproducible synthetic histories with explicit
  validation.
- **CBOLT** (CBO) — the benchmark for official projection authority
  and macro-fiscal integration. The open model makes no claim to
  official scoring; its contribution is a publicly inspectable
  construction, validation, and policy workflow.
- **Morningstar's retirement-outcomes model** — the closest adjacent
  benchmark for retirement-adequacy and LTSS-oriented household
  modeling.

These models are not reasons to avoid the project. They are reasons
to scope it correctly.

The closest open analogue is the **Cato Social Security model**
[@catossmodel2026], an AGPL-3.0 R implementation. It simulates
mortality, fertility, marriage, divorce, and employment as
stochastic transitions on a sample of approximately 10,000
households drawn from the 2007 ASEC, with pre-2007 earnings
histories matched against the 2006 SSA Public Use File and forward
earnings indexed to the Average Wage Index. It produces conventional
long-term Social Security outputs — trust fund ratios, insolvency
dates, 75-year balance — and can score reforms to retirement ages,
bend points, indexing methodology, and benefit credits.

The stated capabilities are deliberately scoped. The model focuses
on OASI without a separate SSDI module, cannot score tax-code
reforms, treats the Windfall Elimination Provision and Government
Pension Offset as limited, holds the labor-force transition matrix
constant from 2024 onward, and depends on simplifying assumptions
for fertility and other dynamics. Running the baseline simulation
requires access to the underlying SSA Public Use File. The
repository does not currently include published validation against
SSA Trustees, MINT, or DYNASIM. A more complete characterization is
in [`existing-models.md`](existing-models.md).

The Cato model shows that an open dynamic Social Security model is
feasible; it does not close the gap this concept addresses. The open
tax-modeling ecosystem already offers production stacks with
calibrated public-data populations, programmatic and AI-callable
APIs, web interfaces, transparent validation, and tax-benefit
integration. The open Social Security layer offers a single narrower
model with none of that combination. That is the specific gap this
project fills.

A more specific signal of demand: some users can already use
PolicyEngine for narrow Social Security-adjacent questions but still
need closed models for broader dynamic analysis. CRFB has
commissioned PolicyEngine for an
[analysis of Social Security benefit-taxation reforms](https://www.policyengine.org/us/taxation-of-benefits-reforms)
(publication forthcoming, 2026 Q2) — a static tax-side question that
the existing open stack already supports. Broader dynamic questions
about actuarial balance, claiming behavior, lifetime distributional
impact, and cohort-specific reform effects still push users toward
closed benchmark models. That is exactly the gap this project
narrows.

## Adjacent applications

The same architecture preserves a path to domains that share the same
longitudinal ingredients:

- retirement adequacy and wealth-sensitive analysis (SCF-linked)
- SSI interactions and poverty analysis
- long-term care and caregiving policy, where disability, wealth, and
  family structure matter over time

These are not phase-one commitments. They are reasons to design the
core architecture well.

The most plausible first adjacent step is a state-specific long-term
care pilot rather than a national dynamic LTSS model. A
state-specific pilot can answer concrete eligibility and spend-down
questions for real families while the harder national dynamic LTC
problem remains a separate, later effort.

## What success looks like, even short of an official model

Success does not require becoming the official federal baseline.
This project is already valuable if it can:

- replicate major baseline Social Security distributions credibly
- support exploratory reform analysis with transparent assumptions
- publish a reusable public longitudinal population asset
- lower the barrier to serious dynamic modeling for outside
  researchers and policy organizations
- create a public benchmark for how lifecycle microsimulation should
  be validated
- expose validated capabilities to AI agents through standard
  interfaces, multiplying the set of users who can run rigorous
  analysis

That is meaningful public value even short of official scoring
status.

## What this is not

This is not "open-source policy analysis in the abstract." It is a
focused public-infrastructure build with an explicit proving ground:

- Social Security first
- public validation first
- productization only after the project earns credibility

That framing matters because it ties the infrastructure work to a
concrete and important policy domain rather than to an open
principle in general.

## Open invitation

The project is at a stage where outside input shapes how it
develops. The most valuable conversations right now are with:

- researchers and modelers with retirement-economics or
  microsimulation expertise
- policy organizations working on solvency, adequacy, or reform
  packages who would actually use validated outputs
- funders interested in public-infrastructure investment rather than
  memo-style research grants
- technical reviewers interested in open validation as a
  methodological contribution

Inquiries and design-partner conversations are welcome.
