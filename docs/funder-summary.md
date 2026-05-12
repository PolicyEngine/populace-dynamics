# Concept Note: An Open Dynamic Model for Social Security

## What this is

This is a concept note for an open, dynamic microsimulation model of
the U.S. Social Security system. The model would combine:

- a synthetic longitudinal population built from public data
- an open rules engine for benefit calculation
- a published validation record that lets outsiders inspect and
  reproduce the model's behavior

The proposed contribution is not just a tool. It is a public
methodological artifact: a fully open dynamic Social Security stack
alongside an inspectable validation record. The point is not to claim
instant parity with closed institutional models. The point is to build
the most transparent public alternative in this space and to validate
it rigorously enough that it becomes useful for research, advocacy,
exploratory policy design, and public education.

## Why this matters

Social Security policy analysis is dominated by models that are
internal to government, tied to restricted administrative records, or
accessible only through institutional relationships. The public
infrastructure for tax microsimulation has matured substantially in
the last decade. The same has not happened for Social Security.

For a program at this scale, the gap is unusually large. Researchers,
smaller policy organizations, journalists, and advocates can discuss
reform packages, but they often cannot inspect or reproduce the
assumptions behind the models that shape those debates. The public
ecosystem is more dependent on institutional trust than it should be.

## What is methodologically new

The contribution is the combination of:

- a public synthetic longitudinal population, calibrated to
  administrative targets
- an open rules engine that integrates benefit calculation natively
- explicit benchmark comparison to DYNASIM, MINT, CBOLT, and
  Morningstar
- a published validation record at the intermediate-state level, not
  only at the headline output level
- a path to productization that follows credibility rather than
  precedes it

No single piece of that bundle is unprecedented in isolation. The
combination does not currently exist for U.S. Social Security
analysis.

## Why this is suddenly achievable

The methodological building blocks have matured:

- Public-data reconstruction of a calibrated cross-sectional
  population is a solved problem. PolicyEngine's Enhanced CPS
  demonstrated that public data plus modern imputation plus
  aggressive calibration can support useful cross-sectional policy
  infrastructure.
- Synthetic population methodology has progressed enough to make
  longitudinal extension plausible without restricted matched data.
- An open rules engine for U.S. tax and benefit policy already
  implements core Social Security calculation logic.
- AI-assisted development substantially reduces the engineering cost
  of infrastructure, validation, replication, and documentation work
  relative to a decade ago.

A serious open dynamic Social Security model would have been
impractical at small-organization scale a decade ago. It is plausible
now.

## Why this is hard

The hard part is not the benefit formula. The hard part is building a
credible longitudinal population that supports:

- lifetime earnings histories
- insured-status determination
- claim-age and disability pathways
- spouse, survivor, and divorced-spouse outcomes
- forward projections that do not drift off course

That is why the project is best understood as a serious
research-and-infrastructure build that uses an existing public
synthetic population platform as a starting point, rather than a thin
Social Security app on top of static data.

## How it relates to existing models

The right comparison object is the full open stack: the longitudinal
population layer, the open rules engine, and the Social Security
application and validation layer. Against that stack:

- **DYNASIM** (Urban Institute) is the main benchmark for breadth,
  maturity, and state richness. The proposed open alternative does
  not match DYNASIM's institutional continuity but offers full
  inspectability and reproducibility.
- **MINT** (SSA) is the main benchmark for the importance of
  earnings-history credibility and administrative-data access. The
  proposed model trades administrative earnings for fully
  reproducible synthetic histories with explicit validation.
- **CBOLT** (CBO) is the main benchmark for official projection
  authority and macro-fiscal integration. The proposed model does not
  claim official scoring authority. Its contribution is making
  construction, validation, and policy workflow publicly inspectable.
- **Morningstar's retirement-outcomes model** is the most relevant
  adjacent benchmark for retirement-adequacy and LTSS-oriented
  household modeling.

Those models are not reasons to avoid this project. They are reasons
to scope it correctly. The right claim is not "we will beat all of
them quickly." The right claim is "we can build the most transparent
public alternative, and we know which components have to earn
credibility first."

## A quantifiable signal of demand

The asymmetry between tax-policy modeling supply and Social Security
modeling supply is a useful proxy for demand.

On the tax side, outside analysts can point to multiple visible
modeling channels that regularly support outside-facing analysis:

- **Tax-Calculator** is an open-source federal tax microsimulation
  model [@taxcalc2026]
- **TaxBrain** provides a public interface to open-source tax models
  [@taxbrain2025]
- **The Tax Policy Center** uses its microsimulation model to analyze
  major individual income tax bills and proposals [@tpcmodelfaq2025]
- **ITEP** publishes federal, state, and local distributional and
  revenue analyses driven by its microsimulation model
  [@itepmodel2025]

On the Social Security side, the benchmark model families are real
but largely closed:

- DYNASIM, MINT, CBOLT, Morningstar
  [@favreault2015; @urban2024dynasim4; @ssa2024mint; @cbo2018; @cbo2024longterm; @look2024retirementoutcomes]

A fair summary is therefore that tax policy has at least **four**
visible modeling channels supporting outside-facing analysis, while
Social Security has **zero** comparably broad public self-service
dynamic microsimulation platforms.

That does not prove that any single closed model is the bottleneck.
But it is strong evidence that public modeling supply is thin
relative to policy interest.

There is also a more specific signal. Some users can already use
PolicyEngine for narrow Social Security-adjacent questions, such as
taxation of benefits, but still need closed models like DYNASIM for
broader dynamic Social Security analysis. CRFB is a useful example of
this pattern. The problem is not lack of policy demand. It is the
absence of an open dynamic model layer.

## What an open dynamic Social Security stack would produce

Concrete deliverables would include:

- a documented public longitudinal synthetic population suitable for
  Social Security analysis and adjacent reuse
- a validated benefit-calculation pipeline integrated with the open
  rules engine
- reform-analysis workflows for a defined set of policy packages
- a public API and a focused user interface
- a permanent open repository containing methods, validation
  artifacts, and documentation
- a published validation record covering earnings histories, family
  structure, disability, claiming, and projection drift

The model would also expose the underlying analysis to AI agents
through standard interfaces (CLI, REST API, MCP server). That makes
it the first dynamic Social Security model an AI agent can call
directly. The accessibility matters: it lowers the cost of
substantive analysis for researchers, journalists, congressional
staff, and others who do not have direct modeling expertise.

## Stage gates and risk management

The project should be understood as stage-gated rather than
all-or-nothing. The principal risk is not coding speed. It is false
confidence — building a model that looks plausible but does not
survive external scrutiny.

The main gates:

- **Stage 1**: are the earnings histories and longitudinal population
  credible enough to support Social Security analysis?
- **Stage 2**: do family, disability, and benefit outputs hold up
  against published benchmarks?
- **Stage 3**: do forward projections remain stable and interpretable?
- **Stage 4**: is the validated subset ready for public
  productization?

If a gate fails, the project stops or narrows rather than proceeding
mechanically. Even a partial completion leaves real public value: a
longitudinal-population methods contribution, benchmark evidence on
competing earnings architectures, validated synthetic baseline panels
for selected uses, and reusable improvements to the underlying open
infrastructure.

## Adjacent applications

The same architecture should preserve a path to domains that share
the same longitudinal ingredients:

- retirement adequacy and wealth-sensitive analysis (SCF-linked)
- SSI interactions and poverty analysis
- long-term care and caregiving policy, where disability, wealth, and
  family structure matter over time

Those are not phase-one commitments. They are reasons to design the
core architecture well.

The most plausible first adjacent step is a state-specific long-term
care pilot rather than a national dynamic LTSS model. A state-specific
pilot can answer concrete eligibility and spend-down questions for
real families while the harder national dynamic LTC problem remains a
separate, later effort.

## What success looks like, even short of an official model

Success does not require becoming the official federal baseline. This
project is already valuable if it can:

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

## Why now

The project is more credible now than it would have been a few years
ago for three reasons.

First, the public-data tax modeling stack has already established a
credible workflow. Enhanced CPS demonstrates that public-data
reconstruction can support useful cross-sectional policy
infrastructure.

Second, synthetic population methodology has matured enough to make
longitudinal extension plausible.

Third, the benchmark landscape is clearer. The public record now
includes substantially more detail about what DYNASIM, MINT, CBOLT,
and Morningstar each optimize for, which makes it easier to scope a
public alternative honestly.

## What this is not

This is not "open-source policy analysis in the abstract." It is a
focused public-infrastructure build with an explicit proving ground:

- Social Security first
- public validation first
- productization only after credibility is earned

That framing matters because it ties the infrastructure work to a
concrete and important policy domain rather than to an open principle
in general.

## Open invitation

The project is at a stage where outside input shapes how it gets
built. The most valuable conversations right now are with:

- researchers and modelers with retirement-economics or
  microsimulation expertise
- policy organizations working on solvency, adequacy, or reform
  packages who would actually use validated outputs
- funders interested in public-infrastructure investment rather than
  memo-style research grants
- technical reviewers interested in open validation as a
  methodological contribution

Inquiries and design-partner conversations are welcome.
