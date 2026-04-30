# Existing Models and What They Teach Us

## Overview

This project does not start from a blank slate. The relevant comparison
set already exists: DynaSim, MINT, CBOLT, PWBM, and a small number of
open or partially open alternatives. The right question is not "how do
we beat them on every dimension?" The right question is "what do they
optimize for, what do they leave inaccessible, and where is there still
room for a public model to matter?"

Four dimensions are especially important:

1. **Credibility**: how much trust the model has earned with agencies
   and policy professionals
2. **Transparency**: how much of the methodology, code, and data logic
   can be inspected publicly
3. **Scope**: what policies and populations the model can analyze
4. **Accessibility**: who can actually use the model in practice

## DynaSim: the benchmark for non-governmental breadth

Urban Institute's DynaSim is the closest thing to the benchmark
non-governmental dynamic microsimulation model for retirement policy
analysis [@favreault2015].

### What DynaSim gets right

- It has decades of accumulated refinement and institutional memory.
- It treats Social Security as part of a broader retirement-income
  system rather than in isolation.
- It supports rich demographic and economic heterogeneity, including
  marriage, disability, and asset dynamics.
- It has produced a large body of policy analysis that researchers and
  funders recognize.

### What DynaSim does not solve for the public

- The code is not open.
- Access typically depends on contracts or institutional relationships.
- Users cannot easily inspect or modify the full modeling pipeline.
- The model is influential, but not independently reproducible in the
  way an open research tool should be.

### What this project should learn from DynaSim

The lesson is not to imitate DynaSim's institutional model. The lesson
is to match its seriousness about validation, demographic detail, and
cross-program context while choosing different tradeoffs on access and
transparency. The longer public-source dossier lives in
[`appendix-dynasim.md`](appendix-dynasim.md) so this chapter can stay
focused on strategic comparison. The more operational side-by-side
comparison lives in
[`benchmark-model-component-matrix.md`](benchmark-model-component-matrix.md).

## MINT: the benchmark for administrative-data credibility

SSA's Modeling Income in the Near Term (MINT) is the strongest reference
point for credibility in Social Security microsimulation
[@smith2010mint; @ssa2024mint].

### Why MINT matters

MINT's defining advantage is access to administrative earnings records.
For older cohorts, that removes much of the uncertainty that public-data
models have to manage. It also gives SSA a natural validation loop:
administrative data continue to arrive, and projection quality can be
checked against realized outcomes over time.

### Why MINT is not a substitute for a public model

- The code is not open.
- External researchers cannot freely reproduce or extend the model.
- Access to the underlying data is heavily restricted.
- The model is designed first for SSA's analytical mission, not for
  broad public use.

### What this project should learn from MINT

MINT is a reminder that the hardest part of this project is not benefit
arithmetic. It is the lifecycle data problem. A public model will not
match MINT's administrative-data advantage in the near term, so it must
compensate with unusually explicit validation and unusually clear
disclosure of uncertainty.

## CBOLT: official fiscal authority, different objective function

CBO's long-term model, often referred to through the CBOLT framework,
serves a different institutional purpose from the one proposed here
[@cbo2018; @cbo2004]. It should not be caricatured as a
representative-agent model. Public CBO materials describe an individual
microsimulation model with administrative earnings records, imputed
family characteristics, annual transitions, and spouse links used for
Social Security benefit calculations [@cbo2018; @cbo2019replacementrates].

### What CBOLT optimizes for

- internal consistency with CBO's broader long-term fiscal outlook
- aggregate budget analysis
- macro-fiscal coherence across policy domains
- distributional Social Security results backed by administrative
  earnings histories

### Why that matters

This comparison helps discipline the project's claims. A public dynamic
microsimulation model should not pretend to outcompete CBO on its own
institutional objective function. The comparative advantage here is
micro-distributional analysis with transparent assumptions, not official
budget authority.

## PWBM: public-facing outputs without full public model access

The Penn Wharton Budget Model shows that public-facing interfaces can be
powerful even when the underlying system remains relatively closed. It
is a useful reminder that accessibility has layers:

- public-facing tools can broaden reach
- but limited code and data transparency still constrain independent
  verification

The implication for this project is straightforward: a web interface is
helpful, but it is not a substitute for an open modeling pipeline.

## The Cato model: proof that open Social Security modeling is possible

The Cato Institute's open-source Social Security model is important even
if it differs methodologically from this proposal. It demonstrates that:

- there is demand for open Social Security modeling
- open-source implementation is not hypothetical
- a smaller or narrower model can still be useful

Its existence also raises the bar. This project cannot claim novelty
just because it is open. The differentiators have to be stronger than
that:

- tighter integration with PolicyEngine's tax-benefit ecosystem
- a larger and better-calibrated public base population
- a stronger validation story
- a clearer path from research pipeline to public interface

## International Open Models

Open dynamic microsimulation is not unique to the United States. Models
such as SimPaths demonstrate that dynamic policy modeling can be built
in the open and documented for outside researchers. That matters less as
a direct methodological template than as proof that a transparent,
research-grade ecosystem can form around this type of work.

## Comparison Table

| Model | Main strength | Main limitation | What we should learn |
|---|---|---|---|
| **DynaSim** | Rich retirement-policy scope and long validation history | Not open or easily accessible | Match the rigor, not the access model |
| **MINT** | Administrative-data credibility | Restricted data and code | Treat validation as the central challenge |
| **CBOLT** | Official long-term fiscal authority plus administrative earnings credibility | Limited public reproducibility of the production pipeline | Do not overclaim on official scoring, administrative-data parity, or macro closure |
| **PWBM** | Public-facing policy communication | Limited transparency beneath the interface | Public tools help, but they are not enough |
| **Cato model** | Open-source proof of concept | Narrower ecosystem and product layer | Openness alone is not the differentiator |

## What Gap Still Exists

No current model combines all of the following:

- open source
- a public-data workflow
- transparent validation artifacts
- a public API
- a public interface
- integration with a broader tax-benefit platform

That is the real gap this project is trying to fill.

## What This Project Should and Should Not Claim

### Plausible claims

- it can become the most transparent dynamic Social Security model in
  the U.S. policy ecosystem
- it can reduce the cost of serious exploratory policy analysis
- it can create a public validation benchmark for lifecycle
  microsimulation
- it can eventually support adjacent domains that share the same
  longitudinal data bottleneck

### Implausible or premature claims

- that it will quickly replace MINT for SSA's purposes
- that it will match CBO's institutional role
- that public-data reconstruction eliminates the value of administrative
  records
- that an interface alone creates credibility

## Bottom Line

Existing models already show what a serious Social Security
microsimulation model looks like. DynaSim shows the value of breadth and
institutional memory. MINT shows the value of administrative earnings
histories. CBOLT shows what official fiscal authority optimizes for.
Open models show that reproducibility is feasible.

The point of this project is to assemble a different bundle of
strengths: public reproducibility, integration with PolicyEngine, and a
validation record strong enough to make the model useful even before it
is trusted as an official benchmark.
