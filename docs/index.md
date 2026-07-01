# An open dynamic Social Security model

::: {.callout-note}
**Stage-gated planning document**

This repository describes a proposed project. It does not claim that a
validated dynamic Social Security model already exists.
:::

## Executive summary

This project proposes an open, dynamic microsimulation model for
Social Security policy analysis — built so that every claim it makes
can be scored against reality. It combines:

- a public synthetic longitudinal population, calibrated to
  administrative targets
- synthetic lifetime earnings histories learned from longitudinal data
- demographic and family transitions needed for auxiliary benefits and
  disability analysis
- benefit calculations computed exactly by an open tax-benefit rules
  engine
- a public scorecard: forecast cells that resolve annually,
  retrodictive backtests, and held-out panel moments
- a public API and user interface for exploratory policy analysis,
  callable from AI agents through standard interfaces

The central claim is not that a public model can instantly replace
SSA, CBO, or Urban Institute tools. The central claim is narrower and
more defensible: public data, modern imputation, and explicit
calibration can support a transparent model whose usefulness is
measured the only way that counts — whether it improves predictions,
scored in public. The model states its own limits up front
([domains-of-validity.md](domains-of-validity.md)) and earns trust
through its resolution record
([scoring-and-resolution.md](scoring-and-resolution.md)).

One naming note, because it is a design decision: this project does
not introduce a named simulator to stand beside DYNASIM or MINT. The
machinery lives in `populace`, PolicyEngine's open microdata stack;
the deliverable is a versioned population artifact with a manifest
and a scorecard. Models were branded when the model was the moat.
Here the artifact and its track record are the product.

## The problem

Social Security is too important to be modeled only behind closed
doors. The main U.S. dynamic models are either internal to government,
tied to restricted administrative records, or accessible only through
contracts. That leaves a major gap between the importance of the
policy questions and the accessibility of the tools used to answer
them.

At the same time, static tax-benefit modeling has already shown that
publicly reproducible microdata can be useful when the pipeline is
carefully engineered and aggressively validated. PolicyEngine's
populace stack demonstrates this at production scale today — built
entirely from primary sources, it became the certified default U.S.
microdata in policyengine.py in 2026 after beating the prior enhanced
CPS on held-out accuracy. The next question is whether that stack can
extend to longitudinal microdata and then support serious Social
Security analysis.

## What this project is

This project is:

- a research and infrastructure effort to build a validated public
  synthetic longitudinal population
- a transparency project for Social Security policy analysis
- the first policy application and validation layer on top of that
  longitudinal population

This project is not:

- a near-term replacement for SSA's MINT or CBO's internal models
- a promise that every behavioral margin can be modeled credibly
- a one-shot 18-month build that goes directly from concept to public
  launch

## Decisions already made

### 1. Build on PolicyEngine's populace microdata stack

The project extends `populace`, PolicyEngine's ML-first microdata
layer, rather than building an isolated Social Security-only
dataset. populace already integrates and calibrates dozens of
surveys and administrative sources and supports the methodological
machinery (synthesis, calibration, sparsification, and
authenticity/privacy evaluation) the Social Security extension
needs. That choice matters because:

- generic population synthesis belongs in populace, not in this
  repository
- populace's cross-sectional layer is already validated against
  large numbers of administrative targets
- this repository can focus on Social Security domain validation and
  policy application rather than rebuilding generic synthesis tools
- future adjacent domains can reuse the same longitudinal population
  asset

### 2. Social Security first, with adjacent interactions preserved

The initial objective is still a Social Security model. That means the
first longitudinal extension of `populace` should include the family
structure, disability, and claiming dynamics needed for serious benefit
analysis. It also preserves interactions with taxes, SSI, and other
means-tested programs through PolicyEngine-US where possible.

The project should be architected so that adjacent domains can be added
later, but it should not dilute early validation by pretending to solve
every lifecycle policy problem at once.

### 3. The scorecard is the product before the product

The strongest reason to fund this work is not the eventual interface.
It is a public scoring record:

- forecast cells that resolve against administrative publications on
  an annual calendar
- retrodictive backtests built from version-pinned data vintages
- held-out panel moments for the population layer itself
- misses published with the same prominence as hits

Without that record, the project would be just another model
description. [scoring-and-resolution.md](scoring-and-resolution.md)
defines the protocol.

### 4. The project should be staffed like a serious build

The current project lead can set direction, but a credible plan assumes
that funded implementation includes dedicated project staff. A dynamic
model of this kind should not rely on fractional attention from already
committed leadership alone.

### 5. Platform validation and policy validation are distinct

This project now has two validation obligations:

- validate longitudinal `populace` as a population asset
- validate Social Security outputs generated from that asset

Those are related, but not identical. A population platform can look
good in generic synthesis metrics and still fail at policy-relevant
Social Security outcomes. Conversely, a narrowly tuned application can
match headline policy outputs while resting on a weak underlying
longitudinal structure. The project has to clear both bars.

### 6. The long horizon ships as a sensitivity surface

The 75-year balance is dominated by assumptions no one has forecast
well. The model therefore never publishes a point depletion date or
75-year balance as a prediction; long-horizon outputs are surfaces
over documented assumption ranges, and every output carries its
domain-of-validity tier as metadata
([domains-of-validity.md](domains-of-validity.md)).

## Deliverables

By the end of the full plan, the project should produce:

- a documented longitudinal `populace` suitable for Social Security
  analysis and adjacent reuse
- a validated benefit-calculation pipeline integrated with
  PolicyEngine-US
- reform-analysis workflows for a defined set of policy packages
- a public API and web interface
- a permanent open repository containing methods, validation artifacts,
  and documentation

## Why this is worth doing even if it never becomes an official model

There is substantial value between "toy model" and "official federal
baseline." A public dynamic model would still matter if it succeeds only
at the following:

- enabling independent replication of standard reform analyses
- helping smaller organizations and researchers test ideas
- providing a public benchmark for assumptions and validation methods
- making distributional tradeoffs easier to inspect
- supporting teaching and training in Social Security policy analysis

Those are meaningful gains even before the model reaches the level of
trust required for official scoring.

## Relationship to adjacent policy domains

The architecture should preserve a path to domains that reuse the same
longitudinal ingredients, especially:

- retirement adequacy and wealth-sensitive analysis
- SSI interactions and poverty analysis
- long-term care and caregiving policy, where disability, wealth, and
  family structure matter over time

That does not mean those domains belong in phase 1. It means the
project should not lock `populace` into a Social-Security-only design
that cannot be extended later.

## Guide to the rest of the book

- [funder-summary.md](funder-summary.md) is a standalone concept note
  describing the project's methodological premise and contribution.
- [domains-of-validity.md](domains-of-validity.md) states which
  questions the model will and will not claim to answer, and why.
- [scoring-and-resolution.md](scoring-and-resolution.md) defines the
  scoring protocol: resolving forecast cells, retrodiction with
  leakage control, and the merge-on-score contribution rule.
- [policy-applications.md](policy-applications.md) explains the user
  needs and policy questions the model should answer.
- [existing-models.md](existing-models.md) positions the project against
  DynaSim, MINT, CBOLT, and other models.
- [benchmark-model-component-matrix.md](benchmark-model-component-matrix.md)
  compares our plan with DYNASIM, MINT, CBO, and Morningstar
  component-by-component using only the public record.
- [data-sources.md](data-sources.md),
  [methodology.md](methodology.md), and
  [operationalizing-longitudinal-construction.md](operationalizing-longitudinal-construction.md)
  go deeper on how lifetime earnings and related state variables would
  actually be constructed.
- [technical-specifications.md](technical-specifications.md) describes
  the required state variables, table structures, and application-layer
  requirements.
- [calibration-targets.md](calibration-targets.md) describes how the
  model earns credibility.
- [evaluation-and-model-selection.md](evaluation-and-model-selection.md)
  defines how the project will choose among competing earnings
  architectures and what counts as passing stage 1.
- [operationalizing-disability-and-claiming.md](operationalizing-disability-and-claiming.md)
  applies the same discipline to SSDI pathways, retirement claiming,
  and auxiliary-benefit timing.
- [operationalizing-family-and-auxiliary-benefits.md](operationalizing-family-and-auxiliary-benefits.md)
  spells out how marriage histories, widowhood, spouse matching, and
  dual-entitlement logic would actually be constructed and validated.
- [operationalizing-mortality-and-projection-drift.md](operationalizing-mortality-and-projection-drift.md)
  covers mortality construction, survivor-relevant death timing, and
  the drift-control logic needed for credible forward projections.
- [roadmap.md](roadmap.md) and
  [risks-and-stage-gates.md](risks-and-stage-gates.md) set a realistic
  implementation and review plan.
