# Building an Open-Source Social Security Dynamic Microsimulation Model

::: {.callout-note}
**Stage-Gated Planning Document**

This repository describes a proposed project. It does not claim that a
validated dynamic Social Security model already exists.
:::

## Executive Summary

This project proposes an open-source dynamic microsimulation model
for Social Security policy analysis, combining:

- a public synthetic longitudinal population, calibrated to
  administrative targets
- synthetic lifetime earnings histories learned from longitudinal data
- demographic and family transitions needed for auxiliary benefits and
  disability analysis
- validated benefit calculations integrated with an open tax-benefit
  rules engine
- a public API and user interface for exploratory policy analysis,
  callable from AI agents through standard interfaces

The central claim is not that a public model can instantly replace
SSA, CBO, or Urban Institute tools. The central claim is narrower and
more defensible: public data, modern imputation, and explicit
calibration may be sufficient to create a transparent model that is
useful for research, teaching, advocacy, exploratory policy design,
and eventually broader policy use if the validation is strong enough.

## The Problem

Social Security is too important to be modeled only behind closed
doors. The main U.S. dynamic models are either internal to government,
tied to restricted administrative records, or accessible only through
contracts. That leaves a major gap between the importance of the
policy questions and the accessibility of the tools used to answer
them.

At the same time, static tax-benefit modeling has already shown that
publicly reproducible microdata can be useful when the pipeline is
carefully engineered and aggressively validated. PolicyEngine's
microplex stack demonstrates this at production scale today. The
next question is whether that stack can extend to longitudinal
microdata and then support serious Social Security analysis.

## What This Project Is

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

## Decisions Already Made

### 1. Build on PolicyEngine's microplex microdata stack

The project extends `microplex`, PolicyEngine's ML-first microdata
layer, rather than building an isolated Social Security-only
dataset. microplex already integrates and calibrates dozens of
surveys and administrative sources and supports the methodological
machinery (synthesis, calibration, sparsification, and
authenticity/privacy evaluation) the Social Security extension
needs. That choice matters because:

- generic population synthesis belongs in microplex, not in this
  repository
- microplex's cross-sectional layer is already validated against
  large numbers of administrative targets
- this repository can focus on Social Security domain validation and
  policy application rather than rebuilding generic synthesis tools
- future adjacent domains can reuse the same longitudinal population
  asset

### 2. Social Security First, With Adjacent Interactions Preserved

The initial objective is still a Social Security model. That means the
first longitudinal extension of `microplex` should include the family
structure, disability, and claiming dynamics needed for serious benefit
analysis. It also preserves interactions with taxes, SSI, and other
means-tested programs through PolicyEngine-US where possible.

The project should be architected so that adjacent domains can be added
later, but it should not dilute early validation by pretending to solve
every lifecycle policy problem at once.

### 3. Validation Is the Product Before the Product

The strongest reason to fund this work is not the eventual interface. It
is the possibility of a public validation record:

- how earnings histories are constructed
- where the synthetic panel matches external benchmarks
- where it fails
- how sensitive results are to model-family and calibration choices

Without that validation layer, the project would be just another model
description.

### 4. The Project Should Be Staffed Like a Serious Build

The current project lead can set direction, but a credible plan assumes
that funded implementation includes dedicated project staff. A dynamic
model of this kind should not rely on fractional attention from already
committed leadership alone.

### 5. Platform Validation and Policy Validation Are Distinct

This project now has two validation obligations:

- validate longitudinal `microplex` as a population asset
- validate Social Security outputs generated from that asset

Those are related, but not identical. A population platform can look
good in generic synthesis metrics and still fail at policy-relevant
Social Security outcomes. Conversely, a narrowly tuned application can
match headline policy outputs while resting on a weak underlying
longitudinal structure. The project has to clear both bars.

## Deliverables

By the end of the full plan, the project should produce:

- a documented longitudinal `microplex` suitable for Social Security
  analysis and adjacent reuse
- a validated benefit-calculation pipeline integrated with
  PolicyEngine-US
- reform-analysis workflows for a defined set of policy packages
- a public API and web interface
- a permanent open repository containing methods, validation artifacts,
  and documentation

## Why This Is Worth Doing Even If It Never Becomes an Official Model

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

## Relationship to Adjacent Policy Domains

The architecture should preserve a path to domains that reuse the same
longitudinal ingredients, especially:

- retirement adequacy and wealth-sensitive analysis
- SSI interactions and poverty analysis
- long-term care and caregiving policy, where disability, wealth, and
  family structure matter over time

That does not mean those domains belong in phase 1. It means the
project should not lock `microplex` into a Social-Security-only design
that cannot be extended later.

## Guide to the Rest of the Book

- [funder-summary.md](funder-summary.md) is a standalone concept note
  describing the project's methodological premise and contribution.
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
