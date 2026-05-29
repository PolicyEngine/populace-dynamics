# Risks and stage gates

## Overview

The main threat to this project is not failure to write code. It is
failure to know when the outputs are not yet good enough. This chapter
defines the principal risks and the stage gates that should govern
continuation.

## Principal risks

### 1. Earnings histories look reasonable but are wrong in policy-relevant ways

It is possible to match simple age-earnings profiles while still getting
mobility, persistence, zero-earnings patterns, or AIME distributions
wrong. That would produce a panel that looks plausible but fails at the
core Social Security task.

### 2. Cross-sectional calibration conflicts with longitudinal validation

A model may be able to match annual aggregates and still distort
longitudinal transitions. Conversely, a model that respects PSID-style
transitions may drift away from administrative totals. Managing that
tension is one of the core research problems of the project.

### 3. Platform work expands faster than policy validation

Now that the project is best understood as making `microplex`
longitudinal, there is a new risk: the population platform can become
technically interesting without yet being decision-useful for Social
Security. That would be real research progress, but it would not by
itself justify strong policy claims.

### 4. Benefit logic works for typical cases but fails at important edges

Social Security policy analysis depends on edge cases, not just median
retired workers. Family benefits, disability pathways, and complex
claiming patterns matter.

### 5. Baseline projections drift

Even a strong baseline panel can become untrustworthy when projected
forward if drift is not controlled explicitly.

### 6. Productization happens before validation is mature

A public interface is attractive and visible, but it can also amplify
weaknesses if the validation record is not ready.

## Stage gate 1: longitudinal microplex quality

The project should advance past stage 1 only if it can show that
longitudinal `microplex` is credible on multiple dimensions:

- age-earnings levels
- dispersion and percentiles
- zero-earnings patterns
- AIME distributions
- mobility and persistence

Failure here is not a minor miss. It is a reason to pause and revise the
methodology.

## Stage gate 2: benefit and family validation

The project should advance past stage 2 only if the synthetic panel
produces benefit outcomes that are good enough for exploratory policy
analysis:

- beneficiary counts by major type
- average benefits
- selected benefit distributions
- family and auxiliary outcomes that are directionally credible

If the model only works for retired worker benefits and breaks down in
family or disability pathways, that should be stated clearly and used to
bound later claims.

## Stage gate 3: projection quality

The project should advance to public reform analysis only if baseline
projections remain stable and interpretable:

- drift is measured and controlled
- near-term projections track published benchmarks reasonably well
- sensitivity to key assumptions is documented

If the model cannot produce a credible baseline projection, it is not
ready for strong claims about reform packages.

## Stage gate 4: public product readiness

The project should release a public interface only if:

- headline examples are reproducible
- limitations are documented in plain language
- the public product is narrower than the full internal research stack
- the validation record is published alongside the interface

## If a gate fails

Failure should still produce useful public outputs. Possible fallback
deliverables include:

- a public methods paper on synthetic longitudinal reconstruction
- a validated baseline panel without a public policy interface
- a narrower model focused on a subset of beneficiary types
- static or semi-dynamic tools built on the same validation assets

The point of stage gates is not to threaten cancellation. It is to avoid
pretending that an unvalidated model is ready for claims it cannot yet
support.

## Explicit near-term non-goals

In the first major phase of the project, the model should not claim to:

- replace SSA's internal models for official estimates
- provide full general-equilibrium feedback analysis
- model every behavioral response in the retirement system
- solve all adjacent lifecycle policy questions at once

Those are legitimate long-run ambitions, but they should not be confused
with early validation milestones.
