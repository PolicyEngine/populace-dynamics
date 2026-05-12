# Development Roadmap

## Overview

This roadmap describes a serious, stage-gated build. The goal is not
to force a public launch on an arbitrary schedule. The goal is to earn
the right to proceed from one stage to the next by passing validation
gates.

The roadmap is therefore organized around capability milestones and
decision points, not chronological phases.

## Workstreams

The project has four core workstreams:

1. **Longitudinal microplex construction**
2. **Social Security integration and validation**
3. **Policy analysis products**
4. **Public API and interface**

These workstreams overlap, but they should not advance at the same pace.
Interface work should lag validation work. Policy analysis should lag
baseline replication. That ordering is intentional.

## Stage 0: Project Setup and Baselines

**Purpose**: lock the main project decisions before large-scale
implementation.

### Deliverables

- Finalized base population platform: `microplex`, with its current
  cross-sectional layer grounded in Enhanced CPS-style construction
- Clear boundary between `microplex` platform work and
  Social Security-specific application work
- Benchmark datasets and target tables assembled
- Initial validation harness for baseline distributions
- Staffing plan finalized for funded implementation
- Written stage-gate criteria approved internally
- Initial design-partner set and first pilot-analysis memo

### Exit Criteria

- Clear specification of which outcomes count as success in stage 1
- No unresolved ambiguity about base population or top-level scope
- At least a small set of external users or partner categories identified
  for the first validated outputs

## Stage 1: Historical Earnings Reconstruction

**Purpose**: determine whether `microplex` can be extended into a
credible longitudinal population asset for Social Security analysis.

### Core Tasks

- Harmonize PSID and related longitudinal sources
- Build at least one conservative production path for earnings-history
  reconstruction inside `microplex`
- Add the first longitudinal state variables and transition machinery to
  `microplex`
- Compare alternative model families where justified
- Validate age-earnings profiles, percentiles, mobility, AIME, and
  correlation structure
- Document where performance is strong and weak

### Deliverables

- Longitudinal `microplex` alpha with earnings histories and core
  longitudinal states
- Validation report on held-out data and external benchmarks
- Recommendation on the production longitudinal architecture

### Exit Criteria

- Longitudinal `microplex` is accurate enough to justify downstream
  benefit modeling
- Validation results are publishable and not merely anecdotal

If these conditions are not met, the project should pause rather than
proceeding mechanically.

## Stage 2: Family, Disability, Claiming, and Benefits

**Purpose**: turn longitudinal `microplex` into a credible Social
Security analysis dataset.

### Core Tasks

- Freeze the minimal production version of longitudinal `microplex`
  chosen at the end of stage 1
- Implement family structure and marital histories needed for auxiliary
  benefits
- Add disability and mortality transitions
- Specify a minimal but concrete claiming model
- Run synthetic records through PolicyEngine-US benefit logic
- Validate beneficiary counts, average benefits, and key distributions

### Deliverables

- Benefit-ready synthetic panel
- Validation report for beneficiary types and benefit amounts
- Replication of a small set of standard baseline tables
- Pilot baseline or reform analyses with clear external use cases

### Exit Criteria

- Benefit results are close enough to published benchmarks to support
  exploratory reform analysis
- Known weaknesses are documented and bounded

## Stage 3: Forward Projection and Reform Analysis

**Purpose**: move from longitudinal `microplex` plus a validated Social
Security layer to a projected dynamic model that can analyze reform
packages.

### Core Tasks

- Add forward aging and cohort entry
- Align near-term and long-term projections to published assumptions
- Test drift control and dynamic calibration
- Replicate selected published reform analyses
- Quantify uncertainty and sensitivity

The operational detail behind this stage now lives in
[operationalizing-mortality-and-projection-drift.md](operationalizing-mortality-and-projection-drift.md).

### Deliverables

- Forward projection pipeline
- Baseline replication against published projection targets
- Reform-analysis workflows for a defined set of policy packages
- External pilot use of validated baseline or reform workflows

### Exit Criteria

- Baseline projection quality is strong enough that reform outputs are
  interpretable
- The model can explain its own uncertainty rather than hiding it
- At least limited outside testing shows the validated outputs are
  legible and useful to non-team users

## Stage 4: Public Productization

**Purpose**: expose validated capabilities to outside users.

### Core Tasks

- Build API endpoints for baseline and reform analysis
- Build a focused public interface
- Publish tutorials, methods notes, and validation artifacts
- Define product boundaries clearly so users know what is and is not
  reliable

### Deliverables

- Public API
- Public web interface
- User documentation
- Public release materials and examples

### Exit Criteria

- External users can reproduce headline examples
- The interface is narrower than the full model, but more reliable
- Validation artifacts are published alongside the product

## Cross-Cutting Deliverables

Throughout the project:

- maintain versioned validation reports
- preserve reproducible data-processing pipelines where licensing
  permits
- document model decisions and reversals
- preserve the separation between reusable `microplex` infrastructure
  and Social Security-specific application code
- collect external review from domain experts

## Adjacent Extension Track

The core project is Social Security-first. However, the architecture
should preserve a path to adjacent modules that can share the synthetic
panel, especially:

- SSI-rich adequacy analysis
- retirement adequacy with wealth and pensions
- long-term care and caregiving policy

The most plausible first LTC step is a static, state-specific rules
pilot rather than immediate national dynamic reform scoring. That kind
of pilot can prove out the policy-rules layer, scenario outputs, and
user demand while the harder national dynamic LTC build remains a later
phase.

Those extensions should follow, not precede, validation of the Social
Security core.

## Summary

The roadmap is deliberately conservative because the main risk in this
project is not coding speed. It is false confidence. A public dynamic
Social Security model becomes valuable by surviving explicit
validation gates, not by reaching a web launch quickly.
