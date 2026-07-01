# Evaluation and model selection

## Why this needs its own chapter

Calibration targets tell us what the model should match. They do not, by
themselves, tell us how we choose among competing longitudinal
architectures.

That distinction matters for this project. The first major technical
question is not whether we can write down a plausible earnings process.
It is whether we can choose a public construction method that is good
enough to justify the next stage of the build. This chapter therefore
defines the evaluation framework for deciding:

- which earnings architecture becomes the production path for
  longitudinal `populace`
- whether the resulting panel is good enough for benefit calculation
- whether the full project has earned the right to advance from stage 1
  to stage 2

In other words, this chapter is about model selection and stage-gate
discipline, not just fit diagnostics.

## The core evaluation principle

The project should not select models on generic imputation quality
alone. The winning method has to perform well on the variables and
failure modes that matter for Social Security.

That implies a layered evaluation rule:

1. **Distributional plausibility**:
   does the synthetic panel resemble observed earnings and work patterns?
2. **Path realism**:
   do full career paths look like plausible work histories rather than
   stitched-together age points?
3. **Program relevance**:
   when the paths are run through PolicyEngine-US, do they produce
   credible insured status, AIME, benefits, and beneficiary types?
4. **Operational viability**:
   is the method stable, explainable, fast enough, and reproducible
   enough to become part of a public platform?

Any method that fails layer 3 should be considered non-production-ready
even if it looks strong on layers 1 and 2.

## Candidate models to evaluate

The project should maintain a clear distinction between benchmark models
and likely production candidates.

| Family | Role in evaluation | Why it stays in scope |
|---|---|---|
| **QRF** | Baseline comparator | Interpretable benchmark and direct comparison to older sequential PE-style methods |
| **ZI-QRF** | Stronger baseline comparator | Isolates the value of explicit zero-inflation without changing the broader modeling family |
| **ZI-QDNN** | Serious candidate | Zero-inflated neural distribution model; plausible production option if refreshed evals support it |
| **ZI-MAF or related flow models** | Serious candidate | Candidate for all-at-once path generation where cross-age correlation structure matters; evaluated against the QRF baseline, not assumed superior |
| **Pathwise sequence models** | Serious candidate | Best fit to the project's long-run architecture if they pass validation |
| **Factorized annual process with calibrated residuals** | Structural benchmark | Useful if a transparent annual-state model beats pure black-box pathwise generation on policy metrics |

The proposal should not prejudge the winner. It should prejudge the
decision rule.

## Evaluation objects

The build should evaluate the project on four distinct objects.

### 1. Variable-level conditional predictions

These are the most local checks:

- zero vs non-zero work outcomes
- conditional positive earnings
- taxable-maximum exceedance
- covered vs noncovered work status

These checks are useful, but insufficient by themselves.

### 2. Full historical paths

These evaluate whether the model can generate realistic longitudinal
careers:

- full age-earnings trajectories
- interruption timing
- persistence of earnings rank
- transitions into and out of zero-covered-earnings years

This is where pathwise models should show their advantage over
age-point baselines.

### 3. Benefit-facing derived variables

These are the first variables that directly speak Social Security's
language:

- quarters of coverage
- years of covered work
- highest-35-years sum
- AIME
- PIA under current law
- replacement rates

This layer is the bridge between earnings-process evaluation and policy
evaluation.

### 4. Full policy outputs

These are the outputs that determine whether the panel is good enough to
support actual policy analysis:

- beneficiary counts by type
- average and percentile benefit amounts
- claiming-age distributions where modeled
- disabled-worker counts and conversions to retirement
- aggregate benefits and tax base

The project should not claim production readiness without clearing this
layer.

## Datasets and holdout design

The evaluation design should make leakage difficult and failure visible.
That means not relying on one train/test split or one aggregate score.

### Panel-data splits

For PSID and other longitudinal sources, the project should use:

- **person holdouts**:
  hold out entire people, not random rows
- **cohort holdouts**:
  hold out birth-cohort bands to test generalization across cohorts
- **late-career holdouts**:
  hold out older workers and retirees, where AIME and claiming fit are
  hardest

The point is to test whether the model can generalize to new people and
new cohort compositions, not merely interpolate within observed
person-years.

### Cross-sectional anchor tests

Because the final use case starts from a cross-sectional `populace`
record, the project should also simulate that workflow directly:

1. collapse a held-out panel person to a pseudo-cross-section at a
   chosen age
2. feed only the information available in the base-year cross section
   into the candidate model
3. reconstruct the person's prior earnings path
4. compare reconstructed quantities with the held-out truth

This test is closer to the actual product than a standard predictive
validation on PSID rows.

### Policy-output holdouts

For beneficiary and benefit outputs, the project should hold out target
tables and cohorts rather than calibrating to everything at once.

That means:

- some targets remain calibration targets
- some become pure validation targets
- the holdout set changes as the benchmark evolves

Otherwise the model can appear stronger than it really is simply because
the evaluation set has been absorbed into calibration.

## Common experimental protocol

All candidate models should be evaluated under a common protocol.

### Shared inputs

Every model should start from:

- the same base cross-sectional records
- the same conditioning variables, where architecture permits
- the same panel training source
- the same weight handling
- the same cohort definitions

If one model gets richer conditioning variables or cleaner sample
construction than another, the evaluation should say so explicitly.

### Multiple seeds

All stochastic candidates should be run across multiple random seeds.
The proposal should report:

- mean performance
- standard deviation or standard error
- failure cases

This is especially important for neural candidates. A model that wins on
mean fit but is unstable across seeds may still be the wrong production
choice.

### Weight-aware evaluation

Since the final platform is population-representative rather than merely
sample-representative, the benchmark should report both:

- unweighted fit within the panel source
- weighted fit against external population targets

This prevents a model from winning only because it matches the quirks of
the training sample.

### Provenance logging

Every evaluation run should save:

- code version
- model config
- data vintage
- calibration target set
- seed
- run time

The project should treat benchmark reproducibility as part of the
deliverable.

## Primary metrics for model selection

The decisive metrics should be the ones that determine Social Security
outcomes.

### Earnings-process metrics

| Metric | Why it matters |
|---|---|
| Covered-work share by age-sex-education | Determines insured status and benefit eligibility |
| Positive covered-earnings distribution | Determines AIME and benefit levels |
| Taxable-maximum incidence | Important for high earners and tax-base realism |
| Zero-covered-years distribution | Central for top-35-years calculations |
| Five-year earnings mobility | Measures lifetime dynamics rather than static fit |
| Cross-age earnings correlation | Tests whether the model preserves long-run rank and persistence |

### Benefit-facing derived metrics

| Metric | Why it matters |
|---|---|
| Quarters of coverage accuracy | Determines retirement and disability eligibility |
| Highest-35-years sum | Direct precursor to AIME |
| AIME distribution | Closest summary of earnings history from Social Security's perspective |
| PIA distribution | Tests benefit arithmetic on top of the generated histories |
| Replacement-rate distribution | Connects lifetime earnings to retirement outcomes |

### Policy-output metrics

| Metric | Why it matters |
|---|---|
| Beneficiary counts by type | Basic system realism |
| Average benefits by type | Tests level accuracy |
| Benefit percentiles | Tests distributional accuracy |
| Aggregate covered earnings and payroll tax base | Fiscal realism |
| Claiming-age distribution where modeled | Needed for reform analysis and timing realism |

## Secondary metrics

A model should not win because it looks cleaner on generic benchmarks
while failing on the policy metrics above. But secondary metrics still
matter.

### Statistical quality metrics

These may include:

- KS or Wasserstein-style distribution distances
- PRDC-style coverage or related synthetic-data metrics
- zero-fraction error
- correlation preservation

They are useful as diagnostics, especially for comparing `populace`
candidate families, but they are not the final decision rule.

### Operational metrics

The production candidate must also be practical:

- training time
- generation time
- memory footprint
- ease of re-running
- brittleness to hyperparameters
- explainability of failure modes

For a public model, operational viability matters more than it would in
a closed internal pipeline.

## Proposed decision rule

The proposal should state the model-selection rule plainly.

### Gate 1: Must-pass criteria

A candidate should be eliminated from production consideration if it
fails any of the following:

- persistent instability across seeds
- major miss on covered-work shares
- major miss on zero-covered-years distribution
- major miss on AIME distribution
- major miss on beneficiary counts by type after benefit calculation

This is the "do not advance" gate.

### Gate 2: Comparative scorecard

For candidates that clear Gate 1, score them on:

- earnings-process fit
- benefit-facing fit
- policy-output fit
- stability
- runtime and reproducibility
- architectural alignment with longitudinal `populace`

The scorecard should be reported as a table, not just prose.

### Gate 3: Production recommendation

The winning architecture should be the one that:

1. clears the must-pass thresholds
2. performs best on the Social-Security-specific metrics
3. is simple enough to explain and maintain publicly

That rule leaves open whether the winner is ZI-QDNN, ZI-MAF, a broader
`populace` sequence model, or a more transparent annual-state process.

## Suggested numeric thresholds for stage 1

The exact tolerances should be refined during implementation, but the
proposal should not avoid numeric commitments altogether.

| Metric | Draft stage-1 threshold |
|---|---|
| Covered-work share by age-sex | within 2 percentage points |
| Share at taxable maximum | within 1 percentage point |
| Zero-covered-years in top 35 | within 0.5 years on average |
| Five-year mobility matrix major cells | within 3 percentage points |
| Cross-age earnings correlation profile | within 0.05 correlation points |
| AIME key percentiles | within 5 percent |
| Beneficiary counts by type | within 1-2 percent |
| Average benefits by type | within 2-3 percent |

The point of this table is not false precision. It is accountability.

## Benchmark outputs the project should publish

Each major benchmark round should produce:

- a machine-readable results table
- a concise written benchmark memo
- per-model cards describing strengths, weaknesses, and failure modes
- a frozen decision note explaining whether the recommended production
  path changed

At least one benchmark output should be designed for outside reviewers,
not just internal iteration.

## How this connects to funding

The evaluation framework is part of the pitch, not a back-office detail.

Funders should be able to see that the project will not:

- quietly pick a convenient method and build around it
- declare success using only generic synthetic-data metrics
- move to a public product before the earnings machinery has passed
  policy-relevant tests

Instead, the proposal should show that a funded year one buys a real
decision:

- whether a public lifetime-earnings construction is credible enough
- which architecture deserves continued investment
- what the residual limitations are even if the answer is "yes"

## Relationship to the refreshed populace evaluations

The `populace` imputation evaluations should feed directly into this
chapter, but they should not be the only evidence.

The right interpretation is:

- refreshed `populace` evals help narrow the candidate set
- Social-Security-specific benchmarks decide the production winner
- the proposal should remain architecture-agnostic until both pieces are
  in hand

That makes the proposal both more honest and more robust.

## Bottom line

This project needs a public benchmark regime, not just a preferred
architecture.

The winning model should be the one that best reconstructs lifetime
earnings in the specific ways Social Security cares about, while
remaining stable and transparent enough to support an open platform.
That is the decision this chapter is designed to make.
