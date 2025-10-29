# Methodology

## Overview

This chapter describes our technical approach to building a synthetic longitudinal panel dataset suitable for dynamic Social Security microsimulation. The core challenge is creating realistic lifetime earnings trajectories and demographic transitions while maintaining cross-sectional accuracy and computational feasibility.

## Conceptual Framework

Our methodology combines three complementary approaches:

1. **Statistical Matching**: Transfer longitudinal structure from PSID to CPS
2. **Machine Learning Imputation**: Quantile regression forests for conditional distributions
3. **Gradient Descent Calibration**: Reweighting to match external targets

This synthesis produces a synthetic panel with:
- Large sample size (from CPS): ~200,000 individuals
- Longitudinal dynamics (from PSID): realistic earnings trajectories
- External validity (from calibration): matches administrative aggregates
- Computational efficiency (from pre-generation): fast policy analysis

## Phase 1: Base Year Cross-Section

### Starting Point: Enhanced CPS

We begin with PolicyEngine's Enhanced CPS (eCPS), which already improves upon raw CPS through:

**Income Imputation**: Filling missing income components using `microimpute`

**Benefit Underreporting**: Correcting for survey underreporting of transfer income

**Tax Unit Construction**: Creating tax filing units from household structure

**Calibration**: Reweighting to match IRS totals and other administrative data

This provides a high-quality cross-sectional base representing the current population.

### Adding Historical Variables

For dynamic modeling, we need variables not in CPS:

**Education**: Already in CPS, but we validate and impute where missing

**Occupation and Industry**: For earnings trajectory modeling

**Health and Disability**: Impute from HRS and NHIS using statistical matching

**Potential Variables**: Predict latent variables that govern dynamics:
- Earnings potential (distinct from current earnings)
- Health status (not just disability)
- Labor force attachment

These "latent" variables will drive longitudinal transitions even when not directly observed.

## Phase 2: Earnings History Imputation

### The Core Challenge

Social Security benefits depend on 35 highest years of earnings, but CPS only observes current year. We need to impute:

- Past earnings for current workers (ages 18-70)
- Future earnings for younger workers (for projections)
- Full lifetime profiles that respect:
  - Age-earnings life cycle
  - Earnings mobility (but not too much)
  - Educational differentials
  - Cohort effects
  - Realistic variance

### Quantile Regression Forest Approach

We use quantile regression forests (QRF) to predict conditional distributions of past/future earnings:

**Why QRF**:
- Predicts full conditional distribution, not just mean
- Non-parametric (captures complex non-linearities)
- Handles high-dimensional predictors
- Preserves heterogeneity across distribution
- Proven performance for distributional imputation

**Training Data**: PSID (1968-present)

**Features** (X variables):
- Current earnings
- Age
- Sex
- Race/ethnicity
- Education
- Marital status
- Number of children
- Occupation
- Industry
- State
- Year (cohort effects)

**Target** (Y variables):
- Earnings at age 25, 30, 35, ..., 65 (separate models)
- Earnings growth rates over 5-year periods
- Career patterns (years with zero earnings)

**Prediction Approach**:

For each CPS individual at age A with current earnings E:

1. Predict conditional distribution of earnings at each age using QRF trained on PSID
2. Sample from predicted distribution to generate earnings history
3. Ensure consistency:
   - If currently age 45, sampled age 45 earnings should match current E
   - Smooth unrealistic year-to-year jumps
   - Respect Social Security maximum taxable earnings
4. Generate multiple imputations for uncertainty quantification

### Year-by-Year Training Strategy

Rather than training one model for all ages, we train separate models for each age:

**Model 25**: Predict earnings at age 25 | features at age 25+

**Model 30**: Predict earnings at age 30 | features at age 30+

...

**Model 65**: Predict earnings at age 65 | features at age 65+

This approach:
- Captures age-specific patterns
- Allows different predictors to matter at different ages
- Prevents impossible trajectories (e.g., starting at $200k at age 22)
- Naturally handles life-cycle earnings profiles

### Cohort-Specific Modeling

Earnings profiles differ across birth cohorts due to:
- Secular wage growth
- Educational expansion
- Industry composition shifts
- Female labor force participation trends

We incorporate cohort effects by:

**Cohort as Feature**: Include birth year in QRF features

**Cohort-Specific Models**: Train separate models by decade of birth if sample sufficient

**Trend Adjustment**: Adjust PSID training data to reflect CPS cohort's economic environment

### Validation of Imputed Histories

We validate imputed earnings histories against multiple benchmarks:

**Age-Earnings Profiles**: Compare average earnings by age to SSA data

**Earnings Distribution**: Check percentiles match SSA earnings statistics

**Earnings Mobility**: Verify transition matrices match PSID quintile mobility

**AIME Distribution**: Distribution of Average Indexed Monthly Earnings should match SSA

**Correlation Structure**: Ensure earnings at different ages have realistic correlations

**Variance Components**: Between-person vs. within-person variance should match PSID

## Phase 3: Demographic Transitions

### Marriage and Divorce

Social Security spousal and survivor benefits require accurate marital history modeling.

**Approach**:
- Estimate discrete-time hazard models from PSID:
  - Marriage entry (for never-married)
  - Divorce
  - Remarriage
- Predictors: age, sex, race, education, earnings, children
- Simulate transitions year-by-year
- Match to marital status distribution at each age (calibration target)

**Spousal Matching**:
- When marriage occurs, match to appropriate spouse
- Use assortative mating patterns from CPS and PSID
- Match on age, education, earnings (with realistic noise)

### Fertility

Children affect earnings (especially for women) and dependency benefits.

**Approach**:
- Estimate birth hazard models from PSID
- Predictors: age, marital status, education, existing children
- Simulate births year-by-year
- Match to fertility rates from NVSS and Census projections

### Disability

SSDI is a major component of Social Security spending.

**Approach**:
- Estimate disability onset hazard from PSID and HRS
- Predictors: age, sex, occupation, prior earnings, health status
- Recovery rates (low but non-zero)
- Match to SSA disability incidence and prevalence rates

### Mortality

Accurate mortality modeling is essential for:
- Survivor benefits
- Lifetime benefit calculations
- Long-run projections

**Approach**:
- Use SSA cohort life tables as base
- Adjust for differential mortality by earnings/education (well-documented in literature)
- Implement mortality improvements over time per SSA assumptions
- Validate against population counts by age

## Phase 4: Forward Projection

Once we have complete histories through base year, we project forward:

### Earnings Projection

For each individual, project future earnings based on:

**Deterministic Component**:
- Age-earnings profile
- Cohort trends
- Aggregate wage growth (per SSA assumptions)

**Stochastic Component**:
- Idiosyncratic shocks (from PSID variance)
- Employment transitions (entry/exit from labor force)
- Disability onset (earnings drop)

**Behavioral**:
- Retirement decision (endogenous based on Social Security rules)
- Labor supply responses to policy reforms (optional extension)

### Demographic Projection

Continue simulating:
- Marriage/divorce transitions
- Fertility (for younger cohorts)
- Disability onset
- Mortality

Match to SSA Trustees intermediate assumptions for aggregate demographic rates.

### Population Growth

New birth cohorts enter model each year:

**Approach**:
- Generate initial cohort from CPS for age 18
- Assign initial education based on trends
- Initialize earnings potential
- Project forward as cohort ages

## Phase 5: Gradient Descent Calibration

After imputation and projection, we calibrate the full synthetic panel.

### Reweighting Framework

We have ~200,000 individuals with CPS survey weights w_i. We want new weights w_i* that:

**Minimize**:
```
Sum_i d(w_i*, w_i)
```

Where d() is a distance function (e.g., Chi-squared distance)

**Subject to**:
```
Sum_i w_i* * X_ik = T_k  for all targets k
```

Where X_ik is characteristic k of individual i, T_k is target k

### Gradient Descent Implementation

**Algorithm**:

1. Initialize: w* = w (CPS weights)
2. Compute: Current weighted totals for all targets
3. Calculate: Gradient of objective with respect to w*
4. Update: w* in direction that reduces distance to targets
5. Project: Ensure w* > 0 and sum(w*) = population
6. Iterate: Until convergence (targets matched within tolerance)

**Advantages**:
- Handles hundreds of targets simultaneously
- Computationally efficient (scales to large datasets)
- Transparent and interpretable
- Proven in PolicyEngine's enhanced CPS construction

### L0 Regularization for Sample Selection

Optional extension: Instead of reweighting, select subset of observations with 0/1 weights.

**Motivation**:
- Computational efficiency (smaller dataset)
- Interpretability (which observations most informative)

**Method**:
- L0 regularization via gradient descent
- Selects subset that best matches all targets
- Can combine with continuous reweighting

See PolicyEngine's L0 package for implementation details.

### Multi-Year Calibration

For each projected year, we calibrate weights to match that year's targets:

**Cross-Sectional**: Age-sex-education distribution, earnings distribution

**Longitudinal**: Transition rates (employment, marriage, etc.)

**Fiscal**: Social Security aggregates (beneficiaries, benefits, revenues)

This prevents drift and ensures long-run accuracy.

## Phase 6: Social Security Benefit Calculation

### PolicyEngine-US Integration

Once synthetic panel is constructed, we leverage PolicyEngine-US's existing Social Security implementation:

**Variables Already Implemented**:
- `social_security_retirement`: Retirement benefits
- `social_security_disability`: SSDI benefits
- `social_security_survivors`: Survivor benefits
- `social_security_dependents`: Dependent benefits
- `taxable_social_security`: Benefit taxation

**Required Inputs** (now available from synthetic panel):
- Lifetime earnings history (35 highest years)
- Date of birth
- Retirement age
- Marital history (for spousal/survivor benefits)
- Disability status
- Number of children (for dependent benefits)

### Calculation Pipeline

For each individual in each year:

1. **Eligibility**: Check quarters of coverage
2. **AIME**: Calculate Average Indexed Monthly Earnings from history
3. **PIA**: Apply bend point formula to AIME
4. **Adjustments**:
   - Early/delayed retirement factors
   - Spousal/survivor benefit rules
   - Disability considerations
5. **Family Benefits**: Dependent and survivor benefits
6. **Taxation**: Federal income tax on benefits (integrated with broader PolicyEngine tax model)

### Reform Modeling

To analyze reforms, we modify PolicyEngine parameters:

**Benefit Formula**: Adjust bend points, replacement rates

**Retirement Age**: Change full/early retirement ages

**Taxation**: Modify benefit taxation thresholds

**Eligibility**: Change quarters required, coverage rules

**Indexing**: Alter wage indexing formulas

PolicyEngine's reform framework enables easy specification and analysis of arbitrary combinations of reforms.

## Computational Efficiency

Dynamic microsimulation can be computationally intensive. Our optimizations:

### Pre-Generation

- Generate synthetic panel once
- Store complete histories
- Policy analysis uses pre-generated panel (very fast)

### Vectorization

- All calculations vectorized using NumPy
- No Python loops over individuals
- Leverage PolicyEngine-Core's efficient simulation engine

### Selective Projection

- Only project variables needed for analysis
- Can skip detailed demographic transitions if just modeling benefit formulas

### Parallel Processing

- Panel construction parallelizable across individuals
- Multiple imputations can run in parallel

### Caching

- Cache intermediate results
- Incremental updates when only some parameters change

## Uncertainty Quantification

Multiple sources of uncertainty:

**Imputation Uncertainty**: Earnings histories are predicted, not observed

**Method**: Multiple imputation (m=5-10 imputations)

**Parameter Uncertainty**: Model parameters estimated from PSID have sampling error

**Method**: Bootstrap PSID sample, re-estimate models

**Projection Uncertainty**: Future unknowable

**Method**:
- Scenario analysis with different assumptions
- Sensitivity to wage growth, mortality, disability rates
- Confidence intervals around SSA's stochastic projections

**Model Uncertainty**: Our model is simplified reality

**Method**:
- Validation against multiple benchmarks
- Comparison to DynaSim/MINT where possible
- Transparent documentation of assumptions

## Validation Strategy

Comprehensive validation at multiple levels:

### Cross-Sectional Validation

**Base Year**:
- Age-sex-education distributions
- Earnings distributions
- Beneficiary counts and average benefits

**Match to**: CPS, SSA Administrative Data

### Longitudinal Validation

**Earnings Dynamics**:
- Age-earnings profiles by cohort
- Earnings mobility matrices
- Variance decomposition

**Match to**: PSID, Published MINT analyses

### Distributional Validation

**Benefit Distribution**:
- Percentiles of benefits by type
- Replacement rates by lifetime earnings
- Progressivity measures

**Match to**: SSA benefit statistics, Academic studies

### Fiscal Validation

**Aggregates**:
- Total benefits by type
- Total covered earnings
- Trust fund projections

**Match to**: SSA Trustees Reports

### External Validation

**Published Results**:
- Compare our reform analysis to published DynaSim results for same reforms
- Compare to CBO cost estimates where available
- Compare to academic studies using MINT

## Summary: Methodological Innovation

Our approach advances the state of practice by:

**Scale**: CPS sample size with PSID dynamics

**Transparency**: Open-source implementation with full documentation

**Flexibility**: Easy to modify assumptions and extend

**Validation**: Comprehensive validation against multiple benchmarks

**Integration**: Seamless integration with PolicyEngine tax-benefit model

**Accessibility**: Public web interface and Python API

**Efficiency**: Pre-generated panel enables fast policy analysis

**Reproducibility**: Anyone can replicate and verify

The next chapter describes the infrastructure and tools that enable this methodology.
