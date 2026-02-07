# Methodology

## Overview

This chapter describes our technical approach to building a synthetic longitudinal panel dataset suitable for dynamic Social Security microsimulation. The core challenge is creating realistic lifetime earnings trajectories and demographic transitions while maintaining cross-sectional accuracy and computational feasibility.

## Methodology Flow

The following diagram illustrates the high-level data flow through our synthetic panel construction process:

```{mermaid}
flowchart TD
    subgraph inputs["Input Data Sources"]
        CPS["CPS ASEC<br/>(Cross-sectional)"]
        PSID["PSID<br/>(Longitudinal)"]
        SSA["SSA Statistics<br/>(Calibration targets)"]
    end

    subgraph processing["Panel Construction"]
        QRF["Train QRF Models<br/>on PSID"]
        IMPUTE["Impute Earnings<br/>Histories to CPS"]
        DEMO["Model Demographic<br/>Transitions"]
        CAL["Gradient Descent<br/>Calibration"]
    end

    subgraph outputs["Outputs"]
        PANEL["Synthetic<br/>Longitudinal Panel"]
        PE["PolicyEngine-US<br/>Benefit Calculations"]
        WEB["Web Interface<br/>& API"]
    end

    PSID --> QRF
    QRF --> IMPUTE
    CPS --> IMPUTE
    IMPUTE --> DEMO
    DEMO --> CAL
    SSA --> CAL
    CAL --> PANEL
    PANEL --> PE
    PE --> WEB
```

## Conceptual Framework

Our methodology combines three complementary approaches:

1. **Statistical Matching**: Transfer longitudinal structure from PSID to CPS
2. **Machine Learning Imputation**: Quantile regression forests for conditional distributions
3. **Gradient Descent Calibration**: Reweighting to match external targets

This synthesis produces a synthetic panel with:
- Representative sample from public survey data (size TBD during proof of concept)
- Longitudinal dynamics (from PSID): realistic earnings trajectories
- External validity (from calibration): matches administrative aggregates
- Computational efficiency (from pre-generation): fast policy analysis

### How This Differs from Existing Models

**vs. DynaSim**: We use public survey base (vs. SIPP with restricted access), QRF imputation (vs. traditional regression), and optimization-based calibration (vs. iterative alignment). Fully reproducible with public data.

**vs. MINT**: We construct fully synthetic panel (vs. matched administrative data). Trade-off: MINT has actual earnings for older cohorts, but isn't publicly replicable. Our approach sacrifices that accuracy for full transparency.

**vs. CBOLT**: We maintain individual-level detail (vs. representative agents), enabling distributional analysis.

See the [Existing Models](existing-models.md#panel-construction-methodology-comparison) chapter for detailed comparison of panel construction and calibration methods.

## Phase 1: Base Year Cross-Section

### Starting Point: Enhanced CPS

We will use PolicyEngine's Enhanced CPS (ECPS) as the base cross-sectional dataset. This decision reflects strong reviewer consensus (Sabelhaus, Fichtner, Ghenis self-review) and several practical advantages:

1. **Proven methodology**: ECPS has already solved the cross-sectional income underreporting problem using the same tools we will apply longitudinally
2. **Integration**: Seamless connection to PolicyEngine-US's existing tax-benefit calculations
3. **Credibility**: Builds on demonstrated success rather than restarting from scratch
4. **Sample size**: ~200,000 individuals provides adequate statistical power for national and state-level analysis

The ECPS improves upon raw CPS through:

**Income Imputation**: Filling missing income components using `microimpute`

**Benefit Underreporting**: Correcting for survey underreporting of transfer income

**Tax Unit Construction**: Creating tax filing units from household structure

**Calibration**: Reweighting to match ~2,800 IRS, Census, and SSA administrative targets

The proof-of-concept phase will validate that ECPS + longitudinal imputation works, rather than re-litigating which base dataset to use. If computational constraints arise with the full ~200,000 sample, we can use L0 regularization to select a representative subsample (e.g., 100,000 individuals) that maintains target accuracy.

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

### Imputation approach: Quantile deep neural networks

Our primary approach uses **zero-inflated quantile deep neural networks (ZI-QDNN)** to predict conditional distributions of past/future earnings. The proof of concept will also evaluate quantile regression forests (QRF) and normalizing flows (Conditional MAF) as alternatives, but early experiments on survey panel data suggest ZI-QDNN outperforms both for this problem.

**Why ZI-QDNN**:
- **Handles zero-inflation natively**: A dedicated logistic head learns P(zero earnings | features) separately from the continuous earnings distribution. This is critical because many person-years have zero earnings (labor force non-participation, unemployment, disability), and standard quantile methods struggle with point masses at zero.
- **Predicts full conditional distribution**: The quantile head outputs conditional quantiles (5th through 95th) for non-zero values, preserving heterogeneity.
- **Faster than alternatives**: In head-to-head experiments on SIPP panel data, ZI-QDNN achieved 3× better trajectory coverage than GRU-based recurrent models while training 2× faster and generating 1.4× faster.
- **Log-space stability**: The quantile head operates in log-space for numerical stability with the wide range of earnings values (from near-zero to hundreds of thousands).
- **Simpler architecture than normalizing flows**: Shared hidden layers with two output heads (zero classification + conditional quantiles) rather than the invertible transformations required by MAF, making training more stable and faster.

**Architecture**:
```
Input (features) → Shared hidden layers (3 × 128 ReLU)
                        ├→ Zero head: Linear → P(zero | features)
                        └→ Quantile head: Linear → ReLU → Linear → τ quantiles (log-space)
```

The loss function combines binary cross-entropy for the zero head with pinball (quantile) loss for the non-zero conditional distribution. At generation time, the model first samples whether a value is zero, then (if non-zero) samples from the learned conditional quantile distribution.

**Why this matters for Social Security**: The zero-inflation structure directly maps to the core modeling challenge. Workers move in and out of the labor force—years with zero covered earnings are common and critical for AIME calculation (since AIME uses the 35 highest years, zero-earnings years only matter if they crowd out higher-earnings years). Getting the zero/non-zero pattern right is arguably more important than getting the exact earnings level right in non-zero years.

**Comparison to QRF** (still evaluated in proof of concept):
- QRF is more interpretable (feature importance, quantile predictions are inspectable)
- QRF is non-parametric and proven for distributional imputation in the Enhanced CPS
- QRF may perform better with smaller training data (PSID has ~10,000 families vs. millions for typical deep learning)
- The proof of concept will compare both on held-out PSID data; if PSID sample size is limiting, QRF may still win despite ZI-QDNN's architectural advantages

**Normalizing flows** (also evaluated): Conditional MAF can learn full joint distributions across multiple target variables simultaneously, which could enable imputing 5–10 years of earnings jointly rather than one year at a time. This may improve correlation structure across years at the cost of training complexity.

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

**Example Implementation**:

```python
from microimpute import QuantileRegressionForest
import numpy as np

# Train QRF model for earnings at age 35
qrf_age35 = QuantileRegressionForest(n_estimators=100)
qrf_age35.fit(
    X=psid_data[["current_earnings", "education", "sex", "race", "birth_year"]],
    y=psid_data["earnings_age_35"]
)

# Predict conditional quantiles for CPS sample
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
predictions = qrf_age35.predict_quantiles(cps_features, quantiles=quantiles)

# Sample from conditional distribution for each individual
def sample_earnings(row, predictions, quantiles):
    """Sample earnings from predicted conditional distribution."""
    u = np.random.uniform()  # Random quantile
    return np.interp(u, quantiles, predictions.loc[row.name])

cps_data["earnings_age_35"] = cps_data.apply(
    lambda row: sample_earnings(row, predictions, quantiles), axis=1
)
```

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

**Spousal Matching** (per Sabelhaus feedback, this deserves more methodological detail):
- When marriage occurs, match to appropriate spouse using a distance-based matching algorithm on age, education, and earnings
- Incorporate assortative mating patterns from CPS married couples and PSID marital transitions
- Preserve spousal earnings correlation (which drives household Social Security wealth)
- Consider a hierarchical synthesis approach: generate household structures top-down (household composition first, then person-level attributes conditional on household type), which naturally preserves realistic family structure and avoids impossible combinations
- Validate matched couple characteristics against CPS distributions of age gaps, educational homogamy, and dual-earner patterns

### Fertility

Children affect earnings (especially for women) and dependency benefits.

**Approach**:
- Estimate birth hazard models from PSID
- Predictors: age, marital status, education, existing children
- Simulate births year-by-year
- Match to fertility rates from NVSS and Census projections

### Disability

SSDI is a major component of Social Security spending (~$150B annually, 8.5 million beneficiaries).

**Approach**:
- Estimate disability onset hazard calibrated to SSA DI incidence rates, which range from ~0.2% at age 25 to ~1.5% at age 60, with higher rates for men than women
- Predictors: age, sex, occupation, prior earnings, health status
- Recovery rates declining with disability duration: approximately 10% in year 1, 5% in year 2, declining to ~3% for longer durations, calibrated to SSA DI termination data
- Age effects on recovery (younger workers more likely to recover)
- Match to SSA disability incidence, prevalence, and termination rates by age and sex

**Additional nuance** (per Sabelhaus and Fichtner feedback):
- Model pre-disability earnings decline (3–5 years of declining earnings before formal SSDI receipt, well-documented in literature)
- Distinguish between disability onset and SSDI award (not all disabled workers receive benefits—application and award rates vary by age and severity)
- Model interaction between disability and early retirement claiming (some disabled workers claim retirement benefits at 62 rather than applying for SSDI)
- Track the 24-month waiting period before Medicare eligibility for SSDI recipients

These refinements are important for policy analysis of disability-related reforms and their interaction with retirement benefit claiming.

### Mortality

Accurate mortality modeling is essential for:
- Survivor benefits
- Lifetime benefit calculations
- Long-run projections
- Distributional analysis (differential mortality by income creates regressive lifetime benefit patterns)

**Approach**:
- Use SSA period life tables as base, providing age-sex-specific mortality probabilities (qx values) for ages 0–119
- Adjust for differential mortality by earnings quintile and education, drawing on Opportunity Insights life expectancy data showing that the top 1% of earners live ~15 years longer than the bottom 1%
- Implement mortality improvements over time per SSA Trustees intermediate assumptions
- Validate against population counts by age and overall life expectancy

**Policy importance** (per Sabelhaus feedback):
Life expectancy gaps between high and low earners have widened substantially in recent decades. This means higher earners receive benefits over more years, generating higher lifetime returns despite the progressive benefit formula. This differential mortality effect is critical for evaluating:
- Retirement age increases (disproportionately affect shorter-lived lower-income workers)
- Lifetime progressivity of the system (may be less progressive than the benefit formula suggests)
- Racial equity (Black men have lower life expectancy → fewer years of benefits)

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

We have N individuals with survey weights w_i. We want new weights w_i* that:

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

**Example Implementation**:

```python
from microcalibrate import GradientDescentCalibrator
import pandas as pd

# Define calibration targets
targets = {
    "population_total": 330_000_000,
    "ss_beneficiaries_65_69": 12_200_000,
    "ss_beneficiaries_70_74": 11_800_000,
    "mean_earnings_age_35_male": 65_000,
    "mean_earnings_age_35_female": 52_000,
}

# Create target matrix from synthetic panel
X = pd.DataFrame({
    "population_total": 1,
    "ss_beneficiaries_65_69": (panel["age"] >= 65) & (panel["age"] < 70) & panel["ss_recipient"],
    "ss_beneficiaries_70_74": (panel["age"] >= 70) & (panel["age"] < 75) & panel["ss_recipient"],
    # ... additional targets
})

# Calibrate weights
calibrator = GradientDescentCalibrator(
    loss="chi_squared",  # Minimize chi-squared distance from original weights
    max_iterations=1000,
    tolerance=0.01  # Match targets within 1%
)

calibrated_weights = calibrator.fit(
    X=X,
    targets=pd.Series(targets),
    initial_weights=panel["original_weight"]
)

panel["calibrated_weight"] = calibrated_weights
```

### Alternative and complementary calibration methods

Beyond gradient descent, several additional calibration approaches may prove useful for different aspects of the longitudinal calibration problem:

**Iterative Proportional Fitting (IPF/raking)**: Well-established method for matching categorical marginal distributions. May be useful for demographic calibration (age-sex-education distributions) where targets are expressed as cell counts.

**Entropy balancing**: Minimizes entropy distance from original weights while exactly matching moment conditions. Produces smoother weight distributions than unconstrained gradient descent and has strong theoretical properties.

**Sparse L0/L1 reweighting**: Instead of adjusting all weights continuously, select a representative subset of observations using L0 regularization.

**Motivation for sparse methods**:
- Computational efficiency (smaller dataset for web deployment)
- Interpretability (which observations are most informative)
- Avoids extreme weight adjustments that can degrade distributional properties

**Method**:
- L0 regularization via gradient descent or iterative reweighted L1
- Selects subset that best matches all targets
- Can combine with continuous reweighting (sparse selection first, then fine-tune)

The proof of concept will evaluate which calibration methods perform best for cross-sectional targets (where gradient descent is proven) versus longitudinal targets (where the loss surface may be more complex and alternative methods could improve convergence).

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

## Summary

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
