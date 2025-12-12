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

### Starting Point: Base Cross-Section

We will evaluate options for the base cross-sectional dataset during the proof of concept phase. One option is PolicyEngine's Enhanced CPS (ECPS), which improves upon raw CPS through:

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

## Why This Approach Will Work: The Enhanced CPS Precedent

A natural skepticism arises: can fully synthetic data really match the quality of administrative records? PolicyEngine has already answered this question affirmatively through cross-sectional analysis.

### The Enhanced CPS Achievement

PolicyEngine's Enhanced CPS (ECPS) is the only publicly available microdata file that produces accurate tax-benefit microsimulation results {cite:p}`ghenis2024`. This achievement directly validates the synthetic data approach we propose to extend to longitudinal analysis.

**The Challenge in Cross-Sectional Modeling**:
All major tax models including Tax Policy Center, Penn Wharton Budget Model, and Tax Foundation rely on the IRS Public Use File. The PUF cannot be publicly shared due to privacy restrictions, creating a reproducibility crisis where researchers cannot verify others' results or independently analyze tax policy proposals.

**Our Solution**:
PolicyEngine developed the Enhanced CPS through a rigorous two-stage methodology. Stage 1 uses quantile regression forests to impute missing or underreported variables from multiple public data sources (PUF via privacy-safe methods, SIPP, SCF, ACS) onto the CPS base. The CPS is cloned to create two copies: one filling missing variables and one replacing existing variables with more accurate imputed values. These copies are concatenated to create the Extended CPS with doubled sample size. Stage 2 applies gradient descent optimization to reweight the Extended CPS, matching over 7,000 administrative targets from IRS Statistics of Income, Census, CBO, and other sources. The optimization uses PyTorch with Adam optimizer, dropout regularization, and log-transformed weights to ensure positivity.

**Validation Results**:
Revenue estimates match Joint Committee on Taxation estimates for major tax reforms. Distributional analysis matches Tax Policy Center's published tables across income deciles. Individual-level calculations validate against actual tax returns where available. Congressional offices use PolicyEngine for actual policy analysis, demonstrating real-world credibility and accuracy. The proof is clear: synthetic data combined with modern machine learning and calibration to thousands of targets produces accuracy comparable to administrative data.

**The Technical Parallel**:
The ECPS methodology directly informs this project. Both employ quantile regression forests for distributional imputation, gradient descent optimization for multi-target calibration, validation against administrative aggregates, and full public reproducibility. The ECPS imputes cross-sectional income detail; this project will impute longitudinal earnings histories. The ECPS calibrates to 7,000+ IRS and Census targets; this project will calibrate to SSA and Census targets. The technical challenges are analogous, and we have already solved them for the cross-sectional case.

### Direct Parallel to This Project

| Dimension | Enhanced CPS (Proven) | This Project |
|-----------|----------------------|--------------|
| **Data dimension** | Cross-sectional (income) | Longitudinal (earnings history) |
| **Restricted gold standard** | IRS PUF | SSA earnings records |
| **Public data source** | CPS | CPS + PSID |
| **ML imputation** | microimpute | QRF (enhanced microimpute) |
| **Calibration** | microcalibrate (gradient descent) | microcalibrate (gradient descent) |
| **Validation targets** | IRS Statistics of Income | SSA statistics |
| **Status** | ✓ Validated, in production | Proposed |

**Key insight**: This is not an untested approach. We're extending a proven methodology from one dimension (current income) to another (lifetime earnings).

### Risk Mitigation

**ECPS development** (2+ years) taught us:
- Which ML methods work for survey imputation
- How to calibrate to hundreds of targets simultaneously
- Validation strategies that build credibility
- Common pitfalls and how to avoid them

**Advantages for this project**:
- Tools already built and tested (microimpute, microcalibrate)
- Team experienced in this exact methodology
- Credibility from ECPS success reduces skepticism
- Know what validation evidence is needed

**The harder problem**: Longitudinal imputation is actually somewhat easier than ECPS cross-sectional imputation:
- PSID has true panel structure (gold standard for training)
- Earnings dynamics well-studied in economics literature
- Strong age-earnings patterns provide structure
- Can validate against published PSID mobility matrices

In contrast, ECPS had to impute hundreds of detailed income components with complex interactions, all in a single year. Lifetime earnings trajectories have cleaner structure and better training data.

### What Could Go Wrong?

**Potential failure modes** (and why they're unlikely):

1. **"QRF imputation produces unrealistic earnings histories"**
   - Mitigation: Validate against PSID hold-out sample before applying to CPS
   - Precedent: ECPS imputation validated extensively before production use

2. **"Calibration can't match all targets simultaneously"**
   - Mitigation: ECPS already handles 100+ simultaneous targets
   - Tool: microcalibrate proven on harder problem (cross-sectional detail)

3. **"Synthetic panel too different from MINT/DynaSim"**
   - Mitigation: Extensive validation against published results
   - Note: MINT/DynaSim also synthetic for younger cohorts

4. **"Computational burden too high"**
   - Mitigation: Pre-generate panel once, reuse for many policies
   - ECPS generation ~8 hours; similar expected here

**Success probability**: Given ECPS precedent, strong team, and proven tools, success is highly likely.

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
