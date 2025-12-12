# Data Sources

## Overview

Building a dynamic Social Security microsimulation model requires multiple data sources that capture cross-sectional population characteristics, longitudinal earnings dynamics, and demographic transitions. This chapter describes the primary data sources we will use.

## Primary Survey Data Sources

### Current Population Survey (CPS)

**Description**: The CPS is the primary source of labor force statistics for the U.S. population. The Annual Social and Economic Supplement (ASEC) provides detailed income and demographic data.

**Key Features**:
- Large sample: ~95,000 households, ~200,000+ individuals
- Annual cross-section (March ASEC)
- Detailed income components including Social Security benefits
- Demographics: age, sex, race, ethnicity, education, family structure
- Employment and earnings information
- Geographic detail: state, metro area

**Strengths for Our Purpose**:
- Large sample enables state-level and demographic subgroup analysis
- Official poverty statistics source ensures public validation
- Available annually since 1962 with consistent time series
- Public use files freely available
- Well-documented and widely used

**Limitations**:
- Cross-sectional only (no panel structure)
- Top-coded high incomes
- Some income underreporting
- No asset or wealth data
- Limited earning history (only current year)

**Our Use**:
- Base year cross-sectional distributions
- Validation of age-earnings profiles
- Calibration targets for population characteristics
- Starting point for synthetic panel construction

### Panel Study of Income Dynamics (PSID)

**Description**: The longest-running longitudinal household survey in the world, following families since 1968.

**Key Features**:
- True panel structure: same families over decades
- Biennial from 1997 (annual before)
- ~9,000 families currently
- Intergenerational linkages
- Detailed earnings histories
- Wealth supplements
- Comprehensive demographic transitions

**Strengths for Our Purpose**:
- **Critical for longitudinal modeling**: True earnings trajectories over careers
- Demographic transitions: marriage, divorce, childbearing, mortality
- Intergenerational links enable family modeling
- Long time series captures cohort differences
- Wealth data for comprehensive retirement security analysis

**Limitations**:
- Smaller sample than CPS (not suitable for state-level detail)
- Sample attrition over time
- Top-coding of high incomes
- Public use files have restricted geographic detail

**Our Use**:
- **Primary source for earnings transition matrices**
- Training data for quantile regression forests
- Validation of lifetime earnings distributions
- Demographic transition modeling
- Calibration of earnings mobility

### Survey of Income and Program Participation (SIPP)

**Description**: Census Bureau survey designed to measure income and program participation, with longitudinal panel structure.

**Key Features**:
- Panel structure: follows individuals for 3-4 years
- Monthly income data
- Detailed program participation (SNAP, SSI, Social Security, etc.)
- Wealth modules
- Sample size: ~20,000-50,000 households (varies by panel)
- New panels fielded every few years

**Strengths for Our Purpose**:
- Detailed benefit receipt data
- Short-term earnings dynamics
- Program participation for validation
- Wealth and asset holdings
- Larger than PSID but retains panel structure

**Limitations**:
- Shorter panels than PSID (3-4 years vs. decades)
- Sample attrition
- Complex survey design
- Public use files have some restrictions

**Our Use**:
- Validation of benefit calculations
- Short-term earnings dynamics
- Program participation rates for calibration
- Supplementary source for transition modeling
- Wealth distribution validation

## Administrative Data Sources (for Validation)

While we cannot directly access individual-level administrative data, we use published aggregate statistics for validation:

### Social Security Administration (SSA)

**Annual Statistical Supplement**:
- Benefit distributions by age, type (retirement, disability, survivors)
- Beneficiary counts and characteristics
- Average benefit amounts
- Earnings distributions of covered workers
- Trust fund financial status

**OASDI Trustees Reports**:
- Long-run actuarial projections
- Demographic assumptions (fertility, mortality, immigration)
- Economic assumptions (wage growth, inflation)
- Financial projections by component

**MINT Documentation**:
- Published analyses using MINT model
- Methodological documentation
- Validation statistics

**Our Use**:
- Primary calibration targets for benefit receipt
- Validation of benefit calculations
- Long-run projection benchmarks
- Demographic assumption alignment

### Internal Revenue Service (IRS)

**Statistics of Income (SOI)**:
- Tax return data aggregates
- Earnings distributions
- Social Security benefit taxation
- Income by source and filing status

**Our Use**:
- Validation of earnings distributions
- Tax calculations on benefits
- High-income earner distributions

### Census Bureau

**Population Estimates and Projections**:
- Annual population by age, sex, race, ethnicity
- Birth and death rates
- International migration

**American Community Survey (ACS)**:
- Large sample (3+ million households)
- Annual cross-sections
- Detailed demographics and geography
- Income and program participation

**Our Use**:
- Population controls for weighting
- Demographic validation
- Geographic distributions

## Supplementary Data Sources

### Health and Retirement Study (HRS)

**Description**: Longitudinal study of Americans over age 50, with detailed wealth and health data.

**Strengths**:
- Excellent wealth measurement
- Health and disability transitions
- Retirement decisions and timing
- Linked to administrative earnings records (restricted access)

**Our Use**:
- Validation of retirement age distributions
- Wealth holdings near retirement
- Disability onset patterns

### American Time Use Survey (ATUS)

**Description**: Time use diary survey providing detailed labor supply.

**Our Use**:
- Validation of labor supply patterns
- Hours worked distributions

### Mortality Data

**National Vital Statistics System**:
- Death rates by age, sex, race, cause
- Life tables

**SSA Actuarial Life Tables**:
- Cohort life tables
- Differential mortality assumptions

**Our Use**:
- Mortality modeling in projections
- Survivor benefit calculations
- Differential mortality by socioeconomic status

## Data Integration Strategy

### Hierarchical Approach

Our data integration follows a hierarchical structure:

1. **Base Population**: Public cross-sectional survey (to be determined during proof of concept)
2. **Longitudinal Structure**: PSID for earnings trajectories and transitions
3. **Validation**: SIPP for program participation; administrative aggregates
4. **Calibration**: SSA statistics for alignment

### Statistical Matching

We employ statistical matching techniques to combine strengths:

- **CPS-PSID match**: Impute longitudinal structure to CPS using PSID patterns
- **Constrained matching**: Preserve CPS cross-sectional distributions
- **Quantile regression forests**: Predict conditional distributions, not just means
- **Multiple imputation**: Propagate uncertainty from matching process

## Data Availability and Reproducibility

All our primary data sources are publicly available:

| Data Source | Access | Cost | License |
|-------------|--------|------|---------|
| CPS ASEC | IPUMS/Census | Free | Public domain |
| PSID | PSID website | Free | Restricted use terms |
| SIPP | Census Bureau | Free | Public domain |
| SSA Statistics | SSA website | Free | Public domain |
| IRS SOI | IRS website | Free | Public domain |

This ensures full reproducibility of our model. Anyone can:
- Download the same source data
- Run our open-source code
- Verify our results
- Modify assumptions
- Extend the model

## Data Limitations and Challenges

### Known Issues

**Income Underreporting**: Survey data underreport income compared to national accounts, particularly:
- Transfer income (SNAP, SSI, etc.)
- Self-employment income
- Interest and dividend income
- High incomes (top-coding)

**Mitigation**: Use external calibration targets from administrative data

**Sample Size**: Panel surveys (PSID, SIPP) have smaller samples than CPS

**Mitigation**: Use CPS for cross-sectional detail; panels for dynamics only

**Attrition**: Panel surveys lose respondents over time, potentially biasing dynamics

**Mitigation**: Attrition weights and validation against full population

**Measurement Error**: All survey data contain measurement error

**Mitigation**: Multiple imputation and sensitivity analysis

### Strategic Choices

**Cross-Sectional vs. Longitudinal Trade-off**:
- CPS: Large but no panel structure
- PSID: Panel structure but small sample

**Our Approach**: Synthetic panel combining CPS size with PSID dynamics

**Administrative Data Access**:
- Ideal: Linked administrative earnings records
- Reality: Not publicly accessible
- Solution: Statistical matching to published aggregates

**Time Period**:
- Most recent data for current policy
- Historical data for validation of dynamics
- Balance: Use recent CPS for base, historical PSID for dynamics

## Data Preparation Pipeline

Our data preparation follows these steps:

1. **Download and Clean**:
   - Harmonized variable names across surveys
   - Consistent coding of demographics
   - Handle missing data
   - Apply survey weights

2. **Match and Impute**:
   - Statistical matching across surveys
   - Quantile regression forest imputation
   - Multiple imputation for uncertainty

3. **Calibrate**:
   - Gradient descent reweighting
   - Align to external targets
   - Validate distributions

4. **Validate**:
   - Compare to administrative aggregates
   - Check internal consistency
   - Sensitivity analysis

5. **Document**:
   - Full documentation of sources
   - Code comments
   - Validation reports

This ensures transparent, reproducible data construction.

## Summary

We leverage best available public data:

- **CPS**: Cross-sectional detail and sample size
- **PSID**: Longitudinal dynamics and transitions
- **SIPP**: Program participation and validation
- **Administrative**: Calibration targets and benchmarks

This multi-source approach enables:
- Large sample with demographic detail
- Realistic longitudinal dynamics
- External validation
- Full public accessibility and reproducibility

The next chapter describes specific calibration targets derived from these sources.
