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

### Survey of Consumer Finances (SCF)

**Description**: Federal Reserve triennial survey providing the gold standard for U.S. household wealth data, with oversampling of high-wealth households.

**Key Features**:
- Comprehensive wealth measurement (financial assets, housing, pensions, debt)
- Oversamples high-net-worth households for accurate tail measurement
- ~6,000 households per wave, triennial since 1989
- Detailed retirement account data (401(k), IRA, DB pensions)

**Why Wealth Matters for Social Security Analysis** (per Sabelhaus feedback):
- Benefit claiming decisions: wealthier households can afford to delay claiming to age 70 for higher benefits
- Replacement rates: Social Security replaces a higher share of consumption for wealth-poor households
- Means-testing proposals: require wealth data to evaluate
- Comprehensive retirement adequacy: Social Security + pensions + savings + housing

**Our Use**:
- Validation of wealth distributions by age and lifetime earnings quintile
- Impute wealth statistically using random forest matching on age, earnings history, education, and marital status (parallel to earnings history imputation from PSID)
- Validate that imputed wealth distributions match SCF aggregates (net worth, financial assets, pension coverage)
- Enable analysis of reforms that interact with wealth (means-testing, claiming behavior)

**Note**: We treat wealth as a validation target rather than a calibration target initially, checking whether our synthetic panel with imputed wealth produces realistic distributions without forcing weights to match SCF. This is because survey-based wealth measures have their own measurement challenges. Full wealth accumulation modeling is an extension beyond the core Phase 1–5 deliverables.

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
- Retirement claiming behavior by AIME quintile

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

### Multi-Survey Fusion Approach

Rather than treating each survey in isolation, we pursue a multi-survey fusion strategy that harmonizes variables across CPS, PSID, and PUF into a unified analytical dataset. This involves:

1. **Common variable schema**: Map each survey's variable names and coding schemes to a shared set of definitions (e.g., CPS `PEARNVAL` → `earnings`, PSID `ER65349` → `earnings`, PUF `E00200` → `wages_and_salaries`)
2. **Cross-survey imputation**: Where one survey has variables that another lacks, use conditional models trained on the richer survey to impute into the target. For example, impute PSID-style longitudinal dynamics into CPS observations, or PUF-style detailed income components into the base population.
3. **Fusion via normalizing flows or QRF**: Train conditional models on the stacked, harmonized data with masking for missing variables. The model learns to generate complete records by conditioning on whichever variables are observed for each survey source.

### Hierarchical Structure

Our data integration follows a hierarchical structure:

1. **Base Population**: Enhanced CPS (primary candidate—see below) providing large sample with calibrated cross-sectional income
2. **Longitudinal Structure**: PSID for earnings trajectories and transition dynamics
3. **Income Detail**: PUF for tax return variables and high-income tail corrections
4. **Validation**: SIPP for program participation; administrative aggregates
5. **Calibration**: SSA statistics for alignment

### Statistical Matching

We employ statistical matching techniques to combine strengths:

- **CPS-PSID match**: Impute longitudinal structure to CPS using PSID patterns
- **CPS-PUF match**: Transfer detailed income and tax variables from PUF to improve income measurement
- **Constrained matching**: Preserve CPS cross-sectional distributions
- **Quantile regression forests**: Predict conditional distributions, not just means
- **Normalizing flows**: Conditional MAF as alternative for variables with zero-inflation or complex multimodal structure
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
