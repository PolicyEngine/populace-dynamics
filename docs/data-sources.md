# Data sources

## Overview

Building a dynamic Social Security microsimulation model means
extending `populace` longitudinally and then using it for Social
Security analysis. That requires multiple data sources that capture
cross-sectional population characteristics, longitudinal earnings
dynamics, and demographic transitions.

PolicyEngine's `populace` stack assembles these primary sources,
along with the administrative aggregates used as calibration targets,
and builds a calibrated synthetic population from them. This chapter
describes the primary sources that feed that pipeline.

## Primary survey data sources

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
- Core cross-sectional input to the current public `populace`
  population layer
- Validation of age-earnings profiles
- Calibration targets for population characteristics
- Starting point for longitudinal extension

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
- **Primary source for longitudinal extension of `populace`**
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

**Why Wealth Matters for Social Security Analysis** (per reviewer feedback):
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

### National Health and Aging Trends Study (NHATS) and National Study of Caregiving (NSOC)

**Description**: Panel studies of older Americans and their caregivers, with detailed information on disability, care needs, care receipt, and unpaid caregiving.

**Strengths**:
- Rich measures of ADLs, IADLs, cognitive impairment, and supervision needs
- Detailed care setting information (informal care, paid home care, residential care)
- Direct measurement of unpaid caregiving hours and caregiver relationships
- Especially useful for validating intermediate LTC states rather than only final fiscal aggregates

**Our Use**:
- Calibrate or validate disability and care-need prevalence among older adults
- Validate transitions between no care, home care, and institutional settings
- Measure caregiver burden, co-residence, and labor-supply spillovers
- Inform static and dynamic modeling of caregiver-support policies

### Medicare Current Beneficiary Survey (MCBS)

**Description**: Survey of Medicare beneficiaries with detailed information on utilization, spending, supplemental coverage, and health status.

**Strengths**:
- Detailed Medicare utilization and out-of-pocket spending
- Information on home health, post-acute care, and other services relevant to older adults with care needs
- Better direct measurement of Medicare-financed services than general household surveys

**Our Use**:
- Validate spending and utilization for Medicare beneficiaries with functional limitations
- Assess interaction between Medicare and LTC proposals such as home-care benefits
- Benchmark static first-pass models of Medicare-at-home style proposals

### National Health Interview Survey (NHIS) and Medical Expenditure Panel Survey (MEPS)

**Description**: Broad household surveys covering health, disability, utilization, and medical expenditures.

**Strengths**:
- Coverage of noninstitutionalized populations below age 65
- Functional limitation and health-status measures outside retirement-age populations
- Expenditure detail useful for near-term medical spending and disability interactions

**Our Use**:
- Support modeling of younger disabled populations who may have LTC needs
- Validate health-status and expenditure gradients outside the Medicare population
- Provide an all-age complement to HRS and NHATS

### CMS Long-Term Care Administrative Sources

**Minimum Data Set (MDS)**:
- Resident assessments for nursing home populations
- Functional status, cognitive impairment, and care needs in institutional settings

**Transformed Medicaid Statistical Information System (T-MSIS)**:
- Medicaid enrollment, service use, and spending
- HCBS and institutional LTSS utilization patterns
- State-by-state variation in Medicaid LTSS programs

**Our Use**:
- Validate institutional care prevalence and resident characteristics
- Benchmark Medicaid LTSS participation, payer mix, and spending
- Anchor state-level LTC rule encoding and downstream model validation

### State LTC Policy Sources

Long-term care policy analysis also requires policy-source data, not just survey microdata. We will need to assemble and version:
- State Medicaid manuals and eligibility rules
- HCBS waiver documentation and assessment criteria
- Home equity and asset treatment rules
- Functional eligibility definitions (ADLs, cognitive impairment, level-of-care tests)
- Spousal impoverishment parameters such as resource allowances, monthly
  maintenance allowances, and personal-needs allowances
- State transfer-penalty divisors, private-pay nursing facility rate
  tables, and Qualified Income Trust or Miller Trust guidance
- Estate recovery guidance, PACE manuals, and Single Entry Point or
  assessment-instrument documentation

These sources play the same role for LTC that statutes, tax forms, and program manuals play in PolicyEngine's existing tax-benefit infrastructure.

This matters because a credible static LTC pilot must do more than check
one income threshold. It should be able to tell a family whether they
are eligible now or soon, what spend-down or trust path would be
required, how spousal protections change the result, and what patient
liability would look like after approval. That level of output depends
on operational state documents as much as on survey data.

For the proposal, two supporting appendices make this more concrete.
[`public-validation-inventory.md`](public-validation-inventory.md)
shows how much of the model can be judged against public evidence before
restricted administrative access is available.
[`appendix-colorado-ltc-rules-packet.md`](appendix-colorado-ltc-rules-packet.md)
shows what a first-pass authoritative source packet looks like for a
state LTC pilot.

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

### Institutional Population Challenge

One reason LTC is hard to model is that no single public dataset adequately covers household populations, caregivers, and institutional residents at the same time. CPS and many other core household surveys exclude most institutional populations. An LTC-ready architecture therefore needs an explicit blended strategy:

1. Household base population from Populace's calibrated CPS-based core and allied surveys
2. Longitudinal aging and wealth dynamics from PSID and HRS
3. Care-need and caregiving detail from NHATS/NSOC and MCBS
4. Institutional population benchmarks from MDS and Medicaid administrative sources

Planning around this challenge early is important because the transition into institutional care is itself one of the core outcomes for LTC policy analysis.

### Multi-Survey Fusion Approach

Rather than treating each survey in isolation, we pursue a multi-survey fusion strategy that harmonizes variables across CPS, PSID, and PUF into a unified analytical dataset. This involves:

1. **Common variable schema**: Map each survey's variable names and coding schemes to a shared set of definitions (e.g., CPS `PEARNVAL` → `earnings`, PSID `ER65349` → `earnings`, PUF `E00200` → `wages_and_salaries`)
2. **Cross-survey imputation**: Where one survey has variables that another lacks, use conditional models trained on the richer survey to impute into the target. For example, impute PSID-style longitudinal dynamics into CPS observations, or PUF-style detailed income components into the base population.
3. **Fusion via normalizing flows or QRF**: Train conditional models on the stacked, harmonized data with masking for missing variables. The model learns to generate complete records by conditioning on whichever variables are observed for each survey source.

### Hierarchical Structure

Our data integration follows a hierarchical structure:

1. **Base population**: Populace's CPS-based core providing a large sample with calibrated cross-sectional income
2. **Longitudinal structure**: PSID for earnings trajectories and transition dynamics
3. **Income detail**: PUF for tax return variables and high-income tail corrections
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

3. **Calibrate the Base Population**:
   - Use survey weights only before longitudinalization
   - Align the base cross-section to external targets
   - Convert to fixed representation factors or replicate counts

4. **Align Dynamic Processes**:
   - Match transition controls through event selection
   - Tune process parameters without independently reweighting linked
     people
   - Preserve spouse, former-spouse, parent-child, and household links

5. **Validate**:
   - Compare to administrative aggregates
   - Check internal consistency
   - Sensitivity analysis

6. **Document**:
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
