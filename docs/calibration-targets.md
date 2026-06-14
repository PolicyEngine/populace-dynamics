# Calibration targets

## Overview

Calibration ensures that longitudinal `populace` matches known
population characteristics and that the Social Security application
layer matches system aggregates. This chapter specifies the targets
the project will use for calibration, their sources, and priority
weighting. `populace` maintains these administrative aggregates as a
versioned target registry — signed facts with standard errors — so
that calibration runs against a single, consistent set of targets and
treats them as uncertainty-weighted evidence rather than exact hits.

## Why calibration matters

Dynamic microsimulation models face two key challenges:

1. **Drift**: Small errors compound over projection periods, causing divergence from reality
2. **Misalignment**: Survey data often differ from administrative totals

Calibration addresses both by matching the synthetic population and its
transition process to external targets while preserving:
- Individual-level heterogeneity
- Correlations across variables
- Distributional properties

Our approach uses weight calibration for the base cross-section, then
uses process alignment for the dynamic panel. Once spouse,
former-spouse, parent-child, and household links exist, arbitrary
person-level reweighting can break the population network.

## Calibration Target Categories

### Demographic Targets (Annual)

These targets ensure our population structure matches Census/SSA projections:

**Age-Sex Distribution**:
- Population counts by single year of age (0-100+) and sex
- Source: Census population estimates/projections
- Priority: **Critical** (fundamental to Social Security modeling)

**Race and Ethnicity**:
- Population by race (White, Black, Asian, Other) and Hispanic ethnicity
- Intersected with age groups (18-29, 30-44, 45-54, 55-64, 65-74, 75+)
- Source: Census population estimates
- Priority: **High** (differential mortality and earnings)

**Marital Status**:
- Married, widowed, divorced, never married by age group and sex
- Source: CPS, ACS aggregates
- Priority: **High** (spousal and survivor benefits)

**Educational Attainment**:
- Less than high school, high school, some college, bachelor's+
- By age group and sex
- Source: CPS, ACS
- Priority: **High** (earnings trajectories)

### Labor Market Targets (Annual)

**Employment Status**:
- Employed, unemployed, not in labor force by age-sex-education
- Source: CPS Labor Force Statistics
- Priority: **High** (earnings determination)

**Earnings Distribution**:
- Mean earnings by age-sex-education
- Percentiles of earnings distribution (10th, 25th, 50th, 75th, 90th, 95th, 99th)
- Source: SSA earnings statistics, CPS
- Priority: **Critical** (determines benefit levels)

**Covered Employment**:
- Share with Social Security covered earnings by age
- Source: SSA Annual Statistical Supplement
- Priority: **Critical** (benefit eligibility)

**Self-Employment**:
- Self-employment share by age and industry
- Source: CPS, IRS SOI
- Priority: **Medium** (different tax treatment)

### Social Security Beneficiary Targets (Annual)

**Beneficiary Counts**:
- Retired workers by age and sex
- Disabled workers by age and sex
- Spouse beneficiaries by age and sex
- Widow(er) beneficiaries by age and sex
- Source: SSA Annual Statistical Supplement, Table 5.A1
- Priority: **Critical** (core model validation)

**Average Benefits**:
- Mean retirement benefit by age of benefit receipt
- Mean disability benefit
- Mean spouse and survivor benefits
- Source: SSA Annual Statistical Supplement
- Priority: **Critical** (benefit accuracy)

**Benefit Distribution**:
- Distribution of benefits (percentiles) by benefit type
- Source: SSA statistics
- Priority: **High** (distributional accuracy)

### Earnings History Targets (Longitudinal)

**Earnings Mobility**:
- Transition matrices between earnings quintiles over 5-year periods
- Intergenerational mobility patterns by geography
- Source: PSID, SSA cohort earnings studies, Opportunity Insights (http://www.equality-of-opportunity.org/data/)
- Priority: **Critical** (lifetime earnings trajectories, geographic variation)

**Opportunity Insights Mobility Data**:
The Equality of Opportunity Project provides essential data for validating earnings mobility patterns:
- Intergenerational income mobility by commuting zone and county
- Earnings outcomes by parental income percentile
- Geographic variation in upward mobility
- Can validate modeled earnings trajectories against observed mobility patterns

**Earnings Growth Profiles**:
- Age-earnings profiles by cohort and education
- Variance of earnings growth by age
- Source: PSID, CPS cohorts
- Priority: **Critical** (benefit determination)

**Career Patterns**:
- Years with covered earnings by age and sex
- Gaps in earnings by reason (child-rearing, disability, etc.)
- Source: MINT documentation, PSID
- Priority: **High** (quarters of coverage)

### Mortality and Disability Targets (Annual)

**Mortality Rates**:
- Age-specific death rates by sex
- Differential mortality by earnings level and geography
- Source: SSA actuarial life tables, National Vital Statistics, Opportunity Insights (https://opportunityinsights.org/paper/lifeexpectancy/)
- Priority: **Critical** (survivor benefits, projection accuracy, inequality analysis)

**Opportunity Insights Life Expectancy Data**:
The Opportunity Insights life expectancy dataset provides critical granularity for modeling differential mortality:
- Life expectancy by income percentile at national, state, county, and commuting zone levels
- Temporal trends (2001-2014) showing diverging life expectancy by income
- Geographic variation revealing localities with smaller or narrowing gaps
- Enables validation of mortality differentials across the earnings distribution
- Essential for accurate lifetime benefit projections and distributional analysis

**Disability Incidence**:
- Disability onset rates by age and sex
- Recovery rates
- Source: SSA disability statistics
- Priority: **High** (SSDI modeling)

### Long-Term Care and Caregiving Targets (Extension Track)

If the model is extended toward LTC, we should add a dedicated validation layer rather than relying only on aggregate program costs.

**Functional Status and Care Need**:
- Prevalence of ADL and IADL limitations by age, sex, and living arrangement
- Cognitive impairment and supervision needs
- Source: NHATS, HRS, MCBS
- Priority: **Critical** within LTC extension (determines care demand)

**Care Setting and Utilization**:
- Share receiving unpaid family care
- Share receiving paid home care
- Share in institutional settings or nursing facilities
- Transition rates between community, home-care, and institutional states
- Source: NHATS/NSOC, MCBS, MDS
- Priority: **Critical** within LTC extension (core behavioral state)

**Medicaid LTSS Participation and Spending**:
- Medicaid LTSS enrollment by age and disability status
- HCBS versus institutional spending shares
- State-level variation in service mix and spending
- Source: T-MSIS, CMS summaries, MACPAC
- Priority: **Critical** within LTC extension (program financing)

**Caregiver Burden and Spillovers**:
- Unpaid caregiving hours by relationship type
- Employment reduction or labor-force exit among caregivers
- Co-residence with care recipient
- Source: NSOC, ATUS, CPS supplements where available
- Priority: **High** within LTC extension (distributional and welfare analysis)

**Asset Depletion and Spenddown**:
- Asset holdings near Medicaid entry
- Distribution of housing and liquid wealth among high-need older adults
- Observed spenddown paths where measurable
- Source: HRS, SCF, Medicaid-linked studies
- Priority: **High** within LTC extension (eligibility and reform incidence)

### Financial and Fiscal Targets (Annual)

**Program Aggregates**:
- Total OASDI benefit payments
- Total covered earnings (tax base)
- Total payroll tax revenue
- Trust fund balances
- Source: SSA Trustees Reports
- Priority: **High** (fiscal accuracy)

**Tax Revenue**:
- Income tax on Social Security benefits
- Source: IRS SOI, Treasury data
- Priority: **Medium** (complete fiscal picture)

## Target Prioritization

Not all targets are equally important. We prioritize:

### Tier 1 (Critical):
- Age-sex distribution
- Earnings distributions and growth
- Beneficiary counts by type
- Average benefits
- Earnings mobility matrices

**Rationale**: Core to Social Security benefit determination

### Tier 2 (High):
- Race, ethnicity, education distributions
- Marital status
- Employment status
- Benefit distributions
- Career patterns
- Mortality differentials

**Rationale**: Important for distributional analysis and demographic modeling

### Tier 3 (Medium):
- Self-employment details
- Tax revenue on benefits
- Regional distributions

**Rationale**: Useful but not central to benefit modeling

For an LTC module, the prioritization changes slightly. Functional status, care setting, and Medicaid LTSS aggregates become Tier 1 within that module because getting those intermediate states wrong can still produce superficially plausible fiscal totals while misleading policy analysis.

## Calibration Approach

### Base-Population Weight Calibration

Before longitudinalization, gradient descent can be used to find survey
or synthetic-population weights that:

**Minimize**: Distance from original CPS weights (preserve representativeness)

**Subject to**: Match base-year calibration targets within tolerance

**Method**:
1. Start with CPS survey weights
2. Compute weighted statistics for each target
3. Calculate gradient of distance with respect to weights
4. Update weights in direction that reduces distance to targets
5. Iterate until convergence

**Advantages**:
- Handles many targets simultaneously
- Preserves weight positivity
- Computationally efficient
- Transparent and interpretable

This is primarily a base-year construction tool. It should not be the
default mechanism for repairing annual dynamic projections after family
relationships and event histories have been simulated.

### Dynamic Process Alignment

For longitudinal projections, we align processes and outputs at multiple
time points:

**Base Year (e.g., 2024)**: Full calibration to current data

**5-Year Intervals**: Align to SSA projected aggregates
- Ensures long-run projections stay aligned
- Prevents drift accumulation
- Uses SSA Trustees Report intermediate assumptions

**Annual Transition Controls**: For near-term projections (10 years),
select events and tune process intercepts so annual transitions match
published controls without breaking household and relationship
consistency

### Retirement claiming behavior targets

**Claiming Age Distribution** (per reviewer feedback):
From SSA administrative data:
- Share claiming at age 62, 63, 64, Full Retirement Age, and 70
- Claiming patterns by AIME quintile (higher earners delay more)
- Spouse vs. own benefit claiming patterns
- Disability-to-retirement conversions at FRA
- Source: SSA Annual Statistical Supplement, Trustees Reports
- Priority: **High** (essential for validating behavioral assumptions and reform analysis)

**Lifetime Benefit Distribution Targets**:
From published MINT analyses:
- Distribution of lifetime Social Security benefits by birth cohort
- Lifetime benefits as share of lifetime earnings (replacement rates) by quintile
- Internal rates of return on Social Security contributions by demographic group
- Source: SSA MINT published analyses, academic studies
- Priority: **Medium** (validation of full lifecycle simulation)

### Validation Against Non-Target Variables

We reserve some variables as validation checks (not calibration targets):

- Wealth distributions (from Survey of Consumer Finances)
- Program participation rates (SNAP, SSI, Medicaid)
- Poverty rates (official and Supplemental Poverty Measure)
- Income inequality measures (Gini, percentile ratios)
- Replacement rates by lifetime earnings quintile
- Family structure outcomes (share receiving spouse vs. own benefits, widow(er) beneficiaries)
- ADL/IADL prevalence and care-setting distributions for LTC extensions
- Caregiver hours and labor-supply responses for caregiver-policy extensions

**Why validation, not calibration?** Survey-based measures like poverty rates suffer from income underreporting—the very problem our methodology corrects. Calibrating to flawed poverty estimates would embed those errors. Instead, we calibrate to administrative data (SSA, IRS) and then *check* whether our corrected income distributions produce more accurate poverty estimates than raw surveys.

**Survey of Consumer Finances (SCF) Wealth Validation** (per reviewer feedback):
The SCF provides the gold standard for U.S. household wealth data. We validate that our synthetic panel produces realistic:
- Net worth distributions by age and lifetime earnings quintile
- Financial asset holdings by age group ($0-$10k, $10k-$50k, $50k-$250k, $250k+)
- Retirement account balances (401(k), IRA) by age and income
- Housing wealth and homeownership rates by age and income
- Pension coverage (DB and DC) by age and employer type

Wealth affects Social Security claiming behavior (wealthier households can afford to delay) and retirement adequacy. While we don't calibrate to SCF targets in the initial model, validating against them ensures comprehensive retirement security analysis and positions the model for future wealth integration.

**Why intermediate LTC validation matters**:
For LTC policy analysis, matching only the final cost estimate is not enough. Two models can generate similar aggregate spending while implying very different paths through disability, family caregiving, paid home care, institutional care, and Medicaid spenddown. Any LTC extension should therefore validate these intermediate states explicitly before being used for structural reforms.

## Data Sources for Targets

| Target Category | Primary Source | Update Frequency | Latest Available |
|----------------|----------------|------------------|------------------|
| Demographics | Census | Annual | 2024 |
| Earnings | SSA Supplement | Annual | 2023 |
| Beneficiaries | SSA Supplement | Annual | 2023 |
| Employment | CPS | Monthly/Annual | 2024 |
| Mortality | NVSS/SSA | Annual | 2022 |
| Projections | SSA Trustees | Annual | 2024 |
| Disability | SSA Statistics | Annual | 2023 |
| LTC need and caregiving | NHATS, HRS, MCBS | Annual/Biennial | Varies |
| Medicaid LTSS | T-MSIS, CMS | Annual | Varies |

## Target Specification Examples

### Example 1: Age-Earnings Profile

**Target**: Mean earnings by single-year age (18-70), by sex

**Source**: SSA Average and Median Earnings Tables

**Data**:
```
Age 25 Male: $45,000
Age 25 Female: $40,000
Age 35 Male: $65,000
Age 35 Female: $52,000
...
```

**Tolerance**: ±2% of target value

**Priority**: Tier 1 (Critical)

### Example 2: Beneficiary Counts

**Target**: Number of retired worker beneficiaries by age group

**Source**: SSA Annual Statistical Supplement, Table 5.A1

**Data**:
```
Age 62-64: 3.5 million
Age 65-69: 12.2 million
Age 70-74: 11.8 million
...
```

**Tolerance**: ±1% of target counts

**Priority**: Tier 1 (Critical)

### Example 3: Earnings Mobility

**Target**: 5-year transition matrix between earnings quintiles

**Source**: PSID analysis, published MINT documentation

**Data** (example):
```
From Q1 → To Q1: 65%
From Q1 → To Q2: 20%
From Q1 → To Q3: 10%
...
```

**Tolerance**: ±3 percentage points

**Priority**: Tier 1 (Critical)

## Calibration Workflow

Our calibration process follows these steps:

1. **Assemble Targets**:
   - Download latest data from all sources
   - Harmonize definitions and formats
   - Document sources and vintages

2. **Base Population Weights**:
   - Start with CPS survey weights
   - Adjust for non-response and coverage

3. **Tier 1 Base Calibration**:
   - Calibrate the base population to critical targets first
   - Ensure core demographics and earnings accurate

4. **Tier 2 Base Calibration**:
   - Add secondary targets
   - Iterate to convergence

5. **Dynamic Alignment**:
   - Select annual events to match aggregate transition controls
   - Tune process parameters without independently reweighting linked
     people

6. **Validation**:
   - Check non-targeted variables
   - Compare to external benchmarks
   - Assess quality of fit

7. **Documentation**:
   - Record all targets and sources
   - Document convergence metrics
   - Report any systematic deviations

## Handling Conflicts

Sometimes targets conflict (perfect match to all is impossible). Our resolution strategy:

**Priority**: Higher-tier targets take precedence

**Tolerance**: Allow small deviations within specified tolerance

**Trade-offs**: Document any necessary compromises

**Sensitivity**: Test impact of alternative calibrations

## Calibration Over Time

As data updates:

**Annual Updates**: Incorporate latest SSA statistics

**Benchmark Revisions**: When Census or SSA revises historical data, rerun calibration

**Assumption Changes**: When SSA Trustees update intermediate assumptions, adjust projections

**Version Control**: Track calibration version with model version

## Expected Outcomes

Successful calibration produces:

- **Accurate Aggregates**: Match SSA benefit totals within 1-2%
- **Correct Distributions**: Match earnings and benefit percentiles
- **Realistic Dynamics**: Plausible earnings trajectories and transitions
- **Valid Demographics**: Representative population structure
- **Fiscal Consistency**: Trust fund projections align with SSA

These ensure confidence in policy analysis results.

## Summary

Calibration targets provide external validation of our synthetic panel:

- **Comprehensive**: Demographics, earnings, benefits, dynamics
- **Hierarchical**: Prioritized by importance to Social Security modeling
- **Transparent**: All targets documented with sources
- **Dynamic**: Calibrated across time periods
- **Validated**: Non-targeted variables as quality checks

[`public-validation-inventory.md`](public-validation-inventory.md)
enumerates the public source stack behind these targets. That appendix
is useful because it shows how much of the validation burden can be met
transparently before any restricted administrative linkage is available.

The next chapter describes the methodology for constructing the synthetic panel subject to these calibration constraints.
