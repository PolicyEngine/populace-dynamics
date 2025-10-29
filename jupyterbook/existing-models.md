# Existing Social Security Microsimulation Models

## Overview

Several organizations have developed sophisticated dynamic microsimulation models for Social Security analysis. This chapter reviews the major existing models, their capabilities, limitations, and how our proposed model would differ.

## DynaSim (Urban Institute)

### Description

DynaSim is Urban Institute's comprehensive dynamic microsimulation model for retirement policy analysis. First developed in the 1970s, it has been continuously updated and is considered the gold standard for non-governmental Social Security modeling.

### Key Features

**Starting Data**: SIPP panels, matched to administrative earnings records where possible

**Projection Period**: Projects population from base year through 75+ years

**Demographic Processes**:
- Birth, death, marriage, divorce
- Immigration
- Educational attainment
- Disability onset and recovery

**Economic Processes**:
- Earnings trajectories by education, race, sex
- Labor force participation
- Retirement decisions
- Asset accumulation and drawdown
- Pension participation

**Policy Detail**:
- Full Social Security benefit rules (retirement, disability, survivors)
- Supplemental Security Income (SSI)
- Medicare
- Private pensions and 401(k)s
- Federal income taxes

### Strengths

- Decades of development and refinement
- Extensive validation against actual outcomes
- Used by researchers and policymakers widely
- Rich demographic and economic detail
- Models behavioral responses

### Limitations

**Access**: Proprietary; requires contracts with Urban Institute, expensive

**Transparency**: Code not open-source; limited documentation of assumptions

**Flexibility**: Cannot easily modify assumptions or add new policies

**Reproducibility**: Published results cannot be independently verified

**Interface**: No public web interface; requires technical expertise

### Use Cases

Urban Institute produces analysis for:
- Congressional offices evaluating Social Security reforms
- Academic researchers studying retirement security
- Advocacy organizations analyzing distributional impacts
- Foundations funding retirement policy research

## MINT (SSA Modeling Income in the Near Term)

### Description

MINT is the Social Security Administration's microsimulation model for projecting retirement income. Developed in collaboration with the Brookings Institution and RAND Corporation.

### Key Features

**Starting Data**: Matched SIPP-administrative earnings records

**Projection Period**: Near-term focus (50-75 years)

**Detail Level**:
- Administrative earnings histories for actual SSA records
- Projected earnings for younger cohorts
- Family structure dynamics
- Wealth accumulation
- SSI and Social Security benefit calculations

### Strengths

- Access to actual SSA administrative data
- High-quality earnings histories for older cohorts
- Official SSA endorsement and use
- Extensive technical documentation
- Regular updates and validation

### Limitations

**Public Access**: Limited; restricted datasets available to researchers

**Code Access**: Not open-source; SAS-based implementation

**Flexibility**: Difficult for external researchers to modify

**Scope**: Focused on SSA programs; limited broader tax-benefit integration

**Behavioral**: Limited modeling of behavioral responses to reforms

### Use Cases

- SSA actuarial projections and policy analysis
- Research collaborations with academic institutions
- Policy briefs on distributional impacts of reforms

## CBOLT (Congressional Budget Office Long-Term Model)

### Description

CBO's dynamic model for long-term budget analysis, including Social Security.

### Key Features

**Scope**: Integrated model of federal budget, demographics, and economy

**Social Security Detail**:
- Benefit calculations
- Trust fund projections
- Interaction with broader fiscal policy
- Macroeconomic feedbacks

**Projection Horizon**: 30+ years (CBO's long-term window)

### Strengths

- Official CBO use for budget analysis
- Integration with broader fiscal modeling
- Macroeconomic consistency
- Regular updates with CBO budget outlook

### Limitations

**Access**: Internal CBO use only; not available to external researchers

**Transparency**: Limited public documentation of methods

**Granularity**: Less distributional detail than DynaSim/MINT

**Reproducibility**: Cannot replicate CBO analysis independently

### Use Cases

- CBO cost estimates for Social Security legislation
- Long-term budget outlook reports
- Analysis of entitlement reform proposals

## Other Models

### Penn Wharton Budget Model (PWBM)

**Description**: Micro-founded dynamic general equilibrium model with Social Security detail

**Strengths**:
- Some public access via web interface
- Integration with broader budget analysis
- Macroeconomic feedbacks

**Limitations**:
- Less distributional detail than DynaSim
- Closed-source implementation
- Limited ability to customize assumptions

### Academic Models

Various researchers have developed specialized models:

- **Gustman-Steinmeier**: Structural retirement model with Social Security
- **Rust-Phelan**: Dynamic programming model of retirement decisions
- **Scholz et al.**: Lifecycle saving and Social Security analysis

These models provide deep insights into specific questions but lack the comprehensive scope needed for general policy analysis.

## Cato Institute's Reported Model

According to the project brief, Cato Institute is reportedly developing a Social Security model. Based on public information:

**Known Details**: Limited public documentation available

**Scope**: Unclear if it will match DynaSim/MINT scope

**Access**: Unknown if it will be publicly accessible or open-source

**Status**: Development stage unknown

Our model would distinguish itself through guaranteed open-source access, public web interface, and integration with PolicyEngine's broader capabilities.

## Panel Construction Methodology Comparison

Understanding the technical approaches used by existing models helps position our methodology and clarify our design choices. This section draws on published technical documentation where available {cite}`favreault2014` {cite}`smith2010` {cite}`butrica2006`.

### DynaSim's Approach

**Starting Data**: SIPP panels (Survey of Income and Program Participation)

**Panel Construction**:
- Begins with SIPP cross-section (~50,000 households)
- Statistical matching to administrative earnings records where possible (restricted access)
- For non-matched individuals: Impute earnings using regression-based methods
- Use transition probability matrices to age population forward
- Estimate transition models (marriage, divorce, death, disability) from historical SIPP panels
- Alignment: Adjust weights periodically to match aggregate totals

**Calibration Method**:
- **Alignment technique**: Adjust transition probabilities to hit aggregate targets
- Example: If too few marriages projected, increase marriage probabilities proportionally
- Calibrate to Census population projections, SSA aggregates, economic forecasts
- Iterative adjustment process, not optimization-based

**Strengths**:
- Real survey base with observed characteristics
- Decades of refinement
- Extensive validation

**Limitations**:
- SIPP sample smaller than CPS
- Alignment can distort individual-level heterogeneity
- Transition models may not capture full earnings dynamics
- Not fully synthetic (depends on which SIPP cohort used)

### MINT's Approach

**Starting Data**: SIPP matched to SSA administrative records (Master Earnings File)

**Panel Construction**:
- **Older cohorts**: Actual administrative earnings histories (gold standard)
- **Younger cohorts**: Project forward using statistical models
- Earnings projection: Quantile regression models by age, education, gender {cite}`butrica2006`
  - Note: MINT pioneered quantile regression for distributional earnings projection
  - Our QRF approach extends this with machine learning and full distribution prediction
- Demographic transitions: Discrete-time hazard models from PSID/SIPP
- Match Survey of Consumer Finances for wealth detail

**Calibration Method**:
- Relies heavily on administrative data (less need for calibration)
- Younger cohort projections aligned to SSA Trustees assumptions
- Wealth distributions calibrated to SCF
- Benefit calculations: Direct application of SSA rules to actual/projected earnings

**Strengths**:
- Real earnings histories for older cohorts (unmatched accuracy)
- Official SSA data access
- Strong validation base

**Limitations**:
- Restricted access (not publicly available)
- SIPP sample size smaller than CPS
- Younger cohort projections still model-dependent
- Cannot replicate without administrative data access

### CBOLT's Approach

**Starting Data**: Representative agent / stylized household types

**Panel Construction**:
- Less granular than DynaSim/MINT
- Representative households by age, income, family structure
- Aggregate earnings profiles rather than individual histories
- Focus on macro consistency over distributional detail

**Calibration Method**:
- Calibrate to National Income and Product Accounts (NIPA)
- Match aggregate labor force participation, earnings
- Demographic projections from Census
- Less individual-level calibration, more aggregate consistency

**Strengths**:
- Computationally efficient
- Macroeconomic consistency
- Integrated with broader fiscal model

**Limitations**:
- Limited distributional detail
- Cannot analyze impacts on specific demographic subgroups
- Less suitable for micro-level reform analysis

### Our Approach: Fully Synthetic Panel

**Starting Data**: CPS ASEC (~200,000 individuals)

**Panel Construction**:
- **Fully synthetic**: No administrative data matching required
- **Quantile regression forests**: Predict full conditional distribution of earnings at each age
- Training: PSID longitudinal data (public use files)
- Generate complete lifetime earnings histories for entire CPS sample
- Demographic transitions: Hazard models estimated on PSID

**Calibration Method**:
- **Gradient descent reweighting**: Optimization-based approach
- Minimize distance from original CPS weights
- Subject to: Match all target variables (earnings distributions, beneficiary counts, etc.)
- Can handle hundreds of simultaneous targets
- Mathematically principled, computationally efficient

**Comparison to Other Methods**:

| Aspect | DynaSim | MINT | CBOLT | Our Model |
|--------|---------|------|-------|-----------|
| **Base Data** | SIPP | SIPP+Admin | Rep. Agents | CPS |
| **Sample Size** | ~50k | ~50k | ~100s | ~200k |
| **Panel Type** | Semi-synthetic | Real+Projected | Stylized | Fully Synthetic |
| **Earnings History** | Regression | Admin+Projection | Aggregate | QRF Imputation |
| **Training Data** | SIPP panels | Admin data | Macro data | PSID |
| **Calibration** | Alignment | Less needed | Aggregate | Gradient descent |
| **Public Replicability** | No | No | No | **Yes** |

**Why Fully Synthetic?**

The choice between real/matched data (MINT), semi-synthetic (DynaSim), and fully synthetic (our approach) involves fundamental trade-offs {cite}`caldwell2017`:

1. **No administrative data required**: Entire methodology reproducible with public data
2. **Larger sample**: CPS sample 4x SIPP, enables state/demographic detail
3. **QRF advantages**: Captures full distribution, not just means; non-parametric flexibility
4. **Modern optimization**: Gradient descent superior to iterative alignment
5. **Open source**: Anyone can validate, modify, extend

**Trade-offs**:

**Advantages over DynaSim/MINT**:
- Fully reproducible (no restricted data)
- Larger sample enables more detailed analysis
- Modern ML methods (QRF) vs. traditional regression
- Optimization-based calibration vs. ad-hoc alignment

**Disadvantages vs. MINT**:
- No actual administrative earnings (MINT has real histories for older cohorts)
- Imputation uncertainty (though we quantify via multiple imputation)
- Requires careful validation since everything is synthetic

**Our position**: We sacrifice MINT's administrative data advantage (which isn't publicly accessible anyway) to gain full transparency and reproducibility while maintaining comparable or superior methodology to DynaSim.

## Comparison Summary

| Feature | DynaSim | MINT | CBOLT | PWBM | Our Model |
|---------|---------|------|-------|------|-----------|
| **Public Access** | Paid only | Limited | None | Web only | Free & Full |
| **Open Source** | No | No | No | No | **Yes** |
| **Web Interface** | No | No | No | Basic | **Full** |
| **Python API** | No | No | No | No | **Yes** |
| **Code Transparency** | Low | Medium | Low | Low | **Full** |
| **Reproducibility** | Limited | Limited | None | Limited | **Full** |
| **Integration** | Standalone | Standalone | Internal | Standalone | **PolicyEngine** |
| **Individual Detail** | High | High | Medium | Medium | **High** |
| **Customization** | Low | Low | None | Low | **High** |
| **Cost** | High | Free* | N/A | Free | **Free** |

*MINT restricted datasets available to approved researchers

## The Gap: An Open-Source Alternative

No existing model provides:

1. **Full Open Source**: Complete transparency in code and assumptions
2. **Public Web Access**: User-friendly interface for non-technical users
3. **API Access**: Programmatic access for researchers and developers
4. **Integration**: Combined with broader tax-benefit modeling
5. **Zero Cost**: Free for anyone to use and modify
6. **Reproducibility**: Anyone can replicate and verify results

This gap creates the opportunity and motivation for our model development.

## What We Can Learn From Existing Models

Despite access limitations, we can leverage public knowledge:

### Methodological Lessons

**From DynaSim**:
- Importance of rich demographic transitions
- Need for behavioral response modeling
- Value of extensive validation
- Alignment techniques for long-run projections

**From MINT**:
- Statistical matching to administrative data
- Earnings imputation methods
- Validation against actual benefit receipt
- Documentation standards

**From CBOLT**:
- Integration with fiscal projections
- Macroeconomic consistency checks
- Communication of uncertainty

### Validation Benchmarks

We can validate our model against:
- Published DynaSim results for similar reforms
- MINT reports and technical papers
- CBO Social Security projections
- SSA actuarial reports and projections
- Academic papers using these models

This provides external validation without requiring access to proprietary models themselves.

## Positioning Our Model

Our model occupies a unique position:

**Accessibility**: First truly open and public dynamic Social Security model

**Integration**: Only model integrated with comprehensive tax-benefit microsimulation

**Transparency**: Full code, data, and assumption transparency

**Extensibility**: Users can modify, extend, and customize freely

**Cost**: Zero cost removes barrier to policy analysis

**Community**: Open-source enables community contributions and improvements

These features make our model complementary to rather than competitive with existing models - we democratize access to sophisticated analysis while existing models continue to serve specialized needs.
