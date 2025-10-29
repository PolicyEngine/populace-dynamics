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
