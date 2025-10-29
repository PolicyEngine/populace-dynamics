# Building an Open-Source Social Security Dynamic Microsimulation Model

## Executive Summary

This document outlines a plan to develop the first open-source, publicly available Social Security dynamic microsimulation model - a tool that would democratize access to sophisticated policy analysis previously available only through proprietary models like DynaSim (Urban Institute) and CBOLT (Congressional Budget Office).

The model will leverage PolicyEngine's existing Social Security rules implementation while creating a synthetic longitudinal panel dataset that tracks individuals across their lifetimes. This approach combines cutting-edge machine learning techniques (quantile regression forests) with traditional microsimulation calibration methods to produce accurate, transparent, and reproducible Social Security policy analysis.

## The Opportunity

Social Security is the largest single program in the U.S. federal budget, with over $1.4 trillion in annual benefits paid to more than 67 million Americans. Yet public access to sophisticated modeling tools for analyzing Social Security reforms remains severely limited:

- **DynaSim** (Urban Institute): Proprietary, expensive, limited public access
- **CBOLT** (Congressional Budget Office): Internal CBO use only, not publicly available
- **MINT** (Social Security Administration): Some public access but limited customization
- **Cato Institute**: Reportedly building something, but scope and availability unclear

A gap exists for an open-source model that provides:
- Full transparency in methodology and assumptions
- Free public access via web interface and Python API
- Individual-level detail for distributional analysis
- Lifetime earnings trajectories for cohort analysis
- Integration with PolicyEngine's existing tax and benefit modeling

## Key Innovation: From Cross-Sectional to Longitudinal

PolicyEngine-US already implements Social Security rules comprehensively, allowing individual-level benefit calculations given earnings history and demographic information. The core modeling challenge is **not** the policy rules themselves, but rather creating realistic synthetic panel data that captures:

1. **Lifetime earnings trajectories**: How do earnings evolve from age 18 to retirement?
2. **Demographic transitions**: Marriage, divorce, childbearing, disability, mortality
3. **Cross-sectional consistency**: Matching current population distributions
4. **Longitudinal consistency**: Realistic earnings mobility and life-cycle patterns
5. **Calibration to external targets**: SSA actuarial projections, historical data, survey panels

This is fundamentally a **data generation problem** that requires:
- Machine learning for imputation (quantile regression forests)
- Statistical matching to longitudinal surveys (PSID, SIPP)
- Gradient descent reweighting for calibration targets
- Validation against administrative data where available

## Significance

This would be a landmark contribution to open policy analysis:

- **First** open-source dynamic Social Security microsimulation model
- Enables researchers, advocates, and policymakers to analyze reforms without proprietary tools
- Integrates with PolicyEngine's web app for public access
- Provides Python package for programmatic access
- Full reproducibility through open-source code and documentation
- Can model reforms at individual level with full distributional impacts
- Tracks effects across entire lifecycles and multiple generations

The model will support analysis of:
- Benefit formula changes
- Retirement age adjustments
- Tax base modifications
- Means-testing proposals
- Progressive indexing
- Minimum benefit proposals
- Effects on poverty, inequality, and adequacy across demographic groups

## This Document

The remainder of this planning document covers:

- **Literature Review**: Academic research on Social Security microsimulation
- **Existing Models**: Detailed comparison of DynaSim, CBOLT, MINT, and other tools
- **Data Sources**: Survey data (PSID, SIPP, CPS) and calibration targets
- **Methodology**: Technical approach to synthetic panel construction
- **Infrastructure**: PolicyEngine tools (microimpute, microcalibrate, enhanced CPS)
- **Team**: Expertise of Max Ghenis, Ben Ogorek, and John Sabelhaus
- **Roadmap**: Milestones and development timeline
