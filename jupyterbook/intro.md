# Building an Open-Source Social Security Dynamic Microsimulation Model

::::{note}
**DRAFT PLANNING DOCUMENT**

This document outlines a proposed project for developing an open-source Social Security dynamic microsimulation model. It represents a planning and design phase, not a completed implementation.
::::

## Executive Summary

This document proposes an open-source Social Security dynamic microsimulation model. The approach combines quantile regression forests {cite:p}`meinshausen2006` for imputing lifetime earnings histories with gradient descent calibration {cite:p}`deville1992` to SSA administrative targets. PolicyEngine has validated this methodology through the Enhanced CPS (ECPS), which applies the same techniques to cross-sectional tax modeling {cite:p}`ghenis2024`. This project extends that work to longitudinal analysis.

## The Problem

Social Security is the largest U.S. federal program—$1.5 trillion in FY2024, serving 67 million beneficiaries {cite:p}`cbo2024longterm,ssa2024facts`. Yet public access to modeling tools for analyzing reforms is limited:

- **DynaSim** (Urban Institute): Proprietary, requires contracts {cite:p}`favreault2015`
- **CBOLT** (CBO): Internal use only {cite:p}`cbo2018`
- **MINT** (SSA): Restricted datasets, closed-source code {cite:p}`ssa2024mint,smith2010mint`
- **Cato model**: Open-source but small sample (10,000 households), no web interface

An analogous problem exists in cross-sectional tax modeling, where major models rely on the restricted IRS Public Use File. PolicyEngine addressed this by developing the Enhanced CPS—a public dataset using QRF imputation and calibration to ~2,800 administrative targets that produces results comparable to PUF-based models.

## Proposed Approach

PolicyEngine-US already implements Social Security rules (OASI, SSDI), SSI, SNAP, Medicaid, taxes, and 50+ other programs. The challenge is generating realistic synthetic panel data with lifetime earnings trajectories and demographic transitions.

### What Already Exists

| Component | Status | Description |
|-----------|--------|-------------|
| **Enhanced CPS** | ✓ Exists | Base population with imputed income, calibrated to ~2,800 targets |
| **microimpute** | ✓ Exists | QRF imputation library |
| **microcalibrate** | ✓ Exists | Gradient descent reweighting |
| **PolicyEngine-US** | ✓ Exists | Benefit rules for SS, SSI, SNAP, Medicaid, taxes |
| **Earnings histories** | **New** | ZI-QDNN/QRF trained on PSID to impute 35-year trajectories |

### Data Sources (All Public)

- **PSID**: Longitudinal earnings data, downloadable from University of Michigan
- **CPS**: Cross-sectional population base
- **SSA Statistical Supplement**: Calibration targets (beneficiaries by age, benefit amounts, etc.)

Once earnings histories are attached to the ECPS population, benefit calculations across all programs run through the existing PolicyEngine-US rules engine—including interactions between Social Security and means-tested programs.

## Key policy questions this model will answer

The model would support analysis of Social Security reforms that are actively debated but difficult to evaluate without dynamic microsimulation:

**Raising the retirement age**: Proposals to increase the Full Retirement Age to 68 or 70 disproportionately affect workers with shorter life expectancy—lower-income workers, those in physically demanding jobs, and Black men. This model can quantify lifetime benefit losses by demographic group, lifetime earnings quintile, and occupation, revealing distributional effects that aggregate models miss.

**Progressive benefit formula changes**: Progressive indexing proposals would reduce benefits for higher earners while maintaining them for lower earners, but implementation requires defining "higher earners" based on lifetime earnings. This model can simulate exact distributional effects across the entire lifetime earnings distribution under different threshold and indexing formula definitions.

**Payroll tax base changes**: Proposals to raise or eliminate the taxable maximum ($168,600 in 2024) affect high earners disproportionately. The model can show effects on both revenue and future benefit entitlements, including interactions with benefit taxation thresholds.

**Effects on women and families**: Social Security spousal and survivor benefits create complex incentives. This model provides individual-level analysis of how benefit rules affect lifetime benefits for divorced women, widows, and single mothers—populations often underserved by existing aggregate models.

**Racial equity analysis**: Differential mortality creates racial disparities in lifetime Social Security returns. Black men have lower life expectancy, receiving benefits over fewer years despite paying taxes at the same rate. This model can quantify these effects and simulate reforms to address them, such as adjusting benefit formulas or retirement ages to account for differential mortality.

Users could examine distributional effects by income, age, race, and gender. Access via web interface and Python API; open-source code enables independent verification.

## Document Overview

Subsequent chapters cover: literature review of dynamic microsimulation; detailed comparison of existing models (DynaSim, CBOLT, MINT); technical specifications for variables and transitions; data sources; calibration methodology; PolicyEngine infrastructure; team expertise; and development roadmap.
