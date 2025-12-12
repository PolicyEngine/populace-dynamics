# Building an Open-Source Social Security Dynamic Microsimulation Model

::::{note}
**DRAFT PLANNING DOCUMENT**

This document outlines a proposed project for developing an open-source Social Security dynamic microsimulation model. It represents a planning and design phase, not a completed implementation.
::::

## Executive Summary

This document outlines a plan to develop the first open-source, publicly available Social Security dynamic microsimulation model. Such a tool would democratize access to sophisticated policy analysis previously available only through proprietary models like DynaSim {cite:p}`favreault2015` and CBOLT {cite:p}`cbo2018`. The model will leverage PolicyEngine's existing Social Security rules implementation while creating a synthetic longitudinal panel dataset that tracks individuals across their lifetimes. This approach combines machine learning techniques, specifically quantile regression forests {cite:p}`meinshausen2006`, with traditional microsimulation calibration methods {cite:p}`deville1992` to produce accurate, transparent, and reproducible Social Security policy analysis.

This is not an untested approach. PolicyEngine has already proven this methodology works through the Enhanced CPS (ECPS), the only publicly available cross-sectional microdata file that produces accurate tax-benefit microsimulation impacts. This project extends that proven methodology from cross-sectional to longitudinal analysis.

## The Opportunity

Social Security is the largest single program in the U.S. federal budget. According to the Congressional Budget Office, Social Security spending totaled approximately $1.5 trillion in fiscal year 2024, representing about 21 percent of total federal expenditures {cite:p}`cbo2024longterm`. The program provides monthly benefits to about 67 million Americans, including retired workers, disabled workers, survivors, and dependents {cite:p}`ssa2024facts`. Despite its fiscal magnitude and centrality to retirement security, public access to sophisticated modeling tools for analyzing Social Security reforms remains severely limited.

### The Problem: Restricted Access to Modeling Tools

The current landscape of Social Security microsimulation models presents significant barriers to transparent policy analysis. DynaSim, developed by the Urban Institute, is proprietary and expensive, with limited public access {cite:p}`favreault2015`. CBOLT, the Congressional Budget Office's long-term model, is used exclusively for internal CBO analysis and not available to external researchers {cite:p}`cbo2018`. MINT, the Social Security Administration's Modeling Income in the Near Term model, provides some public access through restricted datasets available to approved researchers, but the underlying code and methodology remain largely inaccessible {cite:p}`ssa2024mint,smith2010mint`. While the Cato Institute has developed an open-source R-based model, it uses a smaller sample size (10,000 households) and lacks web interface or API integration for broader accessibility.

This restricted landscape creates a critical gap for policy researchers, advocates, and policymakers who need tools that provide full transparency in methodology and assumptions, free public access via web interface and Python API, individual-level detail for distributional analysis, lifetime earnings trajectories for cohort analysis, and integration with comprehensive tax and benefit modeling.

### A Proven Approach: The Enhanced CPS Precedent

PolicyEngine has already demonstrated that fully public, synthetic data can match the accuracy of restricted administrative data for microsimulation purposes. The analogous problem exists in cross-sectional tax-benefit modeling: all major U.S. microsimulation models, including those from the Tax Policy Center, Penn Wharton Budget Model, and Tax Foundation, rely on the IRS Public Use File (PUF). This reliance creates fundamental challenges. The PUF cannot be publicly shared due to privacy restrictions, limiting reproducibility of research. Most models require adding CPS non-filers to the PUF, creating hybrid datasets with complex integration challenges. The restricted nature of the PUF creates barriers to entry for new researchers and prevents independent verification of published results.

PolicyEngine addressed this reproducibility crisis by developing the Enhanced CPS (ECPS), which is the only publicly available cross-sectional microdata file that produces accurate tax-benefit microsimulation impacts {cite:p}`ghenis2024`. The ECPS construction process begins with fully public CPS data, uses machine learning to impute PUF-like detail in a privacy-safe manner through quantile regression forests, applies gradient descent calibration to over 7,000 administrative targets from IRS Statistics of Income and other sources, and achieves comparable or superior accuracy to PUF-based models while enabling full reproducibility and transparency.

The validation results demonstrate that this approach works. PolicyEngine-US produces revenue estimates, distributional analysis, and reform impacts that match or exceed proprietary models, and the entire methodology is open-source and fully documented. The ECPS uses a two-stage process: first, quantile regression forests impute missing or underreported variables from public data sources; second, gradient descent optimization reweights the dataset to match thousands of administrative targets simultaneously.

This project applies the same methodological framework to longitudinal Social Security modeling. Just as ECPS proves we can build accurate cross-sectional files without restricted data, this project will demonstrate that we can build accurate longitudinal panels without administrative earnings records. The methodology is directly analogous: ECPS combines CPS data, machine learning imputation from the PUF, and calibration to produce accurate tax modeling; this project will combine CPS data, machine learning imputation from the Panel Study of Income Dynamics (PSID), and calibration to produce accurate Social Security modeling.

We are not proposing an untested approach. Rather, we are extending a proven methodology that has already gained credibility through real-world validation and use in policy analysis. The infrastructure (microimpute, microcalibrate), the validation framework, and the team expertise all derive from the successful ECPS development.

## Key Innovation: From Cross-Sectional to Longitudinal

PolicyEngine-US already implements Social Security rules comprehensively, allowing individual-level benefit calculations given earnings history and demographic information. The core modeling challenge is not the policy rules themselves, but rather creating realistic synthetic panel data that captures lifetime earnings trajectories, demographic transitions such as marriage, divorce, childbearing, disability, and mortality, cross-sectional consistency with current population distributions, longitudinal consistency with realistic earnings mobility and life-cycle patterns, and calibration to external targets including SSA actuarial projections, historical data, and survey panels.

This is fundamentally a data generation problem that requires machine learning for imputation through quantile regression forests, statistical matching to longitudinal surveys such as PSID and SIPP, gradient descent reweighting for calibration to thousands of targets, and validation against administrative data where available. The technical approach leverages the same tools (microimpute, microcalibrate) that produced the Enhanced CPS, but extends them from cross-sectional to longitudinal dimensions.

### No Access Barriers: 100% Public Data

A critical advantage of this approach is that **all data sources are publicly available**:

- **PSID** (Panel Study of Income Dynamics): Publicly downloadable from the University of Michigan. No application or approval process required for the main dataset.
- **CPS** (Current Population Survey): Freely available from Census Bureau. The Enhanced CPS builds entirely on public CPS microdata.
- **SSA calibration targets**: Published annually in the SSA Annual Statistical Supplement, available at ssa.gov.
- **Mortality data**: Life tables from SSA actuaries and National Vital Statistics are public.

Unlike DynaSim (requires Urban Institute contracts) or MINT (restricted access), anyone can replicate and extend this model from day one.

### What Already Exists

This project is not starting from scratch. PolicyEngine has already built and validated the core infrastructure:

| Component | Status | What It Does |
|-----------|--------|--------------|
| **Enhanced CPS (ECPS)** | ✓ Exists | Base population dataset with imputed income, calibrated to 7,000+ targets |
| **microimpute** | ✓ Exists | QRF imputation library, proven in ECPS development |
| **microcalibrate** | ✓ Exists | Gradient descent reweighting, handles thousands of simultaneous targets |
| **PolicyEngine-US rules** | ✓ Exists | 50+ federal/state programs including Social Security, SSI, SNAP, Medicaid, taxes |
| **Longitudinal earnings histories** | **New** | Training QRF on PSID to impute 35-year trajectories—this is the core new work |

**PolicyEngine-US already implements:**

- **Social Security** (OASI, SSDI) - retirement and disability benefits
- **SSI** (Supplemental Security Income) - means-tested aged/disabled benefits
- **SNAP** (food stamps), **Medicaid**, housing assistance
- **EITC, CTC**, and all federal income taxes
- State income taxes for all 50 states

Once we generate synthetic earnings histories and attach them to the ECPS population, calculating benefits across all these programs is automatic—we simply run the panel through PolicyEngine-US's existing rules engine. This means safety net interactions (how Social Security changes affect SNAP eligibility, SSI benefits, etc.) are included from the start, not as future extensions.

**The actual new development is narrow:** Train QRF models on public PSID data to impute earnings histories, then calibrate to public SSA statistics. Everything else—the base dataset, the imputation tools, the calibration framework, the benefit rules—already exists and is production-tested.

## Significance

This model would represent a landmark contribution to open policy analysis. It would be the first open-source dynamic Social Security microsimulation model, enabling researchers, advocates, and policymakers to analyze reforms without proprietary tools. Integration with PolicyEngine's web application will provide public access through an intuitive interface, while a Python package will offer programmatic access for detailed research. Full reproducibility through open-source code and comprehensive documentation will allow independent verification and extension. The model will support individual-level analysis with full distributional impacts, tracking effects across entire lifecycles and multiple generations.

The model will support rigorous analysis of benefit formula changes, retirement age adjustments, tax base modifications, means-testing proposals, progressive indexing mechanisms, minimum benefit proposals, and effects on poverty, inequality, and retirement adequacy across demographic groups. By removing cost and access barriers, the model will democratize sophisticated Social Security policy analysis.

## This Document

The remainder of this planning document provides comprehensive detail on the technical approach and development plan. The Literature Review chapter examines academic research on Social Security microsimulation and dynamic modeling methodologies. The Existing Models chapter provides detailed comparison of DynaSim, CBOLT, MINT, and other tools, including their construction methodologies and how our approach differs. The Technical Specifications chapter outlines the required variables, transition equations, historical simulation process, and behavioral response mechanisms essential for Social Security dynamic microsimulation. The Data Sources chapter describes survey data including PSID, SIPP, and CPS, as well as calibration targets from SSA administrative data. The Methodology chapter presents the technical approach to synthetic panel construction, including quantile regression forest imputation, demographic transition modeling, and gradient descent calibration. The Infrastructure chapter describes PolicyEngine tools including microimpute, microcalibrate, and the Enhanced CPS, demonstrating how proven tools will be extended to longitudinal analysis. The Team chapter introduces the expertise of Max Ghenis, Ben Ogorek, and John Sabelhaus. Finally, the Roadmap chapter outlines development milestones and the timeline for model construction and validation.
