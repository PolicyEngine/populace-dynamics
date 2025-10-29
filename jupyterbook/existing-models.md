# Existing Social Security Microsimulation Models

## Overview

Several organizations have developed sophisticated dynamic microsimulation models for Social Security analysis. This chapter reviews the major existing models, examining their capabilities, methodological approaches, and limitations. Understanding these models provides essential context for positioning our proposed open-source alternative and clarifying how our methodology both builds upon and differs from established approaches.

## DynaSim (Urban Institute)

The Dynamic Simulation of Income Model (DynaSim), developed by the Urban Institute, represents a comprehensive dynamic microsimulation model for retirement policy analysis {cite}`favreault2015`. First developed in the 1970s and continuously updated through multiple versions (currently DynaSim4), it is widely considered the gold standard for non-governmental Social Security modeling.

### Key Features and Methodology

DynaSim begins with SIPP panels, matched to administrative earnings records where available. The model projects the U.S. population forward from the base year through 75 or more years, incorporating comprehensive demographic and economic processes. The demographic component models birth, death, marriage, divorce, immigration, educational attainment, and disability onset and recovery. The economic component models earnings trajectories differentiated by education, race, and sex, labor force participation and retirement decisions, asset accumulation and drawdown, and pension participation {cite}`favreault2015`.

The policy detail encompasses full Social Security benefit rules including retirement, disability, and survivors benefits, Supplemental Security Income (SSI), Medicare, private pensions and 401(k) accounts, and federal income taxes. This comprehensive scope allows analysis of interactions between Social Security and other retirement income sources.

### Strengths and Impact

DynaSim's decades of development and refinement have produced extensive validation against actual outcomes, establishing credibility with researchers and policymakers. The model provides rich demographic and economic detail and can incorporate behavioral responses to policy changes. Urban Institute has used DynaSim to produce influential analyses for Congressional offices evaluating Social Security reforms, academic researchers studying retirement security, advocacy organizations analyzing distributional impacts, and foundations funding retirement policy research.

### Limitations

Despite its analytical power, DynaSim faces significant access and transparency limitations. Access requires contracts with the Urban Institute, creating cost barriers for many potential users. The code is not open-source, and documentation of assumptions remains limited, constraining transparency. Users cannot easily modify assumptions or add new policies without Urban Institute involvement. Published results cannot be independently verified, limiting reproducibility. The model lacks a public web interface and requires substantial technical expertise to use, restricting accessibility to a small community of specialists.

## MINT (SSA Modeling Income in the Near Term)

The Modeling Income in the Near Term (MINT) model is the Social Security Administration's microsimulation model for projecting retirement income {cite}`smith2010mint` {cite}`ssa2024mint`. Developed through collaboration between SSA and the Urban Institute, with contributions from the Brookings Institution and RAND Corporation, MINT has evolved through multiple versions, with MINT8 representing the current iteration.

### Key Features and Methodology

MINT's distinctive strength lies in its access to matched SIPP-administrative earnings records, providing actual earnings histories from SSA's Master Earnings File. For older cohorts, MINT uses actual administrative earnings histories, representing a gold standard for accuracy. For younger cohorts, the model projects earnings forward using statistical models, including quantile regression approaches pioneered in MINT6 {cite}`butrica2006`. The model incorporates family structure dynamics, wealth accumulation patterns matched to the Survey of Consumer Finances, and comprehensive SSI and Social Security benefit calculations {cite}`smith2010mint`.

The projection period focuses on the near term, typically 50 to 75 years forward. MINT uses discrete-time hazard models estimated from PSID and SIPP data to model demographic transitions. The earnings projection methodology employs quantile regression to preserve distributional characteristics across the earnings distribution, an innovation that influenced subsequent microsimulation model development.

### Strengths and Limitations

MINT's access to actual SSA administrative data provides unmatched accuracy for older cohorts' earnings histories. The model carries official SSA endorsement and benefits from extensive technical documentation and regular updates {cite}`ssa2024mint`. SSA uses MINT for actuarial projections and policy analysis, research collaborations with academic institutions, and policy briefs on distributional impacts of reforms.

However, public access remains limited. While SSA makes restricted datasets available to approved researchers, the underlying code is not open-source and relies on SAS implementation. External researchers find it difficult to modify assumptions or extend the model. The scope focuses primarily on SSA programs, with limited integration of broader tax-benefit interactions. Behavioral modeling of responses to reforms remains more limited than in DynaSim.

## CBOLT (Congressional Budget Office Long-Term Model)

The Congressional Budget Office Long-Term Model (CBOLT) serves as CBO's primary analytical tool for making long-term projections of the economy and federal budget {cite}`cbo2018` {cite}`cbo2004`. These projections extend beyond CBO's standard 10-year budget window and underlie recurring CBO publications including The Long-Term Budget Outlook and Social Security Policy Options.

### Model Structure and Capabilities

CBOLT comprises four integrated components: a demographic model, a microsimulation model, a long-term budget model, and a policy growth model {cite}`cbo2018`. The Social Security component calculates benefits, projects trust fund dynamics, analyzes interactions with broader fiscal policy, and incorporates macroeconomic feedbacks. The projection horizon typically extends 30 or more years, matching CBO's long-term analytical window.

The model emphasizes integration with broader fiscal modeling and macroeconomic consistency. CBO uses CBOLT for official cost estimates of Social Security legislation, long-term budget outlook reports, and analysis of entitlement reform proposals. Regular updates align with CBO's annual budget outlook publications.

### Strengths and Limitations

CBOLT's integration with CBO's broader analytical framework ensures macroeconomic consistency and fiscal coherence. The model carries the authority of official CBO analysis. However, access is restricted to internal CBO use only, with no availability for external researchers. Public documentation of methods remains limited compared to academic models. The level of distributional detail is less granular than DynaSim or MINT, as CBOLT emphasizes aggregate fiscal projections over individual-level heterogeneity. External researchers cannot replicate CBO analyses independently, limiting reproducibility and transparency.

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

Understanding the technical approaches used by existing models provides essential context for positioning our methodology and clarifying our design choices. This section draws on published technical documentation {cite}`favreault2014` {cite}`smith2010mint` {cite}`butrica2006` to examine how different models construct longitudinal panels and calibrate to external targets.

### DynaSim's Approach

DynaSim begins with SIPP panels containing approximately 50,000 households {cite}`favreault2014`. The panel construction process employs statistical matching to administrative earnings records where possible, though access to administrative data is restricted. For individuals without matched administrative records, DynaSim imputes earnings using regression-based methods. The model ages the population forward using transition probability matrices, with transition models for marriage, divorce, death, and disability estimated from historical SIPP panels.

The calibration methodology relies on an alignment technique that adjusts transition probabilities iteratively to hit aggregate targets. For example, if the model projects too few marriages relative to Census targets, marriage probabilities are increased proportionally for the affected demographic groups. This iterative adjustment process calibrates to Census population projections, SSA aggregate totals, and economic forecasts, but does not employ optimization-based methods.

DynaSim's approach offers several strengths: it uses a real survey base with observed characteristics, benefits from decades of refinement and extensive validation, and has established credibility through numerous published studies {cite}`favreault2015`. However, the SIPP sample is substantially smaller than the CPS, limiting detailed geographic and demographic disaggregation. The alignment technique, while practical, can distort individual-level heterogeneity because it adjusts probabilities uniformly within broad groups. Transition models may not capture the full complexity of earnings dynamics observed in longitudinal data. The approach is not fully synthetic, as results depend on which SIPP cohort is used as the starting point.

### MINT's Approach

MINT's methodological foundation rests on SIPP data matched to SSA's Master Earnings File, providing access to actual administrative earnings histories {cite}`smith2010mint`. This access to administrative records fundamentally distinguishes MINT from other models and shapes its construction methodology.

For older cohorts, MINT uses actual administrative earnings histories from SSA records, representing the gold standard for historical earnings accuracy. For younger cohorts who have limited or no administrative history, the model projects earnings forward using statistical models. The earnings projection methodology employs quantile regression models differentiated by age, education, and gender {cite}`butrica2006`. MINT pioneered the use of quantile regression for distributional earnings projection in microsimulation, recognizing that traditional mean regression fails to preserve heterogeneity across the earnings distribution. Our proposed QRF approach extends MINT's innovation by incorporating machine learning and full distribution prediction.

Demographic transitions use discrete-time hazard models estimated from PSID and SIPP data. Wealth distributions are matched to the Survey of Consumer Finances to provide comprehensive retirement resource projections. Benefit calculations apply SSA rules directly to actual or projected earnings histories.

The calibration methodology relies heavily on administrative data, reducing the need for extensive reweighting. Younger cohort projections are aligned to SSA Trustees assumptions regarding future earnings growth, labor force participation, and demographic trends. Wealth distributions are calibrated to SCF aggregates by age and income groups.

MINT's access to real earnings histories for older cohorts provides unmatched accuracy for historical periods and official SSA endorsement establishes credibility {cite}`ssa2024mint`. The strong validation base against actual benefit receipt data demonstrates the value of administrative data access. However, restricted access means the methodology is not publicly replicable. The SIPP sample remains smaller than CPS, limiting some types of detailed analysis. Projections for younger cohorts remain model-dependent, introducing uncertainty that grows with projection horizon. Most fundamentally, researchers cannot replicate MINT analyses without administrative data access, limiting independent verification.

### CBOLT's Approach

CBOLT employs a different methodological framework than DynaSim or MINT, focusing on representative agents and stylized household types rather than individual-level microsimulation {cite}`cbo2018` {cite}`cbo2004`. The panel construction uses representative households categorized by age, income, and family structure, with aggregate earnings profiles rather than individual histories. This approach prioritizes macroeconomic consistency over distributional detail.

The calibration methodology emphasizes alignment to National Income and Product Accounts (NIPA), matching aggregate labor force participation and earnings patterns. Demographic projections derive from Census estimates. The model performs less individual-level calibration than DynaSim or MINT, focusing instead on aggregate fiscal consistency and integration with CBO's broader budget model.

CBOLT's representative agent framework offers computational efficiency and ensures macroeconomic consistency with CBO's broader fiscal projections {cite}`cbo2018`. However, the limited distributional detail prevents detailed analysis of impacts on specific demographic subgroups. The model is less suitable for micro-level reform analysis that requires understanding heterogeneous effects across the income or age distribution.

### Our Approach: Fully Synthetic Panel

Our proposed methodology begins with CPS ASEC data containing approximately 200,000 individuals, substantially larger than SIPP-based models. The approach is fully synthetic, requiring no administrative data matching, which ensures complete public replicability. We employ quantile regression forests (QRF) to predict the full conditional distribution of earnings at each age, extending MINT's quantile regression innovation with modern machine learning methods {cite}`meinshausen2006`. Training uses PSID longitudinal data from public use files, allowing anyone to replicate the methodology. The process generates complete lifetime earnings histories for the entire CPS sample. Demographic transitions use hazard models estimated on PSID, similar to MINT and DynaSim but applied to a larger CPS base.

The calibration methodology employs gradient descent reweighting, an optimization-based approach that minimizes distance from original CPS weights subject to matching all target variables including earnings distributions and beneficiary counts {cite}`deville1992`. This method can handle hundreds or thousands of simultaneous targets, as demonstrated in the Enhanced CPS construction {cite}`ghenis2024`. The approach is mathematically principled, ensuring the solution minimizes a well-defined objective function, and computationally efficient, converging rapidly even with large datasets and many targets.

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

### The Enhanced CPS Precedent: Validation of the Synthetic Data Approach

PolicyEngine has already demonstrated that this methodological approach works for cross-sectional analysis. The same reproducibility challenge exists in tax-benefit microsimulation: all major models including Tax Policy Center, Penn Wharton Budget Model, and Tax Foundation rely on the IRS Public Use File (PUF), which cannot be publicly shared due to privacy restrictions. This creates a fundamental reproducibility crisis in tax policy research.

PolicyEngine addressed this crisis by developing the Enhanced CPS (eCPS), which is the only publicly available cross-sectional microdata file that produces accurate tax-benefit microsimulation impacts {cite}`ghenis2024`. The eCPS construction employs a two-stage methodology. First, quantile regression forests impute missing or underreported variables from the PUF onto the CPS base, creating the Extended CPS with doubled sample size. Second, gradient descent optimization reweights the dataset to match over 7,000 administrative targets from IRS Statistics of Income, Census, and other sources. This approach achieves accuracy comparable or superior to PUF-based models while maintaining full reproducibility and transparency.

Validation demonstrates the effectiveness of this approach. PolicyEngine-US revenue estimates match Joint Committee on Taxation and Treasury estimates for major tax reforms. Distributional analysis matches Tax Policy Center's published tables. Individual-level calculations validate against actual tax returns where available. Congressional offices use PolicyEngine for actual policy analysis, demonstrating real-world credibility.

The parallel to this project is direct. In cross-sectional tax modeling, the gold standard restricted data is the IRS PUF, the common approach uses the PUF but cannot share results publicly, and our eCPS approach combines CPS, machine learning imputation, and calibration to achieve accurate and reproducible results. In longitudinal Social Security modeling, the gold standard restricted data is SSA earnings records, the common approach uses administrative data but cannot share results publicly, and our synthetic panel approach will combine CPS, QRF imputation from PSID, and calibration to achieve accurate and reproducible results.

The eCPS development required over two years of intensive development and validation. This investment produced proven machine learning imputation methods implemented in microimpute, proven calibration methods implemented in microcalibrate, a validated framework for assessing accuracy against administrative targets, and credibility from accurate cross-sectional results that match or exceed proprietary models. This project applies those same tools to the longitudinal dimension, significantly reducing methodological risk. The infrastructure exists, the methods are proven, and the team has experience executing this exact type of project.

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
