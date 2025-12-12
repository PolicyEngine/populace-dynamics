# Internal Advisory Feedback: Social Security Microsimulation Model Planning Document

**From:** John Sabelhaus, Senior Fellow, Wisconsin Center for Financial Security
**Date:** December 12, 2025
**Re:** Technical review of planning document for funder presentation

---

## Executive Summary

This planning document presents a technically sound and ambitious proposal to build the first fully open-source Social Security dynamic microsimulation model. The core methodology—extending the proven Enhanced CPS approach from cross-sectional to longitudinal analysis—is fundamentally solid. The team has the right expertise, and the Enhanced CPS precedent substantially de-risks what would otherwise be a highly uncertain research project.

However, before presenting to funders, several areas require strengthening: (1) more explicit treatment of wealth dynamics and their integration with Social Security analysis, (2) clearer specification of which base cross-sectional dataset will be used and why, (3) deeper engagement with the Survey of Consumer Finances as a calibration source, (4) more realistic discussion of what can be delivered in 18 months versus longer-term extensions, and (5) stronger connection to the academic retirement security literature.

The document is strongest on technical methodology (QRF imputation, gradient descent calibration) and weakest on articulating the broader economic and distributional questions the model will answer. Funders will want to understand not just *how* you'll build it, but *what specific policy questions* it will illuminate that existing models cannot.

---

## Part I: Technical Methodology Assessment

### 1. Quantile Regression Forest (QRF) Approach

**Strengths:**
- The QRF methodology for earnings history imputation is technically appropriate and represents a clear advance over traditional mean regression approaches used in earlier models
- The year-by-year training strategy (separate models for each age) is sophisticated and will better capture life-cycle dynamics than pooled approaches
- Recognition of cohort effects is critical—earnings profiles differ substantially across birth cohorts due to educational expansion, women's labor force participation changes, and structural economic shifts
- The PSID is the right training data source; it's the gold standard for longitudinal earnings dynamics in the U.S.

**Concerns and Recommendations:**

**1.1 Sample Size and Statistical Power:**
The PSID has ~9,000 families currently, but effective sample sizes for specific demographic groups (e.g., Black women born in the 1960s) become quite small. When training separate QRF models for each age (18-70 = 53 models), you're further fragmenting the training sample.

*Recommendation:* Be explicit about minimum cell sizes required for QRF estimation. You may need to pool adjacent ages or use ridge regression/elastic net to borrow strength across ages. Consider publishing a technical appendix showing effective sample sizes by demographic group and age.

**1.2 Censoring and Top-Coding:**
Both PSID and CPS have earnings censoring issues, particularly at high earnings levels. Social Security's maximum taxable earnings cap creates additional censoring. Your methodology discussion doesn't address how you'll handle this.

*Recommendation:* Add a section on handling censored observations. Consider Tobit-style approaches or conditional imputation above the censoring threshold. This matters enormously for distributional analysis—the top 10% of earners pay a disproportionate share of payroll taxes and drive aggregate revenue estimates.

**1.3 Zero Earnings and Labor Force Attachment:**
A substantial share of observations (particularly women with children) have zero earnings in any given year. Traditional QRF struggles with mass points at zero. How will you model the extensive margin (work/not work) versus intensive margin (earnings conditional on working)?

*Recommendation:* Consider a two-stage approach: (1) logistic regression for labor force participation, then (2) QRF for earnings conditional on working. Document the share of person-years with zero earnings and validate against SSA statistics on covered employment.

**1.4 Validation Against Hold-Out PSID Sample:**
The methodology section mentions validation but doesn't provide detail on the hold-out strategy.

*Recommendation:* Commit to k-fold cross-validation on PSID with published results. Show that predicted earnings histories for held-out PSID individuals match actual histories in terms of:
- Mean absolute error by age and demographic group
- Variance of earnings changes
- Autocorrelation structure
- Quantile coverage (do 80% of actual outcomes fall within predicted 10th-90th percentile range?)

This validation is critical for establishing credibility. DynaSim and MINT can point to actual administrative data; you need to demonstrate that QRF imputation from PSID produces comparably realistic trajectories.

### 2. Gradient Descent Calibration

**Strengths:**
- The gradient descent reweighting approach is mathematically principled and proven through the Enhanced CPS development
- Handling 7,000+ targets in the ECPS demonstrates computational feasibility
- The hierarchical prioritization of targets (Tier 1/2/3) is sensible
- Recognition that some targets may conflict and require trade-offs shows appropriate realism

**Concerns and Recommendations:**

**2.1 Longitudinal Calibration Targets:**
Cross-sectional calibration (matching age-sex-earnings distributions in a single year) is conceptually straightforward. Longitudinal calibration (matching transition matrices, earnings mobility patterns) is more complex because you're calibrating correlations across time, not just marginal distributions.

*Recommendation:* Provide more detail on how gradient descent handles longitudinal constraints. For example, how do you ensure that calibrated weights produce realistic 5-year earnings mobility matrices while simultaneously matching cross-sectional earnings percentiles? This may require iterative calibration or explicit weighting of longitudinal vs. cross-sectional targets.

**2.2 Identifiability and Degrees of Freedom:**
With a CPS sample of ~200,000 individuals and potentially 1,000+ calibration targets, you have ample degrees of freedom for reweighting. But add longitudinal dimensions (35 years of earnings history × multiple demographic groups × mobility matrices), and the constraint space becomes enormous.

*Recommendation:* Discuss identifiability explicitly. How many independent degrees of freedom do you really have? What happens when targets conflict? The ECPS calibrates cross-sectional tax data where conflicts are primarily about levels. Longitudinal calibration involves both levels and correlations, which may be harder to reconcile.

**2.3 Calibration Target Quality and Consistency:**
Your calibration targets come from multiple sources: SSA, Census, IRS, NVSS. These sources use different survey methodologies, have different coverage, and may not be perfectly consistent with each other.

*Recommendation:* Add a section on "Calibration Target Quality Assessment." Document:
- Which targets are based on administrative data (high confidence) vs. survey estimates (lower confidence)
- Known inconsistencies between data sources
- How you'll handle conflicts when targets from different sources imply different population totals

For example, CPS and ACS often produce different estimates of the same population characteristics. Which takes precedence?

### 3. Demographic Transitions

**Strengths:**
- The discrete-time hazard model approach for marriage, divorce, fertility, disability, and mortality is standard and appropriate
- Recognition of differential mortality by earnings/education is critical—this drives substantial heterogeneity in lifetime Social Security benefits
- Spousal matching with assortative mating patterns is essential for accurate spousal and survivor benefit modeling

**Concerns and Recommendations:**

**3.1 Mortality and Life Expectancy Differentials:**
This is excellent. The document correctly identifies that differential mortality by socioeconomic status is crucial for distributional analysis. The Opportunity Insights life expectancy data by income percentile provides essential calibration targets.

*Recommendation:* Strengthen this section by being more explicit about the policy importance. Life expectancy gaps between rich and poor have widened dramatically over recent decades. This means:
- High earners receive benefits over more years, getting higher lifetime returns
- Regressive net transfers despite progressive benefit formula
- Changes to retirement ages disproportionately affect lower-income workers

Make this connection explicit in the introduction. It's a compelling policy motivation that existing models don't always handle well.

**3.2 Disability Dynamics:**
The methodology section treats disability as a simple onset hazard with low recovery rates. The reality is more complex:
- SSDI application process has long lags and uncertain outcomes
- Substantial employment transitions before formal disability onset
- Interaction between disability and early Social Security claiming

*Recommendation:* Add nuance to disability modeling. Consider:
- Pre-disability earnings decline (3-5 years before SSDI receipt)
- SSDI application and award rates (not everyone who becomes disabled receives benefits)
- Interaction between disability status and early retirement claiming (many disabled workers claim retirement benefits at 62 rather than applying for SSDI)

This matters for policy analysis of disability reforms.

**3.3 Spousal Matching and Assortative Mating:**
The document mentions assortative mating patterns but doesn't specify the matching algorithm.

*Recommendation:* Be more specific about spousal matching methodology. Will you:
- Use a matching algorithm (e.g., Gale-Shapley stable matching with distance penalty on age/education/earnings)?
- Impute spouse characteristics from conditional distributions?
- Match actual married couples from CPS and preserve their characteristics?

Spousal earnings correlation matters enormously for household Social Security wealth. This deserves more methodological detail.

---

## Part II: Data Sources and Limitations

### 4. Base Cross-Sectional Dataset: A Critical Decision

**Critical Concern:**
The document repeatedly states "base dataset to be determined during proof of concept" and lists options including the Enhanced CPS. This is too vague for a funder proposal. The choice of base dataset has profound implications for sample size, geographic detail, and integration with existing PolicyEngine infrastructure.

**Why This Matters:**
- **CPS ASEC (~200,000 individuals):** Large sample, state-level detail, already integrated with PolicyEngine-US, but income underreporting issues
- **Enhanced CPS:** Addresses underreporting through imputation and calibration, proven accuracy, but doubled sample may create computational challenges
- **SIPP (varies by panel):** Better benefit receipt data, panel structure, but smaller and shorter panel duration
- **Starting fresh with CPS:** Maximum flexibility but loses Enhanced CPS investment

*Recommendation:* **Make a decision before the proposal.** I strongly recommend starting with the Enhanced CPS for these reasons:

1. **Proven methodology:** You've already solved the cross-sectional underreporting problem
2. **Integration:** Seamless connection to PolicyEngine-US's existing tax-benefit calculations
3. **Credibility:** Funders want to build on proven success, not restart from scratch
4. **Sample size:** ECPS's doubled sample (~400,000 individuals) provides ample statistical power

The proof-of-concept phase should validate that ECPS + longitudinal imputation works, not re-litigate which base dataset to use.

**Alternative view:** If computational constraints are severe, you might use a stratified subsample of ECPS for the initial dynamic model (e.g., 100,000 individuals selected via L0 regularization to maintain representativeness). But start with ECPS as the conceptual foundation.

### 5. PSID as Training Data: Strengths and Limitations

**Strengths (well-documented):**
- Longest-running U.S. panel
- True longitudinal structure with intergenerational links
- Detailed earnings histories
- Rich demographic transitions

**Limitations (need more explicit treatment):**

**5.1 Sample Size:**
As noted above, ~9,000 families is small for training 50+ age-specific QRF models across demographic subgroups.

**5.2 Representativeness:**
PSID oversamples low-income families (by design from original poverty study). Even with sample weights, PSID may not perfectly represent the full earnings distribution, especially at the top.

*Recommendation:* Explicitly discuss PSID representativeness issues and how calibration to SSA targets addresses this. Consider supplementing PSID training with SIPP for short-term dynamics and using published earnings mobility matrices from SSA studies.

**5.3 Survey vs. Administrative Earnings:**
PSID earnings are self-reported, not administrative records. There's measurement error and potential underreporting.

*Recommendation:* Acknowledge this limitation and note that calibration to SSA targets will correct for systematic biases. This is another reason why calibration is critical—even with PSID training, you need external validation.

### 6. Survey of Consumer Finances: The Missing Piece

**Critical Gap:**
The document mentions the SCF in passing but doesn't adequately address wealth dynamics. This is a significant weakness for a retirement security model.

**Why Wealth Matters for Social Security Analysis:**

1. **Benefit claiming decisions:** Wealth-rich households can afford to delay claiming to age 70 for higher benefits
2. **Replacement rates:** Social Security replaces a higher share of consumption for wealth-poor households
3. **Distributional analysis:** Comprehensive retirement security analysis requires looking at Social Security wealth + financial wealth + pension wealth + housing wealth
4. **Policy reforms:** Means-testing proposals require wealth data
5. **Adequacy assessment:** Can people afford retirement? Need total resources, not just Social Security

**My Work on Wealth and Social Security:**
At the Federal Reserve Board, I oversaw the Survey of Consumer Finances, which is the gold standard for U.S. household wealth data. The SCF is carefully designed to capture the full wealth distribution, including oversampling high-wealth households. SCF data are essential for:
- Validating wealth distributions by age and lifetime earnings
- Understanding retirement preparedness
- Analyzing distributional effects of Social Security reforms
- Connecting Social Security to broader retirement security

*Recommendation:* **Expand the calibration targets section to include SCF wealth targets.** Specifically:

**Phase 1 (Essential):**
- Net worth distribution by age group and income quintile
- Financial asset holdings by age
- Housing wealth by age and homeownership status
- Ratio of wealth to lifetime earnings by demographic group

**Phase 2 (Extensions):**
- Pension wealth (DB and DC)
- Annuitized wealth (including Social Security wealth)
- Comprehensive retirement wealth

Even if your initial model doesn't have a full wealth accumulation module (which I understand given time constraints), you should at least impute wealth statistically from the SCF and validate that imputed wealth distributions match SCF aggregates. This is parallel to what you're doing with PSID for earnings—statistical matching to transfer distributional properties.

**Technical approach:** Use random forest imputation to transfer SCF wealth distributions to your synthetic panel based on age, earnings history, marital status, and education. Then calibrate to ensure aggregate wealth holdings match SCF.

**Why this matters for funders:** Retirement security analysis that ignores wealth is incomplete. Funders interested in Social Security policy care about total retirement income, not just Social Security benefits in isolation. This is especially important for distributional analysis—Social Security may appear progressive in terms of benefit formula, but combining with wealth reveals a more complex picture.

### 7. Calibration Targets: Strengthening the Framework

**Strengths:**
- Comprehensive list of targets across demographics, labor markets, benefits, earnings dynamics, and mortality
- Appropriate prioritization (Tier 1/2/3)
- Specific sources documented (SSA, Census, IRS, NVSS)
- Inclusion of Opportunity Insights mobility and mortality data is excellent

**Recommendations for Enhancement:**

**7.1 Add Explicit Wealth Targets (as discussed above):**
From SCF:
- Median net worth by age quintile
- Share of households with $0-$10k, $10k-$50k, $50k-$250k, $250k+ in financial assets
- Housing wealth and homeownership rates by age and income
- Pension coverage and DC account balances

**7.2 Add Retirement Claiming Behavior Targets:**
From SSA administrative data:
- Share claiming at 62, FRA, 63-FRA-1, FRA, FRA+1-69, 70
- Claiming patterns by AIME quintile
- Spouse vs. own benefit claiming patterns
- Disability conversions to retirement benefits at FRA

This is essential for validating behavioral responses to reform.

**7.3 Add Lifetime Benefit Distribution Targets:**
From SSA's Modeling Income in the Near Term (MINT) published analyses:
- Distribution of lifetime Social Security benefits by birth cohort
- Lifetime benefits as a share of lifetime earnings (replacement rates)
- Internal rates of return on Social Security contributions by demographic group

These provide validation of your full lifecycle simulation, not just cross-sectional benefit distributions.

**7.4 Add Family Structure Targets:**
From Census and CPS:
- Share of beneficiaries receiving spouse vs. own benefits
- Share of widow(er) beneficiaries
- Dependent child beneficiaries
- Average family maximum benefit scenarios

Family benefit rules are complex, and validation of family structures is essential.

**7.5 Validate Against Published MINT/DynaSim Results:**
Where available, use published results from existing models as informal validation targets:
- MINT projections of Social Security benefits by cohort and quintile
- DynaSim distributional analyses of specific reforms
- CBO long-run Social Security projections

This allows comparison to established models without requiring access to proprietary code.

---

## Part III: Validation Strategy

### 8. Multi-Level Validation Framework

**Strengths:**
- The document outlines validation at multiple levels: cross-sectional, longitudinal, distributional, and fiscal
- Recognition that validation must occur against both historical data and forward projections
- Commitment to external benchmarking against published MINT/DynaSim results

**Recommendations for Strengthening:**

**8.1 Publish Comprehensive Validation Documentation:**
Before public launch, publish a validation report that covers:

**Level 1: Input Data Quality**
- PSID earnings trajectories vs. SSA published cohort earnings studies
- CPS benefit receipt vs. SSA administrative totals
- Demographic distributions vs. Census

**Level 2: Imputation Quality**
- Hold-out PSID validation: do imputed histories match actual histories?
- Quantile coverage tests: are prediction intervals well-calibrated?
- Correlation structure: do imputed histories preserve autocorrelation?

**Level 3: Cross-Sectional Calibration**
- Age-earnings profiles by demographic group
- Beneficiary counts by type and age
- Average benefit amounts by type
- Benefit distributions (percentiles)

**Level 4: Longitudinal Validation**
- Earnings mobility matrices (5-year transitions)
- Variance decomposition (between-person vs. within-person)
- Cohort-specific age-earnings profiles

**Level 5: Fiscal Aggregates**
- Total OASDI benefit payments vs. SSA
- Total covered earnings vs. SSA
- Trust fund projections (10-year and 30-year)

**Level 6: Policy-Relevant Outcomes**
- Replacement rates by lifetime earnings quintile
- Lifetime benefit distribution by cohort
- Distributional progressivity measures
- Poverty rates among elderly by demographic group

**8.2 Establish Success Criteria Thresholds:**
The roadmap lists success criteria (e.g., "earnings distributions within 5% of SSA") but doesn't explain what happens if you miss these targets. Be realistic: achieving ±2% accuracy across hundreds of targets is extremely difficult.

*Recommendation:* Define "acceptable" vs. "target" vs. "stretch" performance:
- **Acceptable:** Core fiscal aggregates within ±5%, distributional measures within ±10%
- **Target:** Core aggregates within ±3%, distributional within ±7%
- **Stretch:** Core aggregates within ±2%, distributional within ±5%

Be clear that even "acceptable" performance would make this the most transparent Social Security model available and a major contribution to the field.

**8.3 Compare to Existing Model Accuracy:**
Don't just compare to administrative data—compare to how well DynaSim and MINT perform on the same validation metrics. This is harder because those models don't publish comprehensive validation statistics, but some are available in technical documentation.

*Recommendation:* Frame validation as: "Our model performs comparably to DynaSim/MINT on available validation metrics, while being fully open-source and reproducible."

---

## Part IV: Technical Specifications and Behavioral Responses

### 9. Historical Simulation and the Jump-Off Problem

**Strengths:**
- The technical specifications chapter (clearly informed by my guidance) correctly identifies the jump-off problem as central to dynamic microsimulation
- Recognition that calibration factors are needed to prevent drift
- Appropriate skepticism about OLG macro models and emphasis on "following the money"

**Additional Recommendations:**

**9.1 Distinguish Historical Validation from Forward Projection:**
Your historical simulation (creating lifetime earnings histories for the current population) is fundamentally different from forward projection (aging the population into the future). Make this distinction clearer:

- **Historical simulation:** Validate against realized outcomes (SSA historical data)
- **Forward projection:** Align to SSA Trustees assumptions but acknowledge uncertainty

**9.2 Stochastic Projections:**
SSA Trustees publishes low-cost and high-cost scenarios in addition to intermediate projections. Your model should be capable of stochastic simulation using alternative demographic and economic assumptions.

*Recommendation:* In Phase 5, implement scenario analysis:
- Low, intermediate, high fertility/mortality/wage growth
- Monte Carlo simulation drawing from distributions of parameters
- Confidence intervals around long-run projections

This is essential for policy analysis—reforms that look sustainable under intermediate assumptions may fail under adverse scenarios.

### 10. Behavioral Responses: Start Simple, Build Incrementally

**Strengths:**
- Appropriate recognition that "textbook optimizing economic behavior" is not required initially
- Focus on essential behavioral margins: labor supply and benefit claiming
- Reduced-form approach using empirical elasticities

**Recommendations:**

**10.1 Initial Model: Minimal Behavioral Response**
For the first version (18-month timeline), I recommend limiting behavioral response to:
- **Benefit claiming:** Empirical hazard model based on age, AIME, spouse status, health
- **No labor supply response:** Assume earnings trajectories are invariant to policy (acknowledge this limitation explicitly)

*Justification:* Labor supply responses to Social Security reforms are small (most workers don't understand the incentives), and modeling them requires strong assumptions. Benefit claiming responses are larger and better documented.

**10.2 Extensions: Add Labor Supply Gradually**
In post-launch extensions, add:
- Retirement age responses to benefit changes (larger and better documented)
- Hours worked responses for workers near retirement (smaller but tractable)
- Lifetime earnings responses (very difficult, long-run general equilibrium effects)

Be clear in the proposal that labor supply responses are a Phase 7+ extension, not part of the 18-month core deliverable.

**10.3 What Matters Most: Distributional Heterogeneity**
The biggest value of microsimulation over OLG models is distributional detail, not behavioral sophistication. Even with zero behavioral response, your model will be valuable for:
- Who wins and who loses from specific reforms?
- How do reforms affect poverty, inequality, and adequacy?
- What are the distributional consequences across race, gender, education, lifetime earnings?

*Recommendation:* Emphasize this in the funder pitch. Existing models (especially CBOLT) sacrifice distributional detail for macroeconomic consistency. Your model makes the opposite trade-off—rich distributional analysis with transparent (if simplified) behavioral assumptions.

---

## Part V: Infrastructure, Team, and Roadmap

### 11. Infrastructure: Building on Proven Foundations

**Strengths:**
- The infrastructure chapter effectively communicates that this isn't starting from scratch
- Clear articulation of how Enhanced CPS validates the synthetic data approach
- Detailed description of microimpute, microcalibrate, and L0 tools

**Recommendations:**

**11.1 Emphasize De-Risking:**
This is your strongest selling point. You're not proposing untested methods—you're extending proven tools to a new dimension.

*Recommendation:* Add a "Risk Mitigation" callout box in the introduction:
> "This project extends PolicyEngine's Enhanced CPS methodology, which has already demonstrated that public data + machine learning + calibration can match restricted administrative data accuracy. The core tools (microimpute, microcalibrate) are battle-tested in production. We're extending proven methodology from cross-sectional to longitudinal analysis, not inventing new methods."

**11.2 Clarify Enhanced CPS as Starting Point:**
As noted earlier, commit to using Enhanced CPS as the base dataset. This makes the infrastructure story cleaner and more compelling.

**11.3 Add Technical Diagram:**
The Mermaid flowchart in the methodology chapter is good, but add a more detailed technical architecture diagram showing:
- Data inputs (CPS/ECPS, PSID, SSA targets)
- Processing pipeline (imputation → calibration → validation)
- Storage (HDF5 files, cloud infrastructure)
- Outputs (API, web interface, Python package)
- Integration with existing PolicyEngine components

### 12. Team Composition: Appropriate Expertise

**Strengths:**
- Max Ghenis: Proven track record with Enhanced CPS, software engineering, project management
- Ben Ogorek: Statistical methods, QRF expertise, validation frameworks
- John Sabelhaus (me): Social Security domain knowledge, Federal Reserve/CBO modeling experience, academic credibility

**Recommendations:**

**12.1 Expand Team Chapter:**
The team chapter is quite brief (1.5 pages). Funders want to understand why this team can succeed where others might fail.

*Recommendation:* Expand to 3-4 pages with:
- More detailed CVs (publications, relevant projects, specific technical skills)
- Track record of collaboration (PolicyEngine development history)
- Advisory board or external reviewers (academics, SSA/CBO connections)
- Diversity of expertise: software engineering + statistics + economics

**12.2 Add Advisory Board:**
Even informal advisors add credibility. Consider listing:
- Academic economists specializing in Social Security (Gary Burtless, Alan Gustman, etc.)
- Former SSA or CBO staff with modeling experience
- Technical experts in microsimulation from other countries (EUROMOD, SimPaths developers)

This isn't about doing the work—it's about showing funders that respected experts endorse the approach.

**12.3 Personnel Time Allocation:**
The roadmap estimates 27 person-months (1.5 FTE) over 18 months. This seems low for the scope described.

*Reality check:*
- Enhanced CPS took 2+ years of intensive development (you mention this)
- Adding longitudinal dimension is at least as complex
- Demographic transitions, validation, web interface all require significant effort

*Recommendation:* Either:
1. Revise timeline to 24 months with same person-months (more realistic), or
2. Reduce scope of 18-month deliverable (e.g., defer web interface to Phase 7), or
3. Add more personnel resources (2.0-2.5 FTE may be more realistic)

My suggestion: Propose 24 months for full deliverable, with 18-month "Version 1.0" that includes API but not full web interface. This manages expectations while maintaining ambitious timeline.

---

## Part VI: Timeline and Milestones

### 13. Roadmap Realism

**Overall Assessment:**
The 18-month timeline is aggressive but potentially feasible given that (1) Enhanced CPS infrastructure exists, (2) team has experience with similar projects, and (3) tools are already built. However, some phases may be too optimistic.

**Specific Concerns:**

**13.1 Phase 1 (Months 1-3): Proof of Concept**
*Timeline:* Realistic
*Concern:* This phase should include the base dataset decision, not defer it

**13.2 Phase 2 (Months 4-6): Full Earnings History Imputation**
*Timeline:* Optimistic
*Concern:* Training 50+ QRF models, validating hold-out performance, implementing consistency checks, and generating multiple imputations is substantial work. MINT's earnings projection methodology (which is simpler than QRF) took years to develop and validate.

*Recommendation:* Consider extending to months 4-7 (4 months instead of 3)

**13.3 Phase 3 (Months 7-9): Demographic Transitions**
*Timeline:* Realistic
*Concern:* Hazard models are relatively straightforward, but spousal matching and disability dynamics add complexity

**13.4 Phase 4 (Months 10-12): Social Security Benefit Calculation**
*Timeline:* Optimistic
*Concern:* PolicyEngine-US already implements Social Security rules, but integrating with synthetic panel, validating all benefit types, and testing edge cases could reveal issues requiring substantial debugging

*Recommendation:* This phase is critical for credibility. Don't rush it. Consider months 10-13 (4 months)

**13.5 Phase 5 (Months 13-15): Forward Projection and Calibration**
*Timeline:* Very Optimistic
*Concern:* This is where the jump-off problem appears. Forward projection with multi-year calibration is conceptually complex and computationally intensive. Debugging drift issues could take substantial time.

*Recommendation:* This deserves 4-5 months (months 13-17 or 18)

**13.6 Phase 6 (Months 16-18): Web Interface and API**
*Timeline:* Realistic if API-only; optimistic if full web interface
*Concern:* PolicyEngine's web app is sophisticated. Adding dynamic Social Security analysis with lifetime visualizations requires non-trivial frontend development.

*Recommendation:* Split this into:
- Phase 6A: API development (months 18-20)
- Phase 6B: Web interface (months 21-24 or post-launch)

**Alternative Timeline Proposal:**

**Version 1.0 (18 months):**
- Phases 1-5: Full synthetic panel with validation (months 1-18)
- Deliverable: Python package, API documentation, validation report
- Sufficient for researchers and policy analysts

**Version 2.0 (24 months):**
- Phase 6: Full web interface and public launch (months 19-24)
- Deliverable: Public-facing web app for non-technical users

This manages expectations while maintaining momentum. Researchers can use the API from month 18, while broader public access follows at month 24.

---

## Part VII: Framing for Funders

### 14. What's Missing: The Policy Motivation

**Critical Feedback:**
The document is technically excellent but doesn't adequately answer: "Why does this matter for policy?" The introduction discusses restricted access to existing models, but it doesn't clearly articulate what **specific policy questions** this model will illuminate.

**Recommended Addition: "Policy Applications" Section**

Add a new section (perhaps after the Executive Summary) titled "Key Policy Questions This Model Will Answer" with concrete examples:

**Application 1: Distributional Effects of Raising the Retirement Age**
Proposals to increase the Full Retirement Age to 68 or 70 are frequently debated. But who is affected?
- Workers with shorter life expectancy (lower-income, minorities) receive benefits over fewer years
- Physically demanding jobs may force earlier claiming with actuarial reductions
- This model can show lifetime benefit losses by demographic group, occupation, and lifetime earnings quintile

**Application 2: Progressive Indexing and Benefit Formula Changes**
Progressive indexing proposals would reduce benefits for higher earners while maintaining benefits for lower earners. But implementation requires defining "higher earners" based on lifetime earnings:
- This model can simulate exact distributional effects across the entire lifetime earnings distribution
- Compare different thresholds and indexing formulas
- Show effects on poverty rates, replacement rates, and adequacy

**Application 3: Means-Testing Proposals**
Some proposals would reduce benefits for wealthy retirees. But means-testing requires data on:
- Total retirement resources (Social Security + pensions + savings + housing)
- This model can integrate wealth data (via SCF imputation) to assess means-testing proposals
- Show behavioral responses (would people save less if benefits are means-tested?)

**Application 4: Effects on Women and Families**
Social Security spousal and survivor benefits create complex incentives:
- How do benefit rules affect lifetime benefits for divorced women, widows, single mothers?
- This model provides individual-level analysis of family benefit rules
- Can analyze reforms to marriage-neutral benefits

**Application 5: Racial Equity Analysis**
Differential mortality creates racial disparities in lifetime Social Security benefits:
- Black men have lower life expectancy → fewer years of benefits
- This model can quantify these effects and simulate reforms to address them
- Example: Adjust benefit formula to account for differential mortality

*Recommendation:* Add this section with 3-5 concrete policy applications. Make them specific, quantitative, and tied to current policy debates. This gives funders concrete examples of what their investment will enable.

### 15. Academic and Policy Impact Framing

**Recommended Addition: "Contribution to the Field" Section**

Add a section explaining why this is a **scientific contribution**, not just a tool:

**Academic Contribution 1: Methodological Validation**
This project tests whether fully synthetic panel data (QRF + calibration) can match restricted administrative data quality. If successful, this methodology could be applied to other domains:
- Pension microsimulation in countries without administrative data
- Health insurance modeling using synthetic panels
- Tax microsimulation in developing countries

**Academic Contribution 2: Open Science in Policy Analysis**
Economics and policy analysis have a reproducibility crisis. Published results from proprietary models cannot be independently verified. This model makes all code, data sources, and assumptions fully transparent:
- Any researcher can replicate results
- Academic papers using this model are fully reproducible
- Enables cumulative scientific progress

**Academic Contribution 3: Democratizing Policy Analysis**
Currently, sophisticated Social Security analysis requires expensive contracts with Urban Institute or internal access at CBO/SSA. This creates inequity:
- Well-funded organizations can do rigorous analysis
- Advocacy groups, journalists, and smaller research institutions cannot
- This model levels the playing field

*Recommendation:* Frame this as contributing to open science and democratic policy analysis, not just building a tool. Funders interested in equity, transparency, and democratic governance will find this compelling.

---

## Part VIII: Specific Recommendations Before Funder Presentation

### Priority Changes (Must Address)

1. **Decide on base dataset:** Commit to Enhanced CPS with explicit justification
2. **Add wealth dimension:** Integrate SCF targets and wealth imputation (at least statistically)
3. **Add policy applications section:** 3-5 concrete examples of policy questions this enables
4. **Revise timeline:** Either extend to 24 months or reduce scope of 18-month deliverable
5. **Expand validation section:** More specific success criteria and comparison to existing models

### Important Additions (Should Address)

6. **Add "Contribution to the Field" framing:** Scientific and democratic motivations
7. **Strengthen team description:** More detailed CVs, advisory board, track record
8. **More realistic discussion of limitations:** What this model won't do (at least initially)
9. **Add wealth calibration targets:** SCF wealth distributions, retirement resources
10. **Expand behavioral response discussion:** Be explicit about what's included vs. deferred

### Nice-to-Have Enhancements (Could Address)

11. **Add comparison table:** This model vs. DynaSim vs. MINT vs. CBOLT (more detailed than current)
12. **Add technical appendices:** Sample size calculations, QRF hyperparameters, calibration algorithms
13. **Add use cases:** Who will use this model and for what purposes?
14. **Add dissemination plan:** How will you reach potential users beyond technical documentation?

---

## Part IX: Final Assessment and Recommendation

### Technical Soundness: Strong

The core methodology is sound and builds on proven techniques. The QRF imputation approach is appropriate and advances beyond traditional regression methods. The gradient descent calibration is mathematically principled and proven through Enhanced CPS. The team has the right expertise.

**Confidence level:** High probability of technical success, conditional on adequate time and resources.

### Feasibility: Moderate

The 18-month timeline is aggressive. Enhanced CPS took 2+ years, and adding longitudinal dimensions increases complexity. The team is experienced, which mitigates risk, but some phases are optimistically scoped.

**Recommendation:** Propose 24-month timeline with 18-month "Version 1.0" (API and validation) and 24-month "Version 2.0" (web interface and public launch).

### Impact Potential: High

This model addresses a real gap—no existing open-source dynamic Social Security model. The reproducibility crisis in policy analysis is genuine. Democratizing access to sophisticated modeling is valuable.

**Concerns:**
- Need stronger articulation of specific policy applications
- Need to address wealth dimension for comprehensive retirement security analysis
- Need to clarify how this model differs from (and complements) existing models

### Funding Recommendation: Support with Modifications

This is a strong proposal that deserves funding. However, before submitting, I recommend:

1. **Extend timeline to 24 months** (or reduce scope of 18-month deliverable)
2. **Add wealth dimension** (SCF calibration targets at minimum)
3. **Strengthen policy motivation** (specific applications section)
4. **Commit to Enhanced CPS** as base dataset
5. **Expand validation framework** (more detailed success criteria)

With these modifications, this becomes a compelling proposal that balances ambition with realism.

### Budget Considerations

The estimated budget of $270k-$370k over 18-24 months seems reasonable for the scope of work. However:

**Personnel:** 1.5 FTE may be insufficient for 18 months. Consider 2.0-2.5 FTE for 24 months.

**Computing:** $8k is probably adequate given Enhanced CPS infrastructure exists.

**Dissemination:** Consider adding budget for:
- Academic conference presentations (2-3 conferences/year)
- Peer-reviewed publication open access fees
- Stakeholder convenings (Congressional staff, think tanks)

**Revised budget estimate:** $350k-$450k over 24 months seems more appropriate.

---

## Conclusion

This is an ambitious but achievable project. The core technical approach is sound, the team has the right expertise, and the Enhanced CPS precedent substantially de-risks what would otherwise be a speculative research project.

The document's greatest strength is the technical methodology—QRF imputation and gradient descent calibration are appropriate and well-explained. The infrastructure chapter effectively communicates that you're building on proven foundations.

The document's greatest weakness is insufficient attention to the wealth dimension and policy motivation. Before presenting to funders, strengthen the framing of why this matters for policy analysis, what specific questions it will answer, and how it complements (rather than competes with) existing models.

With the modifications I've outlined—particularly the policy applications section, wealth dimension, realistic timeline, and base dataset decision—this becomes a compelling proposal that funders should support.

I'm happy to discuss any of these points in more detail and can provide specific text suggestions for key sections.

---

**John Sabelhaus**
Senior Fellow, Wisconsin Center for Financial Security
Former Assistant Director, Division of Research and Statistics, Federal Reserve Board
Former Chief, Long-Term Modeling, Congressional Budget Office
