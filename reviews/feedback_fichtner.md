# Review of Social Security Microsimulation Model Planning Document

**Reviewer**: Jason Fichtner (Perspective)
**Position**: Vice President and Chief Economist, Bipartisan Policy Center
**Former Roles**: Deputy Commissioner of Social Security, Chief Economist at SSA, Associate Commissioner for Retirement Policy at SSA
**Date**: December 2025
**Document Reviewed**: Social Security Dynamic Microsimulation Model Planning Document

---

## Executive Summary

This planning document proposes developing the first open-source dynamic Social Security microsimulation model using synthetic panel data constructed through machine learning imputation and gradient descent calibration. As someone who spent years inside SSA working with administrative data and seeing how policy analysis gets done in practice, I have mixed reactions to this proposal.

**The good news**: The team has thought carefully about methodology, they've already proven their approach works for cross-sectional analysis (the Enhanced CPS), and they understand the validation requirements. The open-source commitment addresses a real transparency gap in Social Security policy analysis.

**My concerns**: Synthetic longitudinal data is fundamentally harder than cross-sectional data. Lifetime earnings trajectories drive everything in Social Security, and small errors compound over 40-year careers. The document is sophisticated but perhaps overconfident about how well synthetic data will match administrative reality.

**Bottom line**: This is worth funding as a research project, but temper expectations. It won't replace SSA's MINT or Urban's DynaSim for official analysis in the next 5 years. It could, however, become a valuable transparency tool and eventually evolve into something policymakers trust - if the validation is rigorous and the team is honest about limitations.

---

## 1. Will This Model Produce Credible Results for SSA, CBO, and Congress?

### Short Answer: Not immediately, but potentially yes within 5-10 years.

### Credibility Requirements from Inside SSA/CBO

Having worked at both SSA and with CBO, I know what it takes for a model to be taken seriously for official analysis:

**Actuarial soundness**: The model must match SSA's 75-year actuarial projections in baseline scenarios, not just for aggregate trust fund balance but for underlying demographic and economic components. The document acknowledges this but underestimates how precisely these projections need to align.

**Benefit accuracy**: Individual benefit calculations must be exact to the dollar for test cases. SSA has hundreds of edge cases - Windfall Elimination Provision, Government Pension Offset, dual entitlement, divorced spouse benefits, child-in-care benefits. The document mentions using PolicyEngine-US's existing Social Security implementation, which is good, but these rules are fiendishly complex and constantly changing.

**Distributional validation**: When SSA or CBO analyzes reforms, we care deeply about who wins and who loses - by lifetime earnings, by cohort, by race/ethnicity, by marital history. Aggregate accuracy isn't enough. The calibration targets in Chapter 4 are reasonably comprehensive, but distributional validation against MINT's published analyses will be the real test.

**Transparency about uncertainty**: Official agencies are deeply risk-averse about model uncertainty. The document mentions uncertainty quantification (multiple imputation, sensitivity analysis), which is appropriate. But here's what matters: when this model produces a cost estimate or distributional table, can the team honestly quantify and communicate the uncertainty bounds? If those bounds are wide, policymakers may prefer established models even if less transparent.

### The Enhanced CPS Precedent: Encouraging but Not Dispositive

The document leans heavily on PolicyEngine's Enhanced CPS success as proof that synthetic data can match administrative quality. This argument has merit but requires nuance.

**What ECPS proves**: You can match cross-sectional tax aggregates (total revenue, distributional tables) using public data + ML + calibration. Congressional offices use PolicyEngine, which demonstrates real-world credibility. This is genuinely impressive.

**What ECPS doesn't prove**: That the same approach works for longitudinal dynamics. Here's why earnings histories are harder:

1. **Dimensionality**: ECPS imputes income in one year. This project imputes 35-47 years of earnings for each person. That's 35-47x more opportunities for errors to accumulate and interact.

2. **Path dependence**: Social Security benefits depend on the specific sequence of earnings over a lifetime, not just the distribution. Two people with identical average earnings but different volatility get different benefits. Can QRF imputation preserve these subtle patterns?

3. **Validation is harder**: For ECPS, you validate against IRS Statistics of Income (published aggregates). For longitudinal earnings, what do you validate against? SSA publishes some cohort statistics, but nothing as granular as IRS SOI. The validation targets in Chapter 4 are sensible but sparser than for cross-sectional tax modeling.

**My assessment**: The ECPS precedent significantly reduces technical risk and demonstrates team capability. But longitudinal Social Security modeling is a harder problem, and the document should be more explicit about this difference rather than treating it as a straightforward extension.

### Timeline to Credibility

**Phase 1 (Years 1-2)**: Proof of concept demonstrates feasibility. Researchers experiment with it. Not yet credible for official estimates.

**Phase 2 (Years 3-5)**: Extensive validation against published MINT and DynaSim analyses for major reforms (raising retirement age, changing bend points, etc.). If results consistently match within reasonable bounds, academic credibility grows. Some Congressional offices might use it for exploratory analysis alongside established models.

**Phase 3 (Years 5-10)**: If validation continues to hold up and the model successfully predicts actual Social Security outcomes (beneficiary counts, average benefits) as new data becomes available, it could gain broader acceptance. SSA might reference it in technical discussions, though probably not for official Trustees Report projections in this timeframe.

**Critical success factor**: Rigorous intellectual honesty about when the model works well and when it doesn't. Nothing kills credibility faster than overselling results or hiding limitations.

---

## 2. Methodology Comparison to Official SSA/CBO Approaches

### What SSA Actually Does (MINT)

The document accurately describes MINT's methodology, but let me emphasize what gives MINT unmatched credibility inside SSA:

**Real administrative earnings for older cohorts**: MINT uses actual W-2 earnings from SSA's Master Earnings File for everyone born before approximately 1960. These aren't imputed, estimated, or modeled - they're the actual numbers SSA will use to calculate benefits. This eliminates uncertainty for people currently approaching retirement.

**Validated projections**: For younger cohorts, MINT projects earnings using quantile regression (the innovation this proposal builds on). But MINT validates these projections against actual earnings as time passes. Each year, MINT's projections for ages 25-35 in prior years can be checked against what those people actually earned. This validation loop isn't available to external models.

**Conservative assumptions**: SSA has a culture of conservative actuarial assumptions. When uncertain, MINT errs toward caution. The document's proposal to use QRF and machine learning is methodologically sophisticated, but sophistication can mask uncertainty. Simpler methods with well-understood error properties sometimes inspire more confidence for official projections.

### What This Model Does Differently

**Fully synthetic approach**: Everything is imputed from public data. No administrative earnings records. This is both the model's greatest strength (full transparency/reproducibility) and its greatest weakness (everything is estimated with uncertainty).

**Modern ML methods**: Quantile regression forests are a clear improvement over traditional regression for distributional imputation. MINT6 pioneered quantile regression; this extends it with random forests' flexibility. Methodologically sound, but ML models are "black boxes" that make some actuaries nervous.

**Optimization-based calibration**: The gradient descent reweighting (from ECPS) is elegant and handles many simultaneous targets. MINT and DynaSim use "alignment" techniques that adjust probabilities iteratively. The document correctly notes this is less principled mathematically. But alignment is transparent and familiar to actuaries. The optimization approach is better but may face initial skepticism.

**Trade-offs I'd emphasize**:
- MINT: Unbeatable for older cohorts, restricted data, not reproducible
- This model: Fully transparent, cutting-edge methods, but everything synthetic
- For researchers: This model is better (reproducibility matters)
- For SSA actuaries: MINT remains gold standard (real data for near-term projections matters more than methodology)

### CBO's CBOLT: A Different Philosophy

CBOLT uses representative agents rather than individual-level microsimulation. The document correctly identifies this as emphasizing "macroeconomic consistency over distributional detail."

**Why CBO does this**: They need to integrate Social Security with broader fiscal policy and macroeconomic feedbacks. Individual heterogeneity matters less for their use case (long-term budget outlook) than ensuring GDP, interest rates, and debt trajectories are internally consistent.

**What this means for the proposed model**: Don't try to compete with CBOLT for macroeconomic analysis. The individual-level microsimulation approach excels at distributional analysis - who wins, who loses from reforms. That's the comparative advantage. The document's skepticism about OLG models (Section on Micro-Macro Interactions in Chapter 7) reflects John Sabelhaus's influence and is refreshingly honest, though that candor may raise eyebrows among some economists.

---

## 3. Calibration Targets: Are They Appropriate?

### Overall Assessment: Comprehensive and Well-Designed

Chapter 4 (Calibration Targets) is one of the strongest parts of the document. The three-tier prioritization is sensible, the targets are well-sourced, and the tolerance levels are reasonable.

### Critical Targets (My Perspective)

**What matters most for actuarial credibility**:

1. **Age-earnings profiles**: Getting these right is fundamental. Benefits depend on the highest 35 years of indexed earnings. If imputed earnings trajectories don't match SSA's actual age-earnings patterns by cohort, everything downstream is wrong. The ±3% tolerance is appropriate for proof of concept but should tighten to ±1-2% for production use.

2. **Earnings mobility matrices**: This is where I worry most. Lifetime benefits depend not just on average earnings but on the correlation structure across years. Someone with stable earnings differs from someone with volatile earnings even if averages match. The Tier 1 classification is correct. The ±3 percentage point tolerance for transition probabilities seems loose - mobility matrices are sticky, and a 3-point error is substantive.

3. **AIME distribution**: This is the ultimate test. Average Indexed Monthly Earnings is what determines benefit levels. If the synthetic panel's AIME distribution matches SSA's published statistics (mean, median, percentiles by cohort and demographics), that's strong evidence the methodology works. I'd suggest making this an explicit Phase 2 validation metric with tight tolerances (±2% for mean, ±5% for percentiles).

4. **Beneficiary counts and average benefits**: These are outputs, not inputs, which makes them excellent validation targets. If the model produces the right number of retired worker, disabled worker, spouse, and survivor beneficiaries with the right average benefits, that demonstrates the full pipeline works. The ±1-2% tolerances are appropriate for official use.

### Targets I'd Add

**Replacement rates by lifetime earnings**: SSA tracks the ratio of initial benefits to career-average earnings by percentile of the lifetime earnings distribution. This directly measures the progressivity of the benefit formula. If your synthetic panel produces the wrong replacement rate schedule, you'll misestimate the distributional impacts of reforms.

**Quarters of coverage by age**: Benefit eligibility requires 40 quarters of covered employment (10 years). The share of each cohort reaching eligibility thresholds matters for disability and survivor benefits. The current document mentions "years with covered earnings" but should explicitly target quarters of coverage.

**Auxiliary beneficiaries**: About 2.8 million people receive Social Security benefits as dependents (children of retirees/disabled workers/deceased workers). Getting family structure and dependent benefits right is tricky, and these are often overlooked in model validation. Should be Tier 2.

**Windfall Elimination Provision (WEP) and Government Pension Offset (GPO) affected populations**: Millions of public sector workers face these provisions. The modeling is complex and requires tracking both Social Security covered and non-covered employment. If PolicyEngine-US already implements WEP/GPO (it should), validation should explicitly test these populations.

### Opportunity Insights Data: Excellent Addition

The document mentions using Opportunity Insights mobility and life expectancy data (Sections in Chapter 4). This is a smart addition that I didn't see in the initial plan. Raj Chetty's team has produced the most comprehensive data on:
- Intergenerational earnings mobility by geography
- Differential mortality by income and location

This data will be invaluable for validating:
- The realism of earnings mobility patterns (does your imputation match observed mobility?)
- Differential mortality (high earners live longer, affecting lifetime benefits)

Including this shows the team is thinking carefully about validation sources beyond SSA's published aggregates.

### Calibration vs. Validation

**Important distinction**: Some targets should be used for calibration (reweight the sample to match them), others for validation (hold out to test if the model works).

The document mentions this ("reserve some variables as validation checks") but could be more explicit about which targets are which. My suggestion:

**Calibration targets**: Demographics (age-sex-education), employment status, current-year earnings distribution, beneficiary counts
**Validation targets**: Earnings mobility matrices, AIME distribution, benefit amounts, replacement rates, poverty rates

This avoids "overfitting" to all available data and provides genuine out-of-sample tests.

---

## 4. Can This Model Analyze Solvency Reforms?

### Short Answer: Yes, but with important caveats.

### Types of Reforms and Model Suitability

**Reforms this model should handle well**:

1. **Benefit formula changes**: Adjusting bend points, changing replacement rates, progressive indexing of benefits. These are pure parameter changes in the benefit calculation. If the underlying earnings histories are realistic, the model should produce credible estimates.

2. **Retirement age increases**: Raising the Full Retirement Age (FRA) or Early Eligibility Age (EEA). This requires modeling claiming behavior, which the document mentions briefly. The validation will be: does the model predict realistic changes in claiming ages when incentives change? Claim age is partially behavioral, not purely mechanical.

3. **Payroll tax changes**: Raising the tax rate, eliminating the taxable maximum, creating a "donut hole" (tax on earnings above $250K). These affect revenue but also benefit calculations (through the AIME). Straightforward to model if earnings distributions are accurate.

4. **Benefit taxation changes**: Increasing taxation of benefits, changing thresholds. Requires integration with income tax modeling, which PolicyEngine-US already does. Should work well.

**Reforms that require caution**:

1. **Means-tested benefit reductions**: If benefits phase out with other income, behavioral responses become critical. Do high earners save more in 401(k)s or Roth accounts to avoid Social Security income tests? Does labor supply change? The document mentions "simple elasticities" for behavioral responses but acknowledges this isn't full structural modeling. I'd recommend clear caveats about second-order behavioral effects for these reforms.

2. **Longevity indexing**: Automatically adjusting benefits based on life expectancy changes. This interacts with differential mortality (rich people benefit more from longevity increases). The model includes differential mortality, which is good. But validating the distributional impacts requires careful comparison to MINT's analyses of similar proposals.

3. **Carve-out personal accounts**: Proposals to redirect payroll taxes into individual accounts fundamentally change the system. This requires modeling account accumulation, investment returns, and annuitization decisions. The current proposal doesn't include a savings module. For these reforms, the model would need major extensions.

### The Fiscal Gap and 75-Year Solvency

The 2024 Trustees Report projects Social Security's 75-year actuarial deficit at 3.50% of taxable payroll. Closing this gap is the central policy question. Can this model credibly analyze combinations of tax increases and benefit cuts to achieve solvency?

**What's needed**:
1. **Long-run projections** that match Trustees' demographic and economic assumptions (fertility, mortality, immigration, wage growth, inflation)
2. **Sensitive revenue estimates** for payroll tax changes
3. **Accurate benefit projections** under current law and reforms
4. **Distributional analysis** showing impacts across age cohorts and lifetime earnings

**My assessment**: The model should be able to do this, but the proof is in validation. I'd recommend:
- Phase 4 includes a validation exercise replicating SSA's 2024 Trustees Report baseline projections
- Phase 5 replicates cost estimates for several well-known reform packages (e.g., Bipartisan Policy Center's Commission on Retirement Security and Personal Savings recommendations, various Congressional proposals)
- Document compares results to SSA and CBO estimates, explaining any differences

If the model produces solvency estimates within 0.2-0.3 percentage points of Trustees' estimates for major reform packages, that would build substantial credibility.

### What I'd Want to See in Reform Analysis

When this model produces a reform analysis for Congressional staff, the output should include:

1. **75-year actuarial balance**: Does this reform close the fiscal gap?
2. **Trust fund depletion date**: When does the trust fund run out under current law and reformed law?
3. **Distributional tables**: Impact on lifetime benefits by percentile of lifetime earnings
4. **Cohort tables**: Impact on different birth cohorts (current retirees, near-retirees, mid-career, young workers)
5. **Demographic breakouts**: Impacts by race/ethnicity, sex, marital status
6. **Poverty and inequality impacts**: Does this reduce or increase elderly poverty? How does it affect Gini coefficients?
7. **Uncertainty ranges**: Confidence intervals or scenario analysis showing sensitivity to assumptions

This is what SSA provides when analyzing reforms. Matching this standard is necessary for credibility.

---

## 5. Validation Against Trustees Report Projections

### This Is the Most Critical Milestone

Everything depends on validation. The Enhanced CPS is credible because it matches IRS data. This model will be credible only if it matches SSA Trustees Report projections and published MINT analyses.

### Validation Strategy: Strong Foundation, Needs More Specificity

Chapter 6 (Methodology) includes a validation section that covers:
- Cross-sectional validation (base year accuracy)
- Longitudinal validation (earnings dynamics)
- Distributional validation (benefit distributions)
- Fiscal validation (trust fund projections)
- External validation (comparison to DynaSim/MINT)

This is the right framework. But I'd push for more concrete commitments:

**Specific validation exercises I'd require**:

1. **Historical backtesting**: Take MINT's 2006 projections for 2020 outcomes. Can your model, calibrated to 2006 data, predict 2020 actual outcomes as well as MINT did? This tests whether your methodology is as good as established models at out-of-sample prediction.

2. **Trustees Report replication**: Can you replicate the 2024 Trustees Report baseline scenario?
   - Trust fund ratio projections (2024-2098)
   - Beneficiary counts by type and age
   - Average benefit amounts
   - Total expenditures and revenues

   Target: Match within 2-3% for near-term (10-year), 5-7% for long-term (75-year). If you can't match baseline, reform estimates won't be credible.

3. **Reform cost estimation**: Pick 5-10 major reforms that SSA or CBO has scored:
   - Raise FRA to 69
   - Eliminate taxable maximum
   - Change COLA from CPI-W to chained CPI
   - Apply 50% benefit cut at trust fund depletion
   - Progressive bend point adjustments

   Compare your cost estimates (as % of payroll, change in actuarial balance) to official estimates. Target: Within 0.2-0.3 percentage points.

4. **MINT distributional comparison**: MINT publishes tables showing replacement rates, benefit levels, and poverty rates by various demographics. Replicate these tables for several birth cohorts and compare. Target: Distributional patterns should match even if levels differ slightly.

### The "Unknown Unknowns" Problem

You can't validate what you don't know is wrong. MINT has 25+ years of refinement. It's captured edge cases and fixed bugs that aren't documented in published papers. Your model will have bugs and oversimplifications you don't yet know about.

**How to address this**:
1. **Red team the model**: Find skeptical actuaries from SSA or other models to probe for problems
2. **Adversarial validation**: Deliberately look for cases where the model fails (very high earners, volatile careers, complex marital histories)
3. **Continuous validation**: As new data becomes available each year, check whether past projections were accurate

### My Biggest Worry: False Precision

Microsimulation models produce precise-looking numbers (e.g., "this reform reduces elderly poverty by 2.37%"). But that precision is often false - the true uncertainty is much wider.

The document mentions uncertainty quantification via multiple imputation and sensitivity analysis. Good. But here's what I'd insist on:

**For any reform estimate, report**:
1. Point estimate (e.g., 2.37% poverty reduction)
2. Standard error from multiple imputation (e.g., ± 0.4%)
3. Sensitivity range from assumption changes (e.g., 1.8% to 2.9% across reasonable assumptions about earnings growth, mortality, etc.)

If the sensitivity range is wide, be honest about it. Policymakers can handle uncertainty; they can't handle being misled by false precision.

---

## 6. Concerns About Methodological Rigor and Potential Misuse

### Concerns About Rigor

**Complexity vs. Transparency Trade-off**: The document emphasizes transparency and open-source as key advantages. But the methodology (QRF imputation, gradient descent calibration, demographic transition models, multiple imputation) is quite complex. Most Congressional staff won't be able to audit the code meaningfully. So "open-source" doesn't automatically mean "transparent" to non-specialists.

**Recommendation**: Invest heavily in documentation and explanation. Create:
- High-level explanation for policymakers (what does this model do, in plain English?)
- Technical documentation for researchers (methods, assumptions, limitations)
- Validation dashboard showing how well the model matches SSA benchmarks
- User guide with worked examples

**The "Synthetic Data" Communication Challenge**: Explaining that the model uses "synthetic" or "imputed" earnings histories rather than real administrative data will be tricky. Some policymakers may dismiss it for this reason alone. Others may not understand what it means.

**Recommendation**: Frame it positively. "This model uses the same approach that successfully matched IRS tax data (the Enhanced CPS), extended to Social Security earnings. Unlike models that use restricted SSA data, anyone can verify and reproduce the results."

**Behavioral Responses: The Weak Link**: The document acknowledges that behavioral responses will use "simple elasticities" rather than full structural models. This is pragmatic but means the model will be less reliable for reforms that induce large behavioral changes (means-testing, major benefit cuts, significant tax increases).

**Recommendation**: Be very explicit about which reforms require caution. For reforms with large behavioral effects, consider partnering with researchers who have structural models to provide complementary estimates.

### Concerns About Potential Misuse

**Cherry-Picking Results**: The model will produce rich output - distributions, percentiles, demographic breakouts. Users could cherry-pick favorable statistics while ignoring unfavorable ones.

**Mitigation**: Design the web interface and API to encourage comprehensive reporting. When someone analyzes a reform, the output should include standard tables (fiscal effects, distributional impacts, poverty/inequality metrics) rather than letting users pick and choose individual statistics.

**Overconfidence in Precision**: Users may treat model estimates as "truth" rather than uncertain projections. This is a problem with all models, but open-source tools are especially prone to misuse by non-experts.

**Mitigation**:
- Always report confidence intervals or uncertainty ranges
- Include prominent disclaimers about limitations
- Provide calibrated uncertainty (if the model says ±X%, that range should actually contain the true value X% of the time)

**Advocacy Use vs. Research Use**: Advocacy groups will use this model to support their preferred policies. That's fine - transparency means everyone has access. But biased use could undermine the model's credibility.

**Mitigation**:
- Remain studiously nonpartisan in official PolicyEngine communications
- Publicly correct egregious misuse (if an advocacy group misrepresents model results)
- Build reputation for intellectual honesty (acknowledge when the model is uncertain or produces unexpected results)

**The "Github Star" Problem**: The document lists "Github stars and forks" as an impact metric. But popularity isn't validation. A model could be widely used and wrong.

**Recommendation**: Focus on validation-based metrics:
- Do academic papers cite this model's estimates?
- Do Congressional offices use it for official analysis?
- When the model makes projections, are they borne out as data becomes available?

These matter more than GitHub stars for credibility.

---

## 7. Specific Technical Concerns

### Quantile Regression Forest Imputation

**The approach**: Train QRF models on PSID to predict earnings at each age conditional on current characteristics. Sample from predicted distributions to generate full earnings histories.

**My concern**: Earnings dynamics in PSID may not match broader population. PSID started as a representative sample in 1968 but has attrition and top-coding issues. QRF trained on PSID will learn PSID patterns, which may differ from SSA administrative data patterns.

**Mitigation**: Validate extensively against SSA published earnings statistics. If age-earnings profiles or mobility matrices don't match, investigate whether PSID representativeness issues are the problem. May need to adjust PSID training data or supplement with SIPP.

### Cohort Effects vs. Time Effects

Earnings trajectories differ across birth cohorts (people born in 1945 had different careers than people born in 1985). The methodology needs to separate:
- Age effects (earnings rise with age due to experience)
- Cohort effects (different generations have different earnings trajectories)
- Period effects (economic conditions affect all cohorts simultaneously)

**My concern**: The document mentions cohort effects but doesn't detail how to separate these three effects, which is a hard identification problem.

**Recommendation**: Be explicit about what assumptions you make to identify cohort vs. time effects. Consider sensitivity analysis showing how results change with different assumptions.

### Demographic Transitions: The Spousal Matching Problem

Social Security spousal and survivor benefits depend on marital history. The methodology needs to:
1. Model marriage/divorce transitions
2. Match individuals to appropriate spouses (assortative mating by education, earnings)
3. Track marital history (who was married to whom, when, for how long)

**My concern**: This is complex and error-prone. If spousal matching is unrealistic, survivor benefit estimates will be wrong.

**Validation**: Compare model outputs to Census/CPS marital status distributions and to SSA's statistics on spouse and survivor beneficiaries. Should be a Tier 1 validation metric.

### Disability Modeling: A Known Hard Problem

SSDI is about 16% of Social Security expenditures. Getting disability right matters. The document mentions estimating disability onset hazard models from PSID and HRS.

**My concern**: Disability is hard to model because it depends on health status (partially unobserved), occupation (physical vs. cognitive work), and partially behavioral (application decisions, which are affected by economic conditions and benefit generosity).

**Recommendation**:
- Validate carefully against SSA disability statistics (incidence by age, sex, education)
- Consider simpler approach initially (exogenous disability rates by demographics) before adding behavioral elements
- Explicitly discuss limitations for disability-related reforms (changing definition of disability, changing disability benefit levels)

### Forward Projection: The "Jump-Off" Problem

The document acknowledges the jump-off problem: transition equations estimated on historical data may not project forward accurately. The solution is calibration to SSA Trustees' assumptions.

**My concern**: How much calibration is acceptable before you're just imposing SSA's projections rather than independently modeling dynamics? There's a continuum:
- No calibration: Model projections diverge from reality
- Light calibration: Adjust aggregate growth rates to match Trustees
- Heavy calibration: Force most outcomes to match Trustees' projections

**Question for the team**: Where on this continuum will you land? If calibration is heavy, you're essentially creating a microsimulation wrapper around SSA's aggregate projections. That's useful for distributional analysis but not an independent forecast. Be explicit about this trade-off.

---

## 8. Comparison to Other Models: Realistic Positioning

### Don't Oversell vs. MINT and DynaSim

The document positions this model as "democratizing access" to Social Security analysis and enabling transparency. That's the right framing. Don't claim to be "better" than MINT or DynaSim - at least not initially.

**Realistic comparative advantages**:
- **Transparency**: Anyone can see assumptions, methodology, code
- **Accessibility**: Free, web interface, no contracts needed
- **Reproducibility**: Anyone can replicate and verify results
- **Integration**: Combined with PolicyEngine's broader tax-benefit modeling
- **Flexibility**: Users can modify assumptions and extend the model

**What this model won't be (at least initially)**:
- **More accurate than MINT**: MINT has real administrative data for older cohorts
- **As comprehensive as DynaSim**: DynaSim has 40+ years of refinement and covers more programs
- **As authoritative as SSA/CBO models**: It takes years to build institutional credibility

**Positioning I'd recommend**: "This model provides the first fully open-source dynamic Social Security analysis capability. It won't replace official models for Trustees Report projections, but it enables researchers, advocates, and policymakers to conduct transparent, reproducible analysis of Social Security reforms."

### The Cato Model: Your Closest Competitor

The Cato Institute's R-based model is open-source and freely available. The document acknowledges this. Key differences:
- Cato uses inverse transform sampling; you use QRF
- Cato has 10K sample; you plan larger (TBD)
- Cato is R; you're Python
- Cato is standalone; you integrate with PolicyEngine

**My take**: These are complementary, not competitive. Different methodologies provide robustness. If Cato and PolicyEngine models produce similar estimates for a reform, that's strong evidence. If they diverge, that's interesting and worth understanding.

**Recommendation**: Engage with the Cato team. Compare methodologies and results. Publish joint validation exercises. This builds credibility for both models and strengthens the open-source ecosystem.

---

## 9. The 18-Month Timeline: Ambitious but Plausible

### Overall Assessment

The roadmap is detailed and shows the team has thought through the development process. The 18-month timeline is ambitious but potentially achievable given:
- Proven team with relevant experience
- Existing tools (microimpute, microcalibrate, PolicyEngine-US)
- Clear milestones and success criteria

### Where Schedule Risk Lies

**Phase 2 (Earnings Imputation)**: If validation doesn't work well initially, this could require multiple iterations. Getting age-earnings profiles and mobility matrices to match SSA data might be harder than expected. Build in contingency time.

**Phase 3 (Demographic Transitions)**: Spousal matching and disability modeling are complex. If these don't validate well, it will delay everything downstream.

**Phase 5 (Forward Projection)**: Long-run projections that match Trustees' assumptions while maintaining individual heterogeneity is technically challenging. This phase might need more than 3 months.

### Realistic Alternative Timeline: 24 Months

Consider budgeting for 24 months instead of 18. The extra 6 months would allow:
- More thorough validation at each phase
- Addressing unexpected challenges (there will be some)
- Building in time for external review and revision
- Red-teaming by SSA/academic actuaries

The difference between "launch with known problems we're still fixing" vs. "launch with solid validation" is worth 6 months.

---

## 10. What Would Make Me Fully Confident in This Model?

Let me be concrete about what would need to happen for me to recommend this model for serious policy analysis:

### Tier 1 Validation (Essential for Any Use)

1. **Trustees Report baseline replication**: Trust fund projections within 3% (10-yr) and 7% (75-yr)
2. **Beneficiary counts**: Within 2% for all major categories (retired workers, disabled workers, spouses, survivors)
3. **Average benefits**: Within 3% for all major categories
4. **Age-earnings profiles**: Match SSA statistics within 2% at each age
5. **AIME distribution**: Match mean within 2%, percentiles within 5%

If these aren't met, the model isn't ready for policy work.

### Tier 2 Validation (Required for Distributional Analysis)

1. **Replacement rates by lifetime earnings**: Match SSA/MINT published tables
2. **Earnings mobility matrices**: Match PSID quintile transition matrices within 5 percentage points
3. **Benefit distributions**: Match SSA percentile statistics within 5%
4. **Differential mortality**: Match Opportunity Insights patterns
5. **Poverty impacts**: Model predictions for elderly poverty rates match Census/SSA data

### Tier 3 Validation (Required for Reform Analysis)

1. **Cost estimation accuracy**: For 10 major reforms scored by SSA or CBO, produce cost estimates within 0.3 percentage points of payroll
2. **Distributional robustness**: Distributional impacts of reforms should qualitatively match published MINT analyses (same directions, similar magnitudes)
3. **Historical backtesting**: Predictions for past periods should match actual outcomes as well as MINT's predictions did

### External Review

1. **SSA actuaries review methodology and validation**: Get formal feedback
2. **Academic peer review**: Publish methodology in peer-reviewed journal (Journal of Policy Analysis and Management, National Tax Journal, etc.)
3. **User community testing**: Beta test with Congressional offices and advocacy groups, collect feedback
4. **Adversarial testing**: Have skeptics try to break the model or find failures

### Uncertainty Quantification

1. **Calibrated confidence intervals**: If the model reports ±X%, that range should be right X% of the time
2. **Scenario analysis**: For major reforms, show sensitivity to key assumptions (wage growth, mortality, disability rates)
3. **Honest limitation reporting**: Documentation clearly states what the model does well and what it does poorly

---

## 11. Final Recommendations

### For the Development Team

1. **Temper the Enhanced CPS analogy**: Yes, ECPS proves your approach works for cross-sectional data. But don't overstate the analogy to longitudinal modeling. Acknowledge that earnings histories are a harder problem.

2. **Invest disproportionately in validation**: Make validation 40% of the effort, not 20%. Credibility depends entirely on how well the model matches SSA benchmarks.

3. **Be obsessively honest about limitations**: When the model doesn't work well, say so. Intellectual honesty builds credibility; overselling destroys it.

4. **Build relationships with SSA actuaries**: Get their input early. They can tell you about edge cases and validation metrics you haven't thought of. If they respect the model, others will follow.

5. **Plan for 24 months, not 18**: The extra time for thorough validation is worth it.

6. **Create a validation dashboard**: Public webpage showing how well the model matches SSA benchmarks, updated as new data becomes available. Transparency about model performance is crucial.

### For Potential Funders

1. **This is worth funding**: The open-source Social Security analysis gap is real, and this team is capable of addressing it.

2. **Manage expectations**: This won't replace MINT or DynaSim for official analysis within 5 years. But it can become a valuable transparency and research tool sooner.

3. **Fund for 24 months**: The 18-month timeline is tight. An extra $100K for 6 more months is a good investment in quality.

4. **Require rigorous validation**: Make continued funding contingent on meeting Tier 1 validation criteria. Don't fund deployment (Phases 6) until validation (Phases 2-5) is solid.

5. **Support external review**: Include budget for SSA actuarial review, peer review publication, and adversarial testing. External validation is essential for credibility.

### For Potential Users

1. **This will be a valuable tool**: If validation succeeds, you'll have free, transparent Social Security analysis capabilities. That's hugely valuable.

2. **Wait for validation**: Don't use early versions (Phases 1-3) for serious policy work. Wait until Tier 1 validation is complete.

3. **Understand uncertainty**: Model estimates will have uncertainty. Ask for confidence intervals. Be skeptical of precise-looking numbers without error bars.

4. **Compare to other models**: For important reforms, compare PolicyEngine estimates to SSA, CBO, DynaSim, or Cato estimates. Convergence across models is reassuring; divergence requires explanation.

---

## 12. Overall Assessment: Cautiously Optimistic

I've spent this review being skeptical and poking holes. That's my job - I've seen too many models that promised more than they delivered. But here's the bottom line:

**This project is worth doing.** The transparency gap in Social Security analysis is real. Advocates, researchers, and smaller Congressional offices need free, accessible tools. PolicyEngine has proven it can build credible microsimulation models (the Enhanced CPS works). The team knows the pitfalls and has thought carefully about methodology.

**Success is plausible but not guaranteed.** Longitudinal earnings imputation is hard. Validation against SSA benchmarks is essential. The timeline is ambitious. But if the team executes well and maintains intellectual honesty, this could become a valuable tool within 3-5 years.

**Start with realistic expectations.** This won't be the new gold standard for Social Security analysis by 2027. It will be an experimental model that's interesting to researchers. Over time, with rigorous validation and continuous improvement, it could earn broader credibility.

**The real test**: In 5 years, will SSA actuaries take this model seriously enough to compare its estimates to MINT's? Will Congressional staff use it alongside established models? Will academic papers cite its estimates as credible? If yes, this project will have succeeded.

I'm cautiously optimistic that it can get there. But it will take longer and require more validation than the current timeline suggests. Budget for 24 months, invest heavily in validation, and maintain intellectual honesty. If you do those things, this could become an important public good.

---

**Jason Fichtner**
Former Deputy Commissioner of Social Security
Vice President and Chief Economist, Bipartisan Policy Center

---

## Appendix: Specific Technical Questions for the Team

1. What is the plan for handling top-coded earnings in PSID? Top-coding affects high earners who are important for benefit calculations above bend points.

2. How will you handle negative or missing earnings (unemployment, out of labor force)? Social Security credits quarters based on earnings thresholds, not just presence of any earnings.

3. What's the plan for self-employment income? It's treated differently for Social Security purposes and is notoriously underreported in surveys.

4. Will the model handle non-covered employment (federal employees before 1984, some state/local workers)? This affects WEP and GPO calculations.

5. How will you handle earnings above the taxable maximum? These don't affect benefits but matter for distributional analysis of payroll tax changes.

6. What's the approach for immigrant populations? Social Security has totalization agreements with other countries affecting benefit calculations for immigrants.

7. Will the model handle disability conversions (SSDI beneficiaries convert to retired worker benefits at FRA)? This is important for long-run projections.

8. How will you model the earnings test for beneficiaries under FRA who continue working? This is complex and affects claiming behavior.

9. What's the plan for handling remarriage rules for widow(er) benefits? Multiple marriages and divorces create complex benefit eligibility patterns.

10. Will the model track children through age 18 for child-in-care benefits? This requires longitudinal family structure tracking.

These details matter enormously for getting Social Security calculations right. I'd want to see the team's plans for each before being fully confident in the model's accuracy.
