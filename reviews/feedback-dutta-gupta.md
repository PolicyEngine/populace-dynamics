# Feedback on Social Security Microsimulation Model Planning Document

**From:** Indivar Dutta-Gupta, Distinguished Visiting Fellow, National Academy of Social Insurance
**Date:** December 12, 2025
**Re:** Review of PolicyEngine Social Security Dynamic Microsimulation Model Proposal

## Executive Summary

This planning document represents a technically sophisticated and methodologically sound proposal for building an open-source Social Security microsimulation model. The transparency, accessibility, and integration with broader tax-benefit analysis are significant democratic advances. However, from the perspective of progressive policy analysis and protecting vulnerable populations, the proposal requires substantial strengthening in several areas:

1. **Equity analysis** needs to be elevated from implicit to explicit across all model components
2. **SSI integration** appears inadequate given its critical anti-poverty role
3. **Distributional granularity** must go beyond standard demographic cuts to capture intersectionality
4. **Political economy considerations** around model use are underspecified
5. **Safety net interactions** need comprehensive treatment, not incremental add-ons

I provide detailed recommendations below, organized by substantive concern. This model has enormous potential to democratize policy analysis—but only if it centers equity and poverty reduction from the design phase forward.

## 1. Does This Model Serve Low-Income Workers and Families?

### Strengths

The model's accessibility represents a democratic breakthrough. By removing the ~$50,000-100,000 barrier to microsimulation analysis and providing a free web interface, this dramatically empowers advocacy organizations working on behalf of low-income communities. Organizations like CLASP, CBPP, and state-level advocacy groups will gain analytical capacity previously monopolized by well-funded think tanks and government agencies. This power shift matters enormously.

The integration with PolicyEngine-US's existing tax-benefit model is strategically important. Social Security does not exist in isolation—low-income families experience it alongside SNAP, SSI, Medicaid, EITC, and housing assistance. The ability to model these interactions will reveal how reforms affect the total safety net, not just one program.

The technical specifications chapter (based on Sabelhaus's guidance) appropriately emphasizes career patterns with earnings gaps due to child-rearing and disability. This attention to non-standard work histories is essential, as low-income workers disproportionately experience employment interruptions.

### Critical Gaps

**1. SSI receives inadequate emphasis throughout**

Supplemental Security Income is mentioned only briefly in existing-models.md and technical-specifications.md, yet SSI represents the primary anti-poverty program for elderly and disabled individuals with limited work histories. The proposal states SSI would be "straightforward" to add because PolicyEngine-US already implements it (technical-specifications.md, lines 76-77). This severely underestimates the challenge.

SSI modeling requires:
- Precise asset testing thresholds and accounting
- State supplementation programs (32 states)
- Complex interaction with Social Security claiming decisions
- In-kind support and maintenance rules
- Representative payee complications
- Administrative barriers and take-up modeling

**The poverty impact analysis is incomplete without comprehensive SSI modeling from Phase 1.** Many low-income workers, particularly women with caregiving gaps and workers with disabilities, depend on SSI to supplement inadequate Social Security benefits. A model that treats this as an "incremental capability" (technical-specifications.md, line 71) will systematically undercount poverty impacts in early versions.

**Recommendation:** Elevate SSI to Phase 4 alongside Social Security benefit calculations. The roadmap currently plans Social Security benefits in months 10-12 but relegates SSI to undefined "extensions." This ordering is backward from a poverty analysis perspective.

**2. Low-wage workers' earnings dynamics may be mischaracterized**

The methodology chapter emphasizes PSID as the primary training data for earnings trajectories. While PSID is the gold standard for longitudinal analysis, its sample size limitations create challenges for modeling low-income populations who experience:
- High earnings volatility and frequent zero-earning years
- Informal economy participation underreported in surveys
- Multiple job-holding and non-standard work arrangements
- Industry and occupational mobility driven by labor market disruptions

The QRF methodology will train on PSID's observed patterns, but if PSID undersamples or mischaracterizes low-wage work, the synthetic panel will perpetuate these biases.

**Recommendation:** Supplement PSID training with:
- SIPP data for short-term earnings volatility among low-income workers
- Opportunity Insights mobility data already mentioned for geographic variation (calibration-targets.md, lines 99-107)
- Explicit validation that bottom quintile earnings transitions match administrative SSA data
- Oversampling or targeted validation for workers with <$15,000 annual earnings

**3. Benefit adequacy metrics are absent from success criteria**

The roadmap's success criteria (roadmap.md, lines 349-387) focus entirely on aggregate accuracy:
- Earnings distributions within 5% of SSA
- Beneficiary counts within 2% of SSA
- Benefit amounts within 3% of SSA

These aggregate metrics can be met while significantly misrepresenting outcomes for low-benefit recipients. A model could perfectly match median benefits while overstating minimum benefits or understating the prevalence of poverty-level benefits.

**Recommendation:** Add distributional success criteria:
- Poverty rates among beneficiaries within 2 percentage points of CPS estimates
- Bottom decile benefit amounts within 5% of SSA statistics
- Validation against SIPP program participation data for dual SSI/SS beneficiaries
- Replacement rate distributions by lifetime earnings quintile

**4. Labor market marginalization is underspecified**

The demographic transitions section (methodology.md, lines 236-288) models marriage, fertility, disability, and mortality but does not explicitly model long-term labor force exit due to:
- Mass incarceration (particularly affecting Black men)
- Opioid epidemic and substance use disorders
- Caregiving responsibilities (particularly affecting women)
- Discriminatory hiring practices

These mechanisms create quarters-of-coverage gaps that result in benefit ineligibility, but the proposal treats labor force participation as a simple employment transition. This is inadequate for populations facing structural barriers.

**Recommendation:** In Phase 3 (demographic transitions), add:
- Incarceration histories linked to race, age, and education, using Bureau of Justice Statistics data
- Explicit caregiving modules beyond fertility (elder care, disabled family member care)
- Validation that quarters-of-coverage distributions by demographic group match SSA administrative data

## 2. Will Distributional Analysis Capture Impacts on Vulnerable Populations?

### Strengths

The calibration targets chapter (calibration-targets.md, lines 27-48) specifies race, ethnicity, educational attainment, and marital status distributions intersected with age groups. This foundational attention to demographic detail is essential.

The inclusion of Opportunity Insights data on differential mortality by income (calibration-targets.md, lines 122-135) represents best practice. Mortality differentials have enormous implications for lifetime benefit receipt, and ignoring them biases progressivity analysis. The fact that this is explicitly included as a "critical" priority tier demonstrates appropriate prioritization.

The technical specifications chapter's discussion of "following the money" (technical-specifications.md, lines 99-103) through distributional tracking of taxes, benefits, and debt ownership is exactly the right framing. This approach aligns with modern distributional national accounts methodology.

### Critical Gaps

**1. Intersectionality is mentioned but not operationalized**

While the proposal tracks race, gender, education, and earnings separately, there is no explicit discussion of intersectional analysis. A Black woman with caregiving responsibilities faces compounding disadvantages that cannot be understood by analyzing race, gender, and employment independently.

The calibration targets table (calibration-targets.md, line 28-47) shows "race by age groups" and "marital status by age and sex" but does not specify targets like "mean earnings for Black women age 35-44 by education." Without such intersectional targets, the model may match marginal distributions while missing joint distributions.

**Recommendation:** In Phase 5 (calibration), add:
- Calibration targets for key intersectional groups: race × gender × education cells
- Validation that Black-white earnings gaps by gender match administrative data
- Hispanic/Latina earnings trajectories validated separately, given distinct immigration and labor force patterns
- Asian American disaggregation where sample size permits (East Asian, South Asian, Southeast Asian communities have very different economic profiles)

**2. Geography matters more than acknowledged**

The proposal mentions state-level detail from CPS (data-sources.md, line 19) but does not emphasize geographic variation in outcomes. Yet research from Opportunity Insights shows that economic mobility varies enormously by commuting zone—a child born into the bottom quintile in San Jose has dramatically different prospects than one born in Charlotte or Milwaukee.

Social Security is a federal program, but geographic variation in:
- Local labor markets and earnings opportunities
- Cost of living (relevant for benefit adequacy, even if formulas are national)
- State tax treatment of benefits (some states exempt SS from taxation)
- Proximity to SSA offices (affecting take-up and administrative burdens)

These factors create heterogeneous impacts that aggregate national analysis obscures.

**Recommendation:**
- Use Opportunity Insights commuting zone mobility data (already cited, calibration-targets.md, line 102) as calibration targets, not just validation
- Report distributional impacts by Census region at minimum
- For major reforms, provide metro-area-level estimates for largest 50 metro areas
- Validate that geographic earnings distributions match Opportunity Insights data

**3. Disability analysis appears superficial**

Disability onset is modeled as a hazard rate (methodology.md, lines 267-275) but the proposal does not discuss:
- Type and severity of disability (SSA uses five-step sequential evaluation)
- Variation in disability screening strictness over time and across adjudicators
- Disparities in SSDI approval rates by race, education, and region
- Work history requirements excluding many disabled individuals (requires 5 of last 10 years)

The validation target (calibration-targets.md, lines 136-142) focuses on aggregate incidence and recovery rates. This is insufficient for understanding distributional impacts, as disability rates vary enormously by occupation, education, and race.

**Recommendation:** In Phase 3 (disability modeling):
- Collaborate with disability policy experts (National Academy of Social Insurance has relevant expertise)
- Validate SSDI approval rates by demographic subgroup against published SSA data
- Model partial disability and workers who leave labor force but don't receive SSDI
- Include validation against SIPP disability status data for non-beneficiaries

## 3. Concerns About Model Use to Justify Benefit Cuts vs. Expansions

This is the most serious political economy concern about the project. The proposal emphasizes transparency and democratization but does not grapple with how the model could be weaponized.

### The Austerity Risk

Dynamic microsimulation models have historically been used to **justify cuts to social insurance**, not expansions. Consider:

1. **CBOLT's role in deficit narratives:** CBO's long-term model consistently produces projections showing Social Security trust fund exhaustion, fueling narratives of crisis and insolvency. These projections drive toward payroll tax increases or benefit cuts, rarely toward solutions like eliminating the cap on taxable earnings or wealth taxation.

2. **DynaSim analyses of progressive reforms:** When Urban Institute models progressive reforms (raising benefits for low earners, caregiver credits), the distributional gains are obvious—but these analyses are typically commissioned by progressive foundations. When commissioned by deficit hawks, the same model analyzes benefit cuts and means-testing.

3. **Penn Wharton Budget Model's framing:** PWBM consistently emphasizes fiscal costs and debt-to-GDP ratios in Social Security analysis, framing benefit adequacy as secondary to solvency. The model's dynamic OLG framework generates large "macroeconomic feedback" effects that make progressive reforms appear more expensive and regressive reforms appear more beneficial.

### The Proposal's Silence is Concerning

The entire planning document—112 pages across 8 chapters—contains **no discussion of how to prevent misuse** or how to ensure the model serves progressive goals. The intro.md (lines 46-47) states the model will analyze "benefit formula changes, retirement age adjustments, tax base modifications, means-testing proposals, progressive indexing mechanisms, minimum benefit proposals, and effects on poverty, inequality, and retirement adequacy."

Notice the framing: cuts (retirement age, means-testing) are listed alongside expansions (minimum benefits) with no normative distinction. This value neutrality is dangerous.

### Why This Matters

Consider a right-wing think tank commissioning PolicyEngine to analyze:
1. Raising full retirement age to 70 (a benefit cut affecting low-income workers with shorter life expectancies)
2. Means-testing benefits above $50,000 income (a cut affecting middle-class beneficiaries)
3. Switching to chained CPI (a real benefit cut over time)

The model will dutifully generate distributional tables showing impacts. The think tank will publish a report emphasizing fiscal savings and downplaying adequacy concerns. Because the model is open-source and "objective," it will carry PolicyEngine's credibility.

How do you prevent this scenario?

### Recommendations

**1. Embed equity metrics in default outputs**

Every reform analysis should automatically report, prominently:
- Poverty impacts (SPM poverty, deep poverty)
- Distributional incidence by income quintile (using comprehensive income, not just benefits)
- Racial equity impacts (Black-white, Hispanic-white benefit gaps)
- Gender equity impacts (particularly widow poverty)
- Effects on workers with caregiving histories
- Effects on workers with disabilities

These metrics should be **unavoidable**—not opt-in extras that users can ignore. Make the web interface lead with distributional impacts before fiscal aggregates.

**2. Default comparisons favor progressive benchmarks**

When a user analyzes a reform, the interface should offer comparisons:
- Status quo vs. reform
- Reform vs. Social Security 2100 Act (benchmark progressive expansion)
- Reform vs. other progressive proposals (caregiver credits, benefit increases)

This frames every analysis within a context of expansion possibilities, not just cuts vs. status quo.

**3. Publish progressive model use cases at launch**

PolicyEngine should commission and prominently feature analyses of:
- Caregiver credit proposals
- Minimum benefit expansions
- Eliminating earnings cap
- Benefits for undocumented workers who paid into the system
- Addressing wealth inequality through benefit progressivity

Establish the narrative that **this tool exists to strengthen Social Security**, not dismantle it.

**4. Governance structure for model updates**

Create an advisory board including:
- Social Security advocacy organizations (Social Security Works, AARP)
- Racial justice organizations (National Urban League, UnidosUS)
- Disability rights organizations (National Disability Rights Network)
- Academic experts committed to social insurance principles

This board reviews major methodological decisions and can raise concerns about modeling choices that systematically bias against vulnerable populations.

**5. Documentation includes normative framing**

The technical documentation should explicitly state:
- Social Security is social insurance, not an investment account or welfare program
- Adequacy is as important as solvency
- Distributional impacts matter more than aggregate fiscal costs
- Progressive reforms are feasible and necessary

Value neutrality in microsimulation is a myth—every modeling choice embeds assumptions. Be explicit about progressive values.

## 4. Open-Source Access and Democratization

### Strengths

This is the proposal's greatest strength. The commitment to full transparency, zero cost, web interface, Python API, and open-source code represents a fundamental democratization of policy analysis tools.

The comparison table (existing-models.md, lines 273-288) starkly illustrates the gap: no existing model provides full open source + web interface + API + zero cost. This combination is genuinely unprecedented.

The Enhanced CPS precedent (intro.md, lines 25-35) demonstrates PolicyEngine's commitment to public goods even when it requires years of development with no immediate revenue. This is credible evidence the team will execute on transparency promises.

### Remaining Concerns

**1. Digital divide in web interface access**

A web interface is more accessible than code, but still assumes:
- Internet access
- Digital literacy
- English language proficiency
- Familiarity with policy concepts

Community-based organizations serving immigrant communities, rural areas, or elderly populations may struggle to use even a user-friendly interface.

**Recommendation:**
- Multi-language interface (Spanish as priority, given Hispanic population)
- Video tutorials with closed captions
- Technical assistance program partnering with community organizations
- Train-the-trainer workshops for advocacy organizations
- API access enabling intermediaries to build custom interfaces

**2. Computational barriers for API users**

While the Python API is open, using it effectively requires:
- Python programming skills
- Understanding of microsimulation concepts
- Computational resources for running analyses

This could recreate inequality between well-resourced think tanks (Heritage, AEI) and under-resourced advocacy groups.

**Recommendation:**
- Provide Google Colab notebooks with pre-built analysis templates
- Host quarterly webinars teaching API use
- Offer cloud compute credits for nonprofit organizations
- Maintain a library of example analyses for common reform types

**3. Interpretation requires policy expertise**

Generating numbers is easy; interpreting them correctly is hard. A microsimulation showing "benefits fall by 15% for top quintile" sounds progressive, but if it's due to means-testing that creates cliffs and poverty traps, the distributional table alone is misleading.

**Recommendation:**
- Detailed guidance documentation on interpreting results
- Red flags for common pitfalls (e.g., "benefit cut" vs "delay in benefit receipt")
- Partnership with academic researchers to provide technical assistance
- Community forum where users can ask interpretation questions

## 5. Missing Elements: Poverty Impacts, SSI, and Safety Net Interactions

### Poverty Measurement

The proposal mentions "poverty" exactly **four times** in 112 pages:
1. Intro (line 47): "effects on poverty" as a model capability
2. Calibration-targets (line 230): poverty as a non-target validation variable
3. Technical-specifications (line 85): Medicaid asset testing
4. Nowhere else

This is an astonishing omission for a Social Security model. Social Security is the **largest anti-poverty program in American history**—lifting 16.5 million elderly individuals and 1.2 million children above the poverty line annually (CBPP data). Yet poverty measurement and validation appear as afterthoughts.

**Recommendation:**
- Add poverty rates as Tier 1 (Critical) calibration targets (calibration-targets.md)
- Validate against Supplemental Poverty Measure (SPM), not just official poverty measure
- Report poverty impacts by demographic group for every reform analysis
- Include deep poverty (<50% poverty line) and near-poverty (<150% poverty line)
- Analyze cliff effects and phase-out interactions that create poverty traps

### SSI Integration (Repeated for Emphasis)

As noted in section 1, treating SSI as an incremental add-on is unacceptable. Nearly 2.2 million elderly and 5.2 million disabled individuals receive SSI. For aged individuals receiving SSI, 67% also receive Social Security—these programs are deeply intertwined.

The proposal states "adding capabilities for Medicare program outcomes is straightforward" while treating SSI as complex (technical-specifications.md, lines 73-79). This is backward: Medicare eligibility is mostly mechanical (age 65+, SSDI after 2 years), while SSI's asset tests, deeming rules, and state supplements are genuinely complex. Yet Medicare affects middle-class and wealthy elderly, while SSI targets the poorest—revealing whose needs the proposal prioritizes.

**Recommendation:** See section 1—elevate SSI to Phase 4.

### Safety Net Interactions Beyond SSI

The proposal acknowledges interactions with SNAP, Medicaid, EITC, and housing assistance but provides no concrete plan for modeling them. The technical-specifications chapter (lines 71-84) describes a modular approach where "adding realistic modules" for various programs is "potentially feasible" but "cutting edge" and requires "substantial additional development effort."

This framing treats comprehensive safety net modeling as aspirational rather than essential. But you cannot analyze Social Security reform impacts on low-income families without modeling:

1. **SNAP:** Social Security benefit increases can push families above SNAP eligibility, reducing total resources
2. **Medicaid:** Asset tests interact with savings behavior and SSI eligibility
3. **Housing assistance:** Benefit increases raise rent in formula-based subsidies
4. **EITC:** Early Social Security claiming can trigger work disincentives interacting with EITC phase-outs

These interactions determine whether a "benefit increase" actually helps the intended population.

**Recommendation:**
- Phase 4 must include basic SNAP, Medicaid, and housing assistance interactions
- PolicyEngine-US already implements these programs for cross-sectional analysis—leverage existing code
- Validation should confirm that dual program participation rates match SIPP data
- Roadmap currently gives 3 months (months 10-12) for all Social Security benefit calculations; expand to 4-5 months and include basic safety net interactions

### Near-Retiree Financial Hardship

The model focuses on lifetime earnings and long-run projections but does not address near-term hardship facing workers approaching retirement with inadequate savings. The wealth module is described as an "extension" (technical-specifications.md, lines 76-79), but understanding retirement security requires knowing:

- Pension coverage and benefit levels (both DB and DC)
- 401(k) balances and contribution rates
- Home equity (often the largest asset for middle-class households)
- Medical debt and other liabilities
- Expected out-of-pocket health costs

Without this, the model cannot address whether Social Security benefits are adequate for actual retirement needs, which vary enormously based on wealth, housing status, and health.

**Recommendation:**
- Phase 5 should include basic wealth imputation from Survey of Consumer Finances
- Validate wealth distributions by age and income against HRS data
- Report retirement security metrics beyond benefit adequacy: benefit + wealth sufficiency for basic consumption

## 6. Additional Methodological Concerns

### Behavioral Responses and Reform Feedback

The technical specifications chapter (lines 56-67) advocates "simple elasticities rather than full structural optimization" for behavioral responses. This is pragmatic, but the proposal underspecifies which elasticities will be used and how they vary across populations.

Low-income workers may have very different labor supply elasticities than high-income workers:
- Credit constraints limit ability to work more when wages rise
- Care responsibilities create non-standard labor supply responses
- Discrimination and occupational segregation constrain options
- Disability affects labor supply independent of wage incentives

Using aggregate elasticities could mischaracterize distributional impacts of reforms affecting labor supply incentives.

**Recommendation:**
- Literature review chapter should survey labor supply elasticities by income, race, gender, and family structure
- Use heterogeneous elasticities in simulation, not single aggregate parameters
- Sensitivity analysis showing how elasticity assumptions affect distributional results
- Transparent documentation of which elasticity estimates are used and why

### Immigration and Undocumented Workers

The proposal mentions immigration as a demographic input (data-sources.md, line 154) but does not discuss:
- Undocumented workers who pay FICA taxes but cannot claim benefits (estimated 3+ million workers)
- Differential benefit access for immigrants based on visa status
- Totalization agreements with other countries
- Language barriers in claiming and administrative processes

These issues disproportionately affect Latino/Hispanic populations and require explicit modeling.

**Recommendation:**
- Phase 3 (demographic transitions) should include immigration status
- Collaborate with immigration policy experts on modeling benefit eligibility rules
- Validate against DHS estimates of undocumented population by demographic characteristics
- Report reform impacts on immigrant populations separately

### Caregiving and Unpaid Work

The model will capture caregiving's effect on earnings (gaps in work history) but not the intrinsic value of care work or its policy implications. Social Security provides spouse and survivor benefits but no direct credit for caregiving years (unlike some European systems with caregiver credits).

This is not just a technical modeling issue—it's a values question about what work society recognizes and compensates. The proposal should explicitly frame caregiver credits as a reform worth analyzing, not a niche extension.

**Recommendation:**
- Default web interface should include caregiver credit reform as a pre-built analysis option
- Documentation should explain why U.S. system disadvantages caregivers (mostly women)
- Model should be capable of analyzing various caregiver credit designs from day one

### Data Privacy and Ethics

The proposal emphasizes using "fully public data" (intro.md, line 27) and avoiding restricted access files. This is methodologically sound and ethically important—it prevents creating another layer of data haves and have-nots.

However, even synthetic data can raise privacy concerns if it enables inferring characteristics of real individuals. While the QRF imputation approach generates plausible but non-real histories, the combination of publicly available features (age, location, education) with imputed earnings could inadvertently reconstruct sensitive information.

**Recommendation:**
- Conduct privacy risk assessment before public release
- Implement disclosure review protocols for granular geographic data
- Document privacy protection measures in technical documentation
- Consider differential privacy techniques if necessary for small-area estimates

## 7. Recommendations Summary

### Immediate (Before Phase 1)

1. **Rewrite intro.md to explicitly center equity and poverty reduction** as core goals, not just technical capabilities
2. **Add SSI to Phase 4** alongside Social Security benefit calculations
3. **Revise success criteria** to include distributional accuracy, not just aggregate accuracy
4. **Establish advisory board** with representation from advocacy organizations

### Phase 1 Modifications (Months 1-3)

5. **Add poverty rate validation** to proof-of-concept deliverables
6. **Include bottom-quintile earnings validation** as success criterion
7. **Supplement PSID with SIPP** for low-income earnings dynamics

### Phase 3 Modifications (Months 7-9)

8. **Add labor market marginalization** modules (incarceration, caregiving detail)
9. **Include intersectional calibration targets** (race × gender × education)
10. **Model immigration status** and benefit eligibility

### Phase 4 Modifications (Months 10-12)

11. **Integrate basic safety net interactions** (SNAP, Medicaid, housing)
12. **Expand from 3 months to 5 months** to accommodate SSI and interactions
13. **Add poverty and benefit adequacy** validation

### Phase 6 Modifications (Months 16-18)

14. **Default interface emphasizes equity metrics** before fiscal aggregates
15. **Include progressive reform benchmarks** in comparison options
16. **Develop multi-language interface** and accessibility features

### Ongoing

17. **Commission progressive use cases** for launch
18. **Provide technical assistance** to advocacy organizations
19. **Publish annual equity audit** of model use and citations

## Conclusion

This proposal represents a significant technical achievement and a genuine contribution to democratizing policy analysis. The team has the capability to execute on the technical vision, and the Enhanced CPS precedent demonstrates a commitment to public goods.

However, **technical excellence is insufficient if the tool perpetuates existing power imbalances** or enables austerity narratives. From a progressive policy perspective, the proposal requires substantial strengthening to ensure it serves the needs of low-income workers, families of color, women, workers with disabilities, and other vulnerable populations.

The core concern is not what the model *can* do—it is what the model *will* be used for. An open-source model is a double-edged sword: it empowers advocacy groups but also arms opponents of social insurance. The proposal's silence on this tension is the single greatest weakness.

I urge the team to:
1. **Center equity explicitly** in all model design, documentation, and interface decisions
2. **Prioritize poverty impacts and SSI** from the start, not as afterthoughts
3. **Build in safeguards** against misuse through default metrics, framing, and governance
4. **Partner with advocates** throughout development, not just at launch

With these modifications, this model could be transformative—shifting power toward the organizations and communities who have been excluded from sophisticated policy analysis. Without them, it risks being another technically sophisticated tool that serves existing power structures.

I remain available for consultation as the project moves forward and am deeply hopeful that PolicyEngine will rise to this challenge. The stakes—Social Security's future and the retirement security of 67 million Americans—could not be higher.

---

**Indivar Dutta-Gupta**
Distinguished Visiting Fellow
National Academy of Social Insurance
(Former President/ED, CLASP)
