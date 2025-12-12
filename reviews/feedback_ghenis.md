# Self-Review: Social Security Dynamic Microsimulation Model Planning Document

**Reviewer**: Max Ghenis (Project Lead)
**Date**: 2025-12-12
**Document Version**: Draft Planning Document

---

## Executive Summary

This planning document presents a compelling vision for building the first open-source Social Security dynamic microsimulation model. However, as the project lead who would be responsible for securing funding and delivering results, I identify several areas requiring honest reassessment before presenting to potential funders.

**Key Strengths**:
- Strong foundation in the proven Enhanced CPS methodology
- Clear differentiation from existing proprietary models
- Realistic technical approach leveraging established tools
- Transparent about being in planning phase

**Critical Gaps**:
- Overconfidence in timeline estimates without proof of concept
- Insufficient discussion of what makes longitudinal modeling genuinely harder
- Missing strategic discussion of base dataset selection
- Weak treatment of behavioral responses and policy feedback effects
- Inadequate acknowledgment of team capacity constraints
- Limited discussion of what happens if validation fails

---

## 1. Technical Accuracy Assessment

### 1.1 ECPS Methodology Extension

**What's Accurate**:
- The ECPS analogy is fundamentally sound: we did prove that synthetic data + QRF + calibration works for cross-sectional analysis
- The two-stage methodology (impute → calibrate) is correctly described
- The tools (microimpute, microcalibrate) do exist and work

**What's Oversimplified**:
- The document claims longitudinal imputation is "actually somewhat easier than ECPS cross-sectional imputation" (methodology.md, line 640). This is dangerously overconfident. Cross-sectional imputation matches current-year distributions. Longitudinal imputation must preserve:
  - Autocorrelation structure across 35+ years
  - Realistic earnings mobility (not too sticky, not too volatile)
  - Cohort effects that vary by education/race/sex
  - Consistency with observed transitions in multiple panel datasets

  The curse of dimensionality is much worse in the longitudinal case. We're predicting joint distributions over time, not marginal distributions at a point.

- The claim that "PSID has true panel structure (gold standard for training)" glosses over serious PSID limitations:
  - Sample size ~9,000 families means thin coverage of specific demographic × education × cohort × earnings level combinations
  - Attrition bias, especially among high earners and minorities
  - PSID oversamples low-income families (by design), requiring careful reweighting
  - Public-use PSID has geographic restrictions limiting state-level analysis

**Missing Technical Complexity**:
- How do we handle the fact that CPS and PSID measure income differently? CPS uses calendar year; PSID's income reference periods have varied over time
- QRF performance degrades when extrapolating beyond training data support. What about cohorts born after 2000 who have no complete PSID earnings histories?
- The document doesn't address the fundamental identification problem: we observe age-period-cohort effects confounded in PSID. How do we separately identify them for projection?

### 1.2 Calibration Approach

**What's Accurate**:
- Gradient descent reweighting is a proven approach (we use it in ECPS)
- The microcalibrate tool exists and works for 100+ targets

**What's Concerning**:
- ECPS calibrates to ~7,000 targets for a single year. This project proposes calibrating a synthetic *panel* with 35+ years of history per person to SSA targets. The targets in calibration-targets.md are well-specified, but the document doesn't address:
  - Do we calibrate the entire joint distribution of (earnings₁₉₈₅, earnings₁₉₈₆, ..., earnings₂₀₂₄) or just marginal distributions by year?
  - If marginal: we'll match aggregate totals but potentially get mobility patterns wrong
  - If joint: the target space is exponentially larger and we lack administrative data on joint distributions

- The proposed approach to "multi-year calibration" (methodology.md, Phase 5) is vague. Do we:
  - Reweight independently each year (destroys longitudinal consistency)?
  - Apply single weight vector to person-level history (limits flexibility)?
  - Use time-varying weights (unprecedented, unclear if gradient descent converges)?

**Missing Methodological Detail**:
- What is the actual loss function for longitudinal calibration? Chi-squared distance in what space?
- How do we handle the tension between matching cross-sectional targets (SSA annual earnings distributions) and longitudinal targets (PSID transition matrices)?
- What if these conflict? The document says "hierarchical prioritization" but doesn't specify the hierarchy

### 1.3 Technical Specifications Chapter (Most Concerning)

**Honest Assessment**: The technical-specifications.md chapter, while written by John Sabelhaus and containing valuable insights, reveals a philosophical tension.

**The Problem**:
- Lines 87-103 dismiss OLG models: "OLG models represent sophisticated mental exercises but have little proven predictive capability. According to OLG logic, the U.S. economy should have collapsed under rising government debt long ago, yet this has not occurred."
- This is editorializing, not technical specification. It's also somewhat of a straw man—most OLG modelers don't claim literal predictive power but rather provide internally consistent frameworks for thinking about policy

- The alternative proposed (lines 99-103): "The most direct approach starts by following the money through the fiscal system... These flows can be aggregated to produce fiscal projections without requiring OLG apparatus."

**Why This Matters**: Potential funders (especially academic foundations or CBO) may read this as dismissive of established methods. While I personally agree OLG models have limitations, the tone here could alienate collaborators.

**What's Actually Missing**: A clear statement of what behavioral responses we *will* model:
- Labor supply elasticities? If yes, what values and from which studies?
- Claiming behavior? The document mentions it's important but doesn't specify the model
- Saving responses to Social Security wealth? Not addressed
- DI application decisions? Mentioned but no model specified

**Recommendation**: Rewrite technical-specifications.md to:
1. Be more respectful of existing approaches while noting our different focus
2. Specify the *minimal* set of behavioral responses we'll include in Phase 1
3. Distinguish what we can do (accounting for first-order responses) from what we can't (general equilibrium effects)

---

## 2. Feasibility and Timeline Assessment

### 2.1 The 18-Month Timeline

**Honest Reality Check**:
- Enhanced CPS development took 2+ years with my full attention and significant help from the team
- That was for cross-sectional imputation, which I understand better than dynamic modeling
- This project proposes 18 months to:
  - Solve a harder technical problem (longitudinal imputation)
  - Develop new demographic transition models
  - Validate against more complex targets
  - Build specialized web interface
  - All while I'm running PolicyEngine and maintaining existing systems

**What the Timeline Underestimates**:

1. **Phase 1 (Months 1-3)**: "Demonstrate feasibility with simplified prototype"
   - This is actually the most critical phase that determines if the whole project is viable
   - Success criteria: "Imputed earnings histories match SSA age-earnings profiles (±5%)"
   - **Problem**: ±5% for *average* age-earnings profiles is easy. The hard part is matching:
     - Variance at each age
     - Autocorrelation across ages
     - Mobility transition matrices
     - Distributional percentiles, not just means
   - **Reality**: This phase could easily take 6 months if we discover QRF doesn't preserve longitudinal structure

2. **Phase 2 (Months 4-6)**: "Complete lifetime earnings imputation"
   - Assumes Phase 1's "simplified prototype" scales up easily
   - But QRF hyperparameter tuning is computationally expensive
   - Training 50+ separate QRF models (one per age) on PSID will reveal age-specific challenges we can't predict
   - Multiple imputation for uncertainty (mentioned as deliverable) adds 5-10x computational cost

3. **Phase 3 (Months 7-9)**: "Demographic transitions"
   - This is actually a different skill set (demography) from what we've proven with ECPS
   - Estimating hazard models is standard, but getting spousal matching right is notoriously difficult
   - Assortative mating patterns are complex and interact with earnings imputation
   - If we get this wrong, spousal benefits will be systematically biased

4. **Phases 4-6**: Assumes everything before succeeded
   - But if we hit fundamental issues in Phases 1-3, the timeline collapses

**More Realistic Timeline**:
- **Minimum**: 24 months with full team commitment
- **Likely**: 30-36 months with realistic team availability and unexpected challenges
- **Conservative**: 36+ months if validation reveals fundamental issues requiring methodology changes

**What We Should Actually Promise**:
- 12 months: Proof of concept with robust validation (Phases 1-2 expanded)
- 24 months: Working model with basic web interface (through Phase 4)
- 36 months: Full-featured system with comprehensive validation (through Phase 6)

### 2.2 Team Capacity Reality

**The Document States** (roadmap.md, lines 265-269):
- Max Ghenis: 12 months (60% time)
- Ben Ogorek: 9 months (50% time)
- John Sabelhaus: 6 months (30% time)

**Honest Assessment of My Availability**:
- I'm CEO of PolicyEngine with responsibility for:
  - Maintaining policyengine-us and policyengine-uk
  - Managing policyengine-api and policyengine-app
  - Fundraising and organizational leadership
  - Public communications and partnerships

- Realistically, I can commit:
  - Year 1: 30-40% time (focused on Phase 1 proof of concept)
  - Year 2: 20-30% time (review and validation)
  - Ongoing: 10-20% maintenance and leadership

**The Implication**:
- We need to hire a dedicated technical lead for this project
- That person needs deep expertise in both microsimulation and dynamic modeling
- Budget estimate should include recruiting and onboarding (3+ months)
- Or: acknowledge this is a 3-4 year project at reduced intensity

### 2.3 What Could Actually Go Wrong

**The Document's Risk Section** (roadmap.md, lines 287-345) is too optimistic.

**Real Risks Not Adequately Addressed**:

1. **"QRF produces unrealistic longitudinal structure"**
   - Document's mitigation: "Validate against PSID hold-out sample"
   - **Reality**: PSID hold-out will tell us if QRF predicts PSID well. It won't tell us if PSID-trained QRF generalizes to CPS, which has different sample design and measurement
   - **What happens if**: QRF matches marginal distributions but gets autocorrelation wrong? We'd need to develop conditional imputation methods (much harder) or abandon QRF for something else (e.g., copula methods, VAR models)

2. **"Calibration can't match all targets simultaneously"**
   - Document's mitigation: "ECPS already handles 100+ simultaneous targets"
   - **Reality**: ECPS targets are cross-sectional and mostly linear in weights. Longitudinal targets like "earnings mobility quintile transition rates" are inherently non-linear in weights. Gradient descent may not converge or may converge to poor local optimum
   - **What happens if**: We might need to relax tolerances significantly (±10% instead of ±2%), which undermines credibility

3. **"Results don't match MINT/DynaSim for test cases"**
   - Document dismisses this: "MINT/DynaSim also synthetic for younger cohorts"
   - **Reality**: If our results for standard reforms (e.g., raise retirement age by 2 years) differ by >20% from DynaSim, we have a credibility problem
   - **What happens if**: We'd need to debug whether differences come from our methodology, their methodology, or different assumptions. This could take months and may not have clear resolution

4. **"Base dataset selection is wrong"**
   - **This risk isn't mentioned at all**
   - The document assumes ECPS (enhanced CPS) is the base but never justifies this vs. alternatives:
     - SIPP: Smaller sample but already has some longitudinal structure
     - ACS: Much larger sample (3M households) but less income detail
     - Synthesize from scratch: Maximum flexibility but no anchor to real survey
   - **What happens if**: We discover 6 months in that CPS sample size isn't sufficient for state-level analysis, or that CPS income measurement issues cause systematic bias in earnings histories

---

## 3. Value Proposition and Differentiation

### 3.1 What's Compelling

**The Document Gets This Right**:
- Open-source vs. proprietary is genuinely important for research reproducibility
- Web interface democratizes access in a way DynaSim/MINT don't
- Integration with PolicyEngine's broader tax-benefit system is unique
- Free access removes cost barrier

**The Market Reality**:
- Congressional offices mostly use CBO's CBOLT (not accessible to us)
- Think tanks with budgets use DynaSim (Urban has relationships)
- Academic researchers can access MINT through SSA restricted data
- Most advocacy groups don't do sophisticated microsimulation

**Our Actual Niche**:
- Academic researchers who want full transparency and replicability
- Smaller think tanks and advocacy groups without DynaSim contracts
- Journalists who need accessible tools for stories
- International researchers studying U.S. Social Security
- Students and educators for teaching purposes

**Honest Question**: Is this a large enough user base to justify $270-370K in development costs?

**The Document Should Add**:
- Specific examples of analysis questions our model enables that aren't possible today
- Case studies of how open-source tax microsimulation (our existing tools) has changed public discourse
- Letters of support from potential users explaining what they'd do with this

### 3.2 The Cato Model (Largely Ignored)

**Document's Treatment** (existing-models.md, lines 110-172):
- Acknowledges Cato model exists, is open-source, uses inverse transform sampling
- Lists its limitations (R implementation, no web interface, 10K sample)
- Claims our model will be different/better

**Honest Assessment**:
- Cato model's existence proves there's demand for open-source SS models
- It also shows the challenge: they built it, but has it had major impact?
- Their methodological choice (inverse transform sampling) vs. our QRF is debatable, not obviously inferior

**What We Should Acknowledge**:
- Cato model demonstrates proof of concept
- Our value-add is NOT just being open-source (they are too) but:
  - Larger sample size (if we choose that)
  - Integration with PolicyEngine ecosystem
  - Python vs. R (larger user base)
  - Web interface for accessibility

**Risk**: Funders might ask "Why not just contribute to Cato's model instead of building from scratch?"

**Our Answer Should Be**: Different methodological approach (QRF vs. inverse sampling), different integration (PolicyEngine), and different target users (web interface for public vs. R for researchers). But we should position as *complementary* not *competitive*.

---

## 4. Missing Elements

### 4.1 Base Dataset Selection Strategy

**Huge Gap**: The document mentions ECPS as one option for base dataset but never rigorously compares alternatives.

**The Decision Matrix We Need**:

| Dataset | Sample Size | Income Detail | Panel Structure | Geographic Detail | Integration with PE |
|---------|-------------|---------------|-----------------|-------------------|---------------------|
| **ECPS** | ~200K | High | None | State | Perfect |
| **CPS ASEC** | ~200K | Medium | None | State | Good |
| **SIPP** | ~50K | Medium | 3-4 years | Limited | Moderate |
| **ACS** | ~3M | Low | None | County | Moderate |
| **Synthetic** | Flexible | Flexible | Flexible | Flexible | Perfect |

**The Real Trade-off**:
- ECPS: Proven cross-sectional quality, but still only 200K observations. Is that enough for state-level dynamic analysis? For analyzing small demographic groups (e.g., Asian female college graduates age 55-64)?
- ACS: 15x larger sample would enable much richer subgroup analysis. But income measurement is coarser and we'd need to rebuild calibration
- Fully synthetic: Maximum flexibility but loses grounding in real survey. Credibility risk.

**What's Missing**: Phase 1 should include explicit comparison of base datasets with quantitative metrics:
- Effective sample size for key subgroups
- Precision of distributional estimates
- Computational feasibility
- Integration complexity

### 4.2 Validation Strategy Details

**What's Described** (methodology.md, lines 543-590):
- Cross-sectional validation: match base year distributions
- Longitudinal validation: match earnings dynamics from PSID
- Distributional validation: match benefit distributions
- Fiscal validation: match SSA aggregates
- External validation: compare to published DynaSim results

**What's Missing**:

1. **Quantitative Success Criteria**:
   - Document says things like "match SSA earnings distributions (±5%)"
   - But ±5% of what? Mean? Median? Every percentile?
   - For every demographic group? Or aggregate?
   - If we match 80% of targets within tolerance but miss 20%, is that success?

2. **Pre-Registered Validation Plan**:
   - We should specify all validation tests *before* seeing results
   - Otherwise risk of data mining: keep tweaking until tests pass
   - Where is the commitment to publish results even if validation fails?

3. **Comparison to Existing Models**:
   - Document says "compare to published DynaSim results for same reforms"
   - But DynaSim's published results often don't show enough detail to fully replicate
   - What if our results differ? How do we diagnose whether we're wrong, they're wrong, or just different assumptions?

4. **Out-of-Sample Validation**:
   - Best validation: predict something we can later observe
   - Example: Use model trained on data through 2019 to predict 2020-2024 outcomes
   - Then compare to actual 2024 SSA statistics
   - This isn't mentioned anywhere

**What We Should Add**: A formal validation protocol document specifying:
- All tests to be performed
- Exact success criteria (numerical)
- Analysis plan if tests fail
- Commitment to public transparency about validation results

### 4.3 Policy Applications and Use Cases

**Weakness**: The document focuses heavily on methodology but under-specifies what users will actually *do* with this model.

**What's Briefly Mentioned**:
- Benefit formula changes
- Retirement age adjustments
- Means-testing proposals
- Progressive indexing

**What's Missing**:

1. **Concrete Example Reforms**:
   - We should work through 2-3 specific reform proposals in detail:
     - Example: "Raise FRA to 68 over 10 years, starting in 2030"
     - Show exactly what model outputs: beneficiary counts by age, average benefits, distributional impacts, fiscal effects
     - Specify what assumptions required (behavioral responses to FRA change? claiming age effects?)

2. **Limitations of Model for Policy**:
   - What questions can our model NOT answer?
   - Example: We can't model general equilibrium wage effects of Social Security reforms
   - Example: We can't model effects of reforms on private pension coverage (unless we build that module)
   - Being honest about limitations builds credibility

3. **Comparison to Existing Tool Capabilities**:
   - SSA's Quick Calculator: simple, fast, individual-level
   - DynaSim: comprehensive, slow, requires Urban contract
   - Our model: where does it fit in the ecosystem?
   - For what use cases is ours the best tool? For what use cases should users still use alternatives?

### 4.4 Sustainability and Maintenance Plan

**Brief Mention** (roadmap.md, lines 451-471):
- Annual updates to base year
- Refresh calibration targets
- Ongoing maintenance

**What's Inadequate**:
- No cost estimate for annual maintenance (it won't be zero)
- No plan for funding ongoing maintenance after initial grant
- No discussion of how to handle legislative changes (e.g., major SS reform passed by Congress)
- No plan for community governance if project becomes popular

**What We Need**:
- Realistic annual maintenance budget: $50-75K/year
  - Update datasets
  - Refresh calibration
  - Bug fixes and user support
  - Legislative tracking

- Sustainability options:
  - Grant renewals (risky - not guaranteed)
  - PolicyEngine general budget (adds to our costs)
  - User fees for enhanced support (contradicts free/open mission)
  - Academic partnership with ongoing funding

**Honest Assessment**: We're much better at building things than maintaining them long-term. We should think carefully about whether we can commit to 5+ years of maintenance before launching this.

---

## 5. Honest Limitations and Challenges

### 5.1 What We Should Acknowledge But Don't

**1. Our Approach Can't Match MINT's Historical Accuracy**

MINT has actual administrative earnings records for older cohorts. We have synthetic histories for everyone. For analysis of current retirees (ages 65+), MINT will always be more accurate than our model. We should be explicit about this rather than glossing over it.

**Our honest pitch**: "For forward-looking policy analysis (reforms affecting future retirees), our model provides transparency and reproducibility that MINT doesn't. For historical validation of current beneficiary characteristics, MINT is superior because it uses actual earnings records."

**2. Small Sample Size for Subgroup Analysis**

CPS has ~200K individuals. That sounds like a lot, but:
- For state-level analysis of specific demographics (e.g., Black college-educated women age 55-64 in Ohio), we might have <50 observations
- Microsimulation variance increases with smaller effective sample size
- Our confidence intervals for state-level effects will be wide

**What we should say**: "National-level analysis will have high precision. State-level and detailed subgroup analysis will have larger uncertainty, which we will quantify through bootstrapping."

**3. Longitudinal Imputation Uncertainty Is Larger Than Cross-Sectional**

ECPS imputation uncertainty is well-understood: we impute missing variables for current year. Validation shows it works well.

Longitudinal imputation is different: we're predicting 35 years of earnings history we never observed. Even with validation, there's fundamental uncertainty:
- Did person age 45 with $80K current earnings have steady growth or volatile history?
- Were they low earners who rose, or high earners who fell?
- This affects benefit calculation (AIME uses 35 highest years)

**What we should say**: "We use multiple imputation to quantify and propagate this uncertainty into policy estimates. Reform effects will have confidence intervals reflecting imputation uncertainty."

**4. Behavioral Responses Are Simplified**

The technical-specifications.md chapter proposes simple elasticities rather than structural modeling. That's a reasonable choice for Phase 1, but we should be explicit:

**What we can do**: First-order effects of reforms holding behavior constant, or with simple elasticity-based adjustments

**What we can't do**:
- Endogenous retirement timing (requires full dynamic programming)
- Forward-looking saving responses
- Labor supply over full lifecycle (requires wealth module)
- General equilibrium wage effects

**Why this matters**: Some reforms (e.g., raising payroll tax cap) have large behavioral responses that simple elasticities won't capture well. We should specify which reforms our model is most vs. least suited to analyze.

### 5.2 The "Following the Money" Alternative to OLG

**Technical-specifications.md, lines 99-103** propose an alternative to OLG models: "following the money through the fiscal system."

**Honest assessment**: This sounds appealing but is under-specified. What does it actually mean?

**If it means**: Track distributional incidence of Social Security taxes and benefits
- **Good news**: PolicyEngine already does this for cross-sectional analysis
- **Challenge**: Extending to lifetime incidence requires solving the same problems OLG models face (what discount rate? what mortality assumptions? what about intergenerational transfers?)

**If it means**: Partial equilibrium analysis ignoring macro feedbacks
- **Good news**: Much simpler and more tractable
- **Challenge**: Potentially misleading for large reforms (e.g., privatization) where GE effects matter

**If it means**: Something else
- **Problem**: We haven't defined it clearly enough

**What we should do**: Either specify this approach precisely (equations, assumptions) or remove the dismissive language about OLG models and simply say "we focus on partial equilibrium analysis."

---

## 6. Integration with PolicyEngine Ecosystem

### 6.1 What's Strong

**The Document Correctly Emphasizes**:
- PolicyEngine-US already implements Social Security rules
- ECPS provides high-quality cross-sectional base
- microimpute and microcalibrate tools exist
- API and web infrastructure operational

**This Is Real Value**: We're not starting from zero. We have foundation to build on.

### 6.2 What's Concerning

**1. Compatibility Burden**

ECPS is actively maintained and updated. Any changes to ECPS affect this model's base. We need:
- Version pinning strategy
- Testing to ensure ECPS updates don't break SS model
- Clear ownership of integration points

**2. PolicyEngine-US Variable Definitions**

PolicyEngine-US defines Social Security variables (social_security_retirement, etc.). These work for cross-sectional analysis. For dynamic modeling, we need:
- Full earnings history (35+ variables per person)
- Marital history over time
- Benefit claiming ages and decisions
- Coordination with other benefit programs

**Document doesn't address**: How do these new variables integrate with existing PolicyEngine-US architecture? Do we extend the core framework or build parallel system?

**3. API Design for Dynamic Analysis**

Current PolicyEngine API is designed for: "Given household characteristics and policy, calculate taxes/benefits"

Dynamic SS model needs: "Given cohort, initial conditions, and reform, simulate lifetime outcomes"

**Different paradigm**: Current API is stateless and deterministic. Dynamic API needs to handle:
- Stochastic simulation (multiple runs for uncertainty)
- Cohort selection and filtering
- Time-series outputs (not just single year)
- Much longer computation time

**Document doesn't specify**: API design for these new use cases. Will we extend existing API or create separate endpoint?

### 6.3 What We Should Add

**Integration Testing Plan**:
- Specify how dynamic model tests against PolicyEngine-US benefit rules
- Define compatibility matrix (which PE-US version with which SS model version)
- Plan for handling PE-US updates that change SS calculations

**Data Flow Architecture**:
- Diagram showing how data moves between:
  - ECPS → SS panel construction → PE-US calculations → API → Web app
- Specify formats at each stage
- Identify caching and optimization points

**Version Management**:
- How do we handle "PolicyEngine-US updated SS rules, but SS model was built with old rules"?
- Rebuild entire panel? Patch calculations? Versioned releases?

---

## 7. Recommended Changes to Document

### 7.1 Structural Changes

**1. Add "Critical Uncertainties" Section**

Before roadmap chapter, add section explicitly listing:
- Unresolved methodological questions
- Assumptions that might not hold
- Decisions that affect feasibility
- Plan for resolving each before committing to Phase 2

**2. Separate "Proof of Concept" from "Full Model" Timeline**

Current timeline mixes exploratory work (Phase 1) with production work (Phases 2-6). Should separate:

**Part 1: Proof of Concept (12 months, $100-150K)**
- Goal: Demonstrate technical feasibility
- Deliverable: Working prototype with validation evidence
- Decision point: Go/no-go for full development based on results

**Part 2: Production Development (conditional on Part 1 success)**
- Goal: Full-featured model with web interface
- Timeline: 18-24 months
- Budget: $200-250K

This is more honest about risk and gives funders/ourselves an exit if PoC reveals fundamental problems.

**3. Expand "Limitations" Section**

Add to intro.md or methodology.md:
- What this model can and cannot do
- Which reforms it's best suited to analyze
- Types of analysis where existing models remain superior
- Quantification of key uncertainties

### 7.2 Content Changes

**Introduction Chapter** (intro.md):
- Tone down "first open-source model" claim (Cato model exists)
- Rephrase as "first open-source model with web interface and PolicyEngine integration"
- Add paragraph on "what makes this hard" to set realistic expectations
- Add explicit statement: "This document presents a plan, not a completed model. Feasibility depends on proof-of-concept results."

**Technical Specifications Chapter** (technical-specifications.md):
- Remove dismissive language about OLG models
- Specify exactly which behavioral responses Phase 1 will include
- Add quantitative specifications (which elasticities, from which sources)
- Distinguish "must have" from "nice to have" features
- Add section on "methodological alternatives considered" to show we've thought through options

**Methodology Chapter** (methodology.md):
- Add subsection: "Why longitudinal imputation is harder than cross-sectional"
- Be explicit about curse of dimensionality challenge
- Add discussion of base dataset selection (ECPS vs. alternatives)
- Specify calibration loss function precisely (not just "gradient descent")
- Add "what could go wrong with each phase" subsections
- Remove claim that longitudinal is "easier" than cross-sectional

**Data Sources Chapter** (data-sources.md):
- Add quantitative comparison of sample sizes for key subgroups across datasets
- Specify geographic granularity available from each source
- Add section on measurement differences across surveys (CPS vs. PSID income concepts)
- Discuss administrative data *we can't access* (SSA MEF) and implications

**Calibration Targets Chapter** (calibration-targets.md):
- Good as is, but add:
- Specify exact tolerance for each target tier (currently vague "±2%")
- Add column showing how many observations in base dataset for each target
- Specify what happens if targets conflict (prioritization algorithm)

**Existing Models Chapter** (existing-models.md):
- Tone down competitive framing vs. DynaSim/MINT
- Position as "complementary" serving different users
- Add more detailed discussion of Cato model (it's our closest comparable)
- Add section: "Use cases where existing models remain superior"

**Infrastructure Chapter** (infrastructure.md):
- Add realistic assessment of each tool's maturity
- microimpute: production-ready ✓
- microcalibrate: production-ready ✓
- L0: research stage, may not be needed
- Specify integration points with PolicyEngine-US
- Add section on version management and compatibility

**Roadmap Chapter** (roadmap.md):
- Extend Phase 1 to 4-6 months (from 3)
- Make Phase 1 a formal decision point
- Add realistic timeline: 24-36 months to full deployment
- Revise budget: $100-150K for PoC, $200-300K for full model
- Add "alternative scenarios" with more pessimistic timelines
- Revise team capacity to realistic levels
- Add section: "What happens if validation fails?"

### 7.3 Additions Needed

**New Section: "Base Dataset Selection"** (in methodology or data-sources):
- Rigorous comparison of ECPS vs. CPS vs. SIPP vs. ACS vs. synthetic
- Quantitative metrics for each
- Decision framework
- Note that Phase 1 should test multiple options

**New Section: "Validation Protocol"** (in methodology or roadmap):
- Pre-registered tests (before seeing results)
- Exact success criteria for each test
- Out-of-sample validation plan
- Commitment to transparency about failures
- Comparison framework for DynaSim results

**New Section: "Policy Applications"** (new chapter or section in intro):
- 2-3 worked examples of reforms
- Specify inputs, outputs, assumptions
- Show what model can and can't answer
- Compare to existing tools' capabilities

**New Section: "Sustainability Plan"** (in roadmap):
- Annual maintenance cost estimate
- Funding strategy for years 2-5
- Governance model for open-source community
- Plan for handling major legislative changes

---

## 8. Strategic Considerations

### 8.1 Funder Perspective

**Who might fund this?**
- Foundations interested in Social Security policy (Arnold, Sloan, Russell Sage)
- Federal agencies (SSA, Census) if framed as methodological research
- Philanthropists interested in government transparency

**What will they ask?**

1. **"Why not just fund Urban Institute to make DynaSim open-source?"**
   - Our answer: Different methodological approach, public data only, web interface
   - Better answer: Urban may not want to open-source their proprietary model (it's revenue source)

2. **"Why should we believe you can do this?"**
   - Strong: ECPS precedent, proven tools, experienced team
   - Weak: No proof of concept for longitudinal imputation yet
   - Fix: Propose PoC first, then full funding conditional on results

3. **"What if it doesn't work?"**
   - Current document doesn't really address this
   - Should add: PoC decision point, contingency plans, useful intermediate outputs even if full vision fails

4. **"How is this different from Cato's model?"**
   - Need stronger answer than "ours will be bigger/better"
   - Real differentiation: PolicyEngine integration, web interface, different users

5. **"Can you actually maintain this long-term?"**
   - Current document is vague on sustainability
   - Need realistic plan for years 2-5 funding

### 8.2 Positioning Strategy

**Current Framing**: "First open-source dynamic Social Security model"

**Problem**: Cato model exists and is open-source

**Better Framing**: "First open-source Social Security model with public web interface, Python API, and integration with comprehensive tax-benefit microsimulation"

**Alternative Framing**: "Extending proven Enhanced CPS methodology from cross-sectional to longitudinal Social Security analysis"

**Value Propositions to Emphasize**:
1. **Reproducibility**: Anyone can verify and extend results (uniquely important for science)
2. **Accessibility**: Web interface democratizes access beyond technical specialists
3. **Integration**: Only model combining SS with full tax-benefit analysis
4. **Transparency**: Full code, data, assumptions public
5. **Cost**: Free removes barrier for smaller organizations

**Value Propositions to De-emphasize**:
1. "Better than DynaSim" (we can't prove this without building it)
2. "Simpler than OLG models" (sounds like we're cutting corners)
3. "Faster development timeline" (sets unrealistic expectations)

### 8.3 Risk Communication

**Current Approach**: Acknowledge risks but emphasize mitigation and ECPS precedent

**Problem**: May sound overconfident to sophisticated funders who know microsimulation is hard

**Better Approach**:
- Frame Phase 1 as genuine research with uncertain outcomes
- Be explicit about what we don't know yet
- Show we've thought through alternatives if first approach fails
- Emphasize learning value even if full vision isn't achieved

**Example Reframing**:

*Instead of*: "This project will create the first open-source dynamic Social Security model."

*Say*: "This project aims to demonstrate whether publicly available data, combined with modern machine learning and calibration methods, can produce dynamic Social Security microsimulation accuracy comparable to models using restricted administrative data. Phase 1 will provide empirical evidence on this fundamental question. If successful, Phase 2 will build production model with web interface."

### 8.4 Academic vs. Applied Positioning

**Current Document**: Blends academic rigor (literature review, methodology) with applied focus (web interface, policy impact)

**Tension**: Academic audiences may find web interface focus insufficiently rigorous. Applied audiences may find methodology detail overwhelming.

**Recommendation**: Develop two versions:
1. **Academic Proposal** (for peer review, foundations): Emphasize methodology, validation, contribution to microsimulation science
2. **Applied Proposal** (for policy funders): Emphasize accessibility, democratization, specific policy questions enabled

Or: Keep single document but with clear "executive summary for policy audiences" vs. "technical appendix for methodologists"

---

## 9. Bottom Line Assessment

### 9.1 Is This Project Feasible?

**My Honest Answer**: Yes, but harder and longer than document suggests.

**Confidence Levels**:
- 90% confident: We can impute earnings histories that match SSA aggregate distributions
- 75% confident: We can preserve realistic longitudinal structure (mobility, autocorrelation)
- 60% confident: We can do this in 18 months
- 90% confident: We can do this in 30 months if we're willing to iterate

**Key Uncertainties**:
1. Does QRF preserve longitudinal structure well enough? (Won't know until we try)
2. Can gradient descent calibrate hundreds of longitudinal targets? (Theoretically yes, but untested)
3. Will results match DynaSim/MINT well enough to be credible? (Depends on tolerance)

### 9.2 Is This Project Worth Doing?

**My Honest Answer**: Yes, even with uncertainties.

**Why**:
- Social Security modeling desperately needs transparency and accessibility
- ECPS precedent shows synthetic data approach can work
- Even partial success (e.g., good national estimates, wide CIs for subgroups) would be valuable
- Methodological contribution to microsimulation field
- Advances open-source policy analysis

**But**: We need to be honest about limitations and realistic about timeline.

### 9.3 What Would Make Me Confident to Proceed?

**Phase 1 Must Demonstrate**:
1. QRF trained on PSID can predict held-out PSID earnings histories with:
   - Mean absolute error <$5K per year
   - Correct autocorrelation structure (within 0.1 of true)
   - Correct transition matrices (within 5 percentage points)

2. Calibration to 50+ simultaneous targets (mix of cross-sectional and longitudinal) converges with:
   - All Tier 1 targets within 2%
   - >80% of Tier 2 targets within 5%
   - Computational time <8 hours

3. Preliminary validation shows:
   - Age-earnings profiles match SSA (±3%)
   - AIME distribution matches SSA (±5%)
   - Benefit estimates for test cases within 10% of SSA calculator

**If Phase 1 Achieves This**: Proceed with confidence to Phase 2

**If Phase 1 Achieves 2/3**: Proceed with caution, address weaknesses

**If Phase 1 Achieves <2/3**: Pause, reassess methodology, possibly pivot to simpler approach

### 9.4 What Success Actually Looks Like

**Optimistic Scenario** (30% probability):
- Model matches DynaSim/MINT for standard reforms
- Web interface gets 1000+ users in first year
- Published in top economics/demography journal
- Congressional offices use for analysis

**Realistic Scenario** (50% probability):
- Model matches SSA aggregates within 5%
- Some differences from DynaSim for specific reforms, but defensible
- Modest web interface usage (100-500 users)
- Published in specialized microsimulation venue
- Valuable for transparency-focused researchers and advocates

**Pessimistic Scenario** (20% probability):
- Model matches aggregates but has large subgroup biases
- Difficult to achieve credibility vs. established models
- Limited user adoption beyond PolicyEngine community
- Useful as methodological proof-of-concept but not production tool

**Prepare for**: Realistic scenario is most likely. Document is implicitly written for optimistic scenario.

---

## 10. Specific Recommendations

### 10.1 Before Seeking Funding

**Do**:
1. Run small pilot study (1-2 months):
   - Train QRF on PSID for ages 25, 35, 45
   - Impute to subsample of CPS
   - Check if longitudinal structure preserved
   - Cost: ~$10K (my time + compute)

2. Write 2-page concept note:
   - Problem: lack of open SS models
   - Solution: extend ECPS to longitudinal
   - Precedent: ECPS works for cross-sectional
   - Risk: longitudinal harder, but pilot shows promise
   - Ask: $100-150K for proof-of-concept

3. Identify 3-5 potential users and get letters:
   - Academic researchers needing transparency
   - Advocacy orgs needing accessible tools
   - Specify what they'd do with model

**Don't**:
1. Promise 18-month delivery (too optimistic)
2. Claim superiority to DynaSim (unproven)
3. Minimize difficulty (undermines credibility)

### 10.2 For Grant Proposal

**Emphasize**:
- ECPS precedent (we've done this before)
- Proven tools (microimpute, microcalibrate)
- Public data only (enables reproducibility)
- Team expertise (me + Ben + John)
- Two-phase structure (PoC, then full model)

**De-emphasize**:
- Competing with DynaSim/MINT (frame as complement)
- Revolutionary methodology (frame as extension)
- Aggressive timeline (be realistic)

**Include**:
- Clear success criteria for Phase 1
- Decision point before Phase 2
- Realistic budget for both phases
- Sustainability plan for maintenance
- Letters of support from users

### 10.3 Document Revision Priorities

**High Priority** (must fix before using for fundraising):
1. Extend Phase 1 timeline to 4-6 months
2. Make Phase 1 a formal decision point
3. Add realistic total timeline (24-36 months)
4. Remove dismissive OLG language
5. Add "limitations and uncertainties" section
6. Tone down "first open-source" claim (acknowledge Cato)
7. Specify base dataset selection as Phase 1 deliverable

**Medium Priority** (improve credibility):
1. Add validation protocol with pre-registered tests
2. Expand policy applications with worked examples
3. Add sustainability plan
4. Improve behavioral response specifications
5. Add integration architecture details

**Lower Priority** (nice to have):
1. More literature citations
2. Additional comparison to international models
3. Extended discussion of advanced features

---

## Conclusion

This planning document presents a compelling vision for democratizing Social Security policy analysis through open-source microsimulation. The core methodology—extending the proven Enhanced CPS approach from cross-sectional to longitudinal analysis—is sound and feasible.

However, the document is written with excessive optimism about timeline and insufficient acknowledgment of genuine technical risks. Before seeking funding, we should:

1. **Run pilot study** to de-risk QRF longitudinal imputation
2. **Revise timeline** to realistic 24-36 months
3. **Separate PoC from production** with explicit decision point
4. **Acknowledge limitations** that build rather than undermine credibility
5. **Specify base dataset selection** as research question, not assumption
6. **Add validation protocol** with pre-registered tests and transparent reporting

The project is worth doing. But we'll be more successful if we're honest with funders (and ourselves) about the challenges, realistic about the timeline, and clear about what we can and cannot promise.

**My Recommendation**: Revise document, run pilot, then seek $100-150K for Phase 1 proof-of-concept. If that succeeds, return for Phase 2 funding with empirical evidence in hand.

---

**Max Ghenis**
Founder & CEO, PolicyEngine
December 12, 2025
