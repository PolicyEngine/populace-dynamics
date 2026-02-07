# Development Roadmap

## Overview

This chapter outlines the development plan for building the open-source Social Security dynamic microsimulation model. We break the project into phases with clear milestones and deliverables.

**Key de-risking factor**: This project extends PolicyEngine's proven Enhanced CPS methodology (the only publicly available microdata producing accurate tax-benefit impacts) from cross-sectional to longitudinal analysis. The core tools (microimpute, microcalibrate), infrastructure (ECPS pipeline), and team expertise are already proven.

## Project Phases

### Phase 1: Foundation and Proof of Concept

**Goal**: Demonstrate feasibility of core methodology with simplified prototype

#### Milestones

**M1.1: Data Preparation**
- Download and clean CPS ASEC (most recent year)
- Download and harmonize PSID (1968-present)
- Extract SSA calibration targets
- Document data sources and preparation

**M1.2: Basic Earnings Imputation**
- Train simple quantile regression forest on PSID
- Impute 10-year earnings history to subset of CPS
- Validate age-earnings profiles
- Compare to SSA earnings statistics

**M1.3: Simple Calibration**
- Implement gradient descent reweighting
- Calibrate to 20-30 key targets
- Validate beneficiary counts
- Document convergence and fit

#### Deliverables
- [ ] Cleaned and documented datasets
- [ ] Working ZI-QDNN imputation pipeline (with QRF comparison)
- [ ] Calibration prototype
- [ ] Validation report comparing to SSA aggregates
- [ ] Technical documentation of methods

#### Success Criteria
- Imputed earnings histories match SSA age-earnings profiles (±5%)
- Calibration matches target variables (±2%)
- Proof of concept demonstrates feasibility

### Phase 2: Full Earnings History Imputation

**Goal**: Complete lifetime earnings imputation with comprehensive validation

#### Milestones

**M2.1: Year-by-Year Imputation Models**
- Train ZI-QDNN (or QRF if PoC favors it) for each age 18-70
- Implement cohort-specific adjustments
- Handle edge cases (zero earnings, gaps)
- Validate on PSID hold-out sample

**M2.2: Earnings History Generation**
- Impute full histories to entire CPS sample
- Implement consistency checks
- Generate multiple imputations
- Calculate AIME distributions

**M2.3: Earnings Validation**
- Compare distributions to SSA statistics
- Validate earnings mobility matrices
- Check correlation structure
- Assess variance components

#### Deliverables
- [ ] Production-ready earnings imputation pipeline
- [ ] Complete earnings histories for base dataset
- [ ] Comprehensive validation report
- [ ] Documentation of imputation methodology (ZI-QDNN / QRF)
- [ ] Published validation notebook

#### Success Criteria
- Age-earnings profiles match SSA data (±3%)
- AIME distribution matches SSA (±5%)
- Earnings mobility matches PSID (±5 percentage points)
- Multiple imputation uncertainty well-calibrated

### Phase 3: Demographic Transitions

**Goal**: Model marriage, divorce, fertility, disability, mortality

#### Milestones

**M3.1: Marital Transitions**
- Estimate marriage/divorce hazard models from PSID
- Implement spousal matching algorithm
- Validate marital status distributions
- Test integration with earnings

**M3.2: Fertility and Family**
- Estimate fertility models from PSID/NVSS
- Assign children to families
- Validate fertility patterns
- Model dependent benefits

**M3.3: Disability and Mortality**
- Implement disability onset models
- Incorporate differential mortality
- Validate against SSA statistics
- Test survivor benefit calculations

#### Deliverables
- [ ] Demographic transition models
- [ ] Family structure simulation
- [ ] Disability and mortality implementation
- [ ] Validation against demographic targets
- [ ] Documentation of demographic methods

#### Success Criteria
- Marital status distributions match Census (±2%)
- Fertility rates match NVSS (±3%)
- Disability prevalence matches SSA (±2%)
- Mortality rates match SSA life tables (±1%)

### Phase 4: Social Security Benefit Calculation

**Goal**: Integrate with PolicyEngine-US for complete benefit modeling

#### Milestones

**M4.1: PolicyEngine Integration**
- Format synthetic panel for PolicyEngine
- Test benefit calculations on sample
- Verify all benefit types
- Validate against SSA calculators

**M4.2: Comprehensive Validation**
- Compare beneficiary counts to SSA
- Validate average benefits by type
- Check benefit distributions
- Test edge cases and special provisions

**M4.3: Reform Modeling**
- Implement several example reforms
- Compare to published analyses where available
- Document reform specification process
- Create user guide for reforms

#### Deliverables
- [ ] Fully integrated synthetic panel in PolicyEngine
- [ ] Benefit calculation validation report
- [ ] Example reform analyses
- [ ] User documentation for reform modeling
- [ ] Comparison to DynaSim results

#### Success Criteria
- Beneficiary counts match SSA (±2%)
- Average benefits match SSA (±3%)
- Benefit distributions match SSA percentiles (±5%)
- Example reforms produce plausible results

### Phase 5: Forward Projection and Calibration

**Goal**: Project population forward with continued calibration

#### Milestones

**M5.1: Projection Framework**
- Implement forward aging
- Project earnings using profiles + shocks
- Simulate demographic transitions
- Add new birth cohorts

**M5.2: Dynamic Calibration**
- Implement multi-year calibration
- Align to SSA Trustees projections
- Prevent drift in long-run projections
- Validate 10-year and 30-year projections

**M5.3: Uncertainty Quantification**
- Multiple imputation inference
- Scenario analysis
- Sensitivity to key assumptions
- Confidence intervals for reforms

#### Deliverables
- [ ] Forward projection pipeline
- [ ] Multi-year calibration system
- [ ] Long-run validation report
- [ ] Uncertainty quantification framework
- [ ] Stochastic projection capability

#### Success Criteria
- 10-year projections match SSA Trustees (±5%)
- Drift controlled through calibration
- Uncertainty quantification well-calibrated
- Projections robust to reasonable assumption changes

### Phase 6: Web Interface and API

**Goal**: Deploy public-facing tools for model access

#### Milestones

**M6.1: API Development**
- Design API endpoints for dynamic analysis
- Implement cohort analysis
- Add lifetime benefit calculations
- Support reform comparisons

**M6.2: Web Interface**
- Design user interface for Social Security analysis
- Implement lifetime profile visualizations
- Create distributional analysis dashboards
- Build reform comparison tools

**M6.3: Documentation and Launch**
- Complete user documentation
- Create tutorials and examples
- Prepare launch materials
- Public release and promotion

#### Deliverables
- [ ] Production API with Social Security endpoints
- [ ] Web application for model access
- [ ] Comprehensive user guide
- [ ] Tutorial videos and examples
- [ ] Public launch and announcement

#### Success Criteria
- API functional and documented
- Web interface user-friendly and responsive
- Positive user feedback
- Growing user adoption

## Ongoing Activities

Throughout the project:

### Code Development
- Follow open-source best practices
- Comprehensive testing (unit, integration, validation)
- Code review by multiple team members
- Continuous integration and deployment

### Documentation
- Maintain up-to-date technical documentation
- Write Jupyter notebooks for methods and validation
- Publish blog posts on progress
- Prepare academic papers

### Validation
- Continuous validation against new data
- Update calibration targets as SSA releases new data
- Track accuracy metrics over time
- External review and feedback

### Community Engagement
- Respond to issues and questions
- Review community contributions
- Present at conferences
- Engage with policy stakeholders

## Risk Management

**Overall risk assessment**: Significantly lower than typical research project due to Enhanced CPS precedent.

PolicyEngine has already proven that public data + ML imputation + calibration can match restricted administrative data quality. We're extending a proven methodology, not testing a new one.

### Technical Risks

**Risk**: Imputation quality insufficient for policy analysis

**Mitigation**:
- Enhanced CPS precedent: same tools (microimpute) already validated
- Extensive validation in Phase 2
- Multiple imputation for uncertainty
- Sensitivity analysis
- Conservative claims about precision
- **Evidence**: ECPS achieves revenue estimate accuracy matching Joint Committee on Taxation

**Risk**: Calibration fails to converge or conflicts arise

**Mitigation**:
- **Enhanced CPS precedent**: microcalibrate already handles 100+ simultaneous targets
- Proven gradient descent methods in production for 2+ years
- Hierarchical target prioritization
- Relaxed tolerances where needed
- Alternative calibration methods if necessary
- **Evidence**: ECPS successfully calibrates to IRS, Census, and SSA targets simultaneously

**Risk**: Computational performance inadequate

**Mitigation**:
- Pre-generation of synthetic panel
- Optimized code and vectorization
- Cloud scaling if needed
- L0 sample selection for reduced dataset

### Data Risks

**Risk**: PSID sample insufficient for training

**Mitigation**:
- Supplement with SIPP for short-term dynamics
- Use published earnings mobility matrices
- Regularization to prevent overfitting

**Risk**: Calibration targets change or unavailable

**Mitigation**:
- Use multiple data sources
- Focus on stable, core targets
- Document assumptions and sensitivity

## Success Metrics

### Technical Metrics

**Accuracy**:
- Earnings distributions within 5% of SSA
- Beneficiary counts within 2% of SSA
- Benefit amounts within 3% of SSA
- Long-run projections within 5% of Trustees

**Performance**:
- Panel generation: <8 hours on standard hardware
- Policy analysis: <1 minute via API
- Web interface: <2 second response times

**Quality**:
- Test coverage >80%
- Code review for all components
- Documented assumptions and limitations
- Reproducible results

### Impact Metrics

**Adoption**:
- 1,000+ unique users in first 6 months
- 100+ policy analyses run
- 10+ external research papers using model

**Visibility**:
- Published in peer-reviewed journal
- Presented at major conferences
- Media coverage in policy outlets
- Congressional office usage

**Community**:
- GitHub stars and forks
- Community contributions
- Issue resolution time
- User satisfaction

## Dependencies and Prerequisites

### Critical Dependencies

**PolicyEngine Infrastructure**:
- Must maintain compatibility with PolicyEngine-Core
- Integration with Enhanced CPS updates
- API infrastructure operational

**Data Availability**:
- CPS ASEC continues to be released annually
- PSID remains publicly accessible
- SSA continues publishing statistics

**Team Availability**:
- Key personnel remain committed
- Can allocate planned time
- Coordination and communication effective

## Maintenance and Updates

After initial launch:

### Annual Updates
- Update base year to latest CPS
- Refresh calibration targets from SSA
- Incorporate legislative changes
- Retrain models on new PSID data

### Continuous Improvement
- Address user feedback
- Fix bugs and issues
- Add new features based on demand
- Improve performance and accuracy

### Long-Term Sustainability
- Seek ongoing funding for maintenance
- Build community of contributors
- Integrate with PolicyEngine's long-term roadmap
- Pursue academic partnerships for improvements

## Resource estimates

### AI-augmented development context

Traditional microsimulation model development timelines—DynaSim, MINT, CBOLT—reflect decades of work by large teams using conventional software engineering. AI-assisted development fundamentally changes the resource calculus. As a concrete reference point: the Cosilico microplex library (normalizing flows, CPS/PSID/PUF data loaders, SSA-calibrated disability and mortality transitions, IPF/entropy/L0 calibration) was built by one person working part-time over approximately two months using Claude Code. The Rules Foundation's statute-encoding infrastructure—AI-assisted translation of legal text into executable code—was built on a similar timeline. These are alpha-quality tools, not production systems, but they demonstrate that AI pair-programming compresses implementation timelines by roughly 5–10× for well-scoped data science and infrastructure tasks.

This compression applies unevenly across task types:

| Task type | AI acceleration | Rationale |
|-----------|----------------|-----------|
| Data pipeline code | 5–10× | Well-defined inputs/outputs, testable, many examples in training data |
| Statistical model implementation | 5–10× | QRF, hazard models, calibration algorithms have clear specifications |
| Validation and testing | 3–5× | AI generates test scaffolding quickly, but interpreting results requires domain judgment |
| Domain modeling decisions | 1–2× | AI assists research and drafting, but Social Security expertise drives decisions |
| Academic writing and documentation | 2–3× | AI drafts, human expert reviews and revises |
| Web/API development | 5–10× | Frontend and backend code generation is a strong AI capability |

Models will continue improving. GPT-5, Claude 4.5 successors, and specialized coding models will likely increase acceleration further over the project's lifetime. We budget conservatively based on current capabilities.

### What already exists (not funded by this project)

Before estimating new work, it's worth noting the substantial infrastructure already built and maintained by PolicyEngine:

| Component | Estimated replacement cost | Status |
|-----------|--------------------------|--------|
| PolicyEngine-US (tax/benefit rules) | $2–5M over 4 years | Production, actively maintained |
| Enhanced CPS (microdata) | $500K–1M | Production, annual updates |
| microimpute (QRF library) | $100–200K | Production |
| microcalibrate (calibration) | $100–200K | Production |
| PolicyEngine-Core (simulation engine) | $1–2M | Production |
| PolicyEngine-API and web app | $1–2M | Production |
| L0 (sparse reweighting) | $50–100K | Research stage |

This infrastructure means the Social Security model is primarily a **data construction project**, not a software platform project. The rules engine, API, and web interface exist. The core challenge is generating the synthetic longitudinal panel.

### Phase-by-phase resource estimates

**Assumptions**: Primary developer working with AI assistance (Claude Code or equivalent). Senior domain expert (Sabelhaus-level) providing guidance and validation review. Statistical methodology advisor (Ogorek-level) for imputation and calibration design. AI acceleration at current (2026) capability levels.

#### Phase 1: Foundation and proof of concept
| Task | Person-months | Notes |
|------|--------------|-------|
| CPS/PSID data preparation | 0.5 | Largely automatable; CPS loader exists, PSID requires harmonization |
| SSA target extraction | 0.5 | Scraping Statistical Supplement tables, structuring ~100 targets |
| ZI-QDNN earnings imputation | 1.0 | Train on PSID, impute 10-year histories to CPS subset |
| QRF and MAF comparison | 0.5 | Evaluate alternatives on same held-out data |
| Simple calibration prototype | 0.5 | microcalibrate already exists; configure for SS targets |
| Validation report | 0.5 | Compare to SSA age-earnings profiles, AIME stats |
| Domain expert review | 0.5 | Sabelhaus/Ogorek review of methodology and results |
| **Phase 1 total** | **4.0** | **~2 months with 2 FTE-equivalent** |

#### Phase 2: Full earnings history imputation
| Task | Person-months | Notes |
|------|--------------|-------|
| Year-by-year QDNN models (ages 18–70) | 1.5 | ~50 models, but largely templated code |
| Cohort-specific adjustments | 0.5 | Birth year effects, secular trends |
| Full history generation + consistency | 1.0 | Multiple imputations, smoothing, constraints |
| Comprehensive validation | 1.0 | AIME distributions, mobility matrices, variance decomposition |
| Methodology documentation | 0.5 | Academic-quality writeup |
| Expert review | 0.5 | |
| **Phase 2 total** | **5.5** | **~3 months with 2 FTE** |

#### Phase 3: Demographic transitions
| Task | Person-months | Notes |
|------|--------------|-------|
| Marriage/divorce hazard models | 1.0 | Estimate from PSID, validate against CPS |
| Spousal matching algorithm | 1.0 | Distance-based matching, assortative mating |
| Fertility models | 0.5 | Simpler; fewer SS implications than other transitions |
| Disability onset/recovery models | 1.0 | Calibrate to SSA DI rates; pre-disability earnings decline |
| Differential mortality | 0.5 | SSA life tables + earnings gradient from Opportunity Insights |
| Integration and validation | 1.0 | All transitions running together, demographic distributions correct |
| Expert review | 0.5 | |
| **Phase 3 total** | **5.5** | **~3 months with 2 FTE** |

#### Phase 4: Social Security benefit calculation
| Task | Person-months | Notes |
|------|--------------|-------|
| Format panel for PolicyEngine | 0.5 | Map synthetic panel variables to PE variable definitions |
| Benefit calculation testing | 1.0 | OASI, SSDI, spousal, survivor, dependent benefits |
| Comprehensive validation | 1.5 | Beneficiary counts, average benefits, distributions vs SSA |
| Example reform analyses | 1.0 | Retirement age, progressive indexing, payroll tax base |
| Documentation | 0.5 | |
| **Phase 4 total** | **4.5** | **~2.5 months with 2 FTE** |

#### Phase 5: Forward projection and calibration
| Task | Person-months | Notes |
|------|--------------|-------|
| Forward aging framework | 1.0 | Project earnings, demographics year-by-year |
| Multi-year calibration | 1.0 | Align to SSA Trustees intermediate assumptions |
| New cohort entry | 0.5 | Birth cohorts, initial conditions |
| Uncertainty quantification | 1.0 | Multiple imputations, scenario analysis |
| Long-run validation | 1.0 | 10-year and 30-year projections vs Trustees |
| **Phase 5 total** | **4.5** | **~2.5 months with 2 FTE** |

#### Phase 6: Web interface and API
| Task | Person-months | Notes |
|------|--------------|-------|
| API endpoints | 1.0 | Extend existing PolicyEngine API |
| Web interface | 1.5 | Lifetime profiles, distributional dashboards, reform tools |
| Documentation and tutorials | 0.5 | User guide, examples, tutorial videos |
| Launch preparation | 0.5 | Blog posts, academic paper draft, outreach |
| **Phase 6 total** | **3.5** | **~2 months with 2 FTE** |

### Total resource summary

| Category | Person-months | Cost estimate (loaded) |
|----------|--------------|----------------------|
| **Engineering/implementation** | 20 | $200–300K |
| **Domain expertise** (Sabelhaus-level) | 4 | $60–80K |
| **Statistical methodology** (Ogorek-level) | 4 | $60–80K |
| **Compute and infrastructure** | — | $10–20K |
| **AI tooling** (Claude Code, etc.) | — | $5–10K |
| **Total** | **~28 person-months** | **$335–490K** |

**Timeline**: 12–15 months from start to public launch, assuming 2 FTE on implementation plus part-time domain and methodology advisors. Phases 1–3 are sequential (hard dependencies). Phases 4–6 can partially overlap.

### Comparison to traditional development

For context on what AI acceleration means here:

| Model | Development time | Team size | Era |
|-------|-----------------|-----------|-----|
| DynaSim | ~10 years ongoing | 5–10 FTE | 1990s–present |
| MINT | ~15 years to maturity | 10+ FTE | 1990s–2010s |
| CBOLT | ~5 years | 5+ FTE | 2010s |
| **This model (estimated)** | **12–15 months** | **2 FTE + advisors** | **2026–2027** |

The 5–10× gap is primarily explained by: (1) existing PolicyEngine infrastructure eliminates ~60% of the traditional scope, (2) AI-assisted development compresses the remaining implementation work, and (3) the project scope is narrower than DynaSim/MINT (Social Security focus vs. full lifecycle). It is **not** because we are cutting corners on validation—the validation phases are budgeted generously because that's where domain expertise matters most and AI helps least.

### Sensitivity and risks to timeline

**Optimistic case** (10 months, $250K): AI capabilities improve faster than expected, PSID data cleaner than assumed, proof of concept validates approach quickly.

**Base case** (15 months, $400K): As estimated above.

**Pessimistic case** (20 months, $600K): PSID harmonization harder than expected, calibration requires novel methods, multiple rounds of expert review reveal fundamental methodology issues.

**Key risk**: The largest uncertainty is whether imputed earnings histories (ZI-QDNN or alternatives) are accurate enough for policy analysis. Early experiments on SIPP panel data are encouraging—ZI-QDNN achieves strong trajectory coverage with realistic zero-inflation patterns—but PSID's smaller sample size (~10,000 families) and the 35-year horizon required for AIME are qualitatively different challenges. Phase 1 is designed as a proof of concept specifically to de-risk this before committing full resources. If Phase 1 validation shows poor AIME distributions, the project should pause for methodology revision rather than proceeding.

## Summary

This roadmap provides a realistic path to creating a valuable public good that democratizes access to sophisticated Social Security policy analysis.

**Phases**: Six phases from proof of concept to web deployment

**Deliverables**: Open-source model, API, web interface, documentation

**Impact**: First open-source dynamic Social Security microsimulation

**Risk**: Manageable with experienced team and proven methods

**Success**: Clear metrics for accuracy, performance, and adoption
