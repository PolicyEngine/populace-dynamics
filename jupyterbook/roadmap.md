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
- [ ] Working QRF imputation pipeline
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

**M2.1: Year-by-Year QRF Models**
- Train QRF for each age 18-70
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
- [ ] Documentation of QRF methodology
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

## Open Questions for Planning

The following questions require further discussion among the project team before finalizing timeline and resource estimates:

### Timeline Considerations

**Phase sequencing:**
- Can phases be parallelized with multiple contributors?
- Which phases have hard dependencies vs. can overlap?
- What is the minimum viable scope for an initial release?

**Development approach:**
- How much can AI-assisted development accelerate traditional timelines?
- Should we pursue a rapid proof-of-concept before detailed planning?
- What is the right balance between speed and validation rigor?

### Resource Considerations

**Team structure:**
- What mix of domain expertise vs. engineering capacity is needed?
- Can existing PolicyEngine infrastructure reduce engineering needs?
- What role should advisory vs. hands-on contributors play?

**Funding strategy:**
- Should we seek a planning grant before full project funding?
- What is the right scope for an initial funding request?
- How do we balance ambition with fundability?

### Scope Considerations

**Core vs. extensions:**
- What constitutes the minimum viable model?
- Which features are essential for credibility vs. nice-to-have?
- How do we prioritize validation breadth vs. depth?

**Integration with PolicyEngine:**
- How much of this is "new" vs. connecting existing pieces?
- SSI, SNAP, Medicaid, etc. already exist in PolicyEngine-US—how does this change scope?
- Can we leverage Enhanced CPS infrastructure more directly?

## Summary

This roadmap provides a realistic path to creating a valuable public good that democratizes access to sophisticated Social Security policy analysis.

**Phases**: Six phases from proof of concept to web deployment

**Deliverables**: Open-source model, API, web interface, documentation

**Impact**: First open-source dynamic Social Security microsimulation

**Risk**: Manageable with experienced team and proven methods

**Success**: Clear metrics for accuracy, performance, and adoption
