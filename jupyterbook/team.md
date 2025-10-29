# Team Expertise

## Overview

This project brings together a team with complementary expertise in microsimulation modeling, survey calibration, retirement economics, and Social Security policy. This chapter describes the qualifications and specific contributions of each team member.

## Max Ghenis - Project Lead and Infrastructure

### Background

**Current Role**: Founder and President, PolicyEngine

**Education**:
- Master of Data Science, UC Berkeley
- MBA, MIT Sloan School of Management

**Relevant Experience**:
- Founded PolicyEngine (2020), building it into a comprehensive open-source tax-benefit microsimulation platform
- Developed PolicyEngine-US-Data infrastructure, including Enhanced CPS construction
- Created and maintained microimpute and microcalibrate packages
- Former Head of Research, UBI Center
- Former Data Scientist at Google

### Expertise

**Microsimulation Modeling**:
- Designed and implemented PolicyEngine-Core (OpenFisca fork)
- Built multi-country microsimulation models (US, UK, Canada, Israel, Nigeria)
- Integrated complex benefit rules and tax systems
- Developed validation and testing frameworks

**Survey Data and Calibration**:
- Enhanced CPS construction using gradient descent reweighting
- Survey imputation using machine learning (microimpute)
- Calibration to administrative targets (microcalibrate)
- L0 regularization for sample selection

**Software Engineering**:
- Open-source development best practices
- API and web application architecture
- Performance optimization and scalability
- Version control and reproducibility

**Policy Analysis**:
- Analyzed hundreds of policy reforms using PolicyEngine
- Collaborated with Congressional offices, advocacy organizations, think tanks
- Communicated technical results to non-technical audiences
- Translated research into accessible web interfaces

### Role in This Project

**Technical Leadership**:
- Overall project architecture and design
- Integration with PolicyEngine ecosystem
- Software development and code review
- API and web deployment

**Data Infrastructure**:
- Enhanced CPS integration
- Calibration pipeline development
- Validation framework
- Performance optimization

**Project Management**:
- Milestone planning and tracking
- Resource allocation
- Community engagement
- Documentation coordination

## Ben Ogorek - Statistical Modeling and Validation

### Background

**Current Role**: Data Scientist, Meta

**Education**:
- PhD in Statistics, Northwestern University
- MS in Statistics, Northwestern University
- BA in Economics, University of Michigan

**Relevant Experience**:
- Senior Data Scientist at Allstate (6+ years)
- Statistical modeling of insurance claims and customer behavior
- Advanced regression and machine learning methods
- Large-scale data analysis and validation
- Academic research in statistical methodology

### Expertise

**Statistical Modeling**:
- Regression analysis (linear, generalized linear, quantile)
- Longitudinal data analysis and panel methods
- Missing data and imputation techniques
- Bayesian statistics and uncertainty quantification
- Causal inference methods

**Machine Learning**:
- Random forests and tree-based methods
- Quantile regression forests for distributional prediction
- Cross-validation and model selection
- Ensemble methods
- Gradient boosting

**Survey Methodology**:
- Complex survey design and weighting
- Calibration and post-stratification
- Variance estimation with survey data
- Non-response and coverage issues
- Statistical matching across data sources

**Validation and Testing**:
- Model validation frameworks
- Diagnostic methods for model checking
- Sensitivity analysis
- Goodness-of-fit assessment
- Out-of-sample prediction evaluation

### Publications and Presentations

- Multiple peer-reviewed publications in statistical journals
- Conference presentations at JSM, UseR!, and other venues
- Active blog on statistical methods and data science

### Role in This Project

**Statistical Methodology**:
- Quantile regression forest implementation for earnings imputation
- Hazard model specification for demographic transitions
- Validation framework design
- Uncertainty quantification methods

**Model Validation**:
- Design validation tests against external benchmarks
- Implement diagnostic checks
- Assess model performance and accuracy
- Sensitivity analysis and robustness checks

**PSID Analysis**:
- Extract earnings dynamics from PSID
- Estimate transition matrices
- Validate imputation against PSID hold-out sample
- Document PSID data preparation

**Quality Assurance**:
- Code review for statistical components
- Verify correctness of statistical methods
- Documentation of statistical assumptions
- Reproducibility testing

## John Sabelhaus - Retirement Economics and Social Security Expertise

### Background

**Current Role**: Senior Fellow, Wisconsin Center for Financial Security

**Former Role**: Assistant Director, Division of Research and Statistics, Federal Reserve Board (2018-2022)

**Education**:
- PhD in Economics, University of Michigan
- MA in Economics, University of Michigan
- BA in Economics, Northwestern University

**Relevant Experience**:
- 30+ years of experience in retirement economics research
- Led research division at Federal Reserve Board overseeing economic modeling
- Senior economist at Congressional Budget Office
- Economics faculty at University of Maryland
- Editor and referee for top economics journals

### Expertise

**Retirement Economics**:
- Lifecycle saving and consumption
- Retirement wealth adequacy
- Pension systems (DB and DC plans)
- Social Security benefit analysis
- Retirement behavior and decision-making

**Social Security**:
- Deep knowledge of OASDI rules and history
- Benefit formula intricacies (spousal, survivor, disability)
- Social Security Administration data and publications
- Actuarial assumptions and projections
- Reform proposals and their impacts

**Household Finance**:
- Survey of Consumer Finances expertise
- Wealth measurement and distribution
- Asset allocation over the lifecycle
- Housing and mortgage markets
- Consumption and income dynamics

**Data and Measurement**:
- Survey data analysis (SCF, PSID, HRS, CPS, SIPP)
- National accounts and aggregate statistics
- Administrative data (Social Security, IRS)
- Data quality and validation
- Measurement error and survey design

### Publications

- 60+ peer-reviewed publications in top economics journals including:
  - American Economic Review
  - Journal of Political Economy
  - Quarterly Journal of Economics
  - Review of Economics and Statistics
- Multiple Federal Reserve Board working papers
- Congressional Budget Office studies

### Role in This Project

**Social Security Expertise**:
- Validate benefit calculations against SSA rules and examples
- Interpret complex Social Security provisions
- Guide modeling of spousal and survivor benefits
- Review disability benefit implementation
- Ensure consistency with SSA actuarial assumptions

**Economic Validation**:
- Assess realism of earnings trajectories
- Validate lifecycle earnings patterns
- Review retirement behavior assumptions
- Check wealth accumulation patterns
- Verify economic relationships and correlations

**Data and Calibration**:
- Specify appropriate calibration targets from SSA/IRS/Census
- Validate against published research using MINT and DynaSim
- Interpret discrepancies between survey and administrative data
- Guide use of HRS, SCF, and other specialized surveys
- Review demographic and economic assumptions

**Research Translation**:
- Connect model to academic literature
- Identify key papers and methodologies to incorporate
- Situate model in context of existing research
- Suggest validation against published results
- Recommend extensions and improvements

**Quality Control**:
- Review overall model design and assumptions
- Assess plausibility of results
- Identify potential issues and biases
- Suggest robustness checks
- Provide senior oversight and guidance

## Complementary Strengths

The team combines:

**Technical Infrastructure** (Ghenis):
- Software engineering and architecture
- Open-source development and deployment
- API and web application development
- Integration with PolicyEngine ecosystem

**Statistical Methodology** (Ogorek):
- Advanced statistical modeling
- Machine learning and imputation
- Validation and diagnostics
- Uncertainty quantification

**Domain Expertise** (Sabelhaus):
- Social Security rules and policy
- Retirement economics
- Data sources and interpretation
- Connection to academic literature

## Collaborative Approach

### Division of Responsibilities

**Data Construction Pipeline** (Ghenis + Ogorek):
- Ghenis: Infrastructure and integration
- Ogorek: Statistical methods and validation
- Both: Calibration implementation

**Earnings Imputation** (Ogorek + Ghenis):
- Ogorek: QRF implementation and PSID analysis
- Ghenis: Integration with CPS and validation framework
- Sabelhaus: Review of earnings patterns and plausibility

**Demographic Modeling** (Ogorek + Sabelhaus):
- Ogorek: Hazard model estimation
- Sabelhaus: Assumptions and benchmarks
- Ghenis: Integration and simulation

**Social Security Benefits** (Sabelhaus + Ghenis):
- Sabelhaus: Rules interpretation and validation
- Ghenis: PolicyEngine implementation
- Ogorek: Statistical validation

**Calibration** (Ghenis + Sabelhaus + Ogorek):
- Sabelhaus: Target specification and prioritization
- Ogorek: Calibration methods
- Ghenis: Implementation and optimization

**Validation** (Ogorek + Sabelhaus):
- Ogorek: Statistical diagnostics
- Sabelhaus: Economic plausibility and external benchmarks
- Ghenis: Implementation and reporting

### Communication and Coordination

**Regular Meetings**:
- Weekly team calls
- Asynchronous updates via Slack/GitHub
- Quarterly in-person meetings

**Code Review**:
- All major components reviewed by at least two team members
- Statistical methods reviewed by Ogorek
- Social Security components reviewed by Sabelhaus
- Infrastructure reviewed by Ghenis

**Documentation**:
- Collaborative documentation in Jupyter Books
- Technical documentation in code
- Academic papers for peer review
- Public blog posts for broader audience

**Quality Assurance**:
- Multiple validation stages
- Cross-checking of results
- External review by advisors
- Community feedback on open-source code

## Advisory and Collaboration

### Academic Advisors

We will seek input from leading researchers in:
- Dynamic microsimulation (authors of DynaSim, MINT papers)
- Social Security policy (academic experts)
- Machine learning for survey data (quantile regression, imputation)
- Lifecycle economics (retirement and saving behavior)

### Policy Stakeholder Engagement

**Congressional Offices**: Feedback on usability and needed features

**Think Tanks**: Urban Institute, CRFB, Brookings, etc.

**Advocacy Organizations**: AARP, Social Security Works, etc.

**Research Community**: Feedback on methodology and validation

## Team Track Record

### PolicyEngine Accomplishments

**Scale**:
- 6 country models (US, UK, Canada, Israel, Nigeria, Australia)
- 100+ state and federal programs modeled
- 1000+ policy parameters
- Millions of simulations run

**Impact**:
- Used by Congressional offices for policy analysis
- Cited in academic papers and policy briefs
- Featured in major media outlets
- Growing user community

**Quality**:
- Extensive validation against administrative data
- Comprehensive test suites (thousands of tests)
- Open peer review and community contributions
- Continuous improvement and updates

**Innovation**:
- First open-source US tax-benefit microsimulation at this scale
- Enhanced CPS with machine learning imputation
- Web interface democratizing access to microsimulation
- API enabling programmatic access

### Prior Social Security Work

**PolicyEngine-US**:
- Comprehensive implementation of OASDI benefit rules
- Retirement, disability, survivor, and dependent benefits
- Benefit taxation
- Spousal and family benefit calculations
- Validated against SSA calculators

**Policy Analysis**:
- Analyzed Social Security reforms for various clients
- Distributional analysis of benefit changes
- Revenue impact estimates
- Integration with broader tax-benefit analysis

**Published Work**:
- Sabelhaus: Extensive Social Security research
- Ghenis: UBI and tax-benefit policy research
- Team: Combined expertise in microsimulation and retirement policy

## Summary

The team brings together:

**Proven Track Record**:
- PolicyEngine's successful microsimulation platform
- Sabelhaus's decades of retirement economics research
- Ogorek's statistical modeling expertise

**Complementary Skills**:
- Software engineering and infrastructure
- Statistical methods and validation
- Social Security and retirement economics

**Collaborative Culture**:
- Open-source development
- Peer review and quality assurance
- Academic rigor with practical implementation

**Committed to Excellence**:
- Comprehensive validation
- Transparent methodology
- Reproducible research
- Community engagement

This team is well-positioned to build the first open-source dynamic Social Security microsimulation model with the quality and rigor needed for serious policy analysis.

The next chapter provides a roadmap for development and deployment.
