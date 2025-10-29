# Infrastructure and Tools

## Overview

Building a dynamic Social Security microsimulation model requires sophisticated infrastructure for data processing, imputation, calibration, and policy simulation. Fortunately, PolicyEngine has already developed much of this infrastructure for its Enhanced CPS construction and tax-benefit modeling. This chapter describes the key tools and infrastructure we will leverage.

## PolicyEngine Ecosystem

Our model builds on PolicyEngine's existing open-source infrastructure:

**PolicyEngine-US**: Tax and benefit rules engine (includes Social Security)

**PolicyEngine-US-Data**: Enhanced CPS construction and microdata processing

**PolicyEngine-API**: REST API for policy simulations

**PolicyEngine-App**: Web interface for policy analysis

**PolicyEngine.py**: Python client for programmatic access

We will extend these with new dynamic modeling capabilities while maintaining integration.

## Key Tools and Libraries

### microimpute: ML-Based Variable Imputation

**Purpose**: Impute missing variables using machine learning

**Repository**: https://github.com/PolicyEngine/microimpute

**Key Capabilities**:
- Random forest imputation
- Quantile regression forests
- Multiple imputation
- Cross-validation and validation
- Preserves correlations and distributions

**Our Use**:
- **Critical for earnings history imputation**
- Impute latent variables (earnings potential, health)
- Fill missing demographic variables
- Generate multiple imputations for uncertainty quantification

**Example Workflow**:
```python
from microimpute import QuantileRegressionForest

# Train model on PSID
qrf = QuantileRegressionForest(
    features=['age', 'education', 'current_earnings'],
    target='earnings_at_age_25'
)
qrf.fit(psid_data)

# Predict distribution for CPS
earnings_dist = qrf.predict_quantiles(
    cps_data,
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
)

# Sample from distribution
cps_data['earnings_25_imputed'] = qrf.sample(cps_data)
```

**Status**: Actively maintained, used in PolicyEngine-US-Data

### microcalibrate: Survey Calibration and Reweighting

**Purpose**: Calibrate survey weights to match external targets using gradient descent

**Repository**: https://github.com/PolicyEngine/microcalibrate

**Key Capabilities**:
- Gradient descent reweighting
- Multiple simultaneous targets
- Constraint satisfaction
- Convergence monitoring
- Performance optimization

**Our Use**:
- **Core calibration tool for synthetic panel**
- Match earnings distributions to SSA data
- Align beneficiary counts to administrative totals
- Cross-sectional and longitudinal calibration
- Year-by-year reweighting in projections

**Example Workflow**:
```python
from microcalibrate import calibrate

# Define targets
targets = {
    'mean_earnings_age_25_male': 45000,
    'mean_earnings_age_25_female': 40000,
    'retired_beneficiaries_age_65': 12_200_000,
    # ... hundreds more targets
}

# Calibrate weights
calibrated_weights = calibrate(
    data=synthetic_panel,
    targets=targets,
    initial_weights=cps_weights,
    method='gradient_descent',
    tolerance=0.01,  # 1% tolerance
    max_iterations=1000
)

# Apply calibrated weights
synthetic_panel['weight'] = calibrated_weights
```

**Status**: Actively maintained, used in Enhanced CPS construction

### L0: Sparse Reweighting and Sample Selection

**Purpose**: L0 regularization for discrete sample selection and sparse reweighting

**Repository**: https://github.com/PolicyEngine/L0

**Key Capabilities**:
- L0 regularized optimization
- Discrete (0/1) weight selection
- Gradient-based optimization
- Sample size reduction while maintaining targets

**Our Use**:
- **Optional**: Select representative subsample for computational efficiency
- Alternative to continuous reweighting
- Identify most informative observations
- Reduce computational burden for web app deployment

**Example Workflow**:
```python
from l0 import sparse_calibrate

# Select 50,000 most informative observations
selected_indices = sparse_calibrate(
    data=synthetic_panel,
    targets=targets,
    n_select=50_000,
    method='l0_gradient'
)

# Create reduced dataset
panel_subset = synthetic_panel[selected_indices]
```

**Status**: Under development, research stage

### PolicyEngine-US-Data: Enhanced CPS

**Purpose**: Construct high-quality microdata from CPS with improved income reporting

**Repository**: https://github.com/PolicyEngine/policyengine-us-data

**Critical Context**: The Enhanced CPS is the only publicly available cross-sectional microdata file that produces accurate tax-benefit microsimulation impacts {cite:p}`ghenis2024`. This achievement provides direct proof that our synthetic data approach works.

**The Cross-Sectional Challenge (Solved)**:
- All major tax models rely on IRS PUF (cannot be shared publicly)
- This creates reproducibility crisis in tax policy research
- PolicyEngine solved this with ECPS using the exact methodology we will extend to longitudinal analysis

**ECPS Construction Process** (Two-Stage Methodology):

**Stage 1: Variable Imputation**
1. Download CPS ASEC, PUF, SIPP, SCF, and ACS from public sources
2. Age all datasets to target year using population growth factors and income indices
3. Clone the aged CPS dataset to create two copies:
   - Copy 1: Fills missing variables (mortgage interest, charitable contributions, capital gains)
   - Copy 2: Replaces existing variables with more accurate PUF values
4. Train quantile regression forests on aged PUF using seven demographic predictors
5. Apply QRF to both CPS copies, sampling from conditional distributions
6. Impute additional variables from SIPP (tip income), SCF (auto loans, wealth), and ACS (property taxes)
7. Concatenate both copies to create Extended CPS (doubled sample size ~400,000 individuals)

**Stage 2: Gradient Descent Reweighting**
1. Construct loss matrix containing households' contributions to over 7,000 calibration targets
2. Define targets from six sources: IRS SOI, Census, CBO/Treasury, JCT, Healthcare, and other administrative data
3. Use PyTorch-based gradient descent optimization with Adam optimizer
4. Apply dropout regularization to prevent overfitting
5. Optimize log-transformed weights to ensure positivity
6. Iterate until convergence: targets matched within tolerance
7. Produce Enhanced CPS with calibrated weights

**Validation Results** (why this matters):
- Revenue estimates match Joint Committee on Taxation
- Distributional tables match Tax Policy Center
- Individual calculations validate against tax returns
- Congressional offices use PolicyEngine for actual policy analysis
- **Proof**: Synthetic data + QRF + 7,000+ targets + gradient descent calibration = accuracy comparable to restricted admin data

**Our Use**:
- **Starting point for our synthetic panel**
- **Proven methodology**: Same tools (microimpute, microcalibrate) that produced ECPS
- **De-risked approach**: Not experimental - extending proven cross-sectional methods to longitudinal
- High-quality cross-sectional base
- Already integrated with PolicyEngine-US

**Key Features**:
- ~200,000 individuals (CPS sample)
- Comprehensive income detail
- Tax unit structure
- Survey weights
- Extensively validated against IRS Statistics of Income

**Extending for Dynamic Model**:
- Add earnings history variables (new: QRF imputation from PSID)
- Add demographic transition tracking (new: hazard models)
- Add longitudinal weights (new: multi-year calibration)
- Maintain compatibility with existing PolicyEngine code

**The Precedent**: If ECPS can match IRS data quality without PUF access, our synthetic panel can match SSA data quality without administrative earnings access. Same methodology, different dimension.

### PolicyEngine-Core: Microsimulation Engine

**Purpose**: Core microsimulation framework (forked from OpenFisca-Core)

**Repository**: https://github.com/PolicyEngine/policyengine-core

**Key Capabilities**:
- Variable and parameter system
- Vectorized calculations
- Entity structure (person, household, tax unit)
- Time period handling
- Reform specification
- Extensive formula primitives

**Our Use**:
- **Calculation engine for Social Security benefits**
- Already implements OASDI rules
- Handles reform specifications
- Efficient vectorized simulation
- Proven reliability and accuracy

**Extension Needed**:
- Add variables for full earnings history
- Enhance longitudinal capabilities
- Support cohort-based analysis

## Additional Open-Source Tools

### Quantile Regression Forest (quantreg)

**Package**: `scikit-garden` or custom implementation

**Purpose**: Predict conditional quantiles for distributional imputation

**Use**: Core of earnings history imputation

### NumPy and Pandas

**Packages**: `numpy`, `pandas`

**Purpose**: Data manipulation and numerical computation

**Use**: Throughout data construction and analysis

### Statsmodels

**Package**: `statsmodels`

**Purpose**: Statistical modeling for hazard models and validation

**Use**:
- Discrete-time hazard models for transitions
- Logistic regression for event modeling
- Diagnostic tests and validation

### Matplotlib and Plotly

**Packages**: `matplotlib`, `plotly`

**Purpose**: Visualization

**Use**:
- Validation charts
- Documentation figures
- Web app visualizations

### Jupyter

**Package**: `jupyter`

**Purpose**: Interactive development and documentation

**Use**:
- Exploratory data analysis
- Documentation notebooks
- Validation reports

## Data Storage and Versioning

### HDF5 for Large Datasets

**Format**: HDF5 (Hierarchical Data Format)

**Purpose**: Efficient storage of large panel datasets

**Advantages**:
- Compressed storage
- Fast random access
- Partial reading (don't need to load entire dataset)
- Metadata support

**Structure**:
```
synthetic_panel.h5
├── demographics/
│   ├── age
│   ├── sex
│   ├── education
│   └── ...
├── earnings/
│   ├── year_1985
│   ├── year_1986
│   └── ...
├── benefits/
│   ├── social_security_retirement
│   └── ...
└── weights/
    └── calibrated_weight
```

### Version Control

**Data Versioning**: Track versions of:
- Source data (CPS vintage, PSID release)
- Imputation models
- Calibration targets
- Final synthetic panel

**Code Versioning**: Git for all code

**Reproducibility**: Every analysis records:
- Code version
- Data version
- Parameter assumptions
- Random seeds (for imputation)

## Cloud Infrastructure

### Computing Requirements

**Development**:
- Local machines sufficient for prototyping
- ~32GB RAM recommended for full dataset

**Production**:
- Cloud compute for panel generation (CPU-intensive)
- Parallel processing across cores/instances
- GPU optional for deep learning extensions

### Deployment

**API**: Google Cloud Platform (existing PolicyEngine infrastructure)

**Web App**: Static hosting for frontend, API backend

**Data**: Cloud storage for synthetic panel versions

**Compute**: On-demand compute for panel regeneration

## Software Architecture

### Modular Design

Our codebase follows modular structure:

```
policyengine-us-data/
├── data/
│   ├── downloads/         # Raw data downloads
│   ├── inputs/           # Processed inputs
│   └── outputs/          # Generated datasets
├── imputation/
│   ├── earnings/         # Earnings history imputation
│   ├── demographics/     # Demographic transitions
│   └── validation/       # Validation code
├── calibration/
│   ├── targets/          # Calibration target definitions
│   ├── weights/          # Reweighting algorithms
│   └── validation/       # Calibration validation
├── simulation/
│   ├── projection/       # Forward projection
│   ├── benefits/         # Benefit calculation
│   └── reforms/          # Reform specifications
└── tests/
    ├── unit/             # Unit tests
    ├── integration/      # Integration tests
    └── validation/       # Validation tests
```

### Integration Points

**With PolicyEngine-US**:
- Synthetic panel formatted as PolicyEngine dataset
- Compatible with existing variable definitions
- Uses same entity structure
- Benefit calculations via PolicyEngine variables

**With PolicyEngine-API**:
- API endpoints for dynamic analysis
- Cohort analysis capabilities
- Lifetime benefit calculations
- Reform comparison

**With PolicyEngine-App**:
- Web interface for model access
- Visualization of lifetime profiles
- Distributional analysis dashboards
- Reform analysis tools

## Development Workflow

### 1. Data Acquisition
```bash
# Download CPS
python scripts/download_cps.py --year 2024

# Download PSID
python scripts/download_psid.py --years 1968-2024

# Download administrative targets
python scripts/download_ssa_data.py
```

### 2. Model Training
```bash
# Train quantile regression forests on PSID
python imputation/earnings/train_qrf.py \
    --input data/inputs/psid.parquet \
    --output models/qrf_earnings.pkl
```

### 3. Imputation
```bash
# Impute earnings histories to CPS
python imputation/earnings/impute_history.py \
    --input data/inputs/cps_2024.parquet \
    --model models/qrf_earnings.pkl \
    --output data/outputs/cps_with_history.h5
```

### 4. Calibration
```bash
# Calibrate weights
python calibration/weights/calibrate_panel.py \
    --input data/outputs/cps_with_history.h5 \
    --targets calibration/targets/ssa_2024.yaml \
    --output data/outputs/synthetic_panel_2024.h5
```

### 5. Validation
```bash
# Run validation suite
python validation/validate_panel.py \
    --input data/outputs/synthetic_panel_2024.h5 \
    --report reports/validation_2024.html
```

### 6. Deployment
```bash
# Package for PolicyEngine
python deployment/package_for_policyengine.py \
    --input data/outputs/synthetic_panel_2024.h5 \
    --output policyengine_us_data/datasets/ss_panel_2024/
```

## Testing Strategy

### Unit Tests

**Imputation**:
- QRF prediction accuracy on held-out PSID sample
- Quantile coverage tests
- Distribution preservation

**Calibration**:
- Target matching within tolerance
- Weight positivity
- Convergence

**Calculation**:
- Social Security benefit formulas
- Edge cases (minimum/maximum benefits)
- Spousal/survivor benefits

### Integration Tests

**End-to-End**:
- Full pipeline from raw CPS to synthetic panel
- Validation against all benchmarks
- Reproducibility (same inputs → same outputs)

### Validation Tests

**External Benchmarks**:
- Match SSA aggregates
- Compare to published DynaSim results
- Validate earnings distributions

### Performance Tests

**Computational**:
- Panel generation time
- Memory usage
- API response times

**Accuracy**:
- Prediction intervals for benefits
- Uncertainty quantification
- Sensitivity analysis

## Documentation Strategy

### Technical Documentation

**Code Documentation**:
- Docstrings for all functions
- Type hints
- Inline comments for complex logic

**Architecture Documentation**:
- System design documents
- Data flow diagrams
- API specifications

### User Documentation

**Web Documentation**:
- Getting started guide
- Methodology documentation
- API reference
- Examples and tutorials

**Academic Documentation**:
- Technical papers
- Validation reports
- Comparison to other models

### Jupyter Books

Like this document:
- Planning and methodology
- Validation and results
- Policy applications

## Open-Source Community

### Contributing Guidelines

**Code Contributions**:
- Issue reporting
- Pull request process
- Code review standards
- Testing requirements

**Data Contributions**:
- Alternative imputation methods
- Additional validation benchmarks
- New calibration targets

**Documentation**:
- Tutorials and examples
- Translation
- Improvements and clarifications

### Governance

**Development**:
- PolicyEngine team leads development
- Community input via issues and PRs
- Regular releases with semantic versioning

**Quality**:
- Comprehensive testing
- Code review
- Continuous integration

**Transparency**:
- Public roadmap
- Open development process
- Community feedback

## Summary

We leverage a rich ecosystem of open-source tools:

**Core Tools** (PolicyEngine-developed):
- `microimpute`: ML imputation
- `microcalibrate`: Gradient descent calibration
- `policyengine-us-data`: Enhanced CPS construction
- `policyengine-core`: Microsimulation engine

**Foundation** (Existing):
- Enhanced CPS as starting point
- Proven data construction pipeline
- Social Security rules already implemented
- Infrastructure for web/API deployment

**Extensions** (To develop):
- Earnings history imputation
- Demographic transition modeling
- Longitudinal calibration
- Dynamic analysis capabilities

This infrastructure foundation accelerates development while ensuring quality, reproducibility, and accessibility.

The next chapter describes the team expertise that will execute this plan.
