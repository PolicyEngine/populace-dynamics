# Infrastructure and Tools

## Overview

Building a dynamic Social Security microsimulation model requires sophisticated infrastructure for data processing, imputation, calibration, and policy simulation. Fortunately, PolicyEngine has already developed much of this infrastructure for its Enhanced CPS construction and tax-benefit modeling. This chapter describes the key tools and infrastructure we will leverage.

## PolicyEngine Ecosystem

Our model builds on PolicyEngine's existing open-source infrastructure:

```{mermaid}
flowchart LR
    subgraph core["Core Libraries"]
        MI["microimpute<br/>QRF Imputation"]
        MC["microcalibrate<br/>Gradient Descent"]
        L0["L0<br/>Sparse Selection"]
    end

    subgraph data["Data Layer"]
        ECPS["Enhanced CPS<br/>(Cross-sectional)"]
        SSM["Social Security Model<br/>(Longitudinal)"]
    end

    subgraph rules["Policy Rules"]
        PEUS["PolicyEngine-US<br/>Tax & Benefit Rules"]
    end

    subgraph interface["User Interface"]
        API["PolicyEngine-API"]
        APP["PolicyEngine-App"]
        PY["policyengine.py"]
    end

    MI --> ECPS
    MC --> ECPS
    MI --> SSM
    MC --> SSM
    L0 --> SSM
    ECPS --> PEUS
    SSM --> PEUS
    PEUS --> API
    API --> APP
    API --> PY

    style SSM fill:#e1f5fe
```

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

The Enhanced CPS uses QRF imputation and gradient descent calibration to ~2,800 administrative targets {cite:p}`ghenis2024`. This project extends the same methodology to longitudinal earnings histories.

**Our Use**:
- Base cross-sectional population
- Proven methodology (microimpute, microcalibrate)
- Already integrated with PolicyEngine-US

**Extensions for Dynamic Model**:
- Add earnings history variables (QRF from PSID)
- Add demographic transitions (hazard models)
- Add longitudinal calibration targets

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

**Structure** (aligned with the 4-table output schema defined in [Technical Specifications](technical-specifications.md#output-dataset-structure)):
```
synthetic_panel.h5
├── person/            # One row per individual (demographics, status)
├── earnings/          # One row per person-year (covered earnings, QC)
├── relationship/      # Family network (marriages, parent-child)
├── event/             # Life events (disability, death, claiming)
├── computed/          # Derived variables (AIME, PIA, eligibility)
└── weights/
    └── calibrated_weight
```

For distribution, CSV or Parquet files (one per table) provide maximum accessibility. For production analysis, HDF5 or a SQL database provides better query performance.

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
- `microimpute`: ML imputation (QRF)
- `microcalibrate`: Gradient descent calibration
- `policyengine-us-data`: Enhanced CPS construction
- `policyengine-core`: Microsimulation engine

**Foundation** (Existing):
- Enhanced CPS as starting point
- Proven data construction pipeline
- Social Security rules already implemented
- Infrastructure for web/API deployment

**Additional Methodological Approaches** (To evaluate during proof of concept):
- **Zero-inflated quantile deep neural networks (ZI-QDNN)**: Primary candidate for earnings imputation, with dedicated zero-inflation head and conditional quantile output; early experiments show 3× better trajectory coverage than recurrent alternatives on survey panel data
- **Normalizing flows**: Conditional Masked Autoregressive Flows (MAF) as alternative for joint multi-year imputation, particularly where cross-year correlation structure matters
- **Multi-survey fusion**: Harmonize CPS, PSID, and PUF into unified datasets using common variable schemas and masked imputation for cross-survey variables
- **Sparse calibration**: IPF (raking), entropy balancing, and L0/L1/L2 sparse reweighting as alternatives to gradient descent for different calibration tasks
- **Demographic transition models**: Discrete-time hazard models for disability onset/recovery (using SSA DI incidence rates), mortality (using SSA period life tables), and marriage/divorce (using CPS/ACS-based rates)
- **Hierarchical household synthesis**: Two-pass household/person generation preserving family structure, tax unit composition, and spousal earnings correlations

**Extensions** (To develop):
- Full earnings history imputation (ZI-QDNN primary, with QRF and normalizing flows as alternatives)
- Spousal matching and assortative mating
- Forward projection with multi-year calibration
- Dynamic analysis API and web interface

This infrastructure foundation accelerates development while ensuring quality, reproducibility, and accessibility.

The next chapter describes the team expertise that will execute this plan.
