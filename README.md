# Social Security Dynamic Microsimulation Model

**Building the first open-source, publicly available Social Security dynamic microsimulation model**

## Overview

This project develops a comprehensive dynamic microsimulation model for Social Security policy analysis, combining:
- PolicyEngine's existing Social Security rules implementation
- Machine learning-based synthetic panel construction
- Quantile regression forests for earnings trajectory imputation
- Gradient descent calibration to administrative targets
- Full integration with PolicyEngine's web app and Python API

This would be the first open-source Social Security model comparable to proprietary tools like DynaSim (Urban Institute) and CBOLT (Congressional Budget Office), democratizing access to sophisticated lifetime benefit analysis.

**This approach is proven.** PolicyEngine's Enhanced CPS is the only publicly available microdata file that produces accurate tax-benefit microsimulation impacts, matching models that use restricted IRS data. This project extends that proven methodology from cross-sectional to longitudinal analysis.

## Documentation

Full planning documentation available as a Jupyter Book:

```bash
cd jupyterbook
myst build --html
myst start
```

Or view chapters directly:
- [Introduction](jupyterbook/intro.md) - Overview and significance
- [Literature Review](jupyterbook/literature-review.md) - Academic foundations
- [Existing Models](jupyterbook/existing-models.md) - Comparison to DynaSim, MINT, CBOLT
- [Technical Specifications](jupyterbook/technical-specifications.md) - Variables, transitions, behavioral responses
- [Data Sources](jupyterbook/data-sources.md) - CPS, PSID, SIPP, SSA data
- [Calibration Targets](jupyterbook/calibration-targets.md) - Validation approach
- [Methodology](jupyterbook/methodology.md) - Technical approach
- [Infrastructure](jupyterbook/infrastructure.md) - Tools and architecture
- [Team](jupyterbook/team.md) - Proposed team leadership
- [Roadmap](jupyterbook/roadmap.md) - Development timeline

## Why This Will Work: The Enhanced CPS Precedent

The same challenge exists in cross-sectional tax modeling: all major models (Tax Policy Center, Penn Wharton, Tax Foundation) rely on the IRS Public Use File, which cannot be publicly shared.

PolicyEngine solved this with the **Enhanced CPS (eCPS)** - the only publicly available microdata producing accurate tax-benefit impacts:
- CPS base + ML imputation + calibration
- Matches Joint Committee on Taxation revenue estimates
- Matches Tax Policy Center distributional tables
- Fully transparent and reproducible

This project applies the same proven methodology to longitudinal analysis:
- **eCPS**: CPS + ML + calibration → accurate tax modeling ✓
- **This project**: CPS + QRF + calibration → accurate Social Security modeling

## Key Innovation

PolicyEngine-US already captures Social Security rules comprehensively. The main challenge is building a **synthetic longitudinal panel** with realistic:
- Lifetime earnings trajectories (birth to retirement)
- Demographic transitions (marriage, divorce, fertility, disability, mortality)
- Cross-sectional accuracy (matches current population)
- Longitudinal consistency (realistic earnings mobility)
- Calibration to SSA projections

## Methodology Summary

1. **Select Base Dataset**: Evaluate options (CPS ASEC, SIPP, or Enhanced CPS) for cross-sectional starting point
2. **Impute Earnings Histories**: Quantile regression forests trained on PSID
3. **Model Demographics**: Hazard models for transitions (marriage, disability, mortality)
4. **Calibrate**: Gradient descent reweighting to match SSA targets
5. **Project Forward**: Year-by-year aging with continued calibration
6. **Calculate Benefits**: Leverage PolicyEngine-US's existing implementation

## Infrastructure

Built on PolicyEngine's open-source tools:
- **microimpute**: Machine learning imputation (quantile regression forests)
- **microcalibrate**: Gradient descent calibration to targets
- **PolicyEngine-US-Data**: Enhanced CPS construction pipeline
- **PolicyEngine-Core**: Microsimulation engine with Social Security rules

## Timeline

**18 months** from start to public launch:
- Months 1-3: Proof of concept
- Months 4-6: Full earnings imputation
- Months 7-9: Demographic transitions
- Months 10-12: Benefit calculation and validation
- Months 13-15: Forward projection and calibration
- Months 16-18: Web interface and API deployment

## Team

- **Max Ghenis**: PolicyEngine founder, infrastructure and integration
- **Ben Ogorek**: PhD Statistics, quantile regression and validation
- **John Sabelhaus**: Former Fed economist, Social Security expert

## Impact

This model will enable:
- **First** open-source dynamic Social Security microsimulation
- Free public access via web interface
- Python API for programmatic analysis
- Full transparency and reproducibility
- Individual-level distributional analysis
- Lifetime benefit calculations across cohorts
- Analysis of reforms (benefit formulas, retirement age, taxation, etc.)

## Project Status

**Current**: Planning and documentation phase

This repository contains the planning documentation. Code development will begin in Phase 1.

## Repository Structure

```
social-security-model/
├── jupyterbook/           # Planning documentation (Jupyter Book)
│   ├── intro.md
│   ├── literature-review.md
│   ├── existing-models.md
│   ├── data-sources.md
│   ├── calibration-targets.md
│   ├── methodology.md
│   ├── infrastructure.md
│   ├── team.md
│   ├── roadmap.md
│   ├── references.bib
│   └── myst.yml
├── README.md              # This file
└── pyproject.toml         # Python package configuration
```

After Phase 1 begins, will add:
```
├── data/                  # Data preparation scripts
├── imputation/           # Earnings history imputation
├── calibration/          # Reweighting and calibration
├── simulation/           # Projection and benefit calculation
└── tests/                # Test suite
```

## Building the Documentation

Requires Python 3.13 and MyST:

```bash
# Install dependencies
pip install -e ".[dev]"

# Build Jupyter Book
cd jupyterbook
myst build --html

# Start local server
myst start
```

View at http://localhost:3004

## Related Projects

### PolicyEngine Infrastructure
- [PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us) - US tax-benefit microsimulation
- [PolicyEngine-US-Data](https://github.com/PolicyEngine/policyengine-us-data) - Enhanced CPS
- [microimpute](https://github.com/PolicyEngine/microimpute) - ML imputation
- [microcalibrate](https://github.com/PolicyEngine/microcalibrate) - Survey calibration
- [L0](https://github.com/PolicyEngine/L0) - Sparse reweighting

### Other Open-Source Social Security Models
- [Cato Institute Social Security Model](https://github.com/kchanwong/social_security_cato_model) - R-based microsimulation using inverse transform sampling methodology

## Citation

If you use this model or methodology, please cite:

```bibtex
@misc{policyengine_ss_model,
  title={Open-Source Social Security Dynamic Microsimulation Model},
  author={Ghenis, Max and Ogorek, Ben and Sabelhaus, John},
  year={2025},
  publisher={PolicyEngine},
  url={https://github.com/PolicyEngine/social-security-model}
}
```

## License

MIT License - See LICENSE file

## Contact

- Max Ghenis: max@policyengine.org
- PolicyEngine: https://policyengine.org
- GitHub Issues: For questions and discussion

## Acknowledgments

This project builds on:
- PolicyEngine's open-source infrastructure
- Academic literature on dynamic microsimulation
- SSA's public data and documentation
- PSID and CPS for panel and cross-sectional data
- Open-source machine learning and statistical tools
