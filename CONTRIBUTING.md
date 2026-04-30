# Contributing to Social Security Model

Thank you for your interest in contributing to the open-source Social Security dynamic microsimulation model!

## Project Status

This project is currently in the **planning and documentation phase**.
The main product right now is a clearer, more honest project plan with
explicit validation gates and decision points.

## How to Contribute

### Documentation Improvements

The most valuable contributions right now are improvements to the planning documentation:

1. Fork the repository
2. Create a feature branch: `git checkout -b improve-docs`
3. Make your changes to files in `docs/`
4. Build and preview locally:
   ```bash
   quarto render docs
   quarto preview docs
   ```
5. Submit a pull request

### Types of Welcome Contributions

- **Clarifications**: Improve unclear explanations
- **Citations**: Add relevant academic references to `references.bib`
- **Data sources**: Suggest additional calibration targets or data sources
- **Methodology**: Propose improvements to the technical approach
- **Scope discipline**: Point out places where the plan overclaims,
  understates risk, or blurs near-term and long-term deliverables
- **Validation**: Suggest better targets, benchmarks, or stop/go
  criteria
- **Typo fixes**: Correct spelling, grammar, or formatting issues

### Future Code Contributions

Once implementation begins (Phase 1), we will welcome contributions to:

- Data preparation scripts
- Imputation algorithms
- Calibration methods
- Validation tests
- Documentation

## Development Setup

### Requirements

- Python 3.10-3.14
- Quarto CLI 1.9+

### Installation

```bash
# Clone the repository
git clone https://github.com/PolicyEngine/social-security-model.git
cd social-security-model

# Install Python dependencies
make install

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Building Documentation

```bash
quarto render docs    # Build HTML
quarto preview docs   # Serve locally
```

## Code Style

- **Python**: Black formatter with 79-character line length
- **Markdown**: Follow existing document structure and formatting
- **Citations**: Use Pandoc citation syntax, e.g. `[@key]` or
  `[@key1; @key2]`

## Pull Request Process

1. Ensure your changes build without errors
2. Run `make format` to format any Python code
3. Update documentation if needed
4. Submit PR with clear description of changes
5. Address any review feedback

## Questions?

- Open a GitHub issue for questions or discussion
- Email: max@policyengine.org

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.
