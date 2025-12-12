# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Current phase**: Planning and documentation only. No implementation code exists yet.

This repository contains a Jupyter Book (MyST 2.0) planning document for an open-source Social Security dynamic microsimulation model. The project will create the first publicly available tool comparable to proprietary models like DynaSim (Urban Institute), MINT (SSA), and CBOLT (CBO).

## Commands

```bash
make install      # pip install -e ".[dev]"
make docs         # cd jupyterbook && myst build --html
make docs-serve   # cd jupyterbook && myst start (serves at localhost:3004)
make format       # black . -l 79
make clean        # remove build artifacts
```

## Documentation System (MyST 2.0)

**This uses Jupyter Book 2.0 / MyST - NOT legacy Jupyter Book 1.x.**

| Task | Command |
|------|---------|
| Build HTML | `myst build --html` |
| Serve locally | `myst start` |
| Config file | `myst.yml` (not `_config.yml`) |

### Citation Syntax

Citations use MyST syntax with the `references.bib` file:
```markdown
{cite:p}`key`     # Parenthetical: (Author 2024)
{cite:t}`key`     # Textual: Author (2024)
{cite:p}`a,b,c`   # Multiple citations
```

Common citation keys: `ghenis2024` (Enhanced CPS), `favreault2015` (DynaSim), `meinshausen2006` (QRF), `deville1992` (calibration)

### Admonitions

```markdown
::::{note}
Content here
::::

::::{warning}
Content here
::::
```

## Writing Standards

This is an **academic planning document** intended for funders and stakeholders. When editing:

- Use formal academic tone with proper citations
- Reference specific data sources (CPS, PSID, SSA administrative data)
- Be precise about methodology (quantile regression forests, gradient descent calibration)
- Include success criteria with quantitative targets (e.g., "±5% of SSA statistics")
- Avoid vague claims; cite evidence or mark as assumptions

## Core Technical Concepts

The model extends PolicyEngine's **Enhanced CPS (ECPS)** methodology:

1. **microimpute**: Train quantile regression forests on PSID panel data
2. **Apply to CPS**: Generate 35-year earnings histories for cross-sectional sample
3. **microcalibrate**: Gradient descent optimization to match SSA administrative targets
4. **PolicyEngine-US**: Calculate benefits using existing Social Security rules

Key insight: ECPS already proved this approach works for cross-sectional tax modeling (matches JCT and TPC estimates). This project extends the same methodology to longitudinal analysis.

## Repository Structure

```
jupyterbook/
├── myst.yml              # MyST config (title, TOC, bibliography)
├── intro.md              # Executive summary and significance
├── literature-review.md  # Academic foundations
├── existing-models.md    # DynaSim, MINT, CBOLT comparison
├── technical-specifications.md  # Variables, transitions, behavioral responses
├── data-sources.md       # CPS, PSID, SIPP, SSA data
├── calibration-targets.md
├── methodology.md        # QRF imputation, gradient descent calibration
├── infrastructure.md     # PolicyEngine tools ecosystem
├── team.md
├── roadmap.md            # 18-month development plan with milestones
├── references.bib        # BibTeX citations
└── _build/               # Generated output (gitignored)
```

## Related PolicyEngine Repositories

| Repository | Purpose | Relevance |
|------------|---------|-----------|
| [microimpute](https://github.com/PolicyEngine/microimpute) | QRF-based imputation | Core tool for earnings history |
| [microcalibrate](https://github.com/PolicyEngine/microcalibrate) | Gradient descent reweighting | Core tool for SSA calibration |
| [policyengine-us-data](https://github.com/PolicyEngine/policyengine-us-data) | Enhanced CPS construction | Proves methodology works |
| [policyengine-us](https://github.com/PolicyEngine/policyengine-us) | US tax-benefit rules | Has Social Security implementation |
| [L0](https://github.com/PolicyEngine/L0) | Sparse reweighting | Optional sample selection |

## Future Code Structure (Phase 1+)

When implementation begins, the repository will add:

```
data/           # Data preparation (CPS, PSID downloads)
imputation/     # Earnings history QRF models
calibration/    # Gradient descent reweighting to SSA targets
simulation/     # Forward projection and benefit calculation
tests/          # Unit and validation tests
```

## Code Style

- Python 3.10-3.13
- Black formatter, 79-char line length
- Ruff linting
- pytest for testing
