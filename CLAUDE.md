# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Current phase**: Implementation, under a locked pre-registered evaluation gate.

This repository is Populace dynamics: an open longitudinal microsimulation layer (paper at populace.dev/papers/dynamics) plus a working implementation in `src/populace_dynamics/`:

- `harness/` — the population-view scoring harness (geometry blocks, PanelView trajectory windows, the moment battery in `moments.py`)
- `data/` — label-verified PSID readers (`family.py` builds the 1968-2022 head/spouse earnings panel with assignment flags; PSID files staged at `~/PolicyEngine/psid-data`)
- `ss/` — the statutory AIME/PIA oracle (parameters load from a policyengine-us checkout; it is the validation oracle for the Axiom rules-engine encodings, not a rules engine)

`gates.yaml` is the pre-registration contract: gate-1 thresholds are **locked** (ratified via PR #33/#39) and change only through a public amendment plus a fresh referee round. `runs/` holds committed evidence artifacts (noise floors, gate runs, sensitivities) pinned by reproduction tests; model candidates are registered on issue #42 before each one-shot run. Do not edit `gates.yaml` thresholds or committed `runs/` artifacts casually.

Gate runs need populace-fit, which pins scikit-learn <1.9 and cannot coexist with the repo `.venv` — use a dedicated venv (`.venv-gate` pattern); the affected tests importorskip `populace.fit`.

## Commands

```bash
make install      # pip install -e ".[dev]"
make docs         # quarto render docs
make docs-serve   # quarto preview docs --port 3004 --no-browser
make format       # black . -l 79
make clean        # remove build artifacts
```

## Documentation System (Quarto)

| Task | Command |
|------|---------|
| Build HTML | `quarto render docs` |
| Serve locally | `quarto preview docs` |
| Config file | `docs/_quarto.yml` |

### Citation Syntax

Citations use Pandoc syntax with the `references.bib` file:
```markdown
[@key]            # Parenthetical: (Author 2024)
@key              # Textual: Author (2024)
[@a; @b; @c]      # Multiple citations
```

Common citation keys: `ghenis2024` (Enhanced CPS), `favreault2015` (DynaSim), `meinshausen2006` (QRF), `deville1992` (calibration)

### Admonitions

```markdown
::: {.callout-note}
Content here
:::

::: {.callout-warning}
Content here
:::
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
docs/
├── _quarto.yml           # Quarto config (title, TOC, bibliography)
├── index.md              # Executive summary and significance
├── literature-review.md  # Academic foundations
├── existing-models.md    # DynaSim, MINT, CBOLT comparison
├── technical-specifications.md  # Variables, transitions, behavioral responses
├── data-sources.md       # CPS, PSID, SIPP, SSA data
├── calibration-targets.md
├── methodology.md        # QRF imputation, gradient descent calibration
├── infrastructure.md     # PolicyEngine tools ecosystem
├── team.md
├── roadmap.md            # 36-month development plan with milestones
├── references.bib        # BibTeX citations
└── _book/                # Generated output (gitignored)
```

## Related PolicyEngine Repositories

| Repository | Purpose | Relevance |
|------------|---------|-----------|
| [microimpute](https://github.com/PolicyEngine/microimpute) | QRF-based imputation | Core tool for earnings history |
| [microcalibrate](https://github.com/PolicyEngine/microcalibrate) | Gradient descent reweighting | Core tool for SSA calibration |
| [policyengine-us-data](https://github.com/PolicyEngine/policyengine-us-data) | Enhanced CPS construction | Proves methodology works |
| [policyengine-us](https://github.com/PolicyEngine/policyengine-us) | US tax-benefit rules | Has Social Security implementation |
| [L0](https://github.com/PolicyEngine/L0) | Sparse reweighting | Optional sample selection |

## Code Layout

```
src/populace_dynamics/   # harness/, data/, ss/ (see Project Status)
gates.yaml               # locked pre-registration contract
runs/                    # committed evidence artifacts (test-pinned)
scripts/                 # floor builders, gate runners, sensitivities
tests/                   # pytest; PSID-dependent tests skip off-machine
paper/                   # the Quarto paper (front door)
```

## Code Style

- Python 3.10-3.14
- Black formatter, 79-char line length
- Ruff linting
- pytest for testing
