# Open-Source Dynamic Microsimulation for Social Security

This repository contains the planning documents for making `microplex`
longitudinal and using it to build a public Social Security policy
model. The goal is not to imitate government models superficially; it
is to build a public, inspectable alternative that can answer serious
policy questions with transparent methods, published validation, and
free access.

## Why This Project Exists

Social Security policy analysis is dominated by models that are either
internal to government or available only through contracts and
specialized relationships. That creates three problems:

1. Researchers and advocates cannot independently reproduce major
   policy estimates.
2. Small organizations are effectively priced out of serious dynamic
   modeling.
3. Public debate defaults to summaries of model output instead of open
   inspection of assumptions, errors, and tradeoffs.

PolicyEngine already solved an analogous cross-sectional problem with
the Enhanced CPS, and `microplex` generalizes that work into a broader
synthetic population dataset and calibration platform. The next step is
to make that population longitudinal. Social Security is the first
serious proving ground because it forces the project to get lifetime
earnings, family structure, disability, and claiming dynamics right.

## Project Principles

- Public-data first: the full pipeline should be reproducible from
  public or broadly accessible sources.
- Validation before ambition: the project earns credibility only by
  passing explicit validation gates, not by promising everything up
  front.
- Platform first, application second: the core population work belongs
  in `microplex`; this repository is the first domain application and
  validation layer on top of it.
- Social Security first: the first application is Social Security, not a
  universal lifecycle simulator.
- Platform thinking: the architecture should preserve a path to
  adjacent domains such as SSI interactions, retirement adequacy, and
  eventually long-term care.
- Honest scope: this project is a serious research and infrastructure
  effort, not an 18-month substitute for SSA or CBO.

## Decisions Already Made

- **Base population platform**: `microplex` is the population dataset
  and synthesis platform, with its current public cross-sectional layer
  grounded in Enhanced CPS-style construction and calibration.
- **Initial focus**: extend `microplex` longitudinally just far enough
  to support lifetime earnings, family structure, disability, claiming,
  and benefit calculation for Social Security reform analysis.
- **Validation standard**: success requires matching baseline Social
  Security distributions and projections closely enough to support
  exploratory policy analysis, while also validating the underlying
  longitudinal population asset itself.
- **Development model**: a stage-gated, approximately 36-month plan,
  with go/no-go checkpoints after each major methodological hurdle.

## What Success Looks Like

**Within 12 months**
- A documented proof of concept extends `microplex` from a public
  cross-sectional population into a credible longitudinal population
  asset.
- The model matches key baseline distributions closely enough to justify
  continuation.
- The repository contains a published validation report, not just a
  methods narrative.

**Within 24 months**
- A validated longitudinal `microplex` can feed Social Security benefit
  calculations.
- Family, disability, and claiming logic are implemented well enough to
  replicate published baseline distributions and selected reform
  analyses.
- External reviewers can inspect the full pipeline and reproduce core
  results.

**Within 36 months**
- A public API and web interface expose the model for exploratory
  policy analysis.
- The model supports a core set of reform packages, cohort analysis,
  and distributional outputs.
- The documentation clearly distinguishes what is production-ready,
  what is experimental, and what remains out of scope.

## Documentation Map

The main planning documents are a Quarto book in [`docs/`](docs/):

- [index.md](docs/index.md):
  executive summary, scope, and core project decisions
- [funder-summary.md](docs/funder-summary.md):
  short funder-facing synthesis of the investment case
- [policy-applications.md](docs/policy-applications.md):
  the concrete policy questions and user needs the model should serve
- [existing-models.md](docs/existing-models.md):
  comparison to DynaSim, MINT, CBOLT, and other models
- [benchmark-model-component-matrix.md](docs/benchmark-model-component-matrix.md):
  component-by-component benchmark comparison
- [data-sources.md](docs/data-sources.md):
  survey, administrative, and policy-rule data inputs
- [methodology.md](docs/methodology.md):
  synthetic-panel construction and modeling approach
- [operationalizing-longitudinal-construction.md](docs/operationalizing-longitudinal-construction.md):
  concrete design for lifetime earnings and longitudinal state construction
- [technical-specifications.md](docs/technical-specifications.md):
  state variables, transitions, and extensions
- [calibration-targets.md](docs/calibration-targets.md):
  targets, validation strategy, and tolerances
- [public-validation-inventory.md](docs/public-validation-inventory.md):
  public benchmark sources for validation
- [evaluation-and-model-selection.md](docs/evaluation-and-model-selection.md):
  model-selection protocol and validation metrics
- [operationalizing-disability-and-claiming.md](docs/operationalizing-disability-and-claiming.md):
  SSDI pathways, claiming behavior, and timing logic
- [operationalizing-family-and-auxiliary-benefits.md](docs/operationalizing-family-and-auxiliary-benefits.md):
  family histories, spouse matching, and auxiliary-benefit logic
- [operationalizing-mortality-and-projection-drift.md](docs/operationalizing-mortality-and-projection-drift.md):
  mortality construction and projection drift controls
- [infrastructure.md](docs/infrastructure.md):
  how `microplex`, PolicyEngine, and supporting libraries fit together
- [team.md](docs/team.md):
  leadership, staffing needs, and review structure
- [implementation-plan-and-budget-logic.md](docs/implementation-plan-and-budget-logic.md):
  work packages, staffing logic, and the funding case for implementation
- [roadmap.md](docs/roadmap.md):
  stage-gated work plan across 36 months
- [risks-and-stage-gates.md](docs/risks-and-stage-gates.md):
  principal risks, stop/go criteria, and fallback deliverables
- [literature-review.md](docs/literature-review.md):
  academic foundations and model-design context
- [appendix-dynasim.md](docs/appendix-dynasim.md):
  source-heavy public dossier on what is and is not knowable about
  DYNASIM
- [appendix-colorado-ltc-rules-packet.md](docs/appendix-colorado-ltc-rules-packet.md):
  first-pass source packet for a possible Colorado long-term-care pilot

## Current Status

This repository is still a planning and documentation repository. There
is no claim that a validated dynamic Social Security model exists yet.
The immediate product is a stronger project plan for a longitudinal
`microplex` and a cleaner validation strategy for its first policy
application.

## Repository Structure

```text
social-security-model/
├── docs/                  # Quarto planning documentation
├── reviews/               # Reviewer feedback used to strengthen the plan
├── README.md
└── pyproject.toml
```

After implementation begins, the repository is expected to add code for
Social Security-specific validation, rules integration, simulation,
tests, and public-facing interfaces. The more generic population-layer
work should live in `microplex` or its related packages.

## Building the Documentation

```bash
quarto render docs
quarto preview docs
```

Python developer tooling can be installed separately with
`pip install -e ".[dev]"`. Quarto itself is provided by the Quarto CLI,
not by the Python package metadata.

## Related Projects

- [PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us)
- [PolicyEngine-US-Data](https://github.com/PolicyEngine/policyengine-us-data)
- microplex (open synthetic population platform; permissively licensed)
- [microimpute](https://github.com/PolicyEngine/microimpute)
- [microcalibrate](https://github.com/PolicyEngine/microcalibrate)
- [Cato social_security_cato_model](https://github.com/kchanwong/social_security_cato_model)

## Contact

- Max Ghenis: max@policyengine.org
- PolicyEngine: https://policyengine.org

## License

MIT License.
