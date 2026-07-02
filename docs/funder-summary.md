# Concept note: Populace dynamics

## What this is

This concept note describes the design of an open, longitudinal
Dynamics layer for `populace`, PolicyEngine's country-agnostic
microdata stack — validated first on the U.S. Social Security
system, and built so that every claim it makes can be scored against
reality. Social Security is the proving ground because it is the
hardest first domain: lifetime earnings, family structure,
disability, and claiming all have to be right. The layer itself is
global by construction, and extends to other countries' pension and
benefit systems as PolicyEngine's country coverage grows.

The premise is George Box's, taken literally: all models are wrong,
and a model is useful only if it improves predictions. So this
project's product is not a brand-name simulator. The machinery lives
in `populace`, PolicyEngine's open microdata stack; the deliverable
is a versioned population artifact with a manifest and a public
scorecard; and this repository holds the Social Security application
and the validation program that grades it. Models made their names
in an era when the model was the moat. Here the artifact and its
track record are the product, and both are public.

## Why this matters

Closed institutional models dominate Social Security policy
analysis. They live inside government, depend on restricted
administrative records, or sit behind institutional relationships.
Tax microsimulation has matured into real public infrastructure over
the past decade. Social Security modeling has not.

For a program at this scale, the gap is unusually large.
Researchers, smaller policy organizations, journalists, and
advocates can debate reform packages, but they cannot inspect,
reproduce, or score the models that shape those debates. The public
ecosystem runs on institutional trust where it could run on
published evidence.

## What this model will not claim

Honesty about scope is the design's first feature, not a caveat
([domains-of-validity.md](domains-of-validity.md) gives the full
argument). The 75-year actuarial balance is dominated by assumptions
nobody has forecast well — fertility, mortality improvement,
immigration. The Trustees' own scenario range is wider than the
headline deficit; CBO's long-term projections differ materially from
the Trustees' for assumption reasons, not arithmetic ones; and
successive Technical Panels have flagged the assumptions as realized
values ran outside the projected path
[@ssa2025trustees; @cbo2024longterm; @technicalpanel2023]. A better
microsimulation does not fix that, because the variance lives in the
inputs.

So this model will not sell a depletion date or a point 75-year
balance. Its outputs come in three tiers, each attached to the
strongest claim it can support:

1. **Distributional analysis under fixed assumptions** — who gains
   and loses from a reform, by cohort and lifetime position, where
   common assumption uncertainty largely cancels in the difference.
2. **Near-term components that resolve** — beneficiary counts,
   average benefits, taxable payroll, claiming ages over roughly a
   ten-year horizon, scored annually against administrative
   publications.
3. **The long horizon as a sensitivity surface** — how outcomes move
   across documented assumption ranges, published as the product
   rather than buried behind a point estimate.

The incumbents bury assumption-dependence in appendices. An open
model can make it the interface.

## Design

The model has four components. This note describes the design first,
then its concrete implementation in the PolicyEngine stack.

### Component 1: synthetic longitudinal population

The population layer must support:

- lifetime earnings histories (the highest-35 earnings record that
  drives benefit calculation)
- marriage, divorce, widowhood, and remarriage histories needed for
  spousal, survivor, and divorced-spouse benefits
- disability onset, recovery, and termination dynamics
- mortality with differential rates by lifetime earnings, education,
  race, and sex
- forward projection with explicit drift control
- coherent household and couple structure preserved across years

The population must be public, reproducible, and scorable against
administrative targets. That rules out dependence on restricted
matched data, which is how MINT operates. It rules in synthesis from
survey panel data, administrative aggregates treated as
uncertainty-weighted facts, and modern imputation — with one weight
per trajectory so that multi-period calibration cannot silently
destroy the panel structure.

The hard part of this project lives almost entirely in component 1.

### Component 2: open tax-benefit rules engine

The rules engine must calculate retirement benefits (AIME, PIA, bend
points, COLA, early and delayed claiming adjustments), disability
benefits, dependent, spousal, and survivor benefits, benefit
taxation, and interactions with means-tested programs — and it must
support reform variants by parameter and formula modification,
vectorized over large panels.

An open rules engine matters not only because users should be able
to inspect the rules, but because analysts must be able to encode
and audit the reforms they test. Closed engines force users to trust
the implementation. Statute is also the deterministic slice of any
forecast: where an output is fixed by law, the engine computes it
exactly, and uncertainty budgets attach only to what is genuinely
uncertain.

### Component 3: scoring and benchmark layer

The project's credential is a public scorecard, not a methods
narrative ([scoring-and-resolution.md](scoring-and-resolution.md)).
Five surfaces: annually resolving component forecasts (beneficiary
counts, benefits, payroll — each cell carrying the resolution rule
that settles it); predictions of the next official projection
revisions; retrodictive backtests with leakage control from
version-pinned data vintages; statutory resolution of enacted
policy; and held-out panel moments for the population layer itself.

Misses publish with the same prominence as hits. The stage gates in
the roadmap are score thresholds: the project advances when the
scorecard says so.

### Component 4: public delivery surface

Three surfaces: a Python library and CLI; a REST API; and an MCP
server so AI agents can run baseline distributions, score reform
packages, and generate cohort-specific outputs through natural
language. Every response carries its validity tier, assumption path,
and calibration history — trust as a number the consumer reads, not
a reputation the producer asserts. We are not aware of an existing
dynamic Social Security model that exposes an AI-callable interface
of this kind.

A public web interface follows validation rather than preceding it.

## What is methodologically new

No single component is unprecedented. The contribution is the
combination:

- a public synthetic longitudinal population, calibrated to
  administrative targets treated as facts with standard errors
- an open rules engine that computes the statutory slice exactly
- a published scorecard — resolving forecast cells, retrodictive
  backtests, and held-out moments — in place of fidelity-only
  validation
- domains of validity as shipped metadata on every output
- a contribution rule inherited from `populace`: changes merge if
  and only if they improve the score on held-out facts, from any
  contributor
- AI-callable interfaces from day one

No equivalent bundle exists for U.S. Social Security analysis.

## Implementation: the PolicyEngine stack

The natural implementation is the PolicyEngine open-source stack.

**Populace** is PolicyEngine's rebuilt, open-source microdata stack
([github.com/PolicyEngine/populace](https://github.com/PolicyEngine/populace),
MIT). It builds a calibrated synthetic population entirely from
primary-source government data (CPS/ASEC, IRS Public Use File,
Survey of Consumer Finances, SIPP, CPS outgoing-rotation groups,
MEPS, and ACS), synthesizes missing variables with weight-aware
conditional models, and calibrates to administrative targets treated
as uncertainty-weighted facts. In June 2026 it replaced
PolicyEngine's enhanced CPS as the certified default U.S. microdata
in policyengine.py, after a matched, symmetric-refit comparison on
41,314 households with a 739-target holdout:

| Metric (lower is better) | Populace | enhanced CPS |
|---|---|---|
| Holdout loss (739 held-out targets) | 0.038 | 0.317 |
| Training loss | 0.190 | 1.089 |
| Full loss | 0.228 | 1.405 |
| Per-target wins | 1,040 | 2,613 (51 ties) |

The asymmetry in the last row is published deliberately: the
enhanced CPS wins more individual targets narrowly, while its
largest misses are far larger — Populace's aggregate loss is an
order of magnitude lower on held-out targets. Publishing the number
that cuts against the headline is the discipline this whole project
runs on. (Source: the release manifest in the Populace repository.)

**The longitudinal extension is designed, not improvised.**
Populace's charter names this project's direction explicitly and
specifies the kernel rules: one weight per trajectory, with
multi-period targets stacked as (target, period) constraint rows
over the same weight vector; entry and exit markers (birth, death,
immigration, emigration) so trajectories contribute to a period only
while present; and a Dynamics operator whose scope includes
immigration and births, not only mortality. Transitions reuse the
stack's existing conditional-model protocol — a transition is
P(state next year | state this year, covariates), the same shape as
the shipped synthesis models — with deterministic hazard tables
where the evidence is tabular (mortality from SSA life tables with
published income gradients, fertility and marriage from vital
statistics) and machine-learned models only where conditional
structure is rich (earnings). Backcasting histories and projecting
forward are the same operator run in either direction.

**PolicyEngine-US** supplies the rules engine — OASDI benefit
calculation, benefit taxation, and means-tested interactions —
through Populace's rules-engine adapter, with Axiom's rules layer as
the next adapter: statute encoded declaratively and compiled to
Rust, a performance boundary that matters when benefit formulas run
over person-periods across hundreds of thousands of trajectories.
In that architecture PolicyEngine is a composition — Axiom rules,
Populace population, and a labeled behavioral scenario layer.
**PolicyEngine-API** and the MCP server are the delivery surface.

The deliverable is a versioned artifact — `populace_us_panel_*` —
with a release manifest and scorecard, certified through the same
path the cross-sectional release already passed. The design is
architecture-agnostic in principle, but this is the only stack where
the cross-sectional foundation has already shipped, won its
benchmark, and carries the governance rule the scoring layer needs.

## The open-modeling landscape

The contrast between U.S. tax microsimulation and U.S. Social
Security microsimulation is a useful proxy for the public
infrastructure gap.

On the tax side, the ecosystem spans a spectrum of openness.

**Openly callable, public-data models**:

- **Tax-Calculator** (Policy Simulation Library): open-source federal
  tax microsimulation model [@taxcalc2026]
- **PolicyEngine**: open-source federal and state tax-benefit
  microsimulation with a calibrated public microdata foundation, a
  REST API, and an interactive web interface [@policyengine2026]

**Source-available, restricted-data models**:

- **Yale Budget Lab Tax-Simulator**: the Budget Lab publishes its
  code on GitHub but the model depends on the IRS Public Use File,
  which the IRS does not release publicly, so outside users cannot
  reproduce production runs [@yaletaxsimulator2026]

**Proprietary models** used for outside-facing analysis:

- **Tax Policy Center** microsimulation model [@tpcmodelfaq2025]
- **ITEP** microsimulation tax model [@itepmodel2025]
- **Tax Foundation** Taxes and Growth model [@taxfoundationtag2025]
- **Penn Wharton Budget Model** [@pwbm2025]

The Social Security side has a much thinner open-modeling layer. The
institutional benchmark models are real and important, but outside
users can reach them only through institutional relationships
[@favreault2015; @urban2024dynasim4; @ssa2024mint; @cbo2018; @cbo2024longterm; @look2024retirementoutcomes]:

- **DYNASIM** (Urban Institute) — the benchmark for breadth,
  maturity, and state richness. The open model will not match its
  institutional continuity, but offers full inspectability and
  reproducibility.
- **MINT** (SSA) — the benchmark for earnings-history credibility
  through administrative data. The open model trades administrative
  earnings for fully reproducible synthetic histories with explicit
  validation.
- **CBOLT** (CBO) — the benchmark for official projection authority
  and macro-fiscal integration. The open model makes no claim to
  official scoring; its contribution is a publicly inspectable
  construction, validation, and policy workflow.
- **Morningstar's retirement-outcomes model** — the closest adjacent
  benchmark for retirement-adequacy and LTSS-oriented household
  modeling.

These models are not reasons to avoid the project. They are reasons
to scope it correctly.

The closest open analogue is the **Cato Social Security model**
[@catossmodel2026], an AGPL-3.0 R implementation. It simulates
mortality, fertility, marriage, divorce, and employment as
stochastic transitions on a sample of approximately 10,000
households drawn from the 2007 ASEC, with pre-2007 earnings
histories matched against the 2006 SSA Public Use File and forward
earnings indexed to the Average Wage Index. It produces conventional
long-term Social Security outputs — trust fund ratios, insolvency
dates, 75-year balance — and can score reforms to retirement ages,
bend points, indexing methodology, and benefit credits.

The stated capabilities are deliberately scoped. The model focuses
on OASI without a separate SSDI module, cannot score tax-code
reforms, treats the Windfall Elimination Provision and Government
Pension Offset as limited, holds the labor-force transition matrix
constant from 2024 onward, and depends on simplifying assumptions
for fertility and other dynamics. Running the baseline simulation
requires access to the underlying SSA Public Use File. The
repository does not currently include published validation against
SSA Trustees, MINT, or DYNASIM. A more complete characterization is
in [`existing-models.md`](existing-models.md).

The Cato model shows that an open dynamic Social Security model is
feasible; it does not close the gap this concept addresses. The open
tax-modeling ecosystem already offers production stacks with
calibrated public-data populations, programmatic and AI-callable
APIs, web interfaces, transparent validation, and tax-benefit
integration. The open Social Security layer offers a single narrower
model with none of that combination. That is the specific gap this
project fills.

A more specific signal of demand: some users can already use
PolicyEngine for narrow Social Security-adjacent questions but still
need closed models for broader dynamic analysis. CRFB has
commissioned PolicyEngine for an
[analysis of Social Security benefit-taxation reforms](https://www.policyengine.org/us/taxation-of-benefits-reforms)
(publication forthcoming) — a static tax-side question that the
existing open stack already supports. Broader dynamic questions
about claiming behavior, lifetime distributional impact, and
cohort-specific reform effects still push users toward closed
benchmark models. That is exactly the gap this project narrows.

## Adjacent applications

The same architecture preserves a path to domains that share the
same longitudinal ingredients:

- retirement adequacy and wealth-sensitive analysis (SCF-linked)
- SSI interactions and poverty analysis
- long-term care and caregiving policy, where disability, wealth,
  and family structure matter over time

These are not phase-one commitments. They are reasons to design the
core architecture well.

The longitudinal machinery itself is generic and lives upstream in
`populace`, whose kernel is country-agnostic. The same extension can
eventually serve other countries' pension and benefit systems;
Social Security is the first application, not the boundary.

The most plausible first adjacent step is a state-specific long-term
care pilot rather than a national dynamic LTSS model. A
state-specific pilot can answer concrete eligibility and spend-down
questions for real families while the harder national dynamic LTC
problem remains a separate, later effort.

## What success looks like

Success does not require becoming the official federal baseline.
The project succeeds if it can:

- build a public calibration record — resolving forecast cells and
  retrodictive backtests — that no closed model publishes
- support reform analysis with transparent assumptions and
  distributional outputs anyone can reproduce
- publish a reusable public longitudinal population asset, governed
  by the merge-on-score rule
- lower the barrier to serious dynamic modeling for outside
  researchers and policy organizations
- expose validated capabilities to AI agents through standard
  interfaces, with validity and calibration shipped as metadata

That is meaningful public value even short of official scoring
status — and it is value that compounds, because every resolved cell
makes the scorecard, and the case for trusting it, longer.

## What this is not

- Not "open-source policy analysis in the abstract" — a focused
  build with a proving ground: Social Security first, public scoring
  first, productization only after the record earns it.
- Not a single-country project — the layer is country-agnostic
  infrastructure under open governance, which is also why it cannot
  be co-owned through bilateral institutional agreements.
- Not a 75-year oracle — the long horizon ships as a sensitivity
  surface, never a point forecast.
- Not a brand-name simulator — the machinery is Populace's, the
  artifact is versioned, and the scorecard is the product.

## Open invitation

The project is at the stage where outside input shapes how it
develops. The most valuable conversations right now are with:

- researchers and modelers with retirement-economics or
  microsimulation expertise
- policy organizations working on solvency, adequacy, or reform
  packages who would use validated outputs
- funders interested in public-infrastructure investment rather than
  memo-style research grants
- technical reviewers interested in open scoring as a methodological
  contribution — corrections to the benchmark chapters are
  especially welcome

Inquiries and design-partner conversations are welcome.
