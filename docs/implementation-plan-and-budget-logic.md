# Implementation Plan and Budget Logic

## Why This Chapter Exists

At this point, the proposal has enough technical depth to answer "what
would you build?" The remaining funding question is "what exactly would
the money buy?"

That question matters because a `$1M-$1.5M` ask will sound too large if
the work is described as:

- writing a better methods paper
- wrapping existing Social Security rules in a thin interface
- extending Enhanced CPS with a small amount of additional imputation

It will sound much more realistic if the work is described correctly:

- making `populace` longitudinal
- validating that longitudinal population against panel and
  Social-Security-specific targets
- building the family, disability, claiming, and benefit logic needed
  for serious policy analysis
- publishing the validation record and a usable interface on top of it

This chapter therefore turns the roadmap into explicit work packages,
staffing assumptions, and budget logic.

## What Funders Would Actually Be Paying For

The funding is not mainly for policy rules. PolicyEngine-US already
provides a large share of the rules infrastructure.

The funding is mainly for six hard things:

1. **longitudinal population construction**
2. **history reconstruction and evaluation**
3. **family, disability, mortality, and claiming state construction**
4. **Social Security-specific validation and benchmark replication**
5. **projection, alignment, and reform workflows**
6. **publication-quality documentation and public delivery**

That is a model-building program, not a documentation refresh.

## Work Package Structure

The cleanest way to present the project is as a set of linked work
packages that map onto the stage gates.

### WP0. Project Setup, Governance, and Baselines

**Purpose**:
lock project scope, staffing, datasets, and stage-gate criteria before
the main build begins.

**Main tasks**:

- finalize source-data inventory and benchmark tables
- define shared evaluation harness and run registry
- settle repository boundaries between `populace` and this repository
- define must-pass metrics for stage 1 and stage 2
- establish advisor and reviewer cadence
- recruit an initial set of external design partners and identify the
  first pilot analyses they would actually use

**Primary outputs**:

- project charter
- benchmark inventory
- evaluation harness skeleton
- written stage-gate memo
- design-partner list and first-use-case memo

**Why it matters**:
without this package, later work becomes hard to evaluate and easy to
oversell.

### WP1. Longitudinal `populace` Base Construction

**Purpose**:
create the first credible longitudinal version of `populace` for
Social Security use.

**Main tasks**:

- harmonize PSID and other longitudinal inputs
- define the person-year state structure
- build baseline history-reconstruction pipelines
- add cohort and age conditioning layers
- integrate calibration hooks and provenance logging

**Primary outputs**:

- longitudinal `populace` alpha
- reproducible data pipeline
- benchmark-ready training and validation datasets

**Why it matters**:
this is the core population-asset work that should outlive the Social
Security application.

### WP2. Earnings Architecture Selection and Validation

**Purpose**:
choose the production earnings-history architecture using
Social-Security-specific metrics rather than generic imputation scores
alone.

**Main tasks**:

- run comparator families such as QRF, ZI-QRF, ZI-QDNN, and pathwise
  `populace` models
- evaluate zero patterns, persistence, mobility, highest-35 sums, AIME,
  and insured status
- run pseudo-cross-sectional holdout tests
- document failure cases and uncertainty

**Primary outputs**:

- model-selection report
- production recommendation memo
- versioned benchmark artifacts

**Why it matters**:
this is where the project earns the right to move from an interesting
synthetic-data effort to a real Social Security input file.

### WP3. Family, Disability, Mortality, and Claiming States

**Purpose**:
turn longitudinal `populace` into a benefit-relevant population rather
than just an earnings panel.

**Main tasks**:

- construct marriage, widowhood, divorce, remarriage, and spouse-link
  histories
- build disability onset and simplified program-pathway states
- add mortality processes and projection controls
- implement claim-age and benefit-entry states
- enforce relational consistency across households and pairs

**Primary outputs**:

- relationship-history layer
- disability and claiming layer
- mortality and benefit-timing layer
- validation tables for family and claimant pathways

**Why it matters**:
without this package, the model may match retired-worker outputs while
failing on widow, spouse, and disability pathways.

### WP4. PolicyEngine-US Integration and Baseline Replication

**Purpose**:
convert the synthetic panel into a real policy-analysis stack.

**Main tasks**:

- map panel outputs into PolicyEngine-US entities and variables
- validate AIME, PIA, beneficiary counts, and benefit amounts
- replicate selected baseline tables from SSA, CBO, DYNASIM, or other
  public references where feasible
- define reliable product boundaries for current-law analysis

**Primary outputs**:

- benefit-ready pipeline
- baseline replication packet
- stage-2 validation memo

**Why it matters**:
this package converts a population asset into something policy users can
actually evaluate.

### WP5. Forward Projection, Alignment, and Reform Workflows

**Purpose**:
move from a validated baseline panel to a usable dynamic reform model.

**Main tasks**:

- add forward aging, cohort entry, and drift control
- align to external assumptions where appropriate
- test baseline projection stability
- build a defined set of reform workflows
- quantify uncertainty and sensitivity for core outputs

**Primary outputs**:

- forward projection pipeline
- projection validation report
- reform-analysis workflows

**Why it matters**:
this is the package that makes the model decision-useful for medium- and
long-run reform analysis.

### WP6. Public Delivery, Documentation, and Productization

**Purpose**:
publish the validated parts of the stack in a way that outside users can
understand and trust.

**Main tasks**:

- package APIs and limited public endpoints
- build a focused interface around validated use cases
- publish tutorials, validation notes, and benchmark comparisons
- write plain-language product boundaries and limitations
- onboard early external pilot users and incorporate feedback on the
  public-facing product layer

**Primary outputs**:

- public API
- public interface
- methods and validation documentation
- public examples and tutorials

**Why it matters**:
without this package, the project remains an internal research asset
rather than a public infrastructure contribution.

### Optional WP7. State LTC Pilot

**Purpose**:
show how the same infrastructure can support a high-value adjacent
domain without pretending that the project has already built a full
national dynamic LTC model.

[`appendix-colorado-ltc-rules-packet.md`](appendix-colorado-ltc-rules-packet.md)
describes the official-source packet that would anchor this work package.

**Main tasks**:

- encode one state's LTSS rules in a production-quality way
- build countable income and asset logic, including exempt-assets logic
- implement spousal impoverishment protections, patient liability, and
  trust or transfer-penalty pathways
- add benefit-valuation and scenario-comparison outputs
- validate against state manuals, public benchmarks, and expert review

**Primary outputs**:

- state LTSS rules layer
- "eligible now or soon?" scenario engine
- spend-down and patient-liability comparison outputs
- state pilot validation memo

**Why it matters**:
this turns LTC from an abstract future possibility into a concrete
adjacent pilot. It also makes clear why even a static LTC extension
requires real policy-engineering and validation capacity rather than a
small amount of spare effort.

## Demand-Side Work Is Also Real Work

The proposal should acknowledge that external uptake will not happen by
accident.

Early demand-side work should include:

- identifying 3-5 likely design partners during setup
- choosing 2-3 concrete pilot analyses with external relevance
- testing whether the earliest validated outputs are actually legible to
  outside users
- using partner feedback to narrow the first public product surface

If the project pursues a state LTC pilot later, this same demand-side
logic becomes even more important. The right early test is whether
families, navigators, legal-aid partners, or aging-policy organizations
actually find outputs like "eligible now or soon?", "Miller Trust
needed", or "likely patient liability" legible and decision-useful.

This is still a technical project, not a partnerships-heavy policy
campaign. But a credible public-infrastructure build should reserve
explicit time for user discovery and pilot use, not assume adoption will
appear automatically after launch.

## Mapping Work Packages to Stages

The work packages and roadmap stages are not identical, but they align
closely.

| Roadmap stage | Main work packages |
|---|---|
| **Stage 0** | WP0 |
| **Stage 1** | WP1 + WP2 |
| **Stage 2** | WP3 + WP4 |
| **Stage 3** | WP5 |
| **Stage 4** | WP6 |

That mapping is useful in funding conversations because it makes the
go/no-go logic legible. If stage 1 underperforms, the project still
produces public technical value through WP1 and WP2 rather than simply
failing silently.

## Recommended Staffing Model

The proposal should distinguish between:

- current leadership and advisory capacity
- the funded execution team required to deliver the work packages

### Core funded team

The recommended execution team is:

1. **Technical lead / research engineer**
   owns the longitudinal `populace` implementation, evaluation harness,
   and reproducibility pipeline
2. **Research economist / quantitative social scientist**
   owns benchmark design, policy interpretation, and stage-gate
   standards
3. **Data engineer / data scientist**
   owns ingestion, harmonization, target assembly, and production-grade
   data workflows
4. **Research assistant support**
   owns benchmark tables, literature synthesis, rule documentation, and
   validation-table assembly

### Leadership and review layer

In addition to the funded team, the proposal should assume:

- project leads providing architectural direction and go/no-go
  decisions
- advisors providing domain review
- external reviewers at major stage gates

This is the minimum structure that makes a 36-month plan credible.

## FTE Logic

The proposal does not need to commit to exact titles or hires today, but
it should show realistic labor intensity.

### Lean but credible build

A lean build still requires roughly:

- `1.0` FTE technical lead over 30-36 months
- `0.75-1.0` FTE research economist over 24-30 months
- `0.5-1.0` FTE data engineer or data scientist over 18-24 months
- `0.5` FTE research assistance over 18-24 months
- fractional leadership and advisor time throughout

This is enough to run the model-building program, but it leaves limited
slack for parallelization and may slow productization.

### More resilient build

A more resilient build would support:

- `1.0` FTE technical lead over the full project
- `1.0` FTE research economist over the full project
- `1.0` FTE data engineer or data scientist for the heavy data and
  validation phases
- `0.5-1.0` FTE research assistance
- `0.25-0.5` FTE frontend or product engineering late in the project
- explicit time for partner development and external pilot feedback
- fractional leadership and external review throughout

This version is much better aligned with the proposal's ambition and
reduces the risk that interface or replication work cannibalizes core
validation.

## Budget Logic

The proposal should frame the budget in categories rather than pretend
that exact hiring costs are settled today.

### Personnel should dominate

For a project like this, the budget should be mostly people.

A reasonable high-level budget structure is:

- **personnel and contractor time**:
  roughly 70-80 percent
- **compute, storage, and engineering tooling**:
  roughly 5-10 percent
- **external review, travel, workshops, and expert consultation**:
  roughly 5-10 percent
- **publication, product design, and interface delivery**:
  roughly 5-10 percent

If the proposal includes institutional overhead or indirect costs, those
should be broken out clearly rather than hidden inside vague line items.

### Why the ask is not smaller

The funding case should say plainly why this is not a `$200K-$400K`
project:

- the central difficulty is building and validating a longitudinal
  population asset
- multiple state blocks have to be built, not just earnings
- benchmark replication and validation are labor-intensive
- the proposal includes publication-quality public delivery, not just an
  internal prototype

That is the difference between a proof of concept and a public model
program.

## Suggested Funding Tiers

The proposal should likely present two credible funding tiers instead of
pretending there is one magic number.

### Tier A: Lean stage-gated build

**Indicative budget**:
approximately `$1.0M-$1.15M`

**What it funds**:

- core execution team with limited parallel slack
- stage 1 and stage 2 fully
- stage 3 selectively
- lighter public productization

**What it is best for**:

- a serious methods-and-validation build
- a smaller set of reform workflows
- a narrower public release

### Tier B: Full validation and delivery build

**Indicative budget**:
approximately `$1.3M-$1.5M`

**What it funds**:

- stronger parallel staffing
- fuller benchmark replication
- more robust stage-3 projection work
- better documentation and public interface delivery
- more structured external review

**What it is best for**:

- the full story this proposal is really aiming at
- a cleaner handoff from research stack to usable public product

This two-tier framing is useful because it avoids making the ask look
arbitrary. It shows what is added when the budget expands.

## Budget by Stage

Funders may also want to know when the spend happens.

### Early spend: stages 0 and 1

The early phases are labor-heavy because they require:

- data assembly
- evaluation harness setup
- longitudinal history construction
- model-family benchmarking

This is where the project either earns continuation or does not.

### Middle spend: stage 2

The middle phase is where complexity expands fastest. Family history,
disability, claiming, and benefit integration are all expensive in
research time because they combine modeling work with validation work.

### Late spend: stages 3 and 4

The late phases shift more spend toward:

- projection and alignment
- reform workflows
- publication and interface work
- user-facing packaging

This is another reason a stage-gated framing helps: productization spend
should be conditional on earlier validation success.

## Why Stage-Gated Funding Helps

The proposal should make clear that the funding structure itself can
reduce risk.

A stage-gated approach allows:

- early investment in the hardest validation problems
- explicit continuation criteria
- useful fallback outputs if a gate fails
- smaller downside risk for the funder than a monolithic all-or-nothing
  build

That is a better story than pretending confidence is already high enough
to justify full productization from day one.

## What a Funder Gets Even if the Full Build Slips

This chapter should also strengthen the downside case.

Even if the full public model is not complete on the original schedule,
the funder can still receive:

- a public longitudinal-population methods contribution
- benchmark evaluations of candidate earnings architectures
- a validated baseline panel for selected Social Security uses
- a benchmark comparison asset for future funders and collaborators
- reusable improvements to `populace` and PolicyEngine infrastructure

That makes the project more legible as research infrastructure, not just
as a risky all-or-nothing product launch.

## Recommended Proposal Language

The cleanest way to pitch the budget is:

1. this is a stage-gated infrastructure and validation build
2. the core cost is specialized labor, not compute
3. the larger ask buys parallelization, stronger validation, and public
   delivery
4. the outputs remain useful even if the project is narrowed at a stage
   gate

That is a much stronger frame than simply saying the work is complex.

## Bottom Line

The `$1M-$1.5M` range is plausible if the proposal is honest about what
is being funded.

It is too large for a thin Social Security app.
It is reasonable for:

- making `populace` longitudinal
- validating the resulting population asset
- building the Social Security-specific family, disability, claiming,
  and benefit layers
- publishing a usable public model interface on top of that stack

That is the right budget logic for this project.
