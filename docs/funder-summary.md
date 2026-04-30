# Funder Summary

## What This Project Is

This project proposes a public dynamic microsimulation stack for Social
Security analysis.

More precisely, it proposes:

- making `microplex` longitudinal
- using that longitudinal population asset to build a Social Security
  analysis layer with PolicyEngine-US
- publishing the methods, validation artifacts, and product boundaries
  openly

The point is not to claim immediate parity with SSA, CBO, or Urban
Institute models. The point is to build the most transparent public
stack in this space and to validate it rigorously enough that it becomes
useful for research, advocacy, exploratory policy design, and public
education.

## Why This Matters

Social Security policy analysis is dominated by models that are:

- internal to government
- tied to restricted administrative data
- available only through institutional relationships or contracts

That creates a basic public-infrastructure gap.

Researchers, smaller policy organizations, journalists, and advocates
can discuss reform packages, but they often cannot inspect or reproduce
the assumptions behind the models that shape those debates. That makes
the policy ecosystem more dependent on institutional trust than it
should be for a program of this scale.

This project is a response to that gap.

## Why This Is Hard

The hard part is not coding the Social Security benefit formula.

The hard part is building a credible longitudinal population that
supports:

- lifetime earnings histories
- insured-status determination
- claim-age and disability pathways
- spouse, survivor, and divorced-spouse outcomes
- forward projections that do not drift off course

That is why the project is best understood as a longitudinal extension
of `microplex`, not as a small Social Security app.

## Why This Team Can Plausibly Do It

The proposal is credible because it does not start from zero.

The relevant ingredients already exist:

- **PolicyEngine-US** already contains a strong open rules layer for tax
  and benefit policy
- **Enhanced CPS** showed that public-data reconstruction and heavy
  calibration can produce useful cross-sectional policy infrastructure
- **microplex** already reframes that work as a public synthetic
  population platform rather than a one-off dataset

The next step is longitudinal.

That is still a major research and engineering effort. But it is a much
more plausible effort than trying to build a public dynamic model from a
cold start.

## What Is Actually New Here

The novelty is not just "open source."

The distinctive contribution is the combination of:

- a public synthetic population platform
- an open rule engine
- explicit benchmark comparison to DYNASIM, MINT, CBO, and Morningstar
- a published validation record at the intermediate-state level
- a path to public productization if the model survives its stage gates

That bundle does not currently exist in the U.S. Social Security policy
ecosystem.

## What the Benchmark Models Tell Us

The project is strongest when it is honest about what the benchmark
models do better today.

- **DYNASIM** is the main benchmark for breadth, maturity, and state
  richness.
- **MINT** is the main benchmark for the importance of earnings-history
  credibility and administrative-data access.
- **CBO** is the main benchmark for official projection authority and
  macro-fiscal integration.
- **Morningstar** is the most relevant adjacent benchmark for
  retirement-adequacy and LTSS-oriented household modeling.

Those models are not reasons to avoid this project. They are reasons to
scope it correctly.

The right claim is not "we will beat all of them quickly." The right
claim is "we can build the most transparent public alternative, and we
know exactly which components have to earn credibility first."

## Why We Expect Real Demand

One useful proxy is the difference between tax-policy modeling supply and
Social Security modeling supply.

On the tax side, there are multiple visible modeling channels that
regularly support outside-facing analysis:

- **Tax-Calculator** is an open-source federal tax microsimulation model
  [@taxcalc2026]
- **TaxBrain** provides a public interface to open-source tax models
  [@taxbrain2025]
- **The Tax Policy Center** uses its microsimulation model to analyze
  major individual income tax bills and proposals
  [@tpcmodelfaq2025]
- **ITEP** states that its federal, state, and local distributional
  analyses and revenue estimates are based on its microsimulation tax
  model [@itepmodel2025]

On the Social Security side, the benchmark models are real but largely
closed:

- DYNASIM
- MINT
- CBO / CBOLT
- Morningstar

Those model families are important, but they do not constitute a broad
public self-service Social Security modeling ecosystem
[@favreault2015; @urban2024dynasim4; @ssa2024mint; @cbo2018; @cbo2024longterm; @look2024retirementoutcomes].

So a fair summary is:

- tax policy has at least **four** visible modeling channels supporting
  outside-facing analysis
- Social Security has **zero** comparably broad public self-service
  dynamic microsimulation platforms

That does not prove that one specific closed model is the entire
bottleneck. But it is strong evidence that public modeling supply is
thin relative to policy interest.

## What the Funding Would Buy

The funding is mainly paying for model construction and validation, not
for policy rules alone.

The core work is:

1. build longitudinal `microplex`
2. select and validate the earnings-history architecture
3. build family, disability, claiming, mortality, and projection layers
4. validate the resulting panel against policy-relevant outputs
5. build forward-projection and reform workflows
6. publish the stack in a usable public form

That is why the budget logic is closer to a research-infrastructure
build than to a conventional policy memo grant.

## What Success Looks Like

Success does not require becoming the official federal baseline.

This project is already valuable if it can:

- replicate major baseline Social Security distributions credibly
- support exploratory reform analysis with transparent assumptions
- publish a reusable public longitudinal population asset
- lower the cost of serious dynamic modeling for outside researchers and
  policy organizations
- create a public benchmark for how lifecycle microsimulation should be
  validated

That is meaningful public value even short of official scoring status.

## What Early Pull Should Look Like

The proposal should not rely only on generic claims that researchers or
advocates will care. It should test that demand directly.

The most plausible early adopters are:

- policy organizations working on solvency and adequacy packages
- researchers who want transparent benchmark and validation assets
- journalists and public communicators who need inspectable
  distributional outputs

The strongest near-term demand milestones would be:

- 3-5 design partners during the setup phase
- 2-3 pilot analyses requested by outside users before full public
  launch
- external testing of validated baseline outputs before stage-4
  productization

That kind of pull would make the infrastructure case much stronger than
relying on long-run aspiration alone.

There is also a more specific signal of demand than generic interest.
Some users can already use PolicyEngine for narrow Social
Security-adjacent questions, such as taxation of benefits, but still
need closed models like DYNASIM for broader dynamic Social Security
analysis. CRFB is a good example of this pattern. That suggests the
problem is not lack of policy demand. It is lack of an open dynamic
model layer.

The same logic may be even stronger for long-term care. Brookings, NASI,
BPC, Urban, and Morningstar all have visible LTSS policy activity or
model-based analysis, but public self-service LTSS modeling
infrastructure appears thinner still
[@brookings2024homecare; @nasi2025ltssdemand; @bpc2025ltss; @favreault2020ltss; @look2025ltss; @look2025wish].
That is one reason the proposal should preserve a path from Social
Security into LTC once the longitudinal population layer is credible.

The most credible first LTC expansion is a static state pilot rather
than a full national dynamic LTSS model. A state-specific pilot could
answer whether a family is eligible now or soon, whether a spend-down or
Miller Trust path is required, how spousal protections change the
result, and what patient liability remains after approval. That is
concrete enough to create partner pull and user testing without
pretending the national dynamic LTC problem has already been solved.

## Why the Project Is Stage-Gated

The proposal uses stage gates because the biggest risk is false
confidence.

The key gates are:

- **Stage 1**:
  are the earnings histories and longitudinal population good enough?
- **Stage 2**:
  do family, disability, and benefit outputs hold up?
- **Stage 3**:
  do forward projections remain stable and interpretable?
- **Stage 4**:
  is the validated subset ready for public productization?

This is a better funding structure than pretending the whole stack is
equally low-risk from day one.

## Why the Budget Range Is Plausible

The `$1M-$1.5M` range is plausible because this is not a narrow policy
tool build.

It is paying for:

- longitudinal data construction
- multi-model evaluation and benchmarking
- family and disability state machinery
- mortality and projection control
- benchmark replication
- public-facing documentation and delivery

Most of the cost is specialized labor. The biggest line item is not
cloud compute; it is the team needed to build, evaluate, and document
the model properly.

## Why the Range Should Probably Be Framed in Tiers

The proposal is stronger if it presents two credible funding levels.

### Lean tier

At roughly `$1.0M-$1.15M`, the project can likely fund:

- the core longitudinal build
- stage-1 and stage-2 validation
- a narrower stage-3 projection effort
- a lighter public release

### Full tier

At roughly `$1.3M-$1.5M`, the project can more comfortably fund:

- stronger parallel staffing
- fuller replication and benchmark comparison
- more robust projection and reform workflows
- more polished public documentation and interface work

This makes the ask easier to evaluate because it shows what gets added
as the budget expands.

## What Reduces the Downside Risk for a Funder

This is not an all-or-nothing product bet.

Even if the full public model takes longer than hoped or narrows after a
stage gate, the funder can still get:

- a public longitudinal-population methods contribution
- benchmark evidence on competing earnings architectures
- a validated synthetic baseline panel for selected uses
- reusable improvements to `microplex` and PolicyEngine infrastructure
- a clearer public benchmark for future Social Security and retirement
  modeling work

That makes the downside much better than a typical bespoke model grant
whose value is mostly private and ephemeral.

## Why Now

This project is more fundable now than it would have been a few years
ago for three reasons.

First, PolicyEngine and Enhanced CPS have already established a credible
cross-sectional public-data workflow.

Second, `microplex` now provides the right abstraction for the next
step: a public population platform rather than a one-off input file.

Third, the benchmark landscape is clearer. We now have a better public
understanding of what DYNASIM, MINT, CBO, and Morningstar each optimize
for, which makes it easier to scope a public alternative honestly.

## Why This Should Not Be Funded as a Generic "Open Data" Project

The proposal should not be pitched as open infrastructure in the
abstract.

It should be pitched as a focused policy-infrastructure build with a
clear proving ground:

- Social Security first
- public validation first
- productization only after credibility is earned

That framing is stronger because it ties the infrastructure work to a
concrete and important policy domain.

## The Main Questions a Funder Should Ask

A serious funder should evaluate the project on a small number of
questions:

1. Does the team have the right base platform and implementation plan?
2. Are the stage gates concrete enough to prevent overclaiming?
3. Is the budget mostly buying down the hardest risks rather than
   subsidizing polish too early?
4. Would the outputs still be valuable even if the project narrows after
   stage 1 or stage 2?
5. If it works, does it create public infrastructure that others can
   build on?

The proposal is strongest when the answer to each of those is clearly
yes.

## Best-Case Outcome

In the best case, this project becomes:

- the most transparent dynamic Social Security modeling stack in the
  United States
- the first serious public longitudinal extension of `microplex`
- a durable platform for adjacent work on SSI, retirement adequacy, and
  eventually long-term care

That is a meaningful infrastructure contribution even in a landscape
that will still include stronger closed models and official baselines.

## Bottom Line

This project is worth funding if the goal is not merely to produce
another policy report, but to create durable public infrastructure for
Social Security modeling.

The right lens is:

- ambitious, but scoped
- technically difficult, but not speculative in the abstract
- risky, but stage-gated
- expensive relative to a memo, but inexpensive relative to the public
  value of a credible open model stack

That is the real investment case.
