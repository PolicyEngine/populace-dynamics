# Appendix: public dossier on DYNASIM

## Purpose

The main comparison chapter treats DYNASIM strategically. This appendix
asks a narrower question: what can an outside researcher actually know
about DYNASIM from public sources alone?

The answer is: quite a lot about scope, inputs, modeled processes, and
policy use cases; much less about the current codebase, full equation
inventory, and day-to-day operating workflow. That distinction matters.
For this repository, DYNASIM is not just a competitor model. It is the
most important benchmark for how much detail a non-governmental dynamic
microsimulation platform can accumulate over time.

## 1. What the public record establishes

| Dimension | What public sources show |
|---|---|
| **Institutional home** | DYNASIM is maintained at the Urban Institute and has been developed there for decades [@favreault2004dynasim3; @favreault2015]. |
| **Current public version** | Urban's current public fact sheet refers to **DYNASIM4** [@urban2024dynasim4]. |
| **Time horizon** | Public materials describe DYNASIM as producing 75-year projections for older adults' economic and health outcomes [@favreault2015; @urban2024dynasim4]. |
| **Starting sample** | Public documentation says the basic DYNASIM4 starting sample is **0.04 percent** of the US population, with an expanded **0.4 percent** version for some work [@urban2024dynasim4]. |
| **Broad scope** | DYNASIM4 projects demographics, work, Social Security, pensions, income, wealth, taxes, health, disability, LTSS, and several other public programs [@favreault2015; @urban2024dynasim4]. |
| **Rule-based components** | Public sources describe sophisticated rule-based calculators for OASDI and SSI, while also noting that some other government benefits are simulated statistically rather than through full eligibility rules [@favreault2015; @urban2024dynasim4]. |
| **Alignment** | Public materials explicitly say some major outcomes are aligned to external targets produced by SSA's Office of the Chief Actuary [@urban2024dynasim4]. |
| **Public access model** | Urban publishes documentation, briefs, and some downloadable tabulations, but not an open-source codebase or a fully self-service public model [@favreault2004dynasim3; @favreault2015; @urban2024dynasim4]. |

## 2. Lineage and versioning

Urban's public primer states that DYNASIM was originally developed in
the 1970s, that DYNASIM2 was built in the early 1980s, and that
DYNASIM3 was a major redevelopment with a newer starting sample,
expanded saving and pension modules, and updated Social Security and SSI
calculators [@favreault2004dynasim3].

The later overview and fact sheet make clear that DYNASIM4 is not a
small patch on top of DYNASIM3. It uses a newer SIPP base and a much
broader health, health spending, and LTSS apparatus [@favreault2015; @urban2024dynasim4].
Publicly, that matters for two reasons:

- DYNASIM has accumulated institutional memory over multiple
  generations rather than being assembled in one grant cycle.
- The public paper trail is versioned and uneven. Older generations are
  often documented in more detail than the current operational model.

## 3. Starting sample, projection horizon, and scale

The 2015 overview notes that DYNASIM4 would be based on the 2004 and
2008 SIPP panels and would start projecting outcomes in 2006
[@favreault2015]. Urban's 2024 fact sheet adds two practical
details that are especially useful for outside readers:

- the basic starting sample is 0.04 percent of the population
- an expanded version uses a 0.4 percent starting sample

The 2015 overview also notes a larger starting sample of 1.056 million
people in 461,000 families, typically processed only through 2040 rather
than through the full long horizon [@favreault2015]. Taken
together, these sources imply that Urban varies the effective scale of
the model depending on the task, but the public record does not fully
document which configuration is standard for which class of analysis.

Urban also publishes a small amount of direct model output, including a
[DYNASIM4 projections-by-birth-cohort page](https://www.urban.org/dynasim4-projections-birth-cohort)
with downloadable tables. That is helpful, but it is still a long way
from a public model release.

## 4. Weights and representation

The public DYNASIM materials describe starting-sample scale and
alignment to external controls, but they do not describe a system of
annual independent person weights that can be recalibrated after family
events occur [@urban2024dynasim4]. That distinction matters for
this project. Once marriage, divorce, fertility, leaving home, and
household splitting are simulated, arbitrary person-level reweighting
would make the relationship network incoherent.

The right design is to treat representation as part of the simulated
population object. Base-year weights can be absorbed into the starting
population, converted into stable replicate counts, or carried as
network-consistent representation factors. Dynamic alignment should then
operate through event selection, process parameters, and
network-preserving resampling rather than by independently changing the
importance of linked people. This is also the methodological warning in
the dynamic-ageing microsimulation literature on weights
[@dekkers2012weights].

## 5. Software structure and processing sequence

The 2015 overview is unusually useful because it exposes part of the
model's processing architecture. Figure 1 describes a **core FORTRAN
model** that loops annually, along with a **SAS postprocessor** that
handles much of the benefit-calculation logic [@favreault2015].

Public documentation also makes clear that DYNASIM is not just one
monolithic equation block. It is a staged simulation with modules that
feed one another. The public processing sequence includes:

- demographic updates such as birth, death, migration, marriage,
  divorce, schooling, and leaving home
- labor-market outcomes such as employment, hours, wages, job change,
  pension coverage, and retirement
- health and disability states including general health, ADLs, IADLs,
  chronic conditions, cognitive impairment, and work limitations
- program calculators for Social Security, SSI, disability insurance,
  taxes, Medicaid, and other payers
- wealth, pension, and insurance updates that feed back into later
  eligibility and spending outcomes [@favreault2015]

For an outside researcher, this is enough to understand DYNASIM's
conceptual architecture, but not enough to recreate the current model
faithfully.

## 6. Data sources visible in public documentation

Urban's public materials are also fairly informative about the data
sources behind DYNASIM. The 2024 fact sheet says DYNASIM4 uses models
estimated on datasets such as the NLSY, PSID, HRS, and SCF, as well as
SIPP linked to administrative data [@urban2024dynasim4].

The longer overview describes those sources in more operational detail.
Publicly documented inputs include:

- **SIPP** as the core starting population for DYNASIM generations
  [@favreault2004dynasim3; @favreault2015]
- **PSID**, older CPS-based matched files, and other sources for
  reconstructing lifetime earnings histories in earlier DYNASIM
  generations [@favreault2004dynasim3]
- **HRS** for late-life health, disability, cognition, and LTSS use
  transitions [@favreault2015]
- **NHATS** for annualizing paid home-care duration in the LTSS module
  [@favreault2015]
- **SCF** and pension-related sources for wealth and retirement-account
  dynamics [@urban2024dynasim4]
- **Genworth** and Medicaid-rate literature for LTSS prices and payer
  costing [@favreault2015]

This matters for transparency. DYNASIM is not based only on public
microdata. Public documentation itself says some model components are
estimated using survey files linked to administrative records
[@urban2024dynasim4]. That is one reason an open-source project
should not assume it can reproduce DYNASIM simply by copying the public
descriptions.

## 7. What DYNASIM models

Urban's public materials make DYNASIM's scope unusually visible.
Between the overview and the 2024 fact sheet, the following categories
are publicly documented:

- demographics and family structure
- education
- employment, hours, wages, sector, and job changes
- pension coverage and benefit accrual
- Social Security, SSI, and disability insurance
- health insurance coverage
- medical spending
- LTSS use, costs, and payer allocation
- taxes
- wealth, including home equity, financial wealth, pension wealth, and
  Social Security wealth [@favreault2015; @urban2024dynasim4]

Two design details are especially important for interpreting DYNASIM's
policy uses:

1. Public documentation describes Social Security and SSI as strongly
   rule-based components.
2. The 2024 fact sheet says the simulation of other government benefits
   is based solely on statistical modeling rather than explicit
   eligibility rules [@urban2024dynasim4].

That means DYNASIM is not best understood as a universal rules engine.
It is a dynamic microsimulation model with some highly detailed
calculators and some more reduced-form program modules.

## 8. Alignment and calibration

Public documentation repeatedly emphasizes that DYNASIM is not a purely
free-running simulation. The 2024 fact sheet explicitly says that some
outcomes such as fertility, mortality, disability, immigration, and
labor force participation are aligned to targets produced by SSA's
Office of the Chief Actuary [@urban2024dynasim4].

That point should shape how this repository talks about DYNASIM.
Urban's benchmark model is not valuable because it avoids alignment. It
is valuable because it combines rich micro-level state transitions with
external anchoring to trusted aggregate forecasts. Any competing public
model will need to be equally explicit about where it aligns and where
it lets the microsimulation run on its own.

## 9. LTSS-specific machinery

For long-term care modeling, the public record is strong enough to be
substantive. The 2015 overview describes a real LTSS module, not a
placeholder [@favreault2015].

### Care settings and transition logic

Public documentation says DYNASIM models:

- nursing home use
- residential care or assisted living
- paid home care

Those equations are estimated on pooled HRS data for respondents ages
65 and older. Predictors include ADL and IADL limitations, self-reported
health, marital status, spouse disability, race and Hispanic origin,
number of children, age, sex, nativity, income, wealth, and prior care
use. Any nursing home care, residential care, and paid home care are
jointly estimated as a trivariate probit with persistence built in
through lags and correlated errors [@favreault2015].

### Intensity and duration

Public documentation also says DYNASIM projects:

- number of nursing home nights
- duration of paid home care
- hours of paid home care

The overview describes zero-truncated negative binomial models for
nursing home nights and home-care hours, plus NHATS-based adjustments to
convert HRS monthly home-care measures into annual quantities
[@favreault2015].

### Prices, payers, and Medicaid

The public LTSS documentation is also unusually concrete about costing:

- private-pay LTSS prices use state-specific Genworth data
- Medicaid LTSS prices rely on published reimbursement summaries
- future LTSS prices are wage indexed
- payer allocation distinguishes out-of-pocket spending, Medicare,
  Medicaid, insurers, and uncompensated or other public care
  [@favreault2015]

The overview further states that DYNASIM reflects a composite set of
state-specific Medicaid eligibility rules, compares those rules against
income and assets, and uses a relatively simple spenddown equation to
link LTSS expenses to Medicaid entry [@favreault2015].

This is enough to say that DYNASIM already covers several pieces that an
LTC-focused public model would need:

- care need and care setting
- service intensity
- payer assignment
- Medicaid interaction
- private long-term care insurance

But the public record also reveals some limitations:

- the 2015 LTSS description focuses on ages 65 and older
- home-care measurement requires an annualization workaround because HRS
  observes paid care only in the prior month
- wealth spenddown is described publicly as relatively simple, not as a
  highly detailed state-specific depletion model [@favreault2015]

## 10. Demonstrated policy uses

DYNASIM's importance is not only methodological. Urban has used it in
published policy analysis across multiple domains. Public sources show
applications in:

- retirement-income and cohort analysis
- Social Security reform and distributional analysis
- racial disparities in retirement outcomes
- LTSS financing proposals
- caregiving and labor-supply consequences [@favreault2015; @favreault2020ltss; @johnson2023caregiving]

This matters because it distinguishes DYNASIM from a merely notional
research model. It is a live production research platform with a long
publication trail.

## 11. Operational clues from public materials

Urban's 2024 fact sheet gives a rare public glimpse into how DYNASIM is
used in practice. It distinguishes between:

- a **baseline** under current law and forecast assumptions
- a **counterfactual** under alternative policy or behavioral
  assumptions

It also says that some baseline-only studies can be completed in days,
whereas new policy scenarios can require weeks of involvement from the
DYNASIM team depending on the complexity of the provisions
[@urban2024dynasim4].

That is an important clue. Publicly, DYNASIM looks less like a
downloadable package and more like an internal expert-operated research
platform.

## 12. What remains opaque

Even after reading the public record closely, several important things
remain unknown or only partially known:

- the full current source code
- the exact version history for each live module
- the current parameter values and full equation inventory
- the testing and quality-assurance workflow
- the exact operational representation of state Medicaid rules
- how run-specific choices differ across Urban publications
- which components rely on restricted linked data in ways an outside
  team cannot reproduce exactly

The DYNASIM3 primer explicitly said that more detailed documentation was
available on request from the authors rather than fully public
[@favreault2004dynasim3]. That is useful context. DYNASIM has
been publicly described for years, but it has not been publicly exposed
as an inspectable end-to-end software artifact.

## 13. Implications for this repository

The public record suggests five practical conclusions for this project:

1. DYNASIM is a serious benchmark for breadth. Any proposal that talks
   about retirement, disability, wealth, or LTC as though this terrain
   is empty will look underinformed.
2. DYNASIM is not an open reference implementation. The gap is not just
   "another model"; it is transparency, inspectability, and the ability
   for outside researchers to reproduce the full pipeline.
3. For Social Security, the real comparison point is not only benefit
   calculation. It is lifecycle data construction, alignment, and
   validation discipline.
4. For LTC, DYNASIM already appears to cover much of the essential
   national-model machinery. A public entrant should differentiate on
   reproducibility, modular policy rules, and better visibility into
   intermediate validation targets.
5. The right ambition is not to dismiss DYNASIM. It is to document, as
   precisely as possible, what DYNASIM already does and then build the
   parts that remain inaccessible to the field.

## Bottom Line

Publicly, DYNASIM is knowable enough to take seriously. Urban has
disclosed its broad architecture, major data sources, alignment logic,
LTSS machinery, and several classes of policy application. What remains
inaccessible is exactly what matters for true reproducibility: the live
code, the full equation system, and the operational workflow.

That is why this repository should treat DYNASIM as both a benchmark
and a boundary marker. It shows how far a non-governmental model can go.
It also shows how much remains unavailable without an explicitly open
alternative.
