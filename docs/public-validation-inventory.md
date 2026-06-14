# Public validation inventory

One of the strongest parts of this proposal is that the model can be
validated extensively against public sources before any restricted
administrative linkage is available. The public record is fragmented,
but it is much richer than many policy-model proposals acknowledge.
That matters for two reasons.

First, it means the project can set milestone-based validation gates
using sources that outside reviewers can inspect themselves. Second, it
reduces the risk that the project becomes blocked on restricted-data
negotiations before it can produce a credible first model.

This chapter inventories the main public or low-friction sources we can
use to validate the model. It is not an exhaustive bibliography. It is
the minimum practical source stack for judging whether longitudinal
`populace` is becoming decision-useful.

Many of these sources are already assembled inside `populace`,
PolicyEngine's microdata stack — the primary-source microdata and the
calibration targets (from CBO, IRS, SSA, Census, and others) that it
draws on. Naming the sources
explicitly here keeps the validation record legible to outside reviewers,
who can inspect the underlying public tables and microdata themselves
rather than taking the data layer on trust.

## Access tiers

The validation stack naturally breaks into four tiers.

- **Open public tables and documentation**: SSA statistical
  publications, Trustees material, Census product pages, CDC mortality
  tables, CMS public program pages, and state Medicaid documentation.
- **Public microdata**: CPS ASEC, ACS PUMS, SIPP, PSID public files,
  HRS public release, NHATS/NSOC public files, and MCBS public-use
  files [@census2025cpsasec; @census2026acs; @census2024sipp; @psid2025home; @hrs2025about; @nhats2025nhats; @cms2025mcbs].
- **Low-friction or application-based research files**: MCBS limited
  data, T-MSIS Analytic Files, and some CMS or SSA products available
  under use agreements rather than full public release
  [@cms2025mcbs; @medicaid2025taf].
- **Restricted administrative linkage**: SSA administrative microdata
  or matched survey-admin files. These could improve precision later,
  but they are not assumed in the first build.

The proposal should be explicit that stage gates are built primarily on
the first two tiers. The third tier is a bonus, not a prerequisite.

## Core source inventory

| Source | Access tier | Best use in this project | Main caveat |
| --- | --- | --- | --- |
| SSA Annual Statistical Supplement [@ssa2025supplement] | Open public tables | National benchmarks for insured status, benefit type, awards, terminations, dual entitlement, and SSI distributions | Aggregate, not person-level |
| SSA DI Annual Statistical Report [@ssa2024diannual] | Open public tables | Disability incidence, beneficiary composition, awards, application outcomes, and return-to-work patterns | Administrative program view, not a full longitudinal panel |
| SSA SSI Annual Statistical Report [@ssa2024ssiannual] | Open public tables | SSI caseload, recipient composition, payment levels, and state patterns | Aggregate, not household microdata |
| OASDI Beneficiaries by State and County [@ssa2024oasdistatecounty] | Open public tables | Geographic validation for beneficiary counts and amounts | No full person-level covariates |
| CPS ASEC [@census2025cpsasec] | Public microdata | Cross-sectional income, demographic, and Social Security income benchmarks | Retrospective income reporting, limited panel structure |
| ACS [@census2026acs] | Public microdata and tabulations | Population geography, demographic structure, and local benchmarking | Limited retirement-program detail relative to SSA files |
| SIPP [@census2024sipp] | Public longitudinal microdata | Program participation dynamics, family composition, wealth, and monthly transitions | Shorter panel horizon than a lifecycle model |
| PSID [@psid2025home] | Public longitudinal microdata | Long-run earnings, wealth, marriage, fertility, and intergenerational structure | Smaller sample, weaker direct program detail |
| HRS [@hrs2025about] | Public longitudinal microdata | Older-age work, claiming, health, and wealth dynamics | Focused on older cohorts |
| NHATS/NSOC [@nhats2025nhats] | Public or application-based microdata | Functional status, care receipt, and caregiving for LTC extensions | Older-adult focus, not all-age LTSS |
| MCBS [@cms2025mcbs] | Public-use and limited data | Medicare beneficiaries, utilization, health status, and older-age spending | Community/facility split matters for use cases |
| NVSS mortality tables [@cdc2025mortalitytables] | Open public tables | Mortality, life expectancy, and cause-of-death structure | Not a direct Social Security mortality table |
| OACT life tables and Trustees material [@ssa2025life; @ssa2025trustees] | Open public tables and reports | Projection alignment, mortality assumptions, taxable payroll, and beneficiary totals | Designed for official projection use, not open micro-simulation replication |
| T-MSIS overview and TAF documentation [@medicaid2025tmsis; @medicaid2025taf] | Open documentation plus research files | Medicaid enrollment, LTSS utilization, state variation, and later LTC validation | Data quality varies by state; research files require more setup |
| MDS 3.0 technical information [@cms2025mds] | Open technical documentation and files | Institutional care benchmarks for nursing-home populations | Not a household survey; linking to other populations is nontrivial |

## Minimum validation set by model block

### Population and household structure

The baseline population should be judged first against Census products
and then against longitudinal surveys.

- Use ACS for geography, age, sex, race, housing, and local population
  structure [@census2026acs].
- Use CPS ASEC for household income structure, filing-unit-adjacent
  relationships, and Social Security income receipt in the base year
  [@census2025cpsasec].
- Use SIPP and PSID to validate whether the synthetic population carries
  plausible family dynamics and wealth heterogeneity, not just plausible
  cross-sectional margins [@census2024sipp; @psid2025home].

### Earnings, coverage, and insured status

This is where the proposal needs the most discipline. Public validation
does not recover exact administrative earnings histories, but it can
still test the outputs that matter for Social Security use.

- Use CPS ASEC and SIPP for observed earnings distributions and zero
  earnings patterns [@census2025cpsasec; @census2024sipp].
- Use PSID for cross-age earnings persistence, volatility, and
  long-horizon rank movement [@psid2025home].
- Use the SSA Annual Statistical Supplement to validate insured-worker
  counts, beneficiary composition, awards, and the downstream results of
  covered-work histories after the rules are applied
  [@ssa2025supplement].

The proposal should continue to treat public validation of `AIME`-like
and benefit-like outputs as a must-pass requirement even before
restricted SSA linkages exist.

### Claiming, auxiliary benefits, and beneficiary status

Public SSA tables are stronger here than many teams realize.

- The Annual Statistical Supplement provides national benchmarks for
  retired workers, spouses, survivors, disabled workers, dual
  entitlement, beneficiary families, awards, terminations, and
  representative payees [@ssa2025supplement].
- OASDI Beneficiaries by State and County provides a public geographic
  cross-check once national totals look reasonable
  [@ssa2024oasdistatecounty].
- CPS ASEC, ACS, SIPP, and PSID help validate the demographic and family
  side of the history-construction problem, especially marriage,
  widowhood, age structure, and co-residence
  [@census2025cpsasec; @census2026acs; @census2024sipp; @psid2025home].

### Disability and SSI

This is one of the strongest public-validation areas because SSA
publishes a surprisingly rich disability record.

- Use the DI Annual Statistical Report for awards, application outcomes,
  diagnosis mix, benefit amounts, return-to-work patterns, and disabled
  beneficiary structure [@ssa2024diannual].
- Use the Annual Statistical Supplement for disability and SSI counts
  embedded in the broader OASDI/SSI system [@ssa2025supplement].
- Use the SSI Annual Statistical Report for recipient composition,
  payment distributions, and blindness/disability basis categories
  [@ssa2024ssiannual].
- Use SIPP for joint means-tested program participation and household
  income context around SSI and disability receipt
  [@census2024sipp].

### Mortality and projection drift

Projection validity should be judged against public official
projections, not only against internally generated smooth paths.

- Use the Trustees Report for topline beneficiary, taxable-payroll, and
  trust-fund alignment targets [@ssa2025trustees].
- Use OACT life tables for Social Security-relevant mortality structure
  [@ssa2025life].
- Use NVSS mortality tables to cross-check mortality by age and sex
  against broader public-health data [@cdc2025mortalitytables].
- Use ACS and CPS for denominator populations when validating drift in
  population structure [@census2026acs; @census2025cpsasec].

### LTC and caregiving extension

The LTC extension would also have a substantial public validation stack,
even if it is thinner and more fragmented than the Social Security one.

- Use HRS for older-age health, work, wealth, and retirement behavior
  [@hrs2025about].
- Use NHATS/NSOC for ADLs, IADLs, cognitive impairment, care setting,
  and family caregiving [@nhats2025nhats].
- Use MCBS for Medicare beneficiaries' health, utilization, and spending
  patterns over time [@cms2025mcbs].
- Use T-MSIS and TAF for Medicaid LTSS participation and state variation
  [@medicaid2025tmsis; @medicaid2025taf].
- Use MDS for nursing-home resident and facility-side functional
  benchmarks [@cms2025mds].

This still does not make LTC easy. It does mean that an LTC extension
can be validated more seriously than most high-level proposals imply.

## Recommended stage-gate use

If this project is funded in phases, the validation stack should be
matched to those phases explicitly.

| Stage | Public sources that should be sufficient to pass the stage |
| --- | --- |
| Stage 0: Base population and synthesis parity | ACS, CPS ASEC, SIPP, PSID |
| Stage 1: Earnings architecture and benefit baseline | CPS ASEC, SIPP, PSID, SSA Annual Statistical Supplement |
| Stage 2: Disability, claiming, and auxiliary-benefit realism | SSA Annual Statistical Supplement, DI Annual Statistical Report, SSI Annual Statistical Report, OASDI by State and County |
| Stage 3: Forward projection credibility | Trustees Report, OACT life tables, NVSS mortality tables, ACS geography checks |
| Optional LTC pilot | Colorado state rule packet, HRS, NHATS/NSOC, MCBS, T-MSIS/TAF, MDS |

That framing matters for funders. It makes clear that the project can be
measured against public evidence all along the way.

## What public sources still cannot solve

This appendix should not be read as claiming that public validation is
enough for every question.

- Public sources will not reproduce exact SSA administrative earnings
  histories at the person level.
- Public sources are weaker on precise application processing,
  adjudication timing, and some program-interaction edge cases.
- LTC remains harder because the public record is split across household
  surveys, facility instruments, Medicaid systems, and state manuals.

But those limitations are different from saying the model cannot be
validated. The more accurate statement is that the project can produce a
substantial public validation record before it reaches the frontier
where restricted administrative data would add the most value.
