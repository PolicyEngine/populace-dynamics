# Technical Specifications

This chapter provides technical specifications for longitudinal
`microplex` and for the Social Security application layer that will sit
on top of it. The relevant build is not a standalone Social
Security-only dataset. It is a reusable longitudinal population asset
plus a policy-analysis layer.

## Required Variables and Transition Equations

Dynamic microsimulation models begin with a longitudinal sample of the
population and age that sample forward through time using stochastic
demographic and economic transition equations. In this project, that
sample should be delivered as longitudinal `microplex`. The specific
variables and equations required depend on the modeling goals. For
Social Security analysis, we need inputs to both payroll tax and benefit
calculations.

### Core Demographic Variables

The demographic inputs directly relevant to Social Security include birth year, which determines retirement eligibility and benefit calculation periods, marital status for spousal and survivor benefit calculations, and family linkages connecting individuals to spouse, children, and parent records in the data structure. These relationships enable modeling of dependent and survivor benefits, which constitute a substantial share of Social Security expenditures.

### Economic Variables

The primary economic input is annual earnings, which determines both payroll tax liability and future benefit levels through the Average Indexed Monthly Earnings (AIME) calculation. Modeling earnings requires capturing labor force participation decisions, educational attainment (which strongly predicts lifetime earnings trajectories), and hours worked (which along with wages determines total earnings). These employment outcomes depend on additional factors including sex, health status (relevant for disability insurance and early claiming decisions), and fertility (particularly relevant for women's labor force participation patterns).

### Complete Variable List

A basic Social Security model thus involves simulating year-by-year and person-by-person:
- Birth and mortality
- Educational attainment
- Marital transitions (marriage, divorce, widowhood)
- Mate-matching (assortative mating patterns)
- Fertility outcomes
- Health status and disability onset
- Labor force participation
- Hours worked
- Earnings (combining participation, hours, and wages)
- Benefit claiming decisions

## Constructing Longitudinal microplex: Historical Simulation

The input dataset for dynamic microsimulation requires longitudinal
histories for all listed variables in a representative sample as of the
base year. Creating and validating that dataset is the central task in
making `microplex` longitudinal. It proceeds through historical
simulation, which is effectively a synthetic data generation exercise
using cross-sectional and longitudinal input data, including
parameterized stochastic earnings shocks and other transition
equations.

### The Historical Simulation Process

Historical simulation works backward from the present to construct plausible lifetime histories. For a 65-year-old in the base year 2024, we need earnings from ages 18-64 (1977-2023). Rather than mechanically imputing these values, we simulate backward using transition equations estimated from longitudinal data sources like PSID. This approach ensures that simulated histories respect observed transition probabilities and correlations across variables.

### Validation Through Historical Matching

Validating the synthetic input file requires matching known cross-sectional and longitudinal moments for the input variables. Cross-sectional validation checks whether the base year distributions match observed data for all key variables. Longitudinal validation examines whether transition rates (job-to-job mobility, marriage rates, etc.) match those observed in panel data. The critical test compares model predictions for outcomes of interest, specifically payroll taxes collected, number of beneficiaries by type, and benefit amounts distributed, to historical outcomes from SSA administrative data.

This validation process differs from simply matching target moments through calibration. Historical simulation validates that the underlying data generation process produces realistic outcomes, not just that we have calibrated weights to hit specific targets. This distinction matters because projections rely on the transition equations continuing to generate plausible outcomes as the population ages forward.

## Projections and the Jump-Off Problem

The same basic machinery used to produce and validate the longitudinal input file is then used to simulate forward through time. All projection models, both micro and macro, face the well-known "jump-off" problem. Earnings processes and transition equations that fit on average during the historical period provide an imperfect prediction when simulating forward from the base year. Structural changes in the economy, demographic shifts, policy changes, and other factors cause projected outcomes to drift from realized aggregates.

### Addressing Projection Drift

Projections generally require adding calibration factors to correct for the jump-off problem. These factors adjust transition probabilities or outcome distributions to ensure that projected aggregates match external benchmarks, such as SSA Trustees' intermediate assumptions for aggregate earnings, labor force participation, mortality rates, and disability incidence. This calibration is key to generating ergodic (non-degenerative) distributions for outcomes of interest and preventing unrealistic drift in long-run projections.

The calibration factors should be applied transparently, with clear documentation of which projections are purely model-driven versus calibrated to external assumptions. This transparency allows users to understand the degree to which results depend on model dynamics versus imposed alignment to official projections.

## Behavioral Responses

Dynamic microsimulation does not generally involve textbook optimizing economic behavior of the sort found in structural models. However, even the simplest models require behavioral responses to ensure realistic policy analysis.

### Essential Behavioral Components

In Social Security modeling, several behavioral responses are essential. Labor force participation and hours worked should vary with wages, reflecting standard labor supply elasticities. Without this response, simulated behavior would be unresponsive to changes in wage rates or tax treatment of earnings, producing unrealistic policy impacts. Benefit claiming decisions should vary with benefit levels across potential retirement ages. Individuals face trade-offs between claiming benefits early with actuarial reductions versus delaying for actuarial increases, and claiming patterns should respond to changes in these incentives.

### Implementation Approach

These behavioral responses can be implemented through simple elasticities rather than full structural optimization. For example, labor supply can respond to net wages using empirically estimated elasticities from the literature, and benefit claiming can follow empirical hazard models that incorporate benefit levels as explanatory variables. This reduced-form approach captures first-order behavioral effects while avoiding the computational complexity and strong assumptions of structural models.

Failure to include these responses means behavior will be unresponsive to benefit formulas or the macroeconomy, severely limiting the model's usefulness for policy analysis. Even simple elasticities substantially improve realism compared to purely mechanical projections.

## Incremental Capabilities and Extensions

The basic Social Security model described above can be extended incrementally to analyze additional programs and interactions.

### Medicare Integration

Adding capabilities for Medicare program outcomes is straightforward because no variables beyond lifetime earnings and health are required inputs. Medicare eligibility depends on age (65+) and disability status (after 24 months of SSDI receipt), both of which are already in the base model. Parts B and D premiums depend on income (determined by MAGI), while out-of-pocket spending can be modeled based on age and health status.

### Medicaid and Long-Term Care

Medicaid and Long-Term Care analysis requires asset testing, introducing additional complexity. This necessitates either imputation of assets based on lifetime earnings and marital history using statistical relationships from the Survey of Consumer Finances and Health and Retirement Study, or development of a saving and wealth accumulation module that explicitly models asset accumulation over the lifecycle. The former approach is simpler to implement initially, while the latter provides richer behavioral content.

Treating LTC as a serious extension requires more than an asset-test flag. The model should be designed to carry an LTSS state vector that can be activated in later phases. Core variables include:
- Functional limitations (ADLs, IADLs, cognitive impairment, need for supervision)
- Care setting (unpaid family care, paid home care, assisted living, nursing facility, no care)
- Financing state (liquid assets, home equity, private LTC insurance, Medicaid eligibility pathway, spenddown status)
- Family spillovers (availability of caregivers, caregiving hours, caregiver employment effects, co-residence)
- Program interactions (SSI, Medicaid LTSS, Medicare home health or home-care benefits, tax-based caregiver supports)

The key design implication is that LTC should not be framed only as a downstream application of the Social Security model. It should be treated as a parallel extension target with its own state variables, transition equations, and validation criteria, while still reusing the common synthetic panel, calibration machinery, and PolicyEngine rules infrastructure.

A practical development sequence is static-first, dynamic-second. An initial LTC workstream can encode state rules and estimate point-in-time eligibility or first-pass fiscal effects. A later workstream can add transitions across disability states, care settings, and spenddown paths. Planning for both stages now reduces the risk of building a Social Security-only architecture that is difficult to extend later.

Even a static-first state pilot is a substantial rules-engine build, not
just a thin eligibility screen. A concrete Colorado-style LTSS scope
would need to encode at least:

- income-cap eligibility tied to SSI-based methodology rather than MAGI
- countable versus exempt assets, including home equity, vehicles,
  burial arrangements, trusts, and retirement accounts
- spousal impoverishment protections such as the Community Spouse
  Resource Allowance and monthly maintenance allowances
- post-eligibility income contribution rules, including personal-needs
  allowances and patient liability toward care costs
- 60-month look-back rules, transfer penalties, and Qualified Income
  Trust or Miller Trust logic
- functional eligibility and pathway logic across institutional care,
  HCBS and PACE, working-disabled buy-in pathways, and youth
  institutional pathways
- estate recovery and home-preservation implications

That in turn implies that a useful LTC product must return more than a
binary flag. It should be able to answer questions such as "eligible now
or soon?", "what spend-down path would make eligibility possible?",
"would a Miller Trust solve the income-cap problem?", "how much income
would still have to be contributed to care?", and "how do home and
spousal protections change the result?". Those outputs are feasible in a
static-first pilot, but they make clear why LTC should be budgeted as a
real extension track rather than a minor add-on.

### Advanced Extensions

More generally, adding realistic modules for pension coverage and benefits, homeownership, saving patterns, detailed taxes, business ownership, and intergenerational transfers is potentially feasible but represents the cutting edge of dynamic microsimulation. These extensions would provide comprehensive lifetime fiscal analysis but require substantial additional development effort.

The incremental approach allows starting with a validated basic model and adding capabilities as resources and priorities dictate. This modularity ensures early versions remain useful for core Social Security analysis while allowing expansion over time.

## Micro-Macro Interactions

Much policy analysis involving dynamic microsimulation begins with a macro model, usually an Overlapping Generations (OLG) framework, with highly simplified budget constraints solved under assumptions of consistent agent expectations and asymptotic elimination of government debt. The micro processes, particularly individual earnings distributions, are then calibrated to match the macro processes from the OLG model.

### Limitations of the OLG approach for this project

OLG models provide internally consistent frameworks for analyzing intertemporal policy trade-offs, and PWBM and other groups have demonstrated their value for fiscal analysis. However, for our distributional microsimulation goals, OLG frameworks involve trade-offs. The assumptions required for tractable solutions—representative agents within cohorts, perfect foresight, and equilibrium convergence—limit the heterogeneity that microsimulation is designed to capture. Calibrating micro processes to match OLG macro outcomes may constrain the distributional detail that is our primary value-add relative to existing models.

### Alternative Approaches

Promising alternatives exist for introducing micro-macro interactions in dynamic microsimulation. Reduced-form dynamic programming can capture behavioral responses without requiring full structural solutions. This approach uses empirical patterns from panel data to inform decision rules, avoiding the strong assumptions of OLG models. More fundamentally, dynamic microsimulation may provide important information about why long-term macro models provide little value for budget analysis.

### Following the Money

The most direct approach starts by following the money through the fiscal system. This means tracking taxes collected from individuals, government benefits paid to individuals, and ownership of government debt across the income and wealth distribution. These flows can be aggregated to produce fiscal projections without requiring OLG apparatus. The distributional patterns emerging from this tracking exercise may reveal why aggregate fiscal projections based on representative agents fail to capture reality.

This approach aligns with PolicyEngine's existing strength in detailed tax and transfer modeling while avoiding contentious assumptions about long-run macroeconomic equilibria.

## Output dataset structure

Longitudinal `microplex` should be organized as a relational dataset
with four linked tables. This structure captures all data elements
required for benefit calculation across all beneficiary types,
maintains family relationships for auxiliary and survivor benefits, and
supports both current-year and forward-projection analysis.

### Table overview

| Table | Grain | Key field(s) | Est. rows (200K sample) |
|-------|-------|-------------|------------------------|
| **PERSON** | One per individual | `person_id` | 200,000 |
| **EARNINGS** | One per person-year | `(person_id, year)` | 4,000,000+ |
| **RELATIONSHIP** | One per relationship | `relationship_id` | 600,000–1,000,000 |
| **EVENT** | One per event | `event_id` | 400,000–800,000 |

All tables link via `person_id`. Relationships link `person_id` to `related_person_id`.

### PERSON table (individual demographics)

One row per individual, capturing status as of the base year (December 31, 2025):

**Core demographics**: `person_id`, `representation_factor`, `date_of_birth`, `sex`, `race_ethnicity`, `education`, `state_of_residence`

`representation_factor` should be fixed or network-preserving after the
base-year population is constructed. It is not a freely recalibrated
annual person weight; arbitrary person-level reweighting would break
spouse, former-spouse, parent-child, and household consistency.

**Vital status**: `vital_status` (alive/deceased), `date_of_death`

**Disability**: `disability_status`, `disability_onset_date`, `disabled_before_22` (for adult disabled child benefits), `current_disability`

**Work history summary**: `total_qc_earned` (quarters of coverage), `fully_insured`, `disability_insured`, `first_year_earnings`, `last_year_earnings`

**Current benefit status**: `receiving_benefits`, `benefit_type` (RET/DIS/SPO/WID/YWI/DWI/CHI/PAR/DIV), `benefit_start_date`, `monthly_benefit_amount`

### EARNINGS table (annual earnings history)

One row per person-year of work, providing the complete earnings history needed for AIME calculation:

**Core fields**: `person_id`, `year`, `covered_earnings` (capped at taxable maximum), `medicare_earnings` (uncapped)

**Employment detail**: `employment_type` (wage/self-employed/government), `qc_earned_year` (0–4)

**Indexing**: `indexed_earnings` (computed: earnings indexed to age 60 using AWI), `awi_year` (National Average Wage Index for the year)

Years with zero earnings have no row (absence implies zero). This keeps the table compact and makes work history gaps visible.

### RELATIONSHIP table (family network)

One row per relationship, capturing all family linkages needed for auxiliary and survivor benefits:

**Core fields**: `relationship_id`, `person_id`, `related_person_id`, `relationship_type` (SPO/DIV/CHI/PAR)

**Marriage detail**: `relationship_start`, `relationship_end`, `end_reason` (divorce/death), `marriage_duration_months` (for the 10-year rule governing divorced spouse benefits), `marriage_number`

**Other**: `biological_relationship` (biological/adopted/step), `dependency_status` (for parent benefits)

Each marriage has two rows (one per spouse). Parent-child relationships have `person_id` as parent, `related_person_id` as child.

### EVENT table (time-varying events)

One row per life event affecting benefit eligibility:

**Core fields**: `event_id`, `person_id`, `event_type` (DTH/DIS/CLM/RMR/FRA/SUS/RSM), `event_date`, `age_at_event`

**Benefit events**: `benefit_type_claimed`, `event_value` (benefit amount), `related_person_id` (for survivor claims)

### Computed variables for benefit calculation

Derived from the base tables for each individual:

**Earnings-based**: `aime`, `pia_base`, `pia_with_cola`, `highest_35_years_sum`, `years_covered_work`, `years_zero_earnings`

**Eligibility**: `full_retirement_age`, `eligible_retirement`, `eligible_disability`, `eligible_survivor`

**Family-based** (require joining through RELATIONSHIP): `spouse_pia`, `ex_spouse_pia`, `deceased_spouse_pia`, `spousal_benefit_amt`, `survivor_benefit_amt`, `family_maximum`, `dual_entitlement_amt`

### Mapping to PolicyEngine-US variables

The 4-table structure maps to PolicyEngine's entity hierarchy:

| Dataset table | PolicyEngine entity | Key variables |
|--------------|-------------------|---------------|
| PERSON | `person` | Demographics, disability, benefit status |
| EARNINGS | `person` (time-series) | `employment_income`, `self_employment_income` |
| RELATIONSHIP | `tax_unit`, `spm_unit`, `family`, `household` | Family structure, filing status |
| EVENT | `person` (time-series) | Claiming decisions, disability onset |

For PolicyEngine integration, the 4-table structure will be flattened into the entity-based format that PolicyEngine-Core expects, with earnings histories stored as person-level arrays and family relationships encoded through entity membership. The benefit calculations themselves (AIME, PIA, spousal benefits, family maximum) then run through PolicyEngine-US's existing Social Security implementation.

### Implementation priority

1. **PERSON + EARNINGS** tables first (sufficient for retired worker benefits—the largest beneficiary category)
2. **RELATIONSHIP** table next (enables spousal, survivor, and child benefits)
3. **EVENT** table last (enables detailed longitudinal analysis and projection)
4. **Computed variables** derived as needed for validation and PolicyEngine integration

## Summary

The technical specifications outlined here provide a roadmap for model
development. We start with clearly defined variables and transition
equations needed for Social Security analysis, but we place them inside
the broader task of building longitudinal `microplex`. Historical
simulation creates and validates the base longitudinal population.
Forward projections incorporate calibration to address the jump-off
problem. Behavioral responses, even if simple, ensure realism.
Incremental capabilities allow starting with core Social Security
analysis and expanding over time. The approach to micro-macro
interactions emphasizes following fiscal flows rather than imposing
questionable OLG structure. The output dataset structure provides a
concrete specification for the longitudinal population asset, organized
to support all benefit types through linked relational tables.

These specifications, developed through decades of experience with dynamic microsimulation, provide a sound foundation for building an open-source model that serves both research and policy analysis needs.
