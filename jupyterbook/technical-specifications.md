# Technical Specifications

This chapter provides detailed technical specifications for the dynamic microsimulation model, drawing on established practices in the field and specific requirements for Social Security analysis.

## Required Variables and Transition Equations

Dynamic microsimulation models begin with a longitudinal sample of the population and age that sample forward through time using stochastic demographic and economic transition equations. The specific variables and equations required depend on the modeling goals. For Social Security analysis, we need inputs to both payroll tax and benefit calculations.

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

## Constructing the Longitudinal Base File: Historical Simulation

The input file for dynamic microsimulation requires longitudinal histories for all listed variables in a representative sample as of the base year. Creating and validating this longitudinal input file proceeds through historical simulation, which is effectively a synthetic data generation exercise using cross-sectional and longitudinal input data, including parameterized stochastic earnings shocks and other transition equations.

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

### Advanced Extensions

More generally, adding realistic modules for pension coverage and benefits, homeownership, saving patterns, detailed taxes, business ownership, and intergenerational transfers is potentially feasible but represents the cutting edge of dynamic microsimulation. These extensions would provide comprehensive lifetime fiscal analysis but require substantial additional development effort.

The incremental approach allows starting with a validated basic model and adding capabilities as resources and priorities dictate. This modularity ensures early versions remain useful for core Social Security analysis while allowing expansion over time.

## Micro-Macro Interactions

Much policy analysis involving dynamic microsimulation begins with a macro model, usually an Overlapping Generations (OLG) framework, with highly simplified budget constraints solved under assumptions of consistent agent expectations and asymptotic elimination of government debt. The micro processes, particularly individual earnings distributions, are then calibrated to match the macro processes from the OLG model.

### Limitations of the OLG Approach

OLG models represent sophisticated mental exercises but have little proven predictive capability. According to OLG logic, the U.S. economy should have collapsed under rising government debt long ago, yet this has not occurred. The assumptions required to generate tractable solutions, including infinite horizons, representative agents within cohorts, and perfect foresight of policy changes, are far from realistic. The calibration of micro models to OLG macro outcomes may introduce biases and unrealistic constraints on distributional analysis.

### Alternative Approaches

Promising alternatives exist for introducing micro-macro interactions in dynamic microsimulation. Reduced-form dynamic programming can capture behavioral responses without requiring full structural solutions. This approach uses empirical patterns from panel data to inform decision rules, avoiding the strong assumptions of OLG models. More fundamentally, dynamic microsimulation may provide important information about why long-term macro models provide little value for budget analysis.

### Following the Money

The most direct approach starts by following the money through the fiscal system. This means tracking taxes collected from individuals, government benefits paid to individuals, and ownership of government debt across the income and wealth distribution. These flows can be aggregated to produce fiscal projections without requiring OLG apparatus. The distributional patterns emerging from this tracking exercise may reveal why aggregate fiscal projections based on representative agents fail to capture reality.

This approach aligns with PolicyEngine's existing strength in detailed tax and transfer modeling while avoiding contentious assumptions about long-run macroeconomic equilibria.

## Summary

The technical specifications outlined here provide a roadmap for model development. We start with clearly defined variables and transition equations needed for Social Security analysis. Historical simulation creates and validates the base longitudinal file. Forward projections incorporate calibration to address the jump-off problem. Behavioral responses, even if simple, ensure realism. Incremental capabilities allow starting with core Social Security analysis and expanding over time. The approach to micro-macro interactions emphasizes following fiscal flows rather than imposing questionable OLG structure.

These specifications, developed through decades of experience with dynamic microsimulation, provide a sound foundation for building an open-source model that serves both research and policy analysis needs.
