# Literature review

## Overview

The relevant literature does not point to one settled recipe for
building a dynamic Social Security model. Instead, it identifies a set
of recurring problems:

1. how to construct plausible lifecycle histories from incomplete data
2. how to preserve heterogeneity rather than collapsing to averages
3. how to keep long-run projections aligned to external benchmarks
4. how to validate models whose key state variables are only partly
   observed

This chapter organizes the literature around those problems rather than
presenting a long inventory of methods.

## 1. Dynamic microsimulation as a framework

Dynamic microsimulation begins with Orcutt's central idea: policy
analysis improves when models follow heterogeneous individuals and
households through time rather than relying only on representative
agents [@orcutt1961].

That framework has since developed into a mature modeling tradition.
Broad reviews emphasize recurring building blocks:

- transition models for demographics and labor-market states
- synthetic or semi-synthetic longitudinal data construction
- alignment or calibration to external aggregates
- explicit treatment of drift over long horizons

These themes are well documented in the microsimulation literature
[@harding1996; @li2013; @van2013].

## 2. The core Social Security problem is lifecycle earnings

Social Security is especially sensitive to lifecycle earnings
measurement. Benefits depend on long-run earnings histories, not on a
single cross-section. That is why cross-sectional earnings are often a
poor proxy for the quantities that matter for benefit determination
[@haider2006].

This literature implies a simple but important point for the project:
the central technical task is not merely "add another income variable."
It is reconstructing a longitudinal earnings process with the right:

- age profile
- dispersion
- persistence
- zero-earnings structure
- cohort pattern

If those ingredients are wrong, downstream benefit estimates will be
misleading even if the model matches headline aggregates.

## 3. Statistical matching and synthetic panel construction

The literature offers two broad strategies for lifecycle data problems:

### Administrative matching

The strongest approach is to match survey records to administrative
earnings data where possible. That is the MINT logic, and it is one
reason MINT has unusual credibility. But this strategy is not publicly
reproducible at scale.

### Synthetic reconstruction

The public-data alternative is to transfer longitudinal structure from a
panel survey or related source into a richer cross-sectional base. This
is the family of methods most relevant to the present project
[@rupp2005; @gouskova2010; @deville2011].

That literature supports the use of synthetic panels, but it also makes
the main risk clear: matching marginal distributions is much easier than
matching the joint structure over time.

## 4. Distributional methods matter more than mean methods

Traditional mean regression is often too weak for lifecycle imputation
because it smooths away the heterogeneity that policy analysis needs.
That is why distributional methods matter.

Quantile-based methods are attractive because they aim to preserve the
shape of the conditional distribution rather than only its center
[@meinshausen2006; @machado2019].

The literature also suggests caution:

- richer models are not automatically better
- smaller panel datasets can punish over-parameterized methods
- interpretability matters when the output will be used for public
  policy analysis

That is why this project treats advanced model families as candidates to
be tested against a conservative baseline rather than assumptions to be
locked in at the proposal stage.

## 5. Calibration is not a cosmetic step

Dynamic microsimulation literature consistently treats alignment or
calibration as central, not optional [@li2013; @deville1992].
Long-run projections drift. Survey totals differ from administrative
aggregates. Transition models are never perfect. Calibration is the
mechanism that keeps the synthetic population anchored to reality.

For this project, the literature implies two requirements:

1. calibration must be explicit and documented
2. validation must extend beyond the targets used in calibration

Otherwise the model can end up matching what it was told to match while
failing on the quantities that actually matter for policy use.

## 6. Validation is the central scientific task

The literature on microsimulation evaluation is clear on one point: the
credibility of the model depends on external validation, sensitivity
analysis, and honest reporting of error [@toder2002; @bourguignon2006; @favreault2016].

That matters especially here because a public synthetic model does not
have the administrative-data privilege of official agency models. The
project therefore needs a stronger validation culture, not a weaker one.
Useful validation targets include:

- current-year earnings distributions
- lifetime or quasi-lifetime benefit distributions
- AIME-related outcomes
- claiming behavior
- poverty and adequacy metrics
- subgroup distributions, not just aggregates

## 7. Distributional analysis is substantive, not decorative

The Social Security literature has long emphasized that aggregate
solvency does not settle the main policy questions. Progressivity,
replacement rates, race and gender disparities, family structure, and
retirement adequacy all matter materially
[@liebman2002; @whitman2011; @tamborini2013].

This has two implications for the project:

1. distributional outputs should be first-class products, not appendices
2. validation should include the subgroup distributions that policy
   debates actually turn on

## 8. Policy applications already motivate the model class

The literature on reform analysis shows why dynamic models are needed in
the first place. Work on privatization, progressive indexing, automatic
adjustment rules, and retirement incentives all depends on lifecycle
information and distributional detail
[@gustman2000; @diamond2003; @auerbach2017].

The lesson is not that one model can answer every reform question
equally well. The lesson is that static or purely aggregate tools leave
out many of the mechanisms people care about when debating Social
Security reform.

## 9. What the literature implies for this project

The literature points toward a disciplined project design:

- start from a strong public cross-section
- reconstruct lifecycle earnings conservatively
- calibrate explicitly to external benchmarks
- validate beyond the calibration targets
- prioritize subgroup and adequacy analysis, not just aggregate fit
- treat uncertainty as part of the output

It also points away from two mistakes:

- assuming that methodological sophistication substitutes for validation
- pretending that open source alone creates credibility

## Bottom Line

The literature supports the ambition of a public dynamic Social Security
model, but only under a strict condition: the project must treat
validation as the main research product. The model earns value by
showing where public-data reconstruction works, where it falls short,
and how those limits affect policy conclusions. That is the standard
this repository now adopts.
