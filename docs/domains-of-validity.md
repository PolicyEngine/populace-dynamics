# Domains of validity

## Why this chapter comes first

All models are wrong; a model earns its keep only where it improves
predictions. This chapter states, before any machinery is described,
which questions this project claims to help with, which it refuses to
answer with false precision, and how users — human or AI — can tell
the difference. Every output the model ships carries this boundary as
metadata, not as a footnote.

## Parameter uncertainty dominates the long horizon

The 75-year actuarial balance is a function of a handful of exogenous
assumptions: total fertility, mortality improvement, net immigration,
real wage growth, and interest rates. Over long horizons, uncertainty
in those inputs — not the fidelity of the microsimulation that
processes them — dominates the answer.

The public record makes this concrete:

- The Trustees' own low- and high-cost scenarios bracket a range of
  75-year balances wider than the intermediate deficit itself, and
  their stochastic intervals are similarly broad [@ssa2025trustees].
- CBO's long-term projections have persistently shown materially
  larger 75-year shortfalls than the Trustees', driven mainly by
  different demographic and economic assumptions rather than
  different benefit arithmetic [@cbo2024longterm].
- Successive Technical Panels convened by the Social Security
  Advisory Board have recommended revising the fertility and
  mortality-improvement assumptions as realized values ran outside
  the intermediate path for extended periods
  [@technicalpanel2023].

Two expert institutions, each with administrative data and decades of
refinement, disagree with each other and have both missed realized
demographic trends. A new model — however well engineered — does not
fix that, because the variance lives in the inputs. Micro-level
precision layered on unforecastable demographics is spurious
precision.

## What this project will not claim

- **A 75-year point forecast of solvency.** The project will not
  market a headline depletion date or a point 75-year balance as a
  prediction. Where such quantities are computed, they are computed
  *conditional on named assumption paths* and presented alongside
  their sensitivity, never as the model's answer.
- **Precision on behavioral response it cannot validate.** Behavioral
  margins (claiming responses to reform, labor-supply feedback) ship
  as clearly labeled scenario inputs with documented ranges, not as
  point estimates wearing model authority.

## What survives, and why

Three output tiers, ordered by the strength of their claim to
usefulness:

### Tier 1: distributional analysis under fixed assumptions

Reform analysis is a difference: outcome under reform minus outcome
under baseline, holding the population and assumption path fixed.
Much of the unforecastable demographic uncertainty is common to both
arms and cancels in the difference. The ingredients this requires —
a calibrated joint distribution of lifetime earnings, family
structure, and differential mortality, plus an exact rules engine —
are the parts of the system that can be validated directly.

The honest caveat, stated rather than buried: the slice of a reform
delta that does *not* cancel — interactions between the reform and
the uncertain dynamics, such as reform-induced claiming shifts or
mortality-gradient interactions — is the least validated slice. The
model labels it as such.

### Tier 2: near-term components that resolve

Over roughly a ten-year horizon, the model's outputs are dominated by
mechanics rather than demographic extrapolation: beneficiary counts
by type, average benefits, covered earnings and taxable payroll,
claiming-age distributions, disability incidence. These quantities
resolve against administrative publications on an annual cycle, so
claims about them can be scored rather than argued
(see [scoring-and-resolution.md](scoring-and-resolution.md)).

### Tier 3: the long horizon as a sensitivity surface

Long-horizon outputs are published as *surfaces over assumptions*,
not points: how the balance, cohort replacement rates, or
distributional outcomes move as fertility, mortality improvement,
and immigration vary across documented ranges. The incumbent
practice is to bury assumption-dependence behind a point estimate
and an appendix. An open model can invert that: make the sensitivity
the product. That is a more honest object than a forecast, and it is
the only long-horizon object whose accuracy does not depend on
predicting the unpredictable.

## Where dynamics genuinely earn their keep

The longitudinal machinery is not decoration. Questions that require
lifetime position — who bears a retirement-age increase by cohort
and lifetime-earnings quintile, how survivor outcomes shift under a
benefit redesign, how insured status evolves for interrupted
careers — cannot be answered from a cross-section. The point of the
tiers is not to shrink the model's ambition; it is to attach each
ambition to the strongest claim it can actually support.

## Validity as metadata

Every API and MCP response carries its tier, its assumption path,
and its calibration history. A downstream agent composing this model
with other tools can weight it accordingly — trust is a number the
consumer reads, not a reputation the producer asserts. The
[scoring-and-resolution.md](scoring-and-resolution.md) chapter
defines where those numbers come from.
