# M6 remarriage transport delta: train-only protocol and findings

- **Lane:** candidate-2 Lane B, remarriage transport delta
- **Authority:** candidate-2 program merged at
  `051b4494ecce9345da14d68488bb2833ed476d22`; independently verified in
  issue-comment `5001901052`
- **Status at this commit:** candidate laws and selector frozen before any
  marital-outcome fit or comparison; findings pending. A separate calendar
  audit inspected only pre-2014 demographic wave availability and did not
  inspect remarriage outcomes or choose any law.
- **Permitted evidence:** real rows whose event year, state year, and required
  interview year are all no later than 2014; synthetic computation
- **Prohibited evidence:** every 2015+ row, truth moment, candidate output, gate
  score, tolerance comparison, and realized post-2014 macro value

## 1. Question and bright line

The ratified program's ninth decision sends the repaired remarriage residual to
a train-only investigation. This report asks whether the incumbent remarriage
hazards systematically over-transport when applied to later dissolved risk
sets on pre-2015 pseudo-holdouts, and whether a small, fully specified estimator
delta is supported by that evidence.

The published `0.263-0.427` 2015-2019 residual band is motivation only. It does
not set a target, grid, cutoff, direction, or choice below. No computation in
this lane may read that window. If the frozen train-only selector returns the
incumbent or no nonzero law is eligible, the recommendation is the program's
designed pause, never a forced adjustment.

This is an analysis protocol, not a model amendment, implementation, gate run,
score, registration, or authorization to change candidate 2. A selected law
would still require a follow-up proposal amendment, referee review, and lock.

## 2. Incumbent mechanism

The current candidate-16 estimator has 60 cells:

```text
5 current-age bands x 3 years-since-dissolution bands
  x 2 origins (divorced/widowed) x 2 sexes.
```

For cell `c`, with F6-weighted event numerator `N_c`, dissolved exposure `E_c`,
and the mean positive dissolved-row weight `wbar`, it sets

```text
h0_c = (N_c + wbar) / (E_c + 2 wbar).
```

Thus the smoothing prior has mean `0.5` and strength `2 wbar`; an empty cell is
exactly `0.5`. Projection transports the resulting table onto each simulated
dissolved person's current age, dissolution duration, origin, and sex. The
landed carrier-episode repair preserves entry-dissolved state through assembly;
this lane does not alter that conformance law.

## 3. Frozen pseudo-holdouts

The expanding boundaries are exactly `{2006, 2008, 2010}`. At boundary `b`:

1. Apply the existing conservative refit truncation. A fit row is admissible
   only if its year is `<=b`; annual marital flows additionally require the
   establishing interview year to be `<=b`.
2. Fit all unchanged family components on admissible rows. Only the remarriage
   table varies across the four laws in section 4.
3. Form the shifted section-2.8 anchor from the earliest positive-weight
   interview in `{b+1,b+3}`. Hold its household and F6 weight fixed.
4. Seed the realized marital state at that person's anchor, including the
   landed entry-dissolved carrier history, and project through `b+4`.
5. Evaluate calendar years `{b+1,...,b+4}` only when the interview required to
   establish that annual row is also no later than 2014, on symmetric
   opening-wave presence support pooled over sex and ages 18-64. The effective
   windows are therefore 2007-2010, 2009-2012, and 2011-2013. Calendar 2014 is
   excluded because its establishing PSID interview is in 2015. Window overlap
   is deliberate recent-history stress, not independent replication.

The prototype must assert before calculation that the largest fit year and
required interview year are at most `b`, and that every evaluation row's year
and required interview year are at most 2014. It must never call the M6 scorer,
read `gates.yaml`, or write a `runs/` artifact.

For Monte Carlo transport, use 40 isolated analysis seeds
`{7200,...,7239}`. All four laws use common random numbers. The two fixed
stability blocks are `{7200,...,7219}` and `{7220,...,7239}`.

## 4. Frozen candidate set

Every empirical prior below is computed only from the boundary's admissible fit
rows. Let `mu` denote a weighted event/exposure hazard and `expit` the logistic
inverse.

### L0: incumbent add-one (no-op)

```text
h_L0,c = (N_c + wbar) / (E_c + 2 wbar).
```

### L1: global empirical-prior smoothing

Replace only the prior mean `0.5` with the pooled training hazard while
retaining the incumbent strength:

```text
mu_global = sum_c N_c / sum_c E_c
h_L1,c = (N_c + 2 wbar * mu_global) / (E_c + 2 wbar).
```

### L2: origin-by-sex empirical-prior smoothing

Use the corresponding training parent hazard for each origin-sex group `g(c)`:

```text
mu_g = sum_{c in g} N_c / sum_{c in g} E_c
h_L2,c = (N_c + 2 wbar * mu_g(c)) / (E_c + 2 wbar).
```

L2 is ineligible at a boundary if any of its four parent groups has fewer than
20 unweighted remarriage events or non-positive weighted exposure.

### L3: L1 plus a recent global log-odds transport shift

First fit L1 on all admissible rows through `b`. On admissible dissolved rows
dated `b-3,...,b`, solve the single weighted intercept equation

```text
sum_i weight_i * (event_i - expit(logit(h_L1,c(i)) + delta_b)) = 0.
h_L3,c = expit(logit(h_L1,c) + delta_b).
```

L3 is ineligible if the recent window has fewer than 20 unweighted events,
non-positive exposure, no finite root, or any non-finite output. If L3 is
selected, its final 2014 parameter is computed by the identical information-
dated rule on the effective 2011-2013 rows (calendar 2014 requires a forbidden
2015 interview) and an L1 table fit through the 2014 information boundary.

There are four laws total, including the exact no-op. There is no continuous
hyperparameter grid and no after-the-fact subgroup, window, cap, or direction
search.

## 5. Frozen diagnostics and selector

For each law and boundary, publish:

- fit rows, fit events, mean weight, parent hazards, empty/thin cell counts,
  cell probability range, and checksums;
- pseudo-future truth dissolved exposure, event numerator, and rate;
- the direct-standardized expected numerator and rate obtained by applying the
  fitted law to the realized dissolved rows, plus weighted Bernoulli deviance.
  A valid same-year dissolve-remarry event (`ysd=0`) is retained in every
  numerator/rate diagnostic but has no entering-year dissolved row under
  `allow_exact_matches=False`; it is therefore excluded only from the
  row-indexed deviance, with its count and F6 weight published. Any unmatched
  event with nonzero YSD is an abort;
- the 40-draw mean projected dissolved exposure, event numerator, and rate;
- projected/truth exposure, numerator, and rate ratios and their natural logs;
- the same transport quantities for each 20-seed block and separately by
  origin, plus carrier-episode state/YSD survival and exact-support checks.

No quantity is compared with a gate tolerance. Define the primary transport
loss only for train selection:

```text
J(L) = mean_b [log(projected_rate_L,b / truth_rate_b)^2],
```

using each law's 40-draw mean projected rate. Also compute the analogous `J`
inside each fixed seed block and the pooled direct-standardized Bernoulli
deviance.

A nonzero law is eligible for selection only if all of the following hold:

1. every required parameter, truth rate, and projected draw is defined at all
   three boundaries, and exact person-year support checks pass;
2. full-draw `J` and both fixed-block `J` values are strictly below L0;
3. absolute projected/truth log-rate error is below L0 at least two boundaries
   and is not above L0 at the remaining boundary;
4. absolute projected/truth log-exposure error is not above L0 at any boundary,
   preventing a rate improvement obtained by worsening dissolved exposure;
5. pooled direct-standardized deviance is below L0 and boundary deviance is
   below L0 at least two of three boundaries without worsening the third.

Among eligible nonzero laws, minimize full-draw `J`. Estimate the Monte Carlo
standard error of the minimum law's `J` from the 40 per-seed losses. Choose the
simplest law within one standard error of that minimum, ordered
`L1 < L2 < L3`. Exact ties use that order. If no nonzero law is eligible, or if
L0 has the lowest loss outside the one-standard-error set, select no-op and
recommend the designed pause.

## 6. Findings

Pending the frozen train-only computation.

## 7. Recommendation

Pending. The only permitted outcomes are a specific law for a later amendment
lane or no-op with a designed pause.

## References

- Ratified candidate-2 program:
  `docs/design/m6_candidate2_program.md`, merged at
  `051b4494ecce9345da14d68488bb2833ed476d22`.
- Candidate-2 verification: issue-comment
  [`5001901052`](https://github.com/PolicyEngine/populace-dynamics/pull/227#issuecomment-5001901052).
- Forensic motivation only: issue #42 comment
  [`4997635883`](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4997635883).
