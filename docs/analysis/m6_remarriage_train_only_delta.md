# M6 remarriage transport delta: train-only protocol and findings

- **Lane:** candidate-2 Lane B, remarriage transport delta
- **Authority:** candidate-2 program merged at
  `051b4494ecce9345da14d68488bb2833ed476d22`; independently verified in
  issue-comment `5001901052`
- **Status at this commit:** complete. The frozen train-only selector chose L0,
  so the finding is no-op and the recommendation is the designed pause.
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

### 6.1 Information-boundary and execution audit

The helper ran with Python 3.13 against the read-only staged PSID directory:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS='ignore::FutureWarning' \
PYTHONPATH=src \
POPULACE_DYNAMICS_PSID_DIR=/Users/maxghenis/PolicyEngine/psid-data \
python3.13 scripts/analyze_m6_remarriage_train_delta.py
```

Its full JSON stdout has SHA-256
`fe914611c1e0f2e96db15e62e49b69907af428379bc6cc5f3d1f3dbb782a540c`.
An independent replay of the same frozen command was byte-identical and
returned the same hash and disposition.
The committed [aggregate result ledger](m6_remarriage_train_only_delta_results.json)
removes only the 480 repetitive per-seed records; it retains source and support
checksums, fit diagnostics, probability ranges, direct diagnostics, draw means,
both fixed blocks, origin results, final parameters, and the selector record.

The staged marriage source is a retrospective product rather than a historical
pre-2015 snapshot. The helper loaded it read-only, severed every post-2014 field
before panel construction, and passed only the sanitized copy to fitting and
selection. The sanitized demographic maximum was interview 2013; the marital
event maximum was 2013. The actual fit maxima at boundaries 2006, 2008, and
2010 were respectively 2005, 2007, and 2009, and the final information-dated
fit maximum was 2013. Every evaluation row carried an asserted establishing
interview no later than 2013. Calendar 2014 was absent because it would require
the forbidden 2015 interview.

The helper did not import or call the M6 scorer, read a gate tolerance, or write
under `runs/`. It emitted the current first-marriage fitter's LBFGS 1,000-
iteration convergence warning once at each boundary; all laws shared that
unchanged fitted component and common random numbers. No 2015-2019 row or
outcome entered any fit, diagnostic, comparison, or choice.

### 6.2 Pseudo-holdout support and truth

| Boundary | Evaluation years | Anchor persons / households | Projected persons | Truth support / dissolved rows | Remarriage rows | Entry-dissolved carriers | Truth rate |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 2006 | 2007-2010 | 24,911 / 8,478 | 8,057 | 18,358 / 1,056 | 46 | 690 | 0.041096 |
| 2008 | 2009-2012 | 25,516 / 8,781 | 7,987 | 17,997 / 781 | 37 | 581 | 0.039751 |
| 2010 | 2011-2013 | 25,973 / 8,994 | 7,389 | 13,458 / 428 | 20 | 419 | 0.032247 |

The F6-weighted truth exposure/numerator pairs were
`12,518,941 / 514,477`, `8,798,740 / 349,758`, and
`4,765,776 / 153,680`. Valid same-year dissolution-remarriage events numbered
1, 4, and 3, with F6 weights 3,250, 28,971, and 15,869. They remain in every
truth numerator and rate. Only the row-indexed deviance omits them, leaving 45,
33, and 17 matchable event rows; no unmatched nonzero-YSD event occurred.

All 480 projected supports exactly matched their boundary's truth person-year
keys. Every one of the 270,400 entry-carrier comparisons preserved dissolved
state and YSD: 40 draws x 4 laws x `(690 + 581 + 419)` carriers. The three
truth-support checksums are recorded in the result ledger.

### 6.3 The incumbent overshoot reproduces on train-only splits

| Boundary | Direct expected / actual numerator | Projected exposure / truth | Projected numerator / truth | Projected rate / truth | Block rate ratios |
|---:|---:|---:|---:|---:|:---|
| 2006 | 1.5260 | 1.0307 | 1.6133 | 1.5674 | 1.6493 / 1.4855 |
| 2008 | 1.6189 | 1.0303 | 1.7057 | 1.6577 | 1.6316 / 1.6837 |
| 2010 | 2.0385 | 1.0583 | 2.2956 | 2.1742 | 2.0650 / 2.2835 |

This is a clear train-only reproduction. L0 overstates the expected numerator
even on the realized dissolved rows, and full projection transports that
overstatement while adding a smaller 3.0-5.8% dissolved-exposure excess. The
pooled rate overshoot appears at every boundary and in both fixed seed blocks;
it is not an artifact of the published 2015-2019 residual.

The origin ledger localizes the result. L0's divorced-origin rate ratios were
1.6712, 1.7329, and 2.0420. Its widowed-origin rate ratios were 0.9718 and
0.9143 in the first two windows; the final window had zero truth widowed
remarriages, so that ratio is undefined and recorded as `null`. Widowed
exposure was nevertheless high by 10.2%, 20.2%, and 16.5%. The pooled numerator
overshoot is therefore primarily divorced-origin hazard transport, while the
global risk set also over-transports widowed exposure.

### 6.4 Frozen candidate comparison

| Law | Rate ratio 2006 | Rate ratio 2008 | Rate ratio 2010 | Full J | Block J1 / J2 | Pooled direct deviance |
|:---|---:|---:|---:|---:|:---|---:|
| L0 | 1.5674 | 1.6577 | 2.1742 | 0.353545 | 0.338605 / 0.369951 | 0.320353 |
| L1 | 1.5395 | 1.6350 | 2.1523 | 0.338462 | 0.324736 / 0.353713 | 0.319624 |
| L2 | 1.5411 | 1.6362 | 2.1562 | 0.339955 | 0.326587 / 0.354810 | 0.319658 |
| L3 | 1.1867 | 1.2389 | 1.5204 | 0.083569 | 0.088679 / 0.080448 | 0.307710 |

Each nonzero law had defined parameters, lowered full and both block losses,
improved the pooled and boundary direct deviance, and met the rate-boundary
rule. None passed the frozen exposure protection:

| Law | Exposure ratio 2006 | Exposure ratio 2008 | Exposure ratio 2010 | Failed selector check |
|:---|---:|---:|---:|:---|
| L0 | 1.030674 | 1.030281 | 1.058255 | baseline |
| L1 | 1.031699 | 1.031067 | 1.058875 | exposure boundaries |
| L2 | 1.031647 | 1.031067 | 1.058713 | exposure boundaries |
| L3 | 1.046868 | 1.045541 | 1.070651 | exposure boundaries |

All three candidate deltas made absolute log-exposure error worse than L0 at
all three boundaries. The strict comparison was fixed before outcomes were
read and did not use a gate tolerance. Consequently, all nonzero laws are
ineligible, the one-standard-error step is not invoked, and the selector
returns L0.

L3's fitted recent-window shifts also become much stronger as the information
boundary advances: -0.3001 from 92 events at 2006, -0.3323 from 65 events at
2008, -0.4500 from 49 events at 2010, and -0.8698 from only 28 effective
2011-2013 events for the final fit. Although L3 most sharply lowers pooled
rate loss, that final shift is not supported as a ratifiable law under the
frozen transport safeguard. A global downward shift also deepens the already
low widowed-origin rate, reaching about 0.72 of truth in the first two splits.

Across each boundary and the final fit, all 60 cells had exposure (zero were
empty) and four were thin under the frozen threshold. L2's four parent groups
and L3's recent window passed their predeclared minimum-event rules. Full
parent hazards, counts, probability ranges, and checksums are in the aggregate
ledger.

### 6.5 Limits of the finding

The three windows overlap and are recent-history stress tests, not independent
replications. The last has only 20 remarriage rows and no widowed-origin event.
The retrospective staged source was sanitizable but is not a contemporaneous
pre-2015 snapshot, and the shared first-marriage fit emitted the convergence
warning noted above. These facts counsel against inventing a fifth law after
seeing the results; they do not change the frozen selector's no-op result.

This report establishes that the overshoot mechanism reproduces before 2015.
It does not establish an admissible corrective law. Choosing one by consulting
the published 2015-2019 residual would cross the campaign bright line.

## 7. Recommendation

**Recommend no-op and the designed pause.** Leave candidate 2, its registered
remarriage law, and every gate surface unchanged. No L1-L3 law is ratifiable
from the frozen train-only evidence because each worsens dissolved-exposure
transport at every pseudo-boundary.

A future lane may propose a new train-only protocol targeted to the divorced-
origin hazard mechanism and explicitly protecting the widowed risk set, but it
must freeze that law and selector before reading outcomes and must not use the
2015-2019 holdout to choose it. This lane supplies no authority for such an
amendment now.

## References

- Ratified candidate-2 program:
  `docs/design/m6_candidate2_program.md`, merged at
  `051b4494ecce9345da14d68488bb2833ed476d22`.
- Candidate-2 verification: issue-comment
  [`5001901052`](https://github.com/PolicyEngine/populace-dynamics/pull/227#issuecomment-5001901052).
- Forensic motivation only: issue #42 comment
  [`4997635883`](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4997635883).
- Aggregate train-only result ledger:
  [`m6_remarriage_train_only_delta_results.json`](m6_remarriage_train_only_delta_results.json).
