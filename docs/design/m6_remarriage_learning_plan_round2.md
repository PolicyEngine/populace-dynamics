# M6 remarriage learning plan, round 2: the per-origin steepening family

- **Plan:** candidate-2 Lane B, pause-resolution learning plan round 2
- **Status:** **SPEC-SOUND FIXES APPLIED — UNRATIFIED.** Docs only; no round-2
  pseudo-holdout outcome or projection has run. The full second referee found
  zero blocking defects and returned SPEC SOUND subject to two should-fixes.
  PR #234 remains draft pending fix verification and ratification.
- **Authority:** the ratified candidate-2 program's designed pause and follow-up
  requirements (`docs/design/m6_candidate2_program.md:842-872,956-994,1038-1046`)
- **Prior learning:** round-1 [findings PR #231][round1-pr] and the
  [VERIFIED adversarial referee record][round1-referee]
- **Revision authority:** the round-2 [REVISE referee record][round2-referee]
- **Second-round verdict:** the [SPEC SOUND referee record][round2-sound-referee]
- **Fit-side validation:** [round-2 validation ledger][round2-validation],
  SHA-256 `26f0fcd3cb026f9811b3560970da6a4967b52c0a952c83545985fe03796cde6b`
- **Frozen evidence domain:** information dated no later than 2014 under the
  field-aware rules in section 4; deterministic fit-side validation and
  synthetic computation
- **Prohibited selection evidence:** every 2015+ row, truth moment, candidate
  output, gate score, tolerance comparison, and realized post-2014 macro value

## 1. Structural opening, revision, and the bright line

Round 1 did not show that no train-supported remarriage correction exists. It
showed that its whole nonzero family was mechanically incompatible with its
exposure safeguard. L1-L3 lowered every cell. Under common random numbers that
could only delay remarriage draw by draw and raise dissolved exposure. Because
L0's train-only projected exposure already exceeded truth at all three
boundaries, frozen selector rule 4 excluded every nonzero law. The referee
verified the structural entailment and resulting no-op. That remains the
opening fact for round 2.

The first round-2 draft escaped that entailment but created its mirror image. A
single global intercept restored total reference-spell area after lowering the
divorced stratum and raising the widowed stratum. Divorced spells dominated the
area functional, so the intercept returned most of the divorced move and
dumped it into every widowed cell. The merged round-1 ledger makes the failure
visible without running a candidate: widowed truth-exposure shares were only
9.4222%, 7.1972%, and 6.4320%, while the 2010 widowed pseudo-window had 40 risk
rows, 306,537 F6 exposure, and zero events. Its R0 projected widowed rate was
0.0669567. Increasing every hazard therefore strictly worsened zero-event
deviance, while draft rule 7 required that deviance not worsen. The first
round-2 family was provably empty.

This revision removes the global transfer rather than exempting one bad
boundary. It makes the origin strata independent:

- divorced hazards receive a frozen early-to-late YSD steepening tilt whose
  intercept and tilt jointly preserve **divorced** reference-spell survivor
  area while delivering an exact post-neutralization mean logit shift of
  `-0.50` or `-0.75`;
- working-age widowed hazards receive either the exact R0 table or a uniform
  `+0.05` logit shift, never any part of the divorced neutralizer; and
- widowed movement is capped by its own positive-event train defect, while a
  separate expected-rate guard replaces deviance comparison where matchable
  widowed events are zero.

The question is now narrow and honest: can divorced timing be front-loaded
enough to correct the reproduced pre-2015 rate overshoot without changing its
origin-specific expected survivor area, while an independently bounded
widowed option protects that origin and its dissolved stock? This is not
described as globally exposure-neutral. The outer projected-exposure guard
remains load-bearing.

The campaign bright line is literal:

> NO 2015-2019 data enters SELECTION. The REPRODUCED train-only (≤2014)
> overshoot is legitimate selection evidence — including the round-1
> train-only rate ratios 1.5674 / 1.6577 / 2.1742. The 2015-2019 holdout
> residual (score band 0.263-0.427 vs tol 0.403) may be CITED as motivation
> ONLY, never as a selection criterion. Replicate round 1's freeze discipline:
> freeze commit before any candidate outcome; runtime asserts.

The holdout band may explain why the ratified program paused
(`docs/design/m6_candidate2_program.md:842-872`). It sets no direction, target,
grid point, loss, eligibility cutoff, tie-break, budget, or disposition here.
No round-2 helper may read it or any row used to compute it.

## 2. What the admissible evidence establishes

The incumbent is a 60-cell table: five age bands by three
years-since-dissolution (YSD) bands by two origins by two sexes. It fits
`(N_c + wbar) / (E_c + 2 wbar)` in every cell
(`src/populace_dynamics/models/family_transitions/components/remarriage.py:26-93`)
and projects the lookup by age, YSD, origin, and sex
(`src/populace_dynamics/models/family_transitions/components/remarriage.py:96-132`).
The simulator compares common uniform draws with the applicable marital
hazards and updates dissolved state on remarriage
(`src/populace_dynamics/models/family_transitions/simulator.py:285-323`).

Round 1 published these selection-legitimate L0 facts on the `≤2014`
pseudo-holdouts:

| Boundary | Direct expected / actual numerator | Pooled rate ratio | Divorced rate ratio | Widowed rate ratio | Pooled exposure ratio | Working-age widowed exposure ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 2006 | 1.5260 | 1.5674 | 1.6712 | 0.9718 | 1.0307 | 1.1024 |
| 2008 | 1.6189 | 1.6577 | 1.7329 | 0.9143 | 1.0303 | 1.2017 |
| 2010 | 2.0385 | 2.1742 | 2.0420 | undefined: zero truth events | 1.0583 | 1.1650 |

The round-1 target and every ratio in this table are explicitly raw-age 18-64,
not all-age quantities. The ledger therefore supplies admissible R0
working-age widowed-stock reference errors
`abs(log(ebar_R0,widowed,b/e_truth,widowed,b))` of
`0.09746736052770962`, `0.18375046620867447`, and
`0.1527426344297312`. Round 2 uses the contemporaneous same-seed R0 as each
boundary's comparator, not these old-seed numeric values as cutoffs.

The exact projected origin log-rate errors retained by the merged ledger are:

| Boundary | Divorced `log(rbar_R0/r_truth)` | Widowed `log(rbar_R0/r_truth)` |
|---:|---:|---:|
| 2006 | +0.5135367879020232 | -0.028644764693506932 |
| 2008 | +0.5497870186488467 | -0.08956860182931886 |
| 2010 | +0.7139462881951797 | undefined: zero truth events |

The direct-standardized overshoot proves that the miss is not solely dynamic
risk-set feedback. The origin ledger localizes the demonstrated excess to
divorced transport. It supports a large divorced correction. It does not
support treating widowed as a balancing reservoir: the two defined widowed
errors are small and opposite the divorced error, and the third window has no
widowed event.

Define the evidence budget

```text
B_W = max(abs(-0.028644764693506932),
          abs(-0.08956860182931886))
    = 0.08956860182931886.
```

This is an aggregate projected log-rate budget, not a fitted widowed cell
residual. The family spends either zero or `0.05` logit. For a uniform positive
logit move `omega`, each fixed-risk hazard obeys

```text
h_omega / h0 = exp(omega) / (1 + h0*(exp(omega)-1)),
```

so the log change in any positive weighted mean hazard lies in `[0, omega]`.
Thus `omega=0.05 < B_W` is evidence-bounded in the ledger's log-rate currency.
It cannot reproduce the old `+0.45` to `+0.94` widowed dump.

Round 1 published no outcome residual by age or YSD cell. Its aggregate ledger
removed the detailed cell records. The revision therefore uses only the fixed
ordinal contrast early / middle / late = `+1 / 0 / -1`. The direction is a
mechanical timing hypothesis: at fixed survivor area, raising earlier hazards
and lowering later hazards can reduce expected event mass because an early exit
removes more later dissolved years. It is not a claim that any named YSD cell
was observed to miss.

The [fit-side validation ledger][round2-validation] is deliberately incapable
of constructing pseudo-holdout truth, projecting a candidate, computing a
candidate outcome diagnostic, or running the selector. It recomputes only R0
fit tables, fit support, reference paths, roots, and construction guards from
the already-admissible sources. Its source checksums match the merged round-1
ledger. All 24 origin-by-YSD support groups across the three boundaries and
final fit have positive exposure; unweighted event counts range from 47 to
1,536. No named law is support-struck.

Round 1 tested three nonzero laws and this revision names four. The findings
and any later registration must disclose seven accumulated nonzero laws across
the two rounds. Every named law counts even if a later pre-outcome structural
guard strikes it. The overlapping windows are recent-history stress tests, not
three independent samples.

## 3. Frozen candidate family

### 3.1 Cells, inputs, no-op, and the steepening contrast

At boundary `b`, a working-age cell is

```text
c = (a, d, o, s)
a in {(18,34), (35,49), (50,64)}
d in {(0,4), (5,9), (10,120)} years since dissolution
o in {divorced, widowed}
s in {female, male}.
```

These are the incumbent bands at
`src/populace_dynamics/models/family_transitions/components/remarriage.py:26-35`
and `src/populace_dynamics/data/transitions.py:137-142`. On admissible fit rows,

```text
h0_c = (N_c + wbar) / (E_c + 2*wbar).
```

`N_c` is F6-weighted remarriage numerator, `E_c` is F6-weighted dissolved
exposure, and `wbar` is the mean positive dissolved-row weight. `R0` returns
that table byte for byte. The investigation must assert R0 table, lookup,
uniforms, transitions, and aggregate output are bit-identical to the incumbent
under the same field-aware boundary construction.

Every basis exposure and delta is restricted by raw age `18 <= age <= 64`
before band indexing. At raw ages outside that range, including 15-17 and 65+,
the applied probability is `h0`. This is load-bearing because the incumbent
band helper clips ages 15-17 into the first `(18,34)` cell
(`src/populace_dynamics/data/transitions.py:110-111`;
`src/populace_dynamics/models/family_transitions/common.py:13-26`). The helper
must branch on raw age first and assert bit-identical R0 probabilities outside
18-64.

Every non-remarriage law and parameter, entry-dissolved carrier history,
event-label rule, truth support, F6 weight, and RNG address remains R0.
Endogenous downstream states may differ and must be published rather than
suppressed.

Define the fixed YSD front-loading contrast

```text
p_(0-4)   = +1
p_(5-9)   =  0
p_(10-120)= -1.
```

For divorced working-age fit exposure at boundary `b`, define

```text
pbar_bD = sum_{c:o=D} E_c*p_d / sum_{c:o=D} E_c
x_bd    = p_d - pbar_bD.
```

The same `E_c`, raw-age domain, origin restriction, and fixed-adjacent pairwise
sum are used here and in the direction statistic

```text
Delta_bo = sum_{c:o} E_c * (logit(h_c)-logit(h0_c)) / sum_{c:o} E_c.
```

Consequently `sum E_c*x_bd / sum E_c` is zero to the frozen arithmetic check.
Positive tilt raises the early band relative to the late band. There is no
fitted `q_od`, `q_oad`, minimum-event hyperparameter, age shape, data-chosen
sign, or post-outcome regrouping.

### 3.2 Per-origin reference paths and arithmetic

Construct separate reference-spell sets `S_bo` for each boundary and origin.
Start with every information-admissible dissolution event with known person,
dissolution year, origin, sex, birth year, administrative censor, and positive
F6 weight. Apply exclusions in this exact order:

1. exclude every spell in an ambiguous duplicate
   `(person_id, dissolution_year, origin)` group;
2. exclude a retained spell whose observed next remarriage is in the same
   calendar year as dissolution because it has no entering-year dissolved row;
3. exclude a spell with an empty potential path after the age/censor boundary.

Publish counts, total excluded F6 weight, and checksums for every category.
Duplicates do not abort the investigation and none of these exclusions removes
an event from the realized truth numerator or rate. The validation found one
divorced duplicate group (two spells, total F6 weight 29.896) and none in the
widowed stratum at every fit. It also publishes the same-year and no-path
counts for every boundary and final fit.

For retained spell `j`, define

```text
U_j = {dissolution_year_j + 1, ...,
       min(b, censor_year_j, birth_year_j + 64)}
W_j = {t in U_j: t >= birth_year_j + 18}.
```

The `+1` is the frozen entering-year convention: dissolution in year `t` is an
event and dissolved state begins in the next person-year
(`src/populace_dynamics/data/transitions.py:30-42,377-420`). The path uses the
fixed dissolution and administrative endpoint and deliberately does not censor
on a later observed remarriage. It is a standardization functional, not a
second population projection. Ages 15-17 may carry unchanged R0 survival into
age 18 but do not enter the working-age sum.

For table `h`, define the origin-specific survivor area

```text
A_bo(h) = sum_{j in S_bo} weight_j *
          sum_{t in W_j} product_{r in U_j:r<t} (1-h_jr).
```

Within each path, update survival in ascending year order; an empty product is
one. Build contributions in ascending
`(origin, person_id, dissolution_year, year)` order. Sum them with this exact
binary64 algorithm: add adjacent pairs, carry an odd final term unchanged, and
repeat until one number remains. The same algorithm governs `E`-weighted
centering and direction sums. A library reduction with an unspecified tree is
forbidden.

The validation's largest per-origin area sum has 55,773 terms, so the pairwise
tree has 16 levels. Including ratio and subtraction, its same-sign summation
bound is `2.2204460492503182e-15`. The ledger additionally reserves
`1.2e-14` for sequential survival products and elementwise arithmetic. The
area/root tolerance is

```text
tau_A = 1e-10.
```

The reserve explains why `tau_A` is separated from ordinary rounding error; it
is not root-acceptance headroom. The worst accepted divorced root is the 2010
`k=0.50` root:
`abs(G)=9.570999548458303e-11=0.9570999548458303*tau_A`. It leaves only
`4.290004515416971e-12`, or `0.04290004515416971*tau_A` (4.29%), before the
acceptance boundary. Root reproduction therefore depends on the exact frozen
arithmetic and runtime identity below, not on the rounding-reserve ratio. This
tolerance remains separate from the selector comparison tolerance of `1e-12`.

### 3.3 Origin targets, root, and exact laws

For divorced origin and strength `k`, define

```text
h_D,c(k,beta) = expit(logit(h0_c) - k + beta*x_bd), raw age 18-64
h_D,c(k,beta) = h0_c,                            otherwise.
```

This is equivalently an origin intercept `alpha_bD=-k` plus steepening tilt
`beta_bD*x_bd`. The two construction targets are

```text
Delta_bD = -k
A_bD(h_D) / A_bD(h0) - 1 = 0 within tau_A.
```

Exposure-centering makes the first equality algebraic. For the second, define

```text
G_bk(beta) = A_bD(h_D(k,beta))/A_bD(h0) - 1.
```

Evaluate `G_bk(0)` and `G_bk(16)` and require `G_bk(0)>0>G_bk(16)`. Set
`lo=0`, `hi=16`, then for iterations 1 through 200:

1. set `mid = lo + (hi-lo)/2`; equality with either endpoint is
   `ROOT_STAGNATION`;
2. evaluate `G_bk(mid)` and return the first midpoint with
   `abs(G_bk(mid)) <= tau_A`;
3. if `G_bk(mid)>0`, set `lo=mid`; otherwise set `hi=mid`.

A non-finite evaluation is `ROOT_NONFINITE`; a failed sign bracket is
`ROOT_NO_BRACKET`; midpoint equality is `ROOT_STAGNATION`; and no accepted
midpoint after iteration 200 is `ROOT_ITERATION_LIMIT`. A disagreement with the
fit-side ledger is `ROOT_VALIDATION_MISMATCH`. Because that ledger already
validates every root on frozen source hashes, any such runtime status aborts the
investigation before selection. It does **not** mark one law ineligible, feed
the no-op, or masquerade as substantive evidence.

The validation ledger also freezes `python_implementation="CPython"`,
`python_version="3.13.12"`, and `numpy_version="2.4.2"`. This follows the
repository's existing exact-string environment identity convention
(`src/populace_dynamics/contract.py:186-200` and
`src/populace_dynamics/models/family_transitions/compatibility.py:368-374`).
Before re-evaluating a root or computing any candidate outcome, the runner must
assert the interpreter implementation/version and `np.__version__` match those
three frozen values exactly. Any mismatch is `ROOT_VALIDATION_MISMATCH` and
aborts the whole investigation before selection; it is never law ineligibility
or a substantive no-op. This identity pin strengthens the freeze and does not
relax `tau_A`, change the first-accepted-midpoint rule, or permit a new root.

For widowed origin, choose `omega in {0.00, 0.05}` independently and define

```text
h_W,c(omega) = expit(logit(h0_c) + omega), raw age 18-64
h_W,c(omega) = h0_c,                       otherwise
alpha_bW     = omega
T_bW(omega)  = A_bW(h_W(omega)).
```

The widowed target is its own table's area, not the divorced residual. Runtime
must reproduce `A_bW(h_W)/T_bW-1` within `tau_A`. `omega=0` is byte-identical
to R0 in every widowed cell. `omega=0.05` moves every working-age widowed logit
by exactly 0.05, below `B_W`; it lowers the validated widowed reference area by
only 1.62-1.67%. No divorced score or root can enter a widowed cell.

The strength grid is exactly `k in {0.50,0.75}`. Because the post-root
`Delta_bD` is exactly `-k`, these endpoints genuinely bracket the round-1
divorced log-rate defects `0.5135368` through `0.7139463`. There is no global
alpha give-back.

The complete family has five laws:

```text
R0                         exact no-op
R_D50_W00 = R1(0.50,0.00) divorced target -0.50; widowed exact R0
R_D75_W00 = R1(0.75,0.00) divorced target -0.75; widowed exact R0
R_D50_W05 = R1(0.50,0.05) divorced target -0.50; widowed +0.05
R_D75_W05 = R1(0.75,0.05) divorced target -0.75; widowed +0.05.
```

The frozen simplicity order is the displayed order. Leaving widowed at R0 is
simpler than spending its separate budget; within a widowed option, the
smaller divorced move is simpler. Consequently `R_D75_W00` precedes
`R_D50_W05`: avoiding any widowed movement outranks choosing the smaller
divorced shift. This is a deliberate governance preference, not an accidental
ordering, and no outcome may reorder it.

The fit-side roots are already published:

| Fit boundary | `beta_D`, k=0.50 | `Delta_D` | residual | `beta_D`, k=0.75 | `Delta_D` | residual |
|---:|---:|---:|---:|---:|---:|---:|
| 2006 | 0.9365218054 | -0.5000000000 | -8.29e-11 | 1.2937688879 | -0.7500000000 | +5.91e-11 |
| 2008 | 0.9253702560 | -0.5000000000 | +8.10e-11 | 1.2791157849 | -0.7500000000 | +2.43e-11 |
| 2010 | 0.9710831894 | -0.5000000000 | -9.57e-11 | 1.3322750581 | -0.7500000000 | +3.46e-12 |
| final 2014 information | 0.9303924022 | -0.5000000000 | +2.89e-11 | 1.2817100482 | -0.7500000000 | -9.27e-11 |

All bracket endpoints have the required signs and all roots converge within
the frozen tolerance. The calibrated divorced cell shifts range from
`-2.2198057` to `+0.4447444` logit, and divorced candidate probabilities remain
between `0.0018317` and `0.2673588`; across the complete family, including the
independent widowed option, the range is `0.0018317` to `0.3800683`. The large
late-cell reduction is disclosed as a
mechanical consequence of preserving area while achieving the mean divorced
target, not as cell-outcome evidence. No post-outcome cap may be invented.

### 3.4 Numerator timing only through the state-consistent hazard

Standalone numerator thinning, relabeling, event-year shifting, censor/window
movement, and count-only calibration remain outside this family. A remarriage
event must coincide with leaving dissolved state and opening the next marriage;
severing numerator and state path would violate carrier/event conformance and
contaminate downstream family histories. Event construction is pinned by the
dissolve-before-marry order and entering-year convention
(`src/populace_dynamics/data/transitions.py:307-315,377-420,473-509`). Any
change to those semantics requires the separate §2.8 ceremony at
`docs/design/m6_candidate2_program.md:835-840,946-954`.

Timing is in scope only through the frozen YSD hazard. The candidate may move a
remarriage and its dissolved-state exit together. Valid same-year `YSD=0`
events remain in every realized/truth numerator and rate, have no row-indexed
expected contribution, and stay outside survivor-area and deviance sums. Their
counts and F6 weights are always published.

## 4. Frozen pseudo-holdouts and information dating

Round 2 copies round 1's field-aware `≤T*` construction. The only intentional
Monte Carlo change is the fresh seed bank below.

The expanding boundaries are exactly `{2006,2008,2010}`. At boundary `b`:

1. Apply conservative refit truncation. A fit row is admissible only if its
   state/event year is `<=b`; annual marital flows also require the establishing
   interview to be `<=b`.
2. Fit all unchanged family components. Only the remarriage table differs.
3. Form the shifted §2.8 anchor from the earliest positive-weight interview in
   `{b+1,b+3}` and hold household and F6 weight fixed.
4. Seed realized marital state, including entry-dissolved carrier history, and
   project only through `max(evaluation_years)`.
5. Evaluate exactly `[2007,2008,2009,2010]`, `[2009,2010,2011,2012]`, or
   `[2011,2012,2013]`, respectively, on symmetric opening-wave presence support
   pooled over sex and ages 18-64.

Calendar 2014 is excluded because its establishing PSID interview is in 2015.
The field-aware refit machinery and dating rules are at
`src/populace_dynamics/engine/refit.py:262-296,351-406,507-544`; the incumbent
refit rule remains
`src/populace_dynamics/engine/refit.py:599-613`.

Retrospective sources are read-only and sanitized by exclusion before any fit,
reference path, truth reduction, or projection:

- demographic rows are limited to interviews no later than 2013;
- a marriage or birth survives only if event year and establishing-report year
  are no later than 2014;
- post-2014 end/separation fields are nulled and `n_marriages` recomputed; and
- deaths after 2014 are nulled.

At each boundary the same rules tighten to `b`. The helper must disclose that
this is not a contemporaneous snapshot: retention of a pre-2015 marriage can
correlate with later panel continuation because a record last established after
2014 is excluded. The convention remains symmetric because it is
exclusion-only and R0 must be bit-equivalent on the same sanitized composition.

The analysis seeds are exactly `{7240,...,7279}`. All five laws use common
random numbers. Blocks are `{7240,...,7259}` and `{7260,...,7279}`. The old
`{7200,...,7239}` outcomes are published motivation only; no round-2 law may be
previewed on them.

The RNG period counts are frozen, not inferred from loose prose:

```text
n_periods(2006) = 4
n_periods(2008) = 4
n_periods(2010) = 3.
```

For seed `s`, instantiate
`ProjectionRNGRegistry(draw_index=s-5200, n_periods=n_periods(b))`, use
`generator(0, ProjectionModule.MARITAL_CORE)` for transition uniforms, and use
`child_generator(0, ProjectionModule.MARITAL_CORE, 1)` for spouse-gap draws.
The root is `s` and `n_periods` is RNG-material because the registry spawns
`n_periods+1` period children
(`src/populace_dynamics/engine/rng.py:16-17,37-99`, especially 61-67).

Candidate order may not alter addresses or insert a law-specific RNG call.
Changed transitions may lawfully change spouse-gap consumption and downstream
active sets; findings publish those differences while asserting registry code,
child addresses, and the raw transition-uniform checksum are unchanged.

## 5. Frozen diagnostics, objective, and selector

### 5.1 Publication record and direct quantities

For every boundary and law, findings publish regardless of disposition:

- fit/support rows, events, exposure, `wbar`, every cell probability, table
  checksum, `pbar`, centered contrast, origin intercepts, divorced root ledger,
  `Delta`, min/max shift, probability range, and both origin area targets;
- reference-spell inclusion and every exclusion count/weight/checksum, path
  counts, pairwise term count, R0 area, target, candidate area, and residual;
- pseudo-future truth dissolved exposure, event numerator, and rate;
- direct expected numerator/rate and F6-weighted Bernoulli deviance;
- all 40 projected exposure, numerator, and rate records, their mean, both
  blocks, ratios, and natural logs;
- the same records by origin and frozen origin-by-age-by-YSD publication groups,
  retaining zero-event groups as explicit zeroes/null ratios; and
- support, carrier, event-label, CRN, input-hash, validation-ledger, and
  R0-equivalence checks.

For law `L` and boundary `b`, direct risk rows `i` use

```text
expected_numerator_Lb = sum_i weight_i*h_L,c(i)
actual_numerator_b = sum_i weight_i*event_i
                     + unmatched_same_year_event_weight_b
expected_rate_Lb = expected_numerator_Lb / sum_i weight_i
actual_rate_b = actual_numerator_b / sum_i weight_i.
```

Same-year events remain in actual numerator/rate but outside row-indexed
deviance. Any unmatched nonzero-YSD event aborts. Deviance is

```text
D_Lb = -2 * sum_i weight_i *
       (event_i*log(h_L,c(i)) + (1-event_i)*log(1-h_L,c(i)))
       / sum_i weight_i.
```

Origin deviances use the same restriction and normalization. Pooled deviance
is the sum of the three pre-normalization numerators divided by the sum of the
three denominators, not an equal-boundary mean.

For widowed origin additionally define, on its row-indexed risk rows,

```text
m_wb = number of positive-weight rows with event_i=1
qdir_Lwb = sum_i weight_i*h_L,c(i) / sum_i weight_i
g_Lwb = log(qdir_Lwb/qdir_R0wb).
```

The truth-defined branch `m_wb>0` versus `m_wb=0` is computed once per boundary
and is law-independent. A same-year-only event does not make row-indexed
deviance informative.

The committed findings may reduce repetitive per-seed records only with a
committed deterministic reducer. It must retain every selector input, the
SHA-256 of full JSON stdout, reducer hash, and enough data to recompute every
eligibility bit and selection without rerunning the model.

### 5.2 Projected quantities, loss, and Monte Carlo uncertainty

Let `K_b` be the boundary's frozen truth-support person-year keys over its exact
effective evaluation years, and define the intervention-aligned subset

```text
K_WA,b = {i in K_b: 18 <= raw_age_i <= 64}.
```

Section 4 already restricts the frozen pseudo-holdout support to raw age 18-64,
so `K_WA,b=K_b`; naming the predicate here prevents an origin restriction from
being read as an all-age stock. For seed `s`, define rather than inherit:

```text
e_Lbs = F6-weighted dissolved exposure of law L summed on K_b
n_Lbs = F6-weighted projected remarriage-event numerator of L summed on K_b
r_Lbs = n_Lbs / e_Lbs.

e_WA,L,widowed,b,s = F6-weighted widowed dissolved exposure of law L
                     summed on K_WA,b
```

Origin versions restrict dissolved rows and remarriage events to that origin.
Truth `e_truth,b`, `n_truth,b`, and `r_truth,b` use the identical keys, years,
weights, and restrictions. This is exactly the round-1 transport aggregation.

Define

```text
rbar_Lb = mean_s r_Lbs
J(L)    = (1/3) * sum_b log(rbar_Lb/r_truth,b)^2

ebar_Lb = mean_s e_Lbs
ebar_Lob = mean_s e_Lobs

rate_error_Lb = abs(log(rbar_Lb/r_truth,b))
exposure_error_Lb = abs(log(ebar_Lb/e_truth,b))
working_age_widow_exposure_error_Lb =
    abs(log(mean_s(e_WA,L,widowed,b,s)/e_WA,truth,widowed,b)).
```

The round-1 ledger re-derives the three R0 working-age reference errors as
`0.09746736052770962`, `0.18375046620867447`, and
`0.1527426344297312`. They document that the scoped guard is material. Because
round 2 uses fresh seeds, eligibility compares each law with round 2's
same-seed R0 rather than importing a cross-bank Monte Carlo cutoff.

Exposure and rate guards use ratios of full-40-seed means, never mean per-seed
ratios or log errors. Each fixed block uses the identical `J` formula on its
20-seed means.

For Monte Carlo uncertainty, delete the same seed `s` from all boundaries,
recompute all three 39-seed rate means and `J_(-s)(L)`, and never re-select
inside a replicate. With `Jjack_bar` their mean,

```text
SE_J(L) = sqrt((39/40) * sum_s (J_(-s)(L)-Jjack_bar)^2).
```

This is Monte Carlo uncertainty only, not data-sampling uncertainty.

For selector comparisons, `x <* y` means `x < y-1e-12` and `x <=* y` means
`x <= y+1e-12`. This tolerance governs loss, deviance, log-error, and budget
comparisons only. It is distinct from `tau_A=1e-10` and from every gate
tolerance. `Lbest` is the exact binary64 minimum; only bit-equal `J` values use
the simplicity order.

### 5.3 Eligibility rules and zero-event handling

R0 is a baseline only if every information, conformance, checksum, and
bit-equivalence assertion passes. Otherwise abort. A nonzero law is eligible
only if all seven rules pass:

1. **Defined and conformant.** Every selector quantity is finite at all three
   boundaries, logarithm arguments are positive, pooled truth rates are
   defined, exact support/carrier/event/RNG checks pass, and candidate tables
   match the fit-side validation. Publication-only ratios may be null when
   truth numerator is zero.
2. **Per-origin construction and budget.** At every boundary,
   `abs(A_bD(h)/A_bD(h0)-1)<=tau_A`,
   `abs(A_bW(h)/T_bW(omega)-1)<=tau_A`,
   `abs(Delta_bD+k)<=1e-12`, and
   `abs(Delta_bW-omega)<=1e-12`. `beta_bD>0`; at least one positive-exposure
   divorced cell rises and one falls. Every widowed working-age cell shift is
   exactly `omega in {0,0.05}`, and `0 <=* g_Lwb <=* omega <=* B_W` on every
   boundary. No mandatory `Delta_widowed>0` rule strikes the `omega=0` control.
3. **Full and block loss.** Full-draw `J` and both block `J` values are `<*` R0.
4. **Boundary rate transport.** `rate_error_Lb` is `<*` R0 at least two
   boundaries and `<=*` R0 at the remaining boundary.
5. **Endogenous exposure protection.** `exposure_error_Lb` is `<=*` R0 at every
   boundary. Origin-specific reference targets do not waive this outer guard.
6. **Direct fit.** Pooled direct deviance is `<*` R0; boundary deviance is `<*`
   R0 at least two boundaries and `<=*` R0 at the third.
7. **Origin protection.** Divorced direct deviance is `<*` R0 at least two
   boundaries and `<=*` R0 at the third. For widowed origin:
   - if `m_wb>0`, direct widowed deviance is `<=*` R0;
   - if `m_wb=0`, widowed deviance is published but not compared, risk exposure
     must be positive, and the separate guard
     `0 <=* g_Lwb <=* omega <=* B_W` must pass.
   `working_age_widow_exposure_error_Lb` is `<=*` the same-seed R0 at
   **every** boundary in both branches.

Rule 5 is intentionally load-bearing: it is the sole genuinely binding
exposure constraint after the reference-area construction. That construction
preserves a full working-age path through age 64, whereas the selector scores a
three- or four-year window. The round-1 windows contained 690 / 581 / 419
entry-dissolved carriers but only 46 / 37 / 20 truth remarriage rows
(`docs/analysis/m6_remarriage_train_only_delta.md:283-287`). Carriers already
at middle or late YSD receive the lowered side of the tilt, which can raise
windowed exposure; new dissolutions receive the front-loaded side and pull the
other way. The sign is therefore carrier-composition- and magnitude-dependent,
not a new foreclosure. A rule-5-driven no-op is the designed result of this
outer guard, not a protocol surprise.

The zero-event branch is not a likelihood exemption chosen after seeing a law.
It is a frozen, truth-defined response to a cell in which Bernoulli deviance is
strictly monotone in hazard and cannot measure improvement toward zero without
re-imposing an R0 veto. The expected-rate guard caps the movement using the
widowed origin's own evidence, while the projected working-age stock guard
protects exactly the raw-age domain this family can move. An all-age widowed
guard is intentionally not used: its 65+ baseline would be nearly invariant to
the candidate and could turn Monte Carlo noise on an immovable stock into a
nominal selector decision.

For every correctly constructed frozen law, the `g/B_W` leg is a conformance
assertion and a restatement of the a-priori level cap, not a data-dependent
selection discriminator. The grid already fixes `omega` to `0` or `0.05`,
`0.05<B_W`, and the uniform logit move mechanically gives
`0<=g_Lwb<=omega`. The assertion detects implementation drift and documents
evidence-boundedness; active outcome protection comes from the positive-event
deviance branch and the working-age stock guard. It must not be credited as an
additional empirical selection filter.

These rules are conjunctions. A law cannot trade a failed exposure, widowed,
or construction guard for a lower `J`.

### 5.4 One-SE/no-op selector and final information fit

If no nonzero law is eligible, select R0 and return
`NO_OP_DESIGNED_PAUSE`. Otherwise:

1. `Lbest` is the eligible nonzero law with minimum full-draw `J`; bit-equal
   ties use the frozen simplicity order.
2. Set `cutoff=J(Lbest)+SE_J(Lbest)`.
3. If `J(R0)<=cutoff+1e-12`, select R0.
4. Otherwise select the first eligible law in simplicity order with
   `J(L)<=cutoff+1e-12`.

R0 therefore wins whenever it is within one Monte Carlo standard error of the
best eligible law, not only when it has the absolute minimum.

If a nonzero law is selected, fit exactly its `(k,omega)` once on all
information admissible at the 2014 boundary. The effective fit maximum remains
2013. Recompute h0, divorced exposure center, both origin reference sets,
`beta_D`, and the widowed target; do not re-select. The values must reproduce
the final-2014 validation row within frozen arithmetic. Publish all 60
probabilities, roots, targets, guards, and hashes.

If final construction or budget checks fail, do not substitute another law,
change a strength, or rerun selection. Publish the selected pseudo-holdout law
and final-fit failure, then continue the designed pause under section 7. A
passing final estimate is still a finding, not authority to edit the registry.

## 6. Freeze commit and runtime enforcement

This revised proposal is unratified. The full second plan referee may require
prospective edits while no round-2 candidate outcome exists. The fit-side
validation commits precede this revision and are narrowly scoped: they can read
admissible fit/reference inputs but cannot construct pseudo-holdout truth,
project a candidate, compute a direct/projected candidate outcome diagnostic,
or run the selector. Their source hashes and all calculations are disclosed.
They do not weaken the outcome freeze.

Before ratification, this branch must be rebased onto current `master` so the
round-1 report and ledger inherited by the validation are present in the same
tree. Do not merge `master` into the draft lane. After the second referee
accepts the complete design, the ratifying merge commit is the selector-freeze
commit. It must precede any helper contact with a round-2 pseudo-holdout truth,
candidate direct outcome, candidate projection, or selector output.

The investigation lane must:

1. record the ratified design commit and SHA-256 of this Markdown file **and**
   the validation JSON in a protocol-lock commit, retaining the validation
   ledger's exact interpreter and NumPy identity;
2. implement mechanically later, exercising only synthetic fixtures;
3. obtain a pre-execution audit confirming laws, inputs, roots, periods, seeds,
   diagnostics, selector, and dispositions match both frozen files; and
4. run once on read-only staged sources and publish full JSON stdout and the
   deterministic reduced ledger regardless of result.

Runtime must assert before selection:

- embedded design commit and both file SHA-256 values equal the lock;
- the interpreter is CPython `3.13.12` and NumPy is exactly `2.4.2`, matching
  the validation ledger, before any root or candidate-outcome computation;
- fit years/interviews are `<=b`, evaluation years are `<=2013`, and required
  evaluation interviews are `<=2014`;
- sanitized maxima, exclusion counts, field-null/recompute counts, and source
  hashes match the input audit and validation ledger;
- no 2015+ row, outcome, macro value, score, or tolerance is reachable by any
  fit, root, diagnostic, eligibility bit, or tie-break;
- boundary list, exact evaluation lists, `n_periods={2006:4,2008:4,2010:3}`,
  law order, `k`, `omega`, `B_W`, contrast, origin targets, root bracket,
  summation algorithm, both tolerances, seeds, and blocks match this plan;
- every fit-support, reference-spell exclusion, path checksum, center, bracket
  endpoint, root, direction, area residual, probability range, and support
  strike matches the validation ledger before candidate outcome computation;
- R0 bit-equivalence, raw-age guards, unchanged component laws, carrier/event
  semantics, RNG addresses, exact support, and selected-law pure-function checks
  pass; and
- the helper never imports or calls the M6 scorer, reads `gates.yaml`, reads a
  `runs/` artifact, writes a file, or emits anything but JSON to stdout.

Dependency/import and file-open audits belong in the findings. If a protocol,
diagnostic, or selector defect appears after first candidate-outcome contact,
stop and disclose the contact and defect. Do not repurpose output: amend,
obtain a new referee decision and freeze, and begin a new investigation.

## 7. Frozen no-op ladder

An R0 selection, empty eligible set, undefined required input, final-fit
failure, or unratifiable finding leaves the pause active. Registration 8
remains forbidden under
`docs/design/m6_candidate2_program.md:971-976,1038-1046`; no nonzero law may be
forced and the holdout may not break a tie.

In particular, a carrier-composition failure of rule 5 returns
`NO_OP_DESIGNED_PAUSE` and proceeds down this ladder. That disposition records
that the load-bearing exposure guard worked as designed; it does not license a
post-outcome relaxation or a reordered family.

The next rung remains fixed: publish a separate W1-style, candidate-blind
surface-question proposal for `remarriage.18-64`. It asks whether one pooled
working-age flow can certify an origin-heterogeneous formation process while
protecting divorced and widowed stocks. It uses truth-only power, vacuity, and
adjacent-pooling ladders. Candidate conformance forensics and both train-only
rounds may motivate the question but cannot set a cell, reducer, tolerance, or
floor. Any surface change needs its own prospective §2.8 and floor ceremony and
cannot retroactively change either learning result.

There is no automatic third hazard family. A later family requires the surface
referee to return the question to model-law learning and a new proposal frozen
without post-2014 outcomes. The pause continues throughout.

## 8. Ceremony and possible disposition

The governance sequence is:

1. **Revised proposal — complete:** this docs-only draft and fit-side ledger
   name the redesigned family, selector, freeze discipline, and ladder; no
   candidate outcome has run.
2. **Full second plan referee — SPEC SOUND:** issue comment 5004566775 found
   zero blocking defects, requested the two should-fixes applied here, and
   identified the three explicit tradeoffs now recorded above.
3. **Fix verification:** independently confirm the working-age stock guard,
   root acceptance margin, numeric runtime identity, and note clarifications.
   PR #234 remains draft throughout this round.
4. **Rebase, ratify, and freeze:** only after authorization, rebase onto
   `master` so round-1 evidence is in-tree, merge only the accepted plan, and
   record both file hashes. PR #234 remains draft until this step is authorized.
5. **Round-2 lane:** implement mechanically, pass synthetic and pre-execution
   audits, run once, and publish all findings regardless of result.
6. **Findings referee:** independently reproduce the ledger, information
   boundary, freeze history, eligibility, and selection.
7. **Disposition:** a verified nonzero law requires a separate prospective
   amendment, implementation, tests, immutable registry spec, and candidate-2
   lock before registration. R0 or an unratifiable law continues the pause and
   triggers section 7.

Selection is learning, not self-executing authorization.

## 9. What this proposal does not change

This proposal does **not**:

- alter candidate 1's frozen valid FAIL, artifact, 0/5 result, or history;
- edit candidate 2, its registry spec, current remarriage law, or authorize
  registration 8;
- edit `gates.yaml`, `runs/`, source/model/test code, truth support, F6 weight,
  a gate/floor tolerance, a registered cell, or an artifact;
- change `remarriage.18-64`, its `0.403` tolerance, or use that tolerance in
  selection;
- alter #226 entry-dissolved conformance, event labels, same-year convention,
  first marriage, divorce, widowhood, or raw-age-outside-18-64 probabilities;
- change or delay candidate-2 amendment 4 or floors-v4, which proceed under
  independent ceremonies; or
- authorize a third family or W1 surface change without section 7 governance.

The program's candidate-1 and surface boundaries remain at
`docs/design/m6_candidate2_program.md:1048-1069`. This proposal changes only
the pre-outcome round-2 learning protocol and its docs validation ledger.

## 10. Decisions for verification and ratification

The verification referee and ratifier must explicitly confirm whether to:

1. accept both published foreclosure proofs as the reason the revised family
   removes global transfer and fitted flattening;
2. ratify the fixed YSD front-loading contrast, independent origin targets,
   exact `Delta_D` strengths, widowed options, four-law grid, and the deliberate
   priority of widowed thrift over divorced magnitude in the simplicity order;
3. accept the fit-side ledger's source match, exclusions, support, root
   feasibility, cell-shift disclosure, acceptance margin, and numeric runtime
   identity;
4. approve exact pseudo-windows, fresh seed bank, and pinned period counts
   `{4,4,3}`;
5. ratify the seven conjunctive rules, including rule 5's carrier-composition
   risk, positive-matchable-event widowed deviance, the `g/B_W` conformance
   role, and the working-age widowed stock guard;
6. retain standalone numerator/timing edits outside the family and preserve
   state-consistent event semantics;
7. require rebase-before-ratification, two-file lock, pre-execution audit,
   runtime asserts, full-output reducer ledger, and stop/re-freeze rule; and
8. preserve the deterministic ladder to the candidate-blind W1 surface
   question with the pause active throughout.

## 11. Non-operative consistency ledger

This review aid is not executable configuration and not a gate edit:

```json
{
  "schema": "m6.remarriage.learning_plan.round2.proposal.v2",
  "status": "spec_sound_fixes_applied_unratified_pending_verification",
  "authority": {
    "program": "docs/design/m6_candidate2_program.md:842-872,956-994,1038-1046",
    "round1_pr": 231,
    "round1_referee_issue_comment": 5003093793,
    "round2_revise_issue_comment": 5003846811,
    "round2_spec_sound_issue_comment": 5004566775,
    "validation_file": "docs/design/m6_remarriage_learning_plan_round2_validation.json",
    "validation_sha256": "26f0fcd3cb026f9811b3560970da6a4967b52c0a952c83545985fe03796cde6b"
  },
  "evidence_cutoff": 2014,
  "pseudo_boundaries": [2006, 2008, 2010],
  "evaluation_years": {
    "2006": [2007, 2008, 2009, 2010],
    "2008": [2009, 2010, 2011, 2012],
    "2010": [2011, 2012, 2013]
  },
  "n_periods": {"2006": 4, "2008": 4, "2010": 3},
  "selection_seeds": {"inclusive_start": 7240, "inclusive_end": 7279},
  "selection_seed_blocks": [[7240, 7259], [7260, 7279]],
  "family": {
    "no_op": "R0",
    "form": "divorced_area_preserving_ysd_frontload_plus_bounded_widowed_level",
    "divorced_effective_shifts": [-0.5, -0.75],
    "widowed_logit_options": [0.0, 0.05],
    "widowed_log_rate_budget": 0.08956860182931886,
    "laws": ["R_D50_W00", "R_D75_W00", "R_D50_W05", "R_D75_W05"],
    "nonzero_laws": 4,
    "cumulative_nonzero_laws_two_rounds": 7,
    "raw_age_delta_domain": [18, 64],
    "outside_raw_age_domain": "exact_R0_probability",
    "ysd_frontload_contrast": [1.0, 0.0, -1.0],
    "divorced_area_target": "R0_per_origin_area",
    "widowed_area_target": "own_uniform_shift_table_area",
    "area_relative_tolerance": 1e-10,
    "sum_algorithm": "fixed_adjacent_pairwise_float64",
    "root_bracket": [0.0, 16.0],
    "root_failure_codes": ["ROOT_NONFINITE", "ROOT_NO_BRACKET", "ROOT_STAGNATION", "ROOT_ITERATION_LIMIT", "ROOT_VALIDATION_MISMATCH"],
    "root_max_iterations": 200,
    "support_struck_named_laws": []
  },
  "selector": {
    "objective": "mean_boundary_squared_log_pooled_rate_error",
    "uncertainty": "40_seed_delete_one_jackknife",
    "comparison_tolerance": 1e-12,
    "widowed_positive_matchable_event_rule": "direct_deviance_no_worse_than_R0",
    "widowed_zero_matchable_event_rule": "no_deviance_comparison_and_direct_expected_rate_log_move_within_omega_and_budget",
    "widowed_exposure_guard": "working_age_no_worse_than_same_seed_R0_every_boundary",
    "rule5_role": "load_bearing_exposure_constraint_carrier_composition_can_produce_designed_no_op",
    "g_BW_role": "construction_conformance_and_apriori_level_cap_not_empirical_selection_filter",
    "simplicity_priority": "widowed_thrift_before_divorced_magnitude",
    "no_op_rule": "select_R0_if_no_eligible_nonzero_or_J_R0_le_best_plus_SE",
    "selected_outcome_if_no_op": "NO_OP_DESIGNED_PAUSE"
  },
  "runtime_numeric_identity": {
    "python_implementation": "CPython",
    "python_version": "3.13.12",
    "numpy_version": "2.4.2",
    "mismatch_status": "ROOT_VALIDATION_MISMATCH"
  },
  "bright_line": "no_2015_2019_data_in_selection",
  "freeze": "ratified_commit_both_file_sha256_and_numeric_runtime_identity_before_any_candidate_outcome",
  "required_integration": "rebase_onto_master_before_ratifying_merge",
  "no_op_next_rung": "separate_candidate_blind_W1_style_remarriage_surface_question",
  "scope": "docs_only_no_candidate_gate_floor_run_or_code_change"
}
```

## References

- Ratified candidate-2 program:
  `docs/design/m6_candidate2_program.md`.
- Merged round-1 report and ledger on `origin/master`:
  `docs/analysis/m6_remarriage_train_only_delta.md` and
  `docs/analysis/m6_remarriage_train_only_delta_results.json`.
- Round-1 findings: [PR #231][round1-pr].
- Round-1 VERIFIED referee: [issue-comment 5003093793][round1-referee].
- Round-2 REVISE referee: [issue-comment 5003846811][round2-referee].
- Round-2 SPEC SOUND referee:
  [issue-comment 5004566775][round2-sound-referee].
- Round-2 fit-side validation: [JSON ledger][round2-validation].
- Pre-repair transport mechanism, motivation only:
  [issue #42 comment 4997635883][forensics4].

[round1-pr]: https://github.com/PolicyEngine/populace-dynamics/pull/231
[round1-referee]: https://github.com/PolicyEngine/populace-dynamics/pull/231#issuecomment-5003093793
[round2-referee]: https://github.com/PolicyEngine/populace-dynamics/pull/234#issuecomment-5003846811
[round2-sound-referee]: https://github.com/PolicyEngine/populace-dynamics/pull/234#issuecomment-5004566775
[round2-validation]: m6_remarriage_learning_plan_round2_validation.json
[forensics4]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4997635883
