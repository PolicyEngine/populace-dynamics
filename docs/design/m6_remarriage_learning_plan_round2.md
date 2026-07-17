# M6 remarriage learning plan, round 2: the exposure-neutral family

- **Plan:** candidate-2 Lane B, pause-resolution learning plan round 2
- **Status:** **PROPOSAL — UNRATIFIED.** Docs only; no investigation has run and
  this document authorizes no model, registry, gate, floor, or run change.
- **Authority:** the ratified candidate-2 program's designed pause and follow-up
  requirements (`docs/design/m6_candidate2_program.md:842-872,956-994,1038-1046`)
- **Prior learning:** round-1 [findings PR #231][round1-pr] and the
  [VERIFIED adversarial referee record][round1-referee]
- **Frozen evidence domain:** information dated no later than 2014 under the
  field-aware rules in section 4; synthetic computation
- **Prohibited selection evidence:** every 2015+ row, truth moment, candidate
  output, gate score, tolerance comparison, and realized post-2014 macro value

## 1. Structural opening and the bright line

Round 1 did not show that no train-supported remarriage correction exists. It
showed that its entire candidate family was mechanically incompatible with its
exposure safeguard. L1-L3 lowered the hazard in every cell. With common random
numbers, uniform lowering weakly delayed remarriage draw by draw and therefore
raised dissolved exposure. Because L0's train-only projected exposure already
exceeded truth at all three boundaries, frozen selector rule 4 then excluded
every nonzero law. The referee verified that structural entailment and the
no-op; it is the opening fact for this plan, not a reason to repeat a uniformly
hazard-lowering family.

Round 2 asks a narrower question: can a small, sign-balanced reshaping of the
working-age remarriage hazard improve the reproduced pre-2015 numerator and
rate transport while preserving a weighted expected dissolved-exposure
functional and protecting realized projected exposure? It tests an exact
no-op, an origin transfer, an origin-by-dissolution-recency reshape, and an
origin-by-age-by-recency reshape. It does not test a global downward shift.

The campaign bright line is literal:

> NO 2015-2019 data enters SELECTION. The REPRODUCED train-only (≤2014)
> overshoot is legitimate selection evidence — including the round-1
> train-only rate ratios 1.5674 / 1.6577 / 2.1742. The 2015-2019 holdout
> residual (score band 0.263-0.427 vs tol 0.403) may be CITED as motivation
> ONLY, never as a selection criterion. Replicate round 1's freeze discipline
> verbatim: freeze commit before any outcome; runtime asserts.

Accordingly, the published holdout band may explain why the ratified program
paused (`docs/design/m6_candidate2_program.md:842-872`). It sets no direction,
target, grid point, loss, eligibility cutoff, tie-break, or disposition in this
plan. No round-2 helper may read it or any row used to compute it.

## 2. What the admissible evidence establishes

The incumbent is a 60-cell table: five age bands by three years-since-
dissolution (YSD) bands by two origins by two sexes. It uses
`(N_c + wbar) / (E_c + 2 wbar)` in every cell
(`src/populace_dynamics/models/family_transitions/components/remarriage.py:26-93`)
and projects the resulting lookup by age, YSD, origin, and sex
(`src/populace_dynamics/models/family_transitions/components/remarriage.py:96-132`).
The simulator compares the same common uniform draw with each applicable
marital hazard and updates dissolved state on remarriage
(`src/populace_dynamics/models/family_transitions/simulator.py:285-323`).

Round 1 supplied the following selection-legitimate facts, all reproduced on
the `≤2014` pseudo-holdouts:

| Boundary | Direct L0 expected / actual numerator | Pooled rate ratio | Divorced-origin rate ratio | Widowed-origin rate ratio | Pooled exposure ratio | Widowed exposure ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 2006 | 1.5260 | 1.5674 | 1.6712 | 0.9718 | 1.0307 | 1.1024 |
| 2008 | 1.6189 | 1.6577 | 1.7329 | 0.9143 | 1.0303 | 1.2017 |
| 2010 | 2.0385 | 2.1742 | 2.0420 | undefined: zero truth events | 1.0583 | 1.1650 |

The direct-standardized numerator overshoot on realized dissolved rows proves
that the train-only miss is not solely dynamic risk-set feedback. The origin
ledger localizes the demonstrated over-formation to divorced-origin transport:
widowed rates were near or below truth in the first two windows, while widowed
exposure was already 10.2-20.2% high. This supports a divorced-down,
widowed-up transfer with a separate widowed protection rule.

That is the full defensible localization. Round 1 published no outcome residual
by age or YSD cell; its aggregate ledger retained probability ranges but removed
the detailed cell records. All 60 cells had exposure and four were thin, but
those are support facts, not evidence that a named age-YSD cell over-formed.
Round 2 therefore freezes a deterministic, fit-side regularizing basis before
it observes any new age/YSD outcome. It may not invent, merge, drop, or reverse a
cell after seeing round-2 pseudo-holdout results.

The three windows overlap and remain recent-history stress tests, not three
independent samples. The family has six nonzero laws after round 1 already
tested three; the findings and any later registration must disclose the
accumulated adaptive search as nine nonzero laws across two rounds, not describe
round 2 as an isolated first search.

## 3. Frozen candidate family

### 3.1 Cells, inputs, and exact no-op

At each pseudo-boundary `b`, let a working-age cell be

```text
c = (a, d, o, s)
a in {(18,34), (35,49), (50,64)}
d in {(0,4), (5,9), (10,120)} years since dissolution
o in {divorced, widowed}
s in {female, male}.
```

These are the incumbent bands at
`src/populace_dynamics/models/family_transitions/components/remarriage.py:26-35`
and `src/populace_dynamics/data/transitions.py:137-142`. For each boundary,
`h0_c` is the exact incumbent add-one fit on that boundary's admissible rows:

```text
h0_c = (N_c + wbar) / (E_c + 2*wbar).
```

`R0` returns that table byte for byte. The investigation must assert R0 table,
lookup, random uniforms, projected transitions, and aggregate output are
bit-identical to the incumbent under the same field-aware truncation at every
boundary. Every basis numerator/exposure and every round-2 delta is restricted
by raw age `18 <= age <= 64` before band indexing. For raw ages outside that
range, including 15-17 and 65+, the applied probability is `h0`. This raw-age
guard is load-bearing because the incumbent band helper clips ages 15-17 into
the first `(18,34)` lookup cell
(`src/populace_dynamics/data/transitions.py:110-111`;
`src/populace_dynamics/models/family_transitions/common.py:13-26`). The helper
must branch on raw age before applying a delta and assert conditional
probabilities are bit-identical to R0 outside 18-64.

Every non-remarriage component law and parameter, entry-dissolved carrier
history, event-label rule, support rule, F6 weight, and RNG address function
remains exactly R0. Endogenous downstream states and outputs need not be equal:
a changed working-age remarriage may lawfully change later marriage,
dissolution, widowhood, or age-65+ risk-set composition, and those differences
must be published rather than suppressed.

The nonzero laws pool sex only when constructing their shape basis; the
incumbent sex-specific `h0_c` remains the offset. On boundary-admissible fit rows
define, for origin `o`, YSD band `d`, and working-age band `a`:

```text
q_od  = (N_od  + wbar) / (E_od  + 2*wbar)
q_oad = (N_oad + wbar) / (E_oad + 2*wbar).
```

`N_g` is F6-weighted remarriage numerator, `E_g` is F6-weighted dissolved
exposure, and `wbar` is the same mean positive dissolved-row weight as R0. The
group sums pool sex; `q_od` also pools working ages. No pseudo-future row enters
these quantities. A group requires positive weighted exposure and at least 20
unweighted remarriage events. A law using any group that misses the requirement
is ineligible; no after-the-fact pooling or fallback group is allowed.

Define a fixed origin transfer

```text
u_divorced = -1
u_widowed  = +1.
```

Define the exposure-centered dissolution-recency flattening basis:

```text
m_o       = sum_d E_od * logit(q_od) / sum_d E_od
r_od      = m_o - logit(q_od)
v_od      = r_od / max(1, max_d abs(r_od)).
```

Define the exposure-centered age-within-recency flattening basis:

```text
m_od      = sum_a E_oad * logit(q_oad) / sum_a E_oad
r_oad     = m_od - logit(q_oad)
w_oad     = r_oad / max(1, max_a abs(r_oad)).
```

Both bases have exposure-weighted mean zero in their stated parent and absolute
value no greater than one. Their sign lowers a locally high fitted logit and
raises a locally low fitted logit. This is a predeclared regularizing reshape,
not a claim that round 1 identified the high or low member as an outcome miss.
All group sums and maxima use ascending origin, YSD, age, and sex order with
IEEE-754 float64 arithmetic.

### 3.2 Reference-spell exposure neutrality

Merely constraining `sum_c E_c h_c` would preserve expected event mass on a
fixed risk set, not dissolved exposure. Round 2 instead neutralizes a weighted
expected survivor-area functional.

For boundary `b`, construct `S_b` from every unique, information-admissible
dissolution spell with known person, dissolution year, origin, sex, birth year,
positive F6 weight, and administrative censor no later than the sanitized
information boundary. Entry-dissolved carrier spells are included. Duplicate
`(person_id, dissolution_year, origin)` keys abort. A spell whose observed next
marriage occurs in the same calendar year as its dissolution is explicitly
excluded from `S_b` because it has no entering-year dissolved row; its count and
F6 weight are published. For retained spell `j`, define its potential path and
working-age exposure years as

```text
U_j = {dissolution_year_j + 1, ...,
       min(b, censor_year_j, birth_year_j + 64)}
W_j = {t in U_j: t >= birth_year_j + 18}.
```

The `+1` implements the frozen entering-year convention: a dissolution in year
`t` is an event while the dissolved state begins in the next person-year
(`src/populace_dynamics/data/transitions.py:30-42,377-420`). The reference path
uses the fixed dissolution entry and administrative endpoint and deliberately
does not censor on the observed remarriage whose exit the hazard models. It is a
standardization functional, not a second population projection. Ages 15-17 may
appear in `U_j` only to carry their unchanged R0 survival into age 18; they do
not enter the working-age exposure sum. A valid same-year dissolve-remarry
(`YSD=0`) remains in the realized event numerator and rate but has neither a
row-indexed expected contribution nor a term in this functional or deviance.

For any table `h`, define

```text
A_b(h) = sum_{j in S_b} weight_j *
         sum_{t in W_j} product_{r in U_j: r < t} (1 - h_{j,r}).
```

The survival product is updated recursively in ascending year order; an empty
product is one. For raw ages outside 18-64, `h_jr=h0_jr`. Accumulation order is
ascending `(person_id, dissolution_year, origin, year)` in float64. Publish the
spell count, same-year exclusion count/weight, total included F6 weight,
potential-path count, working-age path-year count, checksum, `A_b(h0)`, and
every candidate value.

For a raw candidate score `z_c`, set

```text
h_jt(alpha) = expit(logit(h0_c(j,t)) + z_c(j,t) + alpha), raw age 18-64
h_jt(alpha) = h0_c(j,t),                                  otherwise.
```

Let `F(alpha) = A_b(h(alpha))/A_b(h0) - 1`. Require finite `A_b(h0)>0` and at
least one positive-weight reference spell with two working-age risk years;
otherwise the law is ineligible. Evaluate `F(-16)` and `F(+16)` in the frozen
arithmetic order. If exactly one endpoint has `abs(F)<=1e-12`, return it; if both
do, declare a flat/nonunique-root failure. Otherwise require
`F(-16)>0>F(+16)`, set `lo=-16` and `hi=+16`, and run this exact float64
bisection for iterations 1 through 200:

1. set `mid = lo + (hi-lo)/2`; midpoint equality with either endpoint is a
   stagnation failure;
2. evaluate `F(mid)`; return the first midpoint with `abs(F(mid))<=1e-12`;
3. if `F(mid)>0`, set `lo=mid`; otherwise set `hi=mid`.

No accepted midpoint by iteration 200, a non-finite value, a failed bracket, or
a flat/nonunique path makes the law ineligible. There is no library-default root
behavior. For the accepted table, define the origin-direction statistic using
R0 fit exposure on working-age cells:

```text
Delta_o = sum_{c: origin=o} E_c * (logit(h_c) - logit(h0_c))
          / sum_{c: origin=o} E_c.
```

The investigation must assert that at least one positive-exposure working-age
cell rises and one falls, `Delta_divorced < -1e-12`, and
`Delta_widowed > +1e-12`.

This constraint preserves weighted expected dissolved exposure on the frozen
reference spells, so the uniform-lowering entailment cannot recur. It does not
guarantee equality of endogenous projected exposure after remarriage,
re-dissolution, death, and risk-set composition feedback. Section 5 therefore
retains the round-1 outer projected-exposure safeguard; the family is no longer
mechanically foreclosed, but no law receives an automatic pass.

### 3.3 The seven laws and frozen order

The strength grid is exactly `k in {0.25, 0.50}`. Those magnitudes bracket a
half-step and a full-step relative to the round-1 divorced-origin train log
errors (approximately 0.51-0.71); they use no holdout quantity. Before the
neutralizing intercept, the laws are:

```text
R0:       z_c = 0                                           exact no-op
R1(k):    z_c = k * u_o                                     origin transfer
R2(k):    z_c = k * (u_o + 0.5*v_od)                        + YSD reshape
R3(k):    z_c = k * (u_o + 0.5*v_od + 0.5*w_oad)            + age/YSD reshape.
```

Thus the closed family has seven laws including R0, six nonzero laws, two fixed
strengths, and no fitted hyperparameter beyond the deterministic fit-side basis
and exposure-neutralizing root. There is no continuous grid and no post-outcome
subgroup, sign, centering, normalization, minimum-count, root bracket, or cap
choice.

The frozen simplicity order is

```text
R1(0.25) < R1(0.50) < R2(0.25) < R2(0.50)
         < R3(0.25) < R3(0.50).
```

Within a structural form, the smaller move is simpler. An origin-only law is
simpler than adding a fitted recency basis, and the recency law is simpler than
adding age-by-recency structure. Exact numeric ties use this order.

### 3.4 Numerator-side timing laws: only through the hazard

Standalone numerator thinning, relabeling, event-year shifting, censor/window
movement, and count-only calibration are outside this family. A valid
remarriage event must coincide with leaving dissolved state and opening the next
marriage; changing its numerator without changing the corresponding state path
would violate the landed carrier/event conformance law and contaminate downstream
family and auxiliary-benefit histories. Event construction is already pinned by
the dissolve-before-marry tie order and entering-year convention
(`src/populace_dynamics/data/transitions.py:307-315,377-420,473-509`). Any change
to those semantics would leave this learning plan and require the separate
narrow §2.8 ceremony named at
`docs/design/m6_candidate2_program.md:835-840,946-954`.

Timing is in scope only in the state-consistent sense embodied by R2/R3: changing
the YSD hazard can move a remarriage and its dissolved-state exit together while
the survivor-area constraint holds. The pre-repair [transport forensics][forensics4]
show why severing event labels from state/history is not harmless. Same-year
`YSD=0` events remain in every realized/truth numerator and rate, have no
row-indexed expected contribution, and remain excluded from survivor-area and
deviance calculations because no entering-year dissolved row exists. Their
count and F6 weight are always published.

## 4. Frozen pseudo-holdouts and information dating

Round 2 copies round 1's field-aware `≤T*` construction verbatim. The only
intentional Monte Carlo change is the fresh, predeclared seed bank below; no new
window, anchor, support, or dating convention is introduced.

The expanding boundaries are exactly `{2006, 2008, 2010}`. At boundary `b`:

1. Apply the existing conservative refit truncation. A fit row is admissible
   only if its state/event year is `<=b`; annual marital flows additionally
   require the interview that establishes the row to be `<=b`.
2. Fit all unchanged family components on admissible rows. Only the remarriage
   table varies across R0-R3.
3. Form the shifted §2.8 anchor from the earliest positive-weight interview in
   `{b+1,b+3}`. Hold its household and F6 weight fixed.
4. Seed the realized marital state at that anchor, including the landed
   entry-dissolved carrier history, and project through `b+4`.
5. Evaluate `{b+1,...,b+4}` only when the interview required to establish the
   annual row is also no later than 2014, on symmetric opening-wave presence
   support pooled over sex and ages 18-64.

The effective windows remain 2007-2010, 2009-2012, and 2011-2013. Calendar 2014
is excluded because its establishing PSID interview is in 2015. The field-aware
refit machinery and its marriage-event/person-year dating live at
`src/populace_dynamics/engine/refit.py:262-296,351-406,507-544`; this plan changes
neither their law nor the incumbent-spec refit rule at
`src/populace_dynamics/engine/refit.py:599-613`.

The retrospective sources are read-only and sanitized by exclusion before any
panel, fit, reference-spell functional, truth reduction, or projection can see
them:

- demographic rows are limited to interviews no later than 2013;
- a marriage or birth event survives only if both its event year and its
  establishing-report year are no later than 2014;
- surviving report years are clipped, post-2014 marriage end/separation fields
  are nulled, and `n_marriages` is recomputed from the surviving records; and
- deaths after 2014 are nulled.

At each boundary, the same operations are tightened to `b`. The helper must
disclose that this is not a contemporaneous pre-2015 snapshot: whether a
pre-2015 marriage survives can be correlated with later panel continuation
because a record last established after 2014 is excluded. The convention is
still internally valid because it is exclusion-only and R0 is required to be
bit-equivalent under that same sanitized composition.

The round-2 analysis seeds are exactly `{7240,...,7279}`. All seven laws use
common random numbers. The fixed stability blocks are `{7240,...,7259}` and
`{7260,...,7279}`. This fresh contiguous bank avoids selecting new laws on the
same Monte Carlo realizations whose round-1 summaries informed this proposal.
The old `{7200,...,7239}` outcomes remain admissible published motivation, but
no round-2 law may be previewed on them before selection. For seed label `s`,
instantiate `ProjectionRNGRegistry(draw_index=s-5200,
n_periods=max(evaluation_years)-b)`, use
`generator(0, ProjectionModule.MARITAL_CORE)` for transition uniforms, and use
`child_generator(0, ProjectionModule.MARITAL_CORE, 1)` for spouse-gap draws.
This is the exact round-1 address mapping and the inherited root is therefore
`s` (`src/populace_dynamics/engine/rng.py:16-17,37-99`).

Candidate enumeration/order may not alter those addresses or insert a
variant-specific RNG call. The transition-uniform generator starts from the
same state for every law. Changed transitions can lawfully change which people
consume spouse-gap draws and can change downstream active sets and outputs; the
findings publish those differences while asserting the registry code,
component-child addresses, and raw transition-uniform checksum are unchanged.

## 5. Frozen diagnostics, objective, and selector

### 5.1 Publication record

For each boundary and law, the findings must publish, regardless of disposition:

- every fit/support maximum, group count, weighted exposure/numerator, `wbar`,
  `q`, basis value, cell probability, thin/ineligible group, checksum,
  neutralizer `alpha`, and reference-spell `A_b` equality;
- pseudo-future truth dissolved exposure, event numerator, and rate;
- direct-standardized expected numerator and rate from applying the law to the
  realized dissolved rows, plus F6-weighted Bernoulli deviance;
- each of the 40 projected exposure, numerator, and rate records, their mean,
  both fixed-block means, and projected/truth ratios and natural logs;
- the same direct and projected records by origin and by the frozen
  origin-by-working-age-by-YSD groups, including zero-event groups as explicit
  zeroes/null ratios rather than silently removing them; and
- exact support, entry-carrier state/YSD survival, event-label, common-random-
  number, and R0-equivalence checks.

A valid unmatched same-year dissolve-remarry event enters the pseudo-truth
event numerator and rate but has no row-indexed expected contribution. For law
`L` at boundary `b`, freeze the direct quantities as

```text
expected_numerator_Lb = sum_i weight_i * h_L,c(i)
actual_numerator_b = sum_i weight_i * event_i
                     + unmatched_same_year_event_weight_b
expected_rate_Lb = expected_numerator_Lb / sum_i weight_i
actual_rate_b = actual_numerator_b / sum_i weight_i,
```

where `i` indexes entering-year dissolved risk rows and matched events. A valid
same-year event is excluded from row-indexed deviance and `A_b`, with its count
and F6 weight published. Any unmatched nonzero-YSD event aborts. Boundary
F6-weighted Bernoulli deviance is

```text
D_Lb = -2 * sum_i weight_i *
        (event_i*log(h_L,c(i)) + (1-event_i)*log(1-h_L,c(i)))
        / sum_i weight_i.
```

Origin deviances use the identical restriction and normalization. Pooled
deviance is the sum of the three pre-normalization weighted deviance numerators
divided by the sum of their three weight denominators, not an equal-boundary
mean. These definitions are frozen before helper contact; round 2 permits no
post-contact "outcome-neutral" diagnostic amendment.

The committed findings ledger may remove person identifiers and repetitive
per-seed rows only through a committed deterministic reducer. It must retain
the complete aggregate cell ledger, every selector input, the SHA-256 of full
JSON stdout, the reducer version/hash, and enough information for a referee to
recompute every eligibility bit and the selected law without rerunning the
model.

### 5.2 Primary loss and Monte Carlo uncertainty

No diagnostic is compared with a gate tolerance. For law `L`, boundary `b`, and
seed `s`, let `r_Lbs` be the projected pooled remarriage rate and `r_truth,b` the
field-aware pseudo-holdout truth rate. Define

```text
rbar_Lb = mean_s r_Lbs
J(L)    = (1/3) * sum_b log(rbar_Lb / r_truth,b)^2.
```

For projected exposure, define before any eligibility comparison

```text
ebar_Lb  = mean_s e_Lbs
ebar_Lob = mean_s e_Lobs

rate_error_Lb = abs(log(rbar_Lb / r_truth,b))
exposure_error_Lb = abs(log(ebar_Lb / e_truth,b))
widow_exposure_error_Lb =
    abs(log(ebar_L,widowed,b / e_truth,widowed,b)).
```

Rules 4, 5, and 7 use these full-40-seed quantities. They do not average
per-seed ratios or per-seed log errors.

Each fixed-block loss uses the identical formula with its 20-seed mean. For a
Monte Carlo standard error on the same statistic, delete the same seed label
`s` from all three boundaries, recompute all three 39-seed rate means and
`J_(-s)(L)`, and never re-select within a replicate. Let `Jjack_bar` be the mean
of those 40 values and define the frozen delete-one jackknife

```text
SE_J(L) = sqrt((39/40) * sum_s (J_(-s)(L) - Jjack_bar)^2).
```

This one-SE quantity measures Monte Carlo uncertainty only. The overlapping
calendar windows do not supply three independent sampling units, and the
findings may not describe `SE_J` as data-sampling uncertainty.

For every selector comparison define `x <* y` as `x < y - 1e-12` and `x <=* y`
as `x <= y + 1e-12`. This single absolute comparison tolerance applies to all
loss, deviance, and absolute-log-error comparisons below. It is not a gate
tolerance and never references the 2015-2019 band. These conventions govern
eligibility and one-SE cutoff membership, not the `Lbest` argmin: `Lbest` is the
exact float64 minimum, and only bit-equal `J` values are ties resolved by the
frozen simplicity order.

### 5.3 Eligibility rules

R0 is the baseline only if all information, support, conformance, checksum, and
bit-equivalence assertions pass; otherwise the investigation aborts rather
than manufacturing a no-op result. A nonzero law is eligible only if all seven
rules pass:

1. **Defined and conformant.** Every quantity consumed by the objective or
   rules 2-7 is finite at all three boundaries, and every argument to a
   logarithm is strictly positive; the pooled truth rates used by `J`,
   minimum-event rules, exact person-year support, carrier, event-label, RNG,
   and R0-equivalence checks pass. A predeclared publication-only subgroup ratio
   may be null when its truth numerator is zero; no selector rule may read that
   null, and it does not cause ineligibility.
2. **Exposure-neutral and sign-balanced by construction.** At every boundary,
   `abs(A_b(h)/A_b(h0)-1) <= 1e-12`; at least one positive-exposure cell rises
   and one falls; `Delta_divorced < -1e-12` and
   `Delta_widowed > +1e-12`.
3. **Full and block loss.** Full-draw `J` and both fixed-block `J` values are
   `<*` R0.
4. **Boundary rate transport.** Absolute projected/truth pooled log-rate error
   (`rate_error_Lb`) is `<*` R0 at least two boundaries and `<=*` R0 at the
   remaining boundary.
5. **Endogenous exposure protection.** `exposure_error_Lb` is `<=*` R0 at every
   boundary. Reference-spell neutrality does not waive this outer safeguard.
6. **Direct fit.** Pooled direct-standardized deviance is `<*` R0, and boundary
   deviance is `<*` R0 at least two boundaries and `<=*` R0 at the third.
7. **Origin protection.** Divorced-origin direct deviance is `<*` R0 at least
   two boundaries and `<=*` R0 at the third; widowed-origin direct deviance and
   `widow_exposure_error_Lb` are `<=*` R0 at every boundary. Deviance, rather
   than a widowed log-rate, makes the predeclared rule defined when the 2010
   pseudo-window has zero widowed truth events.

These are conjunctions, not a menu. No law can trade a failed exposure or
widowed check for a lower pooled `J`, and no failed group can be removed after
the fact.

### 5.4 One-SE/no-op selector and final information fit

If there is no eligible nonzero law, select R0 and return
`NO_OP_DESIGNED_PAUSE`. Otherwise:

1. Let `Lbest` be the eligible nonzero law with minimum full-draw `J`; exact
   ties use the frozen simplicity order.
2. Set `cutoff = J(Lbest) + SE_J(Lbest)`.
3. If `J(R0) <= cutoff + 1e-12`, select R0. This is the explicit
   no-op-favoring one-SE rule.
4. Otherwise select the first eligible law in the frozen simplicity order with
   `J(L) <= cutoff + 1e-12`.

This wording replaces round 1's unexercised ambiguous phrase: R0 wins whenever
it is within one Monte Carlo standard error of the best eligible nonzero law,
not only when it has the absolute minimum.

If a nonzero structure and strength are selected, fit that exact law once on
all information admissible at the 2014 boundary. The field-aware effective
maximum remains 2013 because calendar-2014 marital flow requires the forbidden
2015 interview. Recompute `h0`, the selected fit-side bases, `S_2014`, and its
neutralizing `alpha` by the identical rules; do not re-select. Publish the final
60 probabilities, basis inputs, root ledger, hashes, and exact no-op comparison.
Reapply every law-definition, group-support, probability, root, neutrality, and
direction guard. If any guard fails, do not substitute another law, change a
strength, or rerun selection: record the pseudo-holdout selection, publish the
final-fit failure, and continue the designed pause under section 7. A passing
final estimate is still a finding, not authority to edit the registry.

## 6. Freeze commit and runtime enforcement

This proposal is not itself ratified. The fable referee may require prospective
edits while no round-2 outcome exists. Once the plan is accepted, its ratifying
merge commit is the selector-freeze commit. That commit must precede any helper
that reads a real remarriage outcome, fits a round-2 law, or projects a
pseudo-holdout.

The investigation lane must reproduce round 1's commit discipline exactly:

1. record the ratified design commit and SHA-256 of this complete Markdown file
   in a protocol-lock commit;
2. implement the helper mechanically in a later commit, exercising only
   synthetic fixtures before committing it;
3. obtain a pre-execution freeze audit confirming that laws, constants,
   pseudo-boundaries, dating, seeds, diagnostics, selector, tie-breaks, and
   no-op disposition still match the ratified file; and only then
4. run the helper on the read-only staged sources and publish full stdout plus
   the deterministic aggregate ledger regardless of outcome.

Runtime must assert, before selection:

- embedded `design_commit` and protocol-file SHA-256 equal the ratified lock;
- fit state/event and required-interview maxima are `<=b` (normally `b-1`),
  every evaluation year is `<=2013`, and every evaluation required interview
  is `<=2014`;
- the sanitized source maxima, exclusion counts, clipped/null field counts,
  recomputed-marriage counts, and source hashes equal the published input audit;
- no 2015+ row, truth moment, candidate output, macro value, score, or tolerance
  is reachable by any fit, basis, root, diagnostic, objective, eligibility bit,
  or tie-break;
- the boundary list, law list/order, strengths, minimum counts, root bracket,
  arithmetic tolerance, seed set, blocks, and CRN checksums equal this plan;
- R0 is bit-equivalent; raw-age-outside-18-64 conditional probabilities,
  non-remarriage component laws/parameters, carrier/event semantics, and RNG
  address functions are unchanged; endogenous downstream differences are
  published; exact support/carrier/YSD/event-label assertions pass; and the
  selected law is a pure function of the frozen ledger; and
- the helper never imports or calls the M6 scorer, never reads `gates.yaml`,
  never reads a `runs/` artifact, writes no file, and emits JSON only to stdout.

The helper's dependency/import audit and file-open audit must be included in the
findings. If any protocol, diagnostic, or selector definition proves defective
after first real-outcome contact, stop and disclose the contact and defect. The
existing output is not silently repurposed: amend the proposal, obtain a new
referee decision and freeze commit, and begin a new investigation. Calling an
edit outcome-neutral is not an exemption.

## 7. Frozen no-op ladder

A round-2 R0 selection, an empty eligible set, an undefined required input, or a
law the findings referee declines to ratify leaves the designed pause active.
Registration 8 remains forbidden under
`docs/design/m6_candidate2_program.md:971-976,1038-1046`; no nonzero law may be
forced and the holdout may not break the tie.

The next rung is fixed now rather than chosen after seeing round 2: publish a
separate W1-style, candidate-blind surface-question proposal for
`remarriage.18-64`. It must ask whether one pooled working-age flow can honestly
certify an origin-heterogeneous formation process while protecting divorced and
widowed dissolved stocks. The question uses truth-only power, vacuity, and
adjacent-pooling ladders; accumulated candidate-1 conformance forensics and both
train-only no-ops may motivate it but cannot set a cell, reducer, tolerance, or
floor. Any origin/stock surface change requires its own prospective §2.8 and
floor derivation/reproduction/lock ceremony and cannot rescue candidate 1 or
retroactively change either learning result.

There is no automatic third hazard family. A later family can exist only if the
surface referee explicitly returns the question to model-law learning and a
new, separately frozen proposal names that family without reading a post-2014
outcome. Until either route is independently ratified and completed, the pause
continues.

## 8. Ceremony and possible disposition

The governance sequence is:

1. **Proposal:** this docs-only draft names the family, selector, freeze
   discipline, and no-op ladder; it runs nothing.
2. **Plan referee:** the fable referee reviews the structural opening,
   bright-line isolation, complete parameterization, exposure functional,
   selector, and scope. Required edits occur before outcome contact.
3. **Ratify and freeze:** merge the accepted plan and record its commit/file
   hash. No investigation starts earlier.
4. **Round-2 lane:** mechanically implement under the frozen selector, pass the
   synthetic/pre-execution audit, execute once on the train-only domain, and
   publish full findings and reducer-derived ledger regardless of result.
5. **Findings referee:** independently reproduce or recompute the ledger,
   verify the information boundary and freeze history, and adjudicate the
   selector's disposition.
6. **Disposition:** a verified nonzero law returns through a separate
   prospective amendment, implementation, tests, immutable registry spec, and
   candidate-2 lock before registration. R0 or an unratifiable law continues
   the pause and triggers section 7's surface-question rung.

Selection is learning, not self-executing authorization. Even a verified
nonzero winner cannot enter candidate 2 directly from the findings branch.

## 9. What this proposal does not change

This proposal does **not**:

- alter candidate 1's frozen, valid FAIL, its artifact, its 0/5 result, or any
  historical claim;
- edit candidate 2, its current remarriage law or immutable registry spec, or
  authorize registration 8;
- edit `gates.yaml`, `runs/`, a model/test/source file, any truth support,
  reducer, F6 weight, cell, tolerance, seed, draw count, conjunction, floor, or
  artifact;
- change `remarriage.18-64`, its `0.403` tolerance, or use that tolerance in
  selection;
- alter the landed #226 entry-dissolved conformance repair, event labels,
  same-year convention, first marriage, divorce, widowhood, or the conditional
  remarriage probability at raw ages outside 18-64;
- change or delay candidate-2 amendment 4 or the floors-v4 lane; both proceed
  independently under their own ratification and lock ceremonies; or
- authorize a third family or a W1-style surface change without the separate
  proposal and governance in section 7.

The program's candidate-1 and surface boundaries remain those at
`docs/design/m6_candidate2_program.md:1048-1069`. Round 2 supplies only a
pre-outcome learning protocol for resolving the active remarriage pause.

## 10. Decisions for the fable referee

The plan referee must explicitly decide whether to:

1. accept the VERIFIED round-1 structural entailment as the reason an
   exposure-neutral/sign-balanced successor is a distinct family;
2. ratify R0-R3, both strengths, group guards, basis/centering rules, survivor-
   area functional, root algorithm, direction checks, and frozen simplicity
   order without adding an after-the-fact branch;
3. approve the exact round-1 field-aware pseudo-holdouts with the fresh pinned
   `{7240,...,7279}` CRN bank and two fixed blocks;
4. ratify the seven eligibility rules, delete-one jackknife, explicit R0-favoring
   one-SE rule, direct/origin protections, and comparison arithmetic;
5. keep standalone numerator/timing edits outside the family while retaining
   state-consistent YSD reshaping and the exact `YSD=0` diagnostic convention;
6. require the ratified merge/file hash, pre-execution freeze audit, runtime
   assertions, full-output/reducer ledger, and stop/re-freeze rule; and
7. ratify the deterministic no-op ladder to the separate W1-style surface
   question, with the candidate-2 pause active throughout.

## 11. Non-operative consistency ledger

This block is a review aid, not executable configuration and not a gate edit:

```json
{
  "schema": "m6.remarriage.learning_plan.round2.proposal.v1",
  "status": "proposal_unratified_no_execution",
  "authority": {
    "program": "docs/design/m6_candidate2_program.md:842-872,956-994,1038-1046",
    "round1_pr": 231,
    "round1_referee_issue_comment": 5003093793
  },
  "evidence_cutoff": 2014,
  "pseudo_boundaries": [2006, 2008, 2010],
  "effective_windows": [[2007, 2010], [2009, 2012], [2011, 2013]],
  "selection_seeds": {"inclusive_start": 7240, "inclusive_end": 7279},
  "selection_seed_blocks": [[7240, 7259], [7260, 7279]],
  "family": {
    "no_op": "R0",
    "forms": ["R1_origin", "R2_origin_ysd", "R3_origin_age_ysd"],
    "strengths": [0.25, 0.5],
    "nonzero_laws": 6,
    "raw_age_delta_domain": [18, 64],
    "outside_raw_age_domain": "exact_R0_probability",
    "reference_exposure_relative_tolerance": 1e-12,
    "root_bracket": [-16, 16],
    "root_max_iterations": 200,
    "group_min_unweighted_events": 20
  },
  "selector": {
    "objective": "mean_boundary_squared_log_pooled_rate_error",
    "uncertainty": "40-seed_delete-one_jackknife",
    "comparison_tolerance": 1e-12,
    "no_op_rule": "select_R0_if_no_eligible_nonzero_or_J_R0_le_best_plus_SE",
    "selected_outcome_if_no_op": "NO_OP_DESIGNED_PAUSE"
  },
  "bright_line": "no_2015_2019_data_in_selection",
  "freeze": "ratified_commit_and_file_sha256_before_any_round2_outcome",
  "no_op_next_rung": "separate_candidate_blind_W1_style_remarriage_surface_question",
  "scope": "docs_only_no_candidate_gate_floor_run_or_code_change"
}
```

## References

- Ratified candidate-2 program:
  `docs/design/m6_candidate2_program.md`.
- Round-1 findings: [PR #231][round1-pr].
- Round-1 VERIFIED referee record: issue-comment
  [5003093793][round1-referee].
- Pre-repair transport mechanism, motivation only: issue #42 issue-comment
  [4997635883][forensics4].

[round1-pr]: https://github.com/PolicyEngine/populace-dynamics/pull/231
[round1-referee]: https://github.com/PolicyEngine/populace-dynamics/pull/231#issuecomment-5003093793
[forensics4]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4997635883
