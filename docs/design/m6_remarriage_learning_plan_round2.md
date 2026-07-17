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
boundary. Ages 65+, every non-remarriage component, entry-dissolved carrier
history, event labels, support, F6 weights, and RNG addresses remain exactly R0
under every round-2 law.

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
`(person_id, dissolution_year, origin)` keys abort. For spell `j`, the potential
working-age dissolved path is

```text
T_j = {dissolution_year_j + 1, ...,
       min(b, censor_year_j, birth_year_j + 64)}.
```

The `+1` implements the frozen entering-year convention: a dissolution in year
`t` is an event while the dissolved state begins in the next person-year
(`src/populace_dynamics/data/transitions.py:30-42,377-420`). The reference path
uses the fixed dissolution entry and administrative endpoint and deliberately
does not censor on the observed remarriage whose exit the hazard models. It is a
standardization functional, not a second population projection. A valid
same-year dissolve-remarry (`YSD=0`) has no entering-year dissolved row and is
excluded from this row-indexed functional exactly as section 5 excludes it from
row-indexed deviance; it remains in every event numerator and rate.

For any table `h`, define

```text
A_b(h) = sum_{j in S_b} weight_j *
         sum_{t in T_j} product_{r in T_j: r < t} (1 - h_{c(j,r)}).
```

The empty product at the first at-risk year is one. Cells after age 64 use R0
and do not enter `A_b`. Accumulation order is ascending `(person_id,
dissolution_year, origin, year)` in float64. Publish the spell count, total F6
weight, path-year count, checksum, `A_b(h0)`, and every candidate value.

For a raw candidate score `z_c`, set

```text
h_c(alpha) = expit(logit(h0_c) + z_c + alpha),  age 18-64
h_c(alpha) = h0_c,                              age 65+.
```

Choose the unique scalar `alpha` that solves `A_b(h(alpha)) = A_b(h0)` by
deterministic bisection on `[-16, +16]`, for at most 200 iterations. The root is
accepted only when the endpoints bracket it and
`abs(A_b(h) / A_b(h0) - 1) <= 1e-12`; otherwise the law is ineligible. The
investigation must assert that at least one positive-exposure working-age cell
rises and one falls and that the fit-exposure-weighted mean logit delta is
strictly negative for divorced cells and strictly positive for widowed cells.

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
the survivor-area constraint holds. Same-year `YSD=0` events remain in every
numerator/rate diagnostic and remain excluded only from the row-indexed
survivor-area and deviance calculations because no entering-year dissolved row
exists. Their count and F6 weight are always published.

[round1-pr]: https://github.com/PolicyEngine/populace-dynamics/pull/231
[round1-referee]: https://github.com/PolicyEngine/populace-dynamics/pull/231#issuecomment-5003093793
