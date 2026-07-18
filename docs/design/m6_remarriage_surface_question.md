# M6 remarriage surface question: can one pooled flow certify two origins?

Status: **PROPOSAL ONLY — NO SURFACE DISPOSITION, GATE EDIT, OR
REGISTRATION AUTHORITY**.

This is the fixed next rung after the ratified R0 / `NO_OP_DESIGNED_PAUSE`
finding.  It proposes the candidate-blind referee question and the truth-only
instruments needed to answer it.  It does not run those instruments, select a
surface, or change a byte of the candidate or gate contract.

## 1. Authority and the exact question

The governing charge is
`docs/design/m6_remarriage_learning_plan_round2.md:788-799`:

> The next rung remains fixed: publish a separate W1-style, candidate-blind
> surface-question proposal for `remarriage.18-64`. It asks whether one pooled
> working-age flow can certify an origin-heterogeneous formation process while
> protecting divorced and widowed stocks. It uses truth-only power, vacuity,
> and adjacent-pooling ladders. Candidate conformance forensics and both
> train-only rounds may motivate the question but cannot set a cell, reducer,
> tolerance, or floor. Any surface change needs its own prospective §2.8 and
> floor ceremony and cannot retroactively change either learning result.
>
> There is no automatic third hazard family. A later family requires the
> surface referee to return the question to model-law learning and a new
> proposal frozen without post-2014 outcomes. The pause continues throughout.

The question is therefore not whether a known candidate can clear the current
cell.  It is:

> On information available by the 2014 cutoff, does the pooled working-age
> remarriage flow have enough identification, power, and stock sensitivity to
> support its certification claim, or must governance change the surface or
> return to model-law learning?

The current cell remains `remarriage.18-64`.  Its tolerance remains `0.403`.
That number is an immutable contract fact, not an input here: no surface
instrument may load it, compare with it, rank by it, or use it as a stop rule.

## 2. Why the question is timely, but not targeted

The two ratified train-only rounds and the completed conformance investigation
motivate asking whether origin pooling is the right certification surface.  The
second round also demonstrated that the working-age stock guard was
load-bearing, and the six-ceremony reducer ladder ended in a byte-reproduced
ratified no-op without forcing a law.  Those are narrative reasons to ask the
question only.

No candidate result supplies a target.  In particular, this proposal does not
import a candidate-1 score, a scored residual, an origin residual, a rate ratio,
a reducer choice, a tolerance, or a floor from either learning round or the
conformance record.  The ratified no-op remains a learning result; a later
surface finding cannot reinterpret it as a successful or failed untested law.

This is the W1 discipline: diagnose the claim and the power of the surface
prospectively, preserve every historical verdict, and place any ratifying flip
in a later, separate ceremony.

## 3. Candidate-blind evidence firewall

The authorized surface-question analysis may read only truth inputs whose
information was established by the 2014 cutoff, the source code needed to
construct those truth records, and this ratified design lineage.  It must not
instantiate a projection engine or a candidate.

### 3.1 Allowed truth inputs

The analysis rebuilds its inputs from the readers underlying:

- `populace_dynamics.data.panels.demographic_panel()`;
- `populace_dynamics.data.marriage.marriage_history()`;
- `populace_dynamics.data.deaths.read_death_records()`; and
- `populace_dynamics.data.transitions.build_marital_panel()`.

The field-aware dating rule is the one documented at
`docs/design/m6_remarriage_learning_plan_round2.md:430-468`:

- a demographic interview must be no later than 2013;
- a marital event and its establishing report must be no later than 2014;
- later end or separation fields are nulled before construction;
- later deaths are nulled; and
- any annual interval requiring a later establishing interview is excluded.

The analysis uses exactly three non-overlapping five-year windows:

```text
W1 = [1999, 2000, 2001, 2002, 2003]
W2 = [2004, 2005, 2006, 2007, 2008]
W3 = [2009, 2010, 2011, 2012, 2013].
```

Five annual flow intervals match the registered flow-score horizon.  Working
backward from the last cutoff-safe interval creates the three most recent
disjoint windows without reading an event, rate, candidate result, or either
train-only window.  No row occurs in two windows.  Calendar 2014 is excluded
because its establishing PSID interview is later than the cutoff.

For year `y`, its opening wave is the greatest actual PSID interview year not
later than `y`; require that it be `y` or `y - 1`.  A person or same-year event
contributes only when that person is present with positive weight in the
event-year opening wave.  There is no rolling, expanding, or overlapping
evaluation window.

The exclusions occur before a rate, count, partition, floor diagnostic, or
stock quantity is computed.  The analysis publishes source hashes, retained
and excluded counts by reason, the last admissible field date, and a static
file-open and import audit.

### 3.2 Prohibited contact

The analysis must not read or import:

- a candidate artifact, candidate registry specification, or projection
  output;
- either train-only results artifact or any candidate-outcome field in its
  reports;
- the M6 scorer, runner, or projection engine;
- `gates.yaml` or a `runs/` floor artifact; or
- any row, score, residual, or truth moment established after the cutoff.

The helper must contain no candidate module import and no general-purpose path
argument.  Its output is a new, never-overwritten truth-only artifact.  A
static import audit, file-open audit, and runtime denylist are required to make
the firewall machine-checkable.

The prior plans may be cited for information dating and governance.  Their
outcomes do not enter any cell, reducer, diagnostic bar, or disposition bit.

## 4. Canonical truth estimands

The truth builder emits separate risk and event tables.  The risk table has one
record for each admissible annual person interval:

```text
(person_id, analysis_household_id, year, raw_age, sex, origin,
 at_risk, weight)
```

`origin` is the marital state at the start of the interval and is exactly one
of `divorced` or `widowed`.  `at_risk` is one only for that dissolved state.
Raw age is restricted before band assignment:

```text
18 <= raw_age <= 64.
```

Presence is conditioned symmetrically on the opening admissible interview.  In
each window, a person's F6 weight comes from that person's first
positive-weight opening interview for the window and remains fixed for all
five intervals.  A person without a window anchor is excluded from both tables
in that window.

The correlation unit is not a person's first household.  Build an undirected
graph over every person appearing in any of the three windows.  Add an edge
when two people share an opening-wave PSID household in a window or are the two
members of a certified marriage, remarriage, divorce, or widowhood record whose
event year is in a window.  `analysis_household_id` is the minimum `person_id`
in the transitive connected component.  The component is constructed once
over all three windows and is immutable across seeds and windows.  This closes
later-forming couples, former partners, and repeated persons onto one split
side.  The artifact publishes every edge class, component size, and checksum.

The event table has one record per certified remarriage event:

```text
(person_id, analysis_household_id, year, raw_age, sex,
 entering_origin, has_entering_risk_row, weight)
```

Ordinary events take `entering_origin` from the start-of-year risk row.  Every
event must satisfy the same event-year opening-wave presence and positive
window-anchor rule as its risk table.  A same-calendar-year dissolution
followed by remarriage has no entering
dissolved row.  It remains in the numerator, as the ratified event convention
requires, and takes its origin from the certified dissolution type.  Ambiguous
origin, duplicate event, or a non-same-year event without a matching risk row
is a designed abort.  The artifact publishes same-year event count and weight
by origin separately from row-matched events.

For a truth cell `c`, define

```text
E_c = sum_{i in risk_c} weight_i * at_risk_i
N_c = sum_{j in events_c} weight_j
r_c = N_c / E_c.
```

The artifact also publishes unweighted at-risk rows and events.  A rate is
undefined if its exposure or numerator is nonpositive; undefined values are
never silently smoothed, pooled after the fact, or replaced with zero.

## 5. The exhaustive origin-preserving adjacent-pooling ladder

The registered working-age atoms are ordered by age:

```text
[18-29], [30-44], [45-64].
```

The evaluated surface crosses those atoms with sex and entering origin.  The
ladder never treats divorced and widowed as adjacent.  Origin pooling occurs
only in the incumbent pooled control, never inside a proposed origin-aware
rung.

For each origin, enumerate these rungs in order:

1. age atom by sex;
2. sex-pooled age atoms;
3. every two-block contiguous age partition after sex pooling:
   `[18-29] | [30-64]` and `[18-44] | [45-64]`; and
4. the sex-pooled working-age block `[18-64]`.

Every contiguous partition is evaluated.  A hand-picked subset is invalid;
the v2-to-v3 M6 floor correction showed why enumeration completeness is a
governance requirement
(`docs/design/m6_projection_engine.md:3421-3424,3430-3436`).

A rung covers an origin only if every age in the working-age domain belongs to
exactly one required cell.  It is not enough for one convenient cell at a rung
to clear.  A joint surface covers the question only if both origins have full
coverage.

Name the rungs `0`, `1`, `2a`, `2b`, and `3` in the written order and map them
to order ranks `0`, `1`, `2`, `3`, and `4`.  Enumerate the full Cartesian
product of the five divorced rungs and five widowed rungs.  Mixed rungs are
allowed: power may support a finer divorced partition than widowed partition,
or the reverse.

Order the twenty-five joint surfaces by this total key:

```text
(-number_of_required_cells,
 max(divorced_rung_rank, widowed_rung_rank),
 divorced_rung_rank,
 widowed_rung_rank)
```

This is the complete joint order; no truth result changes it.

The analysis publishes all twenty-five joint surfaces, including failed and
undefined surfaces.  `SELECTED_ORIGIN_SURFACE` is the first surface in this
fixed order that clears §7.4's exact coverage-and-power predicate.  If none
clears, it is null and `ORIGIN_SURFACE_READY` is false.

## 6. Truth-only power ladder

The power calculation follows the pre-lock M6 precedent at
`docs/design/m6_projection_engine.md:3283-3303,3342-3350` but is rebuilt on the
cutoff-safe truth above.  It does not read a frozen M6 floor.

For split seeds `0` through `99`, assign fixed correlation components, not
persons, to two equal-probability halves.  Use the same component assignment in
all three windows.  For every window `w` and required cell compute

```text
r_Acws = N_Acws / E_Acws
r_Bcws = N_Bcws / E_Bcws
d_cws  = abs(log(r_Acws / r_Bcws)).
```

For cell `c` and window `w`, over all defined seeds compute

```text
m_cw     = mean_s(d_cws)
s_cw     = sample_sd_s(d_cws)
sigma_cw = sqrt(m_cw**2 + s_cw**2)
u_cw     = decimal_half_even(m_cw + 3 * s_cw, places=3).
```

`u_cw` is a non-operative diagnostic bar, not a proposed gate tolerance.  The
artifact retains its unrounded binary64 operands, converts them through their
decimal strings, and quantizes to `Decimal("0.001")` with `ROUND_HALF_EVEN`,
matching `evaluation.derive_tolerance`.  It publishes the exact conversion and
rounding record.

A cell is power-admissible only when all of these hold in every window:

- all one hundred half-split scores are defined;
- every split has at least twenty unweighted certified event-table records in
  its weaker half; F6-weighted event mass cannot satisfy this count;
- `sigma_cw` is finite and positive; and
- the uncapped diagnostic bar is strictly below `ln(1.5)`.

For a complete evaluated surface, compute its draw-noise-free half-normal
operating characteristic separately in each window:

```text
p_cw    = 2 * Phi(u_cw / sigma_cw) - 1
p_seed,w = product_c(p_cw)
p_4of5,w = sum_{j=4..5} choose(5, j) * p_seed,w**j
           * (1 - p_seed,w)**(5 - j).
```

The surface is not power-admissible unless `p_4of5,w >= 0.90` in all three
windows.  There is no pooling of events, sigmas, or operating characteristics
across windows.  This is a diagnostic application of the standing weak-power
rule, not a floor lock.  The artifact publishes the per-cell, per-window, and
surface arithmetic for every rung.

The incumbent pooled control is recomputed through the same truth-only
machinery.  Its current tolerance is neither read nor substituted for `u_cw`.

## 7. Vacuity and origin-heterogeneity tests

Power alone does not establish that an origin split is informative.  The
surface artifact therefore answers both statistical and structural vacuity.

### 7.1 Household-clustered origin signal

Compute an age-and-sex-standardized truth contrast independently of the
selected origin rung.  Let `g` range over the six registered raw age-by-sex
atoms.  In window `w`, define common-overlap weights

```text
a_gw = min(E_divorced,gw, E_widowed,gw)
lambda_gw = a_gw / sum_g(a_gw).
```

A stratum with zero exposure for either origin receives zero weight.  Each
sum below runs only over positive-overlap strata.  Each window must retain
positive common exposure and positive standardized rates:

```text
r_std,ow = sum_g(lambda_gw * N_ogw / E_ogw)
theta_w = log(r_std,divorced,w / r_std,widowed,w)
theta = (theta_W1 + theta_W2 + theta_W3) / 3.
```

Equal window weight prevents a larger historical sample from setting the
signal.  For each unique `analysis_household_id`, delete that entire connected
component from all three windows and recompute the overlap weights, the three
window contrasts, and `theta`.  With `H` components and mean delete-one
estimate `theta_bar`, the fixed household-cluster jackknife standard error is

```text
se_theta = sqrt((H - 1) / H
                * sum_h((theta_without_h - theta_bar)**2)).
```

Every delete-one standardized rate must be positive and finite.
`ORIGIN_SIGNAL` is true exactly when the two-sided interval

```text
[theta - 1.96 * se_theta, theta + 1.96 * se_theta]
```

excludes zero and all three full-data `theta_w` values have the same sign as
`theta`.  Otherwise it is false.  The artifact publishes every stratum rate,
overlap weight, window contrast, delete-one estimate, interval, sign check, and
the Boolean.  This adjusts the origin comparison for age and sex composition,
clusters all rows and events from one connected component together, and does
not treat overlapping random splits as independent inferential replicates.  It
uses no candidate prediction or candidate score.

### 7.2 Pooled-invariance envelope

A pooled rate has a structural null space.  For each window
`w = [y0, y1, y2, y3, y4]`, define from its truth tables, for origin `o` and
year `t`:

```text
S0_ot = weighted start-of-year at-risk stock
M0_ot = weighted row-matched remarriage events
U_ot  = weighted same-year events with no entering risk row.
```

Give each dissolved spell the immutable key
`(person_id, dissolution_year, origin)`.  For `t` from `y0` through `y3`,
`I_o,t+1` is the weight of spell keys active in next year's origin stock but
not active in the current stock.  This includes a new dissolution, a later
same-origin spell even when the person was already in that origin at `t`, and
aging into the working-age domain.  Define the fixed non-remarriage exit share
`q_ot` by the identity

```text
S0_o,t+1 = (S0_ot - M0_ot) * (1 - q_ot) + I_o,t+1.
```

Thus

```text
q_ot = 1 - (S0_o,t+1 - I_o,t+1) / (S0_ot - M0_ot).
```

The event order is fixed: start-of-year dissolved stock, row-matched
remarriage, death/censor/age-out or other support exit, then entries into the
next start-of-year stock.  Same-year dissolution-remarriage events never enter
`S0` and remain in `U`.  Each denominator above must be positive and every
`q_ot` must lie in `[0, 1]`; otherwise the truth construction aborts before a
surface referee sees an outcome.

The observed annual pooled truth rate is

```text
r_pool,t = (M0_Dt + M0_Wt + U_Dt + U_Wt)
           / (S0_Dt + S0_Wt).
```

Now seed `S_o,y0 = S0_o,y0`, keep `I`, `q`, and `U` fixed, and allow the
row-matched origin event masses `m_ot` to vary in all five window years,
subject to

```text
m_Dt + m_Wt + U_Dt + U_Wt
    = r_pool,t * (S_Dt + S_Wt)
0 <= m_ot <= S_ot
S_o,t+1 = (S_ot - m_ot) * (1 - q_ot) + I_o,t+1  for t < y4
S_o,t+1 >= 0                                             for t < y4.
```

These linear equalities and inequalities define the complete feasible
pooled-invariance polytope.  The observed path `S0, M0` must be a feasible
point.  The analysis uses a deterministic linear-program reducer to find, for
each origin, the minimum and maximum

```text
A_ow = sum_{t=y0..y4} S_ot
Q_ow = S_o,y4 / (S_D,y4 + S_W,y4).
```

`A_ow` is cumulative working-age dissolved-stock exposure.  `Q_ow` is the
origin's share of the two dissolved stocks at the start of the terminal
interval; its denominator must be positive.  Aging into the domain is in `I`;
aging out, death, censoring, and other support loss are in `q`.  Panel
disappearance and later re-entry remain visible in the published `q` and `I`
ledgers rather than being relabeled as marital transitions.

The `A_ow` extrema are ordinary linear objectives.  `Q_ow` has a positive
linear denominator and is reduced through the standard Charnes--Cooper
transformation: augment the state vector with a constant coordinate, set
`z = x / denominator` and `v = 1 / denominator`, impose the transformed linear
constraints and `denominator(z) = 1`, then minimize or maximize the transformed
origin numerator.  The transformed witness must map back to a feasible point
of the original polytope.

The solver evaluates each window's full polytope, not a chosen grid or
perturbation.  Preserving every annual pooled rate is stronger than preserving
only the aggregate cell.  Stock movement in this smaller null space is
therefore invisible to the incumbent pooled score.

Before any truth contact, the helper's source freeze must pin the linear
program library and version, algorithm, variable and constraint order,
presolve and scaling settings, feasibility and optimality tolerances,
unbounded/infeasible status mapping, and deterministic handling of degenerate
optima.  Synthetic polytopes with analytic extrema and multiple optima must
pass independently.  The surface referee must approve those bindings before
the authorized run; they cannot be inferred from output or changed on retry.

### 7.3 Truth-only stock margins

For the same one hundred correlation-component half splits, compute from each
half's truth risk table, by window and origin:

- cumulative working-age dissolved-stock exposure, scored by absolute log
  ratio between halves; and
- terminal working-age dissolved-stock share, scored by absolute difference
  between halves.

Reduce each with the same decimal half-even `mean + 3 * sample_sd` diagnostic
rule.  Call the resulting bars `u_Aow` and `u_Qow`.  These are stock-variability
margins only, not proposed gate floors.  Against the observed full-truth
`A0_ow` and `Q0_ow`, set

```text
mask_Aow = max(abs(log(Amin_ow / A0_ow)),
               abs(log(Amax_ow / A0_ow))) > u_Aow
mask_Qow = max(abs(Qmin_ow - Q0_ow),
               abs(Qmax_ow - Q0_ow)) > u_Qow
STOCK_MASK = any(mask_Aow or mask_Qow
                 for w in {W1, W2, W3}
                 for o in {divorced, widowed}).
```

All extrema and baseline quantities must be positive and finite where a log
or share requires them.  A missing quantity aborts the analysis before a
disposition.  The comparisons are strict and use the frozen reducer's exact
binary64 outputs; no post-result epsilon or rounding substitution is allowed.

This avoids treating a nearly inherited end stock as independent evidence.
It asks the narrower identification question: can a path invisible to the
pooled flow move an origin stock by more than truth-side sampling variation?

### 7.4 Surface vacuity flags

`POOLED_CONTROL_READY` is true exactly when, in every one of the three windows,
the recomputed single pooled cell:

- has all one hundred half-split scores defined;
- has at least twenty unweighted certified event records in every weaker half;
- has finite positive `sigma_cw` and an uncapped `u_cw` below `ln(1.5)`; and
- has a one-cell `p_4of5,w` at least `0.90`.

An origin-aware joint surface clears exactly when:

- it covers all working ages for both origins;
- every required cell clears the half-split support and cap rules in every
  window;
- the joint operating characteristic clears the weak-power rule in every
  window; and
- the full ladder and every undefined value are published.

`ORIGIN_SURFACE_READY` is true exactly when `SELECTED_ORIGIN_SURFACE` is the
first clearing joint surface in §5's fixed order; otherwise both are false and
null, respectively.  The artifact separately records
`POOLED_CONTROL_READY`, `ORIGIN_SIGNAL`, `STOCK_MASK`, and
`ORIGIN_SURFACE_READY`.  It may not collapse a failed flag into a favorable
headline.

An input-hash, information-date, event-semantics, arithmetic, solver,
undefined-value, or audit failure is `ANALYSIS_ABORT`: publication is required,
but the surface referee receives no surface finding and reaches no
disposition.  It requires a new proposal and freeze.  This is a process
incident, not a fourth surface disposition.

## 8. Referee decision semantics

The truth-only artifact is evidence for a surface referee, not a self-executing
gate edit.  For a valid artifact, the four Boolean flags reduce to exactly one
of the three authorized dispositions in this binding order:

```text
if not POOLED_CONTROL_READY:
    SURFACE_CHANGE_WARRANTED
elif (ORIGIN_SIGNAL or STOCK_MASK) and ORIGIN_SURFACE_READY:
    SURFACE_CHANGE_WARRANTED
elif (ORIGIN_SIGNAL or STOCK_MASK) and not ORIGIN_SURFACE_READY:
    RETURN_TO_MODEL_LAW_LEARNING
else:
    POOLED_SURFACE_STANDS
```

The first branch says the current surface itself needs a prospective rescope;
the eventual ceremony may replace it, narrow it, or demote it.  The second says
truth establishes a reason for origin protection and can price a complete
replacement.  The third says the pooled surface is substantively incomplete
but truth cannot safely price an origin replacement.  The final branch keeps
the aggregate cell because neither a truth-resolvable origin signal nor a
material stock mask is established.

The referee verifies and ratifies the computed mapping or rejects the artifact
as an `ANALYSIS_ABORT`.  It cannot substitute a disposition, choose a different
rung, relax a Boolean, or invent a fourth surface outcome after seeing results.

The referee must state the certification scope plainly.  A pooled-surface
finding certifies only aggregate working-age remarriage flow.  It does not
silently acquire a claim about either origin's hazard or stock.

## 9. What each disposition requires next

Under the currently operative §10 decision 9, none of the three surface
dispositions independently authorizes registration 8.  A separate prospective
decision-9 re-adjudication may change that registration disposition before or
after this inquiry reports.  This fixed surface question proceeds regardless;
it neither waits for nor pre-commits that referee's answer.

### 9.1 Pooled surface stands

`POOLED_SURFACE_STANDS` changes no gate byte, cell, reducer, floor, tolerance,
or candidate.  It answers the surface question with no change and supplies
evidence to the separate §10 decision-9 re-adjudication.  It does not itself
lift the pause or authorize registration 8.  The decision-9 referee must still
decide whether registration with a candid modal-failure forecast is warranted
under the unchanged contract.

### 9.2 A surface change is warranted

`SURFACE_CHANGE_WARRANTED` is a design finding only.  No surface change may
govern any registration without a separate prospective §2.8 and floor ceremony
that must:

1. propose the exact certification claim, cell set, reducers, support, weights,
   split unit, undefined rule, conjunction, and candidate-score semantics;
2. create a new, never-overwritten truth-only floor artifact and environment
   sidecar without copying any diagnostic value from this inquiry;
3. publish every split, support count, floor input, operating characteristic,
   vacuity check, source hash, code hash, and artifact hash;
4. obtain independent byte reproduction, an adversarial referee round, fixes,
   and a separate verification round;
5. specify the exact prospective §2.8 and gate-contract edit, certification
   narrowing, history entry, implementation guard, and mutation tests;
6. ratify and lock that amendment before candidate-2 registration; and
7. update the registration forecast and contract inventory to the ratified
   surface while preserving all other prerequisites.

This proposal pre-commits none of the ceremony's resulting cell bytes,
reducers, floor values, tolerances, or lock hashes.  The ceremony may itself
pause.  It may not overwrite or retrospectively apply a floor.

Under current decision 9, registration 8 remains forbidden until that
independent ceremony and the separate decision-9 adjudication both clear.  If
a separately ratified decision-9 amendment has already authorized and locked
candidate 2 before this finding, the later surface change is prospective only:
it cannot alter the registered bytes, the one-shot, or that candidate's result.
It instead governs the next eligible registration or successor contract.

### 9.3 Return to model-law learning

`RETURN_TO_MODEL_LAW_LEARNING` does not name or authorize a third family.  It
opens the plan's third-family gate only.  A later docs-only proposal must freeze
its law, inputs, information dating, search size, selector, no-op rule, stock
guards, and publication record before any outcome contact.  It must use no
post-cutoff outcome to choose a law or break a tie.

Under current decision 9, registration 8 remains forbidden while that proposal,
learning run, findings review, and any resulting prospective amendment are
incomplete.  A separately ratified decision-9 re-adjudication may change the
registration disposition, but it cannot turn this third-family gate into a law,
change bytes after registration, or make later learning retroactive.  Another
no-op cannot be converted into a forced law.

## 10. Ceremony for this question

The sequence is:

1. review and ratify this docs-only specification without truth execution;
2. implement the isolated truth-only helper and its denylist audits in a new
   lane;
3. freeze its source, inputs, runtime, output schema, and one authorized run;
4. publish the complete ladder, power, vacuity, stock-envelope, and audit
   artifact regardless of result;
5. obtain independent reproduction and a surface referee disposition; and
6. follow exactly one branch in §9 through its own prospective governance.

The future lane must use these never-overwritten paths and schema names:

- helper: `scripts/analyze_m6_remarriage_surface_question.py`;
- artifact: `docs/analysis/m6_remarriage_surface_question_results.json`;
- report: `docs/analysis/m6_remarriage_surface_question.md`;
- environment sidecar:
  `docs/analysis/m6_remarriage_surface_question_environment.json`; and
- artifact schema: `m6.remarriage.surface_question.v1`.

Any path or schema change requires amendment and re-freeze before truth contact.

No candidate run, gate edit, or registration belongs in this question lane.
The W1 proposal/referee/separate-flip sequence is the precedent; see
`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md:329-368,518-531`.

## 11. Scope and non-retroactivity

This proposal does not:

- alter candidate 1's frozen FAIL, artifact, or historical contract;
- reinterpret either ratified train-only round or its no-op disposition;
- alter candidate 2 or its two model deltas;
- edit `remarriage.18-64`, its reducer, or its `0.403` tolerance;
- edit a gate, floor, artifact, registry specification, source file, test, or
  run;
- authorize a third hazard family, a surface flip, or registration 8; or
- delay independent candidate-2 amendment or floor ceremonies.

Any later surface change "cannot retroactively change either learning result."
For this fixed surface ladder, its truth-only analysis, and its own downstream
governance, the plan is explicit: "The pause continues throughout."  That
constraint supplies no answer to the separate prospective decision-9
re-adjudication and does not delay the surface question if that adjudication
proceeds in parallel.
