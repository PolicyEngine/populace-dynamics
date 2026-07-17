# M6 candidate-2 program — earnings persistence, first-marriage transport, and the mandatory floors finding

- **Program id:** `2026-07-16-m6-candidate2`
- **Gate:** `gate_m6` (the temporal-holdout projection-drift gate)
- **Candidate:** 2; proposed fresh artifact `runs/gate_m6_candidate2_v1.json`
- **Status:** PROPOSAL for the fable referee. This is the proposal step, not a
  contract amendment, implementation, registration, run, score, or floor flip.
- **Forensic authority:** issue #42 [forensics round 4, comment
  4997635883][forensics-4], registered in [comment 4997460916][forensics-reg]
  after the [candidate-1 grading, comment 4997458179][grading].
- **Frozen evidence:** candidate 1's published artifact at commit
  `8ff7b14fa89f021c4951ddfbd2102f795ff4de21`, SHA-256
  `546a9739f8d1c7d21a91a07eb902c8af9bda92cdaa8f7917f312894f6a861b24`
  ([immutable artifact][candidate1-artifact], [PR #225][candidate1-pr]). It is
  read-only evidence and is not run, re-scored, or rewritten here.
- **Scope:** DOCS ONLY. This proposal changes no code, test, `gates.yaml` cell,
  floor, tolerance, run artifact, truth-side reducer, support rule, or verdict.
- **Candidate-ladder precedent:** W1 left candidate 1's FAIL intact, measured
  mechanisms before choosing candidate-2 deltas, registered only those deltas,
  byte-carried everything else, and wrote a fresh artifact; see the [W1
  candidate-2 registration][w1-c2-reg], [grading][w1-c2-grade], and the standing
  [search-disclosure / pass-verification rule][search-rule].

## 1. Disposition and design home

Candidate 1 is a valid, published **FAIL**: 6 of 11 gated cells failed, no gate
seed passed, and every failed cell failed on all five seeds. Candidate 2 is a
prospective successor, never a reinterpretation of that record.

The proposed candidate-2 package has two model deltas and one conformance
prerequisite:

1. replace the unbounded first-marriage age-by-cohort extrapolation with a
   regularized, support-aware interaction whose cohort deviation is flat beyond
   observed cohort-age support; make non-convergence a hard pre-scoring abort;
2. amend §2.7 prospectively to add a train-selected rank-refresh mixture to
   positive-to-positive earnings transitions, damping conditional-rank memory
   while preserving participation, the §2.7.6 rank-to-level marginal, its inverse,
   and `I_proj`; and
3. merge and verify the separate entry-dissolved remarriage conformance repair
   before candidate 2 is registered or scored.

The §2.8.3a power finding is orthogonal to model quality. The referee must choose
whether registration 8 keeps the conservative frozen tolerances with the power
gap disclosed or follows a new closed-domain floors ceremony. Section 6
recommends the ceremony.

### 1.1 Why this is a sibling program, not a §2.8 edit

`docs/design/m6_projection_engine.md` is ratified domain law. Its §2.8 harness
already requires the realized entry state and dissolution duration to survive
projection assembly (`docs/design/m6_projection_engine.md:943-994`); the
remarriage defect violates that existing law. Repairing it without changing
support, events, cells, or tolerances is conformance, not a new harness design.
The first-marriage proposal is a new candidate-family spec. The earnings proposal
does require a prospective amendment, but to §2.7's stochastic law, not §2.8's
harness.

A separate program document therefore keeps four acts distinct: proposal,
adversarial review, any ratifying amendment/lock ceremony, and the later candidate
implementation and run. This mirrors the W1 proposal-to-flip sequence: the W1
proposal explicitly moved no threshold
(`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md:1-11`), preserved the prior verdict
(`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md:329-346`), and required
a separate ratifying flip
(`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md:362-368` and
`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md:518-531`). If the
referee finds that the remarriage repair cannot be implemented while preserving
the current event/support law, a **narrow §2.8 amendment must be proposed in a
different PR before implementation**; this document does not pre-authorize one.

## 2. Frozen evidence and the no-self-rescue boundary

### 2.1 Published outcome

The frozen artifact records both preflights passing, nonzero across-draw
dispersion for every scored seed-cell, and this 11-cell result:

| Cell | Candidate-1 result | Worst score / tolerance |
|---|---:|---:|
| `first_marriage.18-29\|female` | FAIL, 5/5 seeds | 4.18× |
| `earn_mob_h1_diag` | FAIL, 5/5 | 3.47× |
| `remarriage.18-64` | FAIL, 5/5 | 2.75× |
| `earn_autocorr_lag2` | FAIL, 5/5 | 2.13× |
| `earn_dlog_mean.prime` | FAIL, 5/5 | 1.77× |
| `earn_p10.prime` | FAIL, 5/5 | 1.61× |
| `divorce.18-44` | pass, 5/5 | 0.452× |
| `incidence.20-66` | pass, 5/5 | 0.436× |
| `recovery.20-66` | pass, 5/5 | 0.442× |
| `earn_dlog_sd.older` | pass, 5/5 | 0.799× |
| `earn_zero_rate.older` | pass, 5/5 | 0.509× |

The one-shot is consumed. Candidate 1 remains a FAIL under its registered v3
contract under either floors resolution in §6. No candidate-2 fit, floor, or
artifact may be written over a candidate-1 path.

### 2.2 First-marriage transport finding

Forensics reproduced all 100 published first-marriage draw rates bit-for-bit
(maximum absolute difference zero) and found no support-key, duplicate-key,
event-origin/state, age, dating, or F6-weight defect. The dominant mechanism is a
reachable age-by-cohort extrapolation:

- the youngest fitted birth-decade level for females (1990) ends at age 23 in the
  `≤2014` fit;
- its fitted hazard is `0.0538753` at age 23, `0.999999928` at 24, and `1.0` at
  25+, while the female-1980 hazards at 23/24/25 are
  `0.07321/0.07420/0.07193`;
- LBFGS exhausted `max_iter=1000` and recorded `converged=False`;
- direct standardization on truth support gives `0.359–0.372` against truth
  `0.0451–0.0543`; simulated depletion lowers the scored projection only to
  `0.184–0.200`.

At the ages that expose the failure, the five-seed mean truth→projection rates are
age 24 `0.060341→0.991002`, age 25 `0.059226→0.940790`, and age 26
`0.059623→0.361957`. The current design constructs spline, sex, spline-by-sex,
cohort dummies, and spline-by-cohort columns and predicts without a support guard
(`src/populace_dynamics/models/family_transitions/components/first_marriage.py:77-110`).
Its fitter caps LBFGS at 1,000 iterations but merely
records the failed convergence state and returns the model
(`src/populace_dynamics/models/family_transitions/components/first_marriage.py:113-162`);
projection
then supplies future age and birth decade directly
(`src/populace_dynamics/engine/marital.py:159-178`).

The amendment-3g entrant hypothesis is refuted. The eligible class is 2,638 people
(2,850 less 212 who never reach 15 by 2022), but only 157 gated female
person-years—2.25% of truth-target weight, with zero truth events—enter this cell.
Removing the class from projection only makes every score worse (mean score
change `+0.025142780`); its projected rate `0.00426–0.01285` dampens the pooled
failure. Candidate 2 therefore keeps amendment 3g and the truth support intact.

### 2.3 Earnings finding

Across all 100 seed-by-draw observations, p10 and mean growth are too low while
mobility-diagonal mass and lag-2 autocorrelation are too high. Counts beyond the
locked tolerance are 92/100, 95/100, 100/100, and 100/100 respectively. The
five-seed mean projected/truth ratios are `0.739`, `0.557`, `1.181`, and `1.246`;
the persistence gaps are 11.5–19.2 draw standard deviations. The conditional-rank
memory, not Monte Carlo noise, dominates.

The registered `I_proj` slope is `0.0225137847` (2.2769% annual growth). A static
replacement with realized NAWI would raise the level by only `0.006479` log point
in 2016 and `0.031005` in 2018. It explains at most 8.7–12.3% of the p10 miss,
leaves at least 53.8–67.8% of the mean-growth gap even under its largest possible
credit, and has exactly zero effect on the two persistence statistics under the
registered static decomposition. The exact-zero statement does not extend to a
full-law change because nominal 2016 earnings also enter the 2018 participation
gate. It does establish that realized NAWI is not a persistence repair.

The current implementation makes the pinned source of memory concrete:
`W_CURRENT=1.0`, `W_PRIOR=0.5`, `W_ANCHOR=0.25`, `LAMBDA_FIXED=0.1`, and `k=25`
(`src/populace_dynamics/engine/forward_earnings.py:47-65`); the target distance
uses current rank, prior rank, and the blended anchor coordinate
(`src/populace_dynamics/engine/forward_earnings.py:983-1044`),
then the generated 2016 rank is recovered from its level and threaded into 2018
(`src/populace_dynamics/engine/forward_earnings.py:1135-1193`). There is no
evidence that the interior-exact `CellMarginal.quantile`/`rank` round trip is
coded incorrectly (`src/populace_dynamics/engine/forward_earnings.py:926-963`).

### 2.4 Mandatory earnings-domain floor finding

The report-only truth-side self-check covers 13,561 domain persons and 45,606
earnings rows. It finds:

| Cell | Frozen locked tolerance | Closed-domain derivation |
|---|---:|---:|
| `earn_autocorr_lag2` | 0.087 | 0.087 |
| `earn_dlog_mean.prime` | 0.043 | 0.043 |
| `earn_dlog_sd.older` | 0.269 | 0.279 |
| `earn_mob_h1_diag` | 0.052 | 0.054 |
| `earn_p10.prime` | 0.221 | 0.284 |
| `earn_zero_rate.older` | 0.163 | 0.168 |

The frozen-v3 earnings OC is `p_seed=0.9347`, `p_gate=0.9626`; applying its locked
tolerances on the actual closed domain yields `p_seed=0.8760`,
`p_gate(≥4/5)=0.8809`. That is 1.91 percentage points below the 0.90 floor and a
faithful-candidate false-fail probability of 11.91%—the disclosed “about 12%
under-power” finding. Closed-domain-derived tolerances yield an earnings-only
`p_seed=0.9300`, `p_gate=0.9575`. No tolerance reaches a metric cap and neither
vacuity nor near-tautology fires. The §2.8.3a near-unpassable trigger does.

## 3. Controlling law, quoted verbatim

This section quotes rather than paraphrases the pins whose classification the
referee must adjudicate.

### 3.1 §2.7 forward-law and certification pins

From `docs/design/m6_projection_engine.md:490-497`:

> The forward law **mirrors the certified candidate-11 conditioning structure**,
> reversed in time, and is **fit from scratch on `≤2014` forward wave tuples** — it is
> **not** an inversion of the certified backward conditional (retiring concern (i)).
> Because it is a `≤2014`-fit conditional-rank law applied out-of-sample to the
> `2014→2016` and `2016→2018` steps — not a period-specific marginal carried past its
> last fit cell — it does not extrapolate a marginal (retiring concern (ii)):
> applying a `≤2014` transition law out-of-sample is exactly what the temporal
> holdout scores.

From `docs/design/m6_projection_engine.md:579-587`:

> **2.7.3 Certification framing.** The forward generator is a **different stochastic
> law** than gate-1 certified (forward conditional ≠ backward conditional), so it is
> **NOT** covered by `gate_1`; **no `gate_1` certificate transfers to it.** It enters
> `gate_m6`'s scored run as one composed module of the engine and is **first-certified
> there**, on the temporal holdout, exactly like the composition itself (§2.6) — its
> `≤2014` fit makes it leakage-safe by construction, and the gate scores its
> 2016/2018 output against realized PSID earnings (§4.5). §4.2 adds it to the refit
> enumeration; decision 10 records that a `gate_m6` pass is the forward generator's
> sole certification.

The exact conditioning and transferred hyperparameters are also pinned at
`docs/design/m6_projection_engine.md:508-537`: `k=25`, distance weights
`1/0.5/0.25`, `λ=0.1`, `u_w`, the participation architecture, and
`rank_{t+2} | rank_t, rank_{t−2}, u_A, age` with its two-step memory ramp.

### 3.2 §2.7.6 rank-to-level, projected-index, and inverse pins

From `docs/design/m6_projection_engine.md:687-701`:

> - **(b2) NAWI-normalized calendar-invariant marginal, re-indexed — PINNED.** Fit a
>   single `CellMarginal` per `age_bin` (the certified `fit_cell_marginals` /
>   `_plotting_positions` machinery `candidate5b:283-313` verbatim) on the **pooled
>   `≤2014`** positive earnings (one `CellMarginal` per `age_bin` on the `[25,64]`
>   grid, §2.7.6.5), each divided by its wave's wage index (NAWI-normalized to a
>   common scale). De-index a drawn rank `u`: normalized level `ĝ =
>   Qhat_pos_agebin.quantile(u)`, then the nominal target-year level is
>   `ĝ × I_proj(target_year)` (§2.7.6.3). This **keeps the certified de-index
>   MECHANISM** — it re-fits a **new** `age_bin`, NAWI-normalized object through the
>   **same `CellMarginal` machinery** (`fit_cell_marginals` / `_plotting_positions`,
>   `candidate5b:283-313`), not the certified `(age_bin, period)` object itself (a
>   `CellMarginal` quantile — R2-faithful). It is fully `≤2014`-derivable, is **not** a
>   period-specific marginal (calendar-invariant, so §2.7.1-consistent), and has an
>   inverse for the re-rank (interior-exact, §2.7.6.4). `p0` (the zero share) and the participation gate are unchanged; a
>   drawn zero stays a zero (non-participation).

From `docs/design/m6_projection_engine.md:703-719`:

> **2.7.6.3 The wage index `I_proj` — `≤2014`-derivable, pinned; the leakage
> prohibition.** `I_proj(2016)` and `I_proj(2018)` are the wage-index (SSA average-
> wage index / NAWI — the `ss/params.py` series the M2 `taxable_payroll_convention`
> and AIME indexing use) **projected from `≤2014`**, never the realized 2016/2018
> values. Pinned form: an **OLS log-linear fit of `ln(NAWI_y)` on `y` over the
> trailing decade `y ∈ [2005, 2014]`**, extrapolated to 2016/2018 (equivalently:
> apply the `≤2014` trailing-decade geometric-mean annual wage-growth rate forward
> from `NAWI_2014`). Justified from `≤2014` diagnostics only: a decade window
> averages out biennial sampling noise while reflecting the recent (post-2005)
> wage-growth regime and excluding pre-2005 structural breaks; the two scored steps
> consume only `I_proj(2016)` and `I_proj(2018)`. **NAWI publication lag, disclosed:**
> the SSA average-wage index for a year is published ~2 years later, so `NAWI_2013`
> and `NAWI_2014` are *published* after 2014; but the AWI is a **final, unrevised**
> series (SSA does not restate prior AWI values once set), so a `NAWI_y` with
> `y ≤ 2014` is a fixed function of `≤ 2014` wages — leakage-safe. Only its
> publication *date* is post-`T*`, not its *information*; no realized 2016/2018 wage
> outcome enters `I_proj`.

From `docs/design/m6_projection_engine.md:721-734`:

> **Leakage prohibition (pinned).** Using any **realized** post-`T*` NAWI / wage-
> index value on the **scored** path is PROHIBITED — it would leak realized wage
> growth into the gated level cells (`earn_p10`, `earn_p50`, the Δlog-earnings SD,
> the mobility diagonal). Concretely: the projection frame **carries a realized
> `nawi` column** (the aging step rolls `nawi_by_year → nawi`, `steps.py:145-156`);
> the scored forward earnings de-index MUST NOT read it for 2016/2018 — it uses the
> generator's own `I_proj`. The realized `nawi` is admissible only for the
> **non-scored** seam / `taxable_max` plumbing and for the **report-only alignment
> path**. The gated path is the **un-aligned** projection (decision 9, §4.8: "gate
> the un-aligned drift"; the alignment layer is report-only, the M2
> `calibration_disclosure` lesson); realized-index alignment — snapping the
> projected level onto realized NAWI — may appear only on the report-only path,
> disclosed per run within the **maximum alignment displacement** (adjudication 9,
> §4.8).

From `docs/design/m6_projection_engine.md:744-763`:

> - **fit-population marginal CDF — PINNED.** Recover `rank_t = rhat_agebin( ℓ /
>   I_proj(wave_of_ℓ) )` — normalize the carried nominal level by its wave's
>   `I_proj`, then evaluate the **same `age_bin` `CellMarginal`'s inverse rank map
>   `rhat`** used to de-index in §2.7.6.2 (`candidate5b` `CellMarginal.rank`). It is
>   the **inverse** of §2.7.6.2 — both are `np.interp` on the one `(wtil, yval)` grid —
>   **exact on the strictly-monotone interior**. At the corners it is
>   **deterministic but inexact**: `quantile` clamps to `[ymin, ymax]` and `rank` to
>   `[0.001, 0.999]` (`RANK_CLAMP_LO/HI`), and a flat-`yval` tie maps back to the first
>   `wtil` (`candidate5b:250-259`; the certified code tracks this corner mass as
>   `corner_bottom` / `corner_top`). **Consequence for the 2018 re-rank:** `rank(ℓ)`
>   is a pure deterministic function of the carried level, so **chain integrity holds**
>   (the 2018 draw conditions on a well-defined `rank_t`); the inexactness is only a
>   **bounded rank perturbation at the corners** (a tail level clamps into
>   `[0.001, 0.999]`; a tie collapses to one grid rank), never a break in
>   reproducibility. A carried zero maps to the `p0` / zero-anchor (Q0) regime.
>   **Conditioning-consistency (R2):** the
>   certified chain's ranks **are** `CellMarginal` CDF positions (it draws `u_prev`
>   and de-indexes via `cell.quantile`), so composing as-estimated requires the
>   deployed re-rank to reference the **fit-population marginal**, exactly what the
>   pinned rule does — not the projected frame's own cross-section.

The generated-lag memory itself is pinned at
`docs/design/m6_projection_engine.md:767-774`, and §2.7.6 closes by requiring an
implementation with “zero design choices”
(`docs/design/m6_projection_engine.md:817-826`).

### 3.3 §2.8.3a scoring-domain and floors pin

The operative support paragraph at
`docs/design/m6_projection_engine.md:1843-1849` is reproduced verbatim:

> - **Gated-earnings scoring support = realized support ∩ the earnings domain,
>   symmetric on both sides.** The forward chain is evaluated (per F1) for every
>   realized-present **in-domain** person-period regardless of simulated death; the
>   truth side (`rate_a`) is restricted to the **same** domain, so projection and
>   truth carry identical support and the §2.8.4 identity guard holds. Later
>   earnings-entrants are excluded from **both** sides (report-only open additions),
>   not scored survivor-/entrant-conditioned.

The complete floor-consistency and two-directional rule at
`docs/design/m6_projection_engine.md:1850-1879` is reproduced verbatim:

> - **Floor consistency — the empirical delta (replaces the "holds by construction"
>   expectation).** Verified against the staged PSID through the frozen floor
>   machinery: of the frozen v3 floor's gated-earnings support (**13,163** persons at
>   `{2016, 2018}`, prime+older), **2,722 (~21 %: 2,199 prime, 590 older)** are later
>   earnings-entrants **outside** the 2014-anchored domain — the frozen floor
>   **over-included open-additions** in the gated earnings cells (a lineage artifact
>   of `build_anchor_frame`'s multi-wave anchor merge; the flows presence-condition
>   those openers at their anchor, but the 2014-anchored chain cannot project them).
>   So point-(6)'s "no gated earnings cell's realized support includes any excluded
>   person" is **empirically false** for the two level cells and the change cells; it
>   **does** hold for non-head/spouse members (no earnings row → in no earnings cell).
>   The consequence is **conservative**, not a leak: scoring the closed-panel domain
>   against the frozen full-support tolerance faces ~`√(N/N_domain)` inflated
>   half-split noise (~1.17× prime, ~1.06× older), so the effective earnings
>   tolerances are **tighter** than the closed-panel warrants (harder to pass, no
>   false-PASS risk). Bind it with a **harness self-check**: recompute the
>   domain-restricted (closed-panel) earnings floor half-splits and publish the
>   tolerance / earnings-OC delta vs the frozen v3 floor (report-only — the frozen
>   tolerances remain the gated contract, applied conservatively). The escalation is
>   **two-directional**, honoring §4.9's both-directions weak-power check: route to
>   the **floors-ceremony finding** rule (adjudication-7 / §4.9 — a ceremony finding,
>   never a silent redesign; a possible earnings-floor re-derivation on the
>   closed-panel domain) if **either** (a) the recomputed closed-panel earnings OC
>   falls **below the 0.90** weak-power floor (near-**unpassable** — the modal
>   direction here, since the frozen tolerances were non-vacuity-certified on the
>   full support and the domain half-split noise is almost-certainly larger), **or**
>   (b) the domain-restricted surface trips the §4.9 **vacuity** guard in the other
>   direction (near-**tautological**: any domain-restricted gated tolerance lands at
>   the `ln(1.5)` cap, or the domain earnings OC approaches 1 with a near-unfailable
>   cell). Both directions publish; neither is silently absorbed.

## 4. First-marriage transport law for candidate 2

### 4.1 Options adjudicated

| Option | Disposition | Reason |
|---|---|---|
| Clamp the **whole hazard** to the nearest cohort-age boundary | reject as the primary law | It prevents the explosion but freezes the global age profile too; supported information from older cohorts at ages 24–29 would be discarded. |
| Penalized smooth global age/sex curve plus a regularized cohort deviation, with the cohort deviation held flat beyond that cohort's support | **adopt** | It borrows the age pattern from supported cohorts while preventing an unsupported age×cohort interaction from running away. |
| Global monotonicity over all ages | reject | First-marriage hazards rise and then fall; a global sign restriction is substantively wrong. The adopted zero-slope boundary on only the unsupported cohort deviation is the narrow shape constraint. |
| Exclude out-of-support people from projection or scoring | reject | The cell is reachable, 3g is not causal, and exclusion would change truth support/floors after seeing a candidate failure. §2.7.6.5's principle applies by analogy: the model must cover the scored surface. |

This choice follows the campaign's estimation-versus-determinism norm. Novel
age×cohort combinations are explicitly diagnosed rather than mistaken for
interpolation ([Bartley et al. 2019][bartley]; [NIST extrapolation
guidance][nist]). Penalized multidimensional hazard surfaces are an established
way to borrow strength while controlling unstable interactions ([Dantony et al.
2024][dantony]; [Eilers and Marx 1996][eilers-marx]). Shape-constrained additive
models support the narrower boundary-slope constraint, but do not justify a
globally monotone marriage hazard ([Pya and Wood 2015][pya-wood]).

### 4.2 Proposed fitted law

Candidate 2 keeps the current outcome, never-married risk set, F6 weights, fixed
knots `{20,22,25,30,40}`, sex definition, birth-decade definition, and `≤2014`
truncation. It replaces only the unguarded cohort interaction and its fit-strength
selection.

For cohort `c`, let `[a_c^-, a_c^+]` be the minimum and maximum ages with positive
fit weight. Let `B(a)` be the current standardized restricted-cubic-spline basis,
and define

```text
a_c* = clip(a, a_c^-, a_c^+)
logit h(a, s, c) = α + B(a)β + sδ + s B(a)β_s + γ_c + B(a_c*)b_c.
```

The global age and age-by-sex terms use the actual target age when that age lies
inside pooled training support. Only the cohort-specific deviation uses `a_c*`.
Thus a female in the 1990 cohort at age 24 receives the supported global
female-age-24 curve plus the 1990 deviation evaluated at its age-23 boundary; the
1990 interaction cannot extrapolate from `0.0538753` to nearly 1. If target age is
outside pooled age support, every age term uses the nearest pooled boundary. If a
target birth decade is newer or older than every fitted decade, its cohort main
effect, support interval, and deviation come from the nearest fitted decade. No
row is excluded and no truth-side value enters this rule.

All non-intercept coefficients retain an L2 penalty. Its strength is selected
before registration from

```text
C_GRID = {0.0001, 0.0003, 0.001, 0.003, 0.01,
          0.03, 0.1, 0.3, 1.0}
PSEUDO_BOUNDARIES = {2006, 2008, 2010}
```

At pseudo-boundary `b`, fit only rows whose dating information is available by
`b`, then evaluate weighted Bernoulli deviance at `b+1…b+4`, applying the same
support-extension rule. Choose the `C` with the lowest mean pseudo-holdout
deviance; ties within `1e-12` choose the smaller `C` (stronger regularization).
Publish every rung, convergence state, support count, deviance, and the selected
value before registration 8. No 2015+ outcome or candidate-1 cell score may enter
selection.

### 4.3 Convergence is a hard designed abort

Every candidate-grid fit and the final `≤2014` fit uses LBFGS with
`max_iter=10_000`, `tol=1e-8`, deterministic row ordering, and the registered
standardization. A grid rung that reaches the ceiling or emits a convergence
warning is ineligible and remains in the published selection ledger. The run
**must abort before any projection, truth read, score, or candidate-artifact
write** if the selected full fit:

- reaches `max_iter`, reports unsuccessful termination, or emits a convergence
  warning;
- has a non-finite coefficient, linear predictor, or probability; or
- cannot reproduce its registered support ledger and selected-`C` checksum.

An iteration ceiling is not evidence of successful termination; the optimizer
reports those separately ([scikit-learn `LogisticRegression`][sklearn-logit];
[SciPy optimization contract][scipy-minimize]). The current `converged=False`
state therefore becomes a designed abort, not metadata attached to a model that
is allowed to score. An abort is recorded publicly and requires a new registration
after a reviewed fix; it is not converted to a FAIL or silently refit with a new
law.

### 4.4 Required disclosures and acceptance

Before score assembly the fresh artifact/sidecar must publish, for each sex,
cohort, and target age:

- fitted support endpoints, in-support count/event weight, and whether the global
  or cohort term was boundary-evaluated;
- convergence status, iterations, warning count, selected `C`, all grid results,
  and deterministic design-matrix checksum;
- the age-18–29 predicted hazard table, including female-1990 ages 23–29; and
- counts and F6 weight of in-support, age-out-of-support, and unseen-cohort rows in
  each gated first-marriage cell.

These are transport diagnostics, not alternate scores. Candidate 2 still clears
`first_marriage.18-29|female` only by the locked scoring rule on the unchanged
truth support. The current candidate-16 spec and its historical registry remain
immutable; implementation must add a new candidate spec rather than mutate the
frozen implementation
(`src/populace_dynamics/models/family_transitions/registry.py:319-347,374-397`).

## 5. Earnings persistence and the prospective §2.7 amendment

### 5.1 Within-law versus amendment adjudication

**Adjudication: the persistence repair requires a prospective §2.7 amendment.**
Changing the distance weights, `λ`, two-step/anchor memory, donor selection,
refresh probability, participation-rank coupling, or RNG consumption changes the
pinned conditional transition law. It is not a refit of estimates under an
unchanged spec. R1 calls changes to fitted parameters/internal paths surgery and
R2 requires as-estimated conditioning
(`docs/design/m6_projection_engine.md:374-384`). §4.2 likewise distinguishes
refitting a frozen spec from selecting a new one
(`docs/design/m6_projection_engine.md:2722-2760`).

The amendment can be narrow. Candidate 2 leaves §2.7.6.2's pooled
NAWI-normalized `CellMarginal`, §2.7.6.3's projected `I_proj` and leakage fence,
§2.7.6.4's inverse CDF, age support, biennial timing, frame state, and
participation gate unchanged. It amends the positive-continuation rank transition
and RNG registry only. Realized post-2014 NAWI stays **PROHIBITED**; the static
decomposition shows it would not solve persistence anyway.

### 5.2 Options adjudicated

| Option | Disposition | Reason |
|---|---|---|
| Substitute realized 2016/2018 NAWI | prohibited | It leaks the holdout, explains at most 12.3% of p10, and cannot move the static persistence measures. |
| Change `CellMarginal` rank↔level mapping | reject | The inverse is interior-exact and forensics found no implementation error; this would disturb a pinned mechanism without targeting the dominant channel. |
| Refit or recouple the participation gate in candidate 2 | defer | Participation can contribute to level/growth, but the hard localization is conditional-rank memory and the older zero-rate cell passed. Do not spend two structural degrees of freedom at once. |
| Remove all prior/anchor conditioning | reject | It is a large uncalibrated discontinuity and would discard useful persistence rather than estimate how much to retain. |
| Train-selected refresh mixture for positive continuers | **adopt** | It directly attenuates rank memory, preserves the existing conditional draw when not refreshed, and keeps the positive marginal/rank-to-level law unchanged. |

Longitudinal microsimulation commonly separates persistent and transitory earnings
components and estimates them from observed histories ([CBO 2006, Appendix
D][cbo-earnings]). PSID evidence also warns that persistence varies with age
([Karahan and Ozkan 2013][karahan-ozkan]). Candidate 2 nevertheless uses one
global refresh share rather than adding age-specific parameters: the limited
pre-2014 pseudo-holdouts should estimate one new degree of freedom first.

### 5.3 Proposed rank-refresh mechanism

Participation is drawn exactly as now. Re-entry from zero is drawn exactly as
now. For a person who is positive at `t`, participates at `t+2`, and would receive
the current conditional donor rank `u_cond`, candidate 2 adds:

```text
B_refresh ~ Bernoulli(q*)
u_out = u_fresh  if B_refresh = 1
        u_cond   otherwise.
```

`u_fresh` is a weighted one-record draw from all positive `u_tp2` ranks in the
same target five-year age bin among `≤2014` positive-to-positive forward pairs.
The pool is sorted by `(person_id, period_tp2)` and uses `weight_tp2`; it is
unconditioned on current rank, prior rank, `u_A`, and `u_w`. It therefore adds a
transitory rank innovation while drawing from the same fitted positive-rank
population that the existing conditional donors use. The unchanged
`rank_to_level` then maps `u_out` through the same age-bin `CellMarginal` and
`I_proj`; the next step re-ranks that level through the unchanged inverse.

The §2.7 amendment must add two isolated per-person/per-period substreams:

```text
4: memory-refresh-gate
5: memory-refresh-rank
```

The existing gate/donor/re-entry streams `{1,2,3}` and their draw order remain
unchanged. A refresh therefore cannot perturb participation or the incumbent
conditional donor draw. Odd years still consume no RNG.

### 5.4 Train-only estimation of `q*`

Estimate, disclose, and freeze `q*` before registration 8:

```text
Q_GRID = {0.00, 0.05, ..., 0.95, 1.00}
PSEUDO_BOUNDARIES = {2006, 2008, 2010}
SELECTION_DRAW_SEEDS = {6200, ..., 6219}
```

For each pseudo-boundary, refit using data available by that boundary, project
two biennial steps, and calculate train-analogue versions of all six gated
earnings concepts. Derive standardizers from person-disjoint half-splits within
that pseudo-holdout. A `q` is feasible only if its older-worker dlog-SD deviation
is no more than one pseudo-floor sigma worse than `q=0` at every boundary;
participation and therefore older zero-rate must be bit-identical by construction.
Among feasible values, minimize the sum of squared standardized errors for
`p10.prime`, `dlog_mean.prime`, `mob_h1_diag`, and `autocorr_lag2` across the three
boundaries. Ties within `1e-12` choose the smaller `q` (least departure from the
current law).

The selection ledger must publish every `q`, cutoff, simulated moment,
standardizer, feasibility result, objective, selected value, effective search
size, and checksum. The chosen `q*`, pool construction, and new substream codes
then enter the prospective §2.7 amendment and candidate-2 registration. No
post-2014 row, realized post-2014 macro value, candidate-1 seed score, or
candidate-2 score may enter this estimation.

### 5.5 What a result would mean

This is not a guarantee that candidate 2 passes. If the train-only rule selects
`q*=0`, the honest conclusion is that pre-2014 pseudo-holdouts do not support the
proposed damping; the referee should not force a nonzero value from the 2016/2018
failure. If a nonzero value is selected and candidate 2 still fails, the fresh
artifact becomes the next forensic record. A candidate-2 pass first-certifies
this amended forward law only on the registered 2016/2018 cells; it does not
transfer gate-1 certification or validate a 2100 earnings projection.

## 6. The mandatory floors finding

### 6.1 The two registered resolutions

| Resolution | What stays/moves | Honest benefit | Honest cost |
|---|---|---|---|
| **A. Retain frozen v3 tolerances** | Keep `runs/m6_holdout_floors_v3.json`, its SHA, all six tolerances, and `gates.yaml` byte-identical. Registration 8 and every candidate-2 report disclose `p_seed=0.8760`, `p_gate=0.8809`, the 1.91pp floor deficit, and 11.91% faithful false-fail probability. | Preserves the ratified artifact and is conservative: harder to pass, no false-PASS direction. | Knowingly scores the closed domain with a floor derived on a larger domain; violates the campaign's stated 0.90 power aspiration and accepts about 12% under-power. |
| **B. Registered closed-domain floors ceremony** | Build a new, never-overwritten truth-only artifact (proposed `runs/m6_holdout_floors_v4.json`) on exactly `realized support ∩ 2014 earnings domain`; rederive the six earnings tolerances, recompute earnings and combined OC/vacuity, adversarially reproduce it, SHA-pin it, and conduct a separate full lock ceremony before registration 8. Flows/cell definitions/reducers remain unchanged. | Prices noise on the population actually scored; the self-check's provisional earnings OC is `0.9575` and no vacuity flag fires. | Moves tolerances after candidate 1 has been observed, so governance must prove truth-only/candidate-blind derivation and preserve candidate 1's historical contract. It costs a full ceremony and may delay candidate 2. |

Neither resolution is authorized by this proposal. Silent threshold movement,
editing v3 in place, or applying v4 retrospectively to candidate 1 is prohibited.

### 6.2 Recommendation

**Recommend resolution B, the registered closed-domain floors ceremony.** The
scorer's domain is now empirically known and §2.8.3a expressly routes the `0.8809
< 0.90` result to a ceremony finding. Re-deriving on that exact truth domain is
more coherent than knowingly carrying a lineage mismatch into a new candidate,
provided the change is candidate-blind, independently reproduced, and locked
before registration 8. Candidate 1 remains a published FAIL under v3; there is no
self-rescue.

The ceremony must not simply copy the six provisional tolerances. It must rebuild
the closed-domain half-splits from the registered truth-only derivation, publish
all seeds and support counts, recompute the **combined** 11-cell OC with the
unchanged flow floors, re-run both power directions and caps, and stop if any
existing floor invariant fails. Only the later ratifying lock PR may edit
`gates.yaml` or add the new floor artifact. If the referee selects resolution A,
the 11.91% under-power disclosure becomes a mandatory registration/artifact field,
not a footnote.

## 7. Remarriage §2.8 conformance repair — separate prerequisite

The remarriage failure cannot be used to tune a hazard until the assembly path
conforms to existing law. Forensics finds a smaller direct-standardized hazard
overshoot (`0.05110–0.05457` versus truth `0.03562–0.03927`), but the dominant
current defect is lost history: projected dissolved exposure is only
12.37–13.29% of truth while the event numerator is 33.58–39.95%.

The code reads entry-divorced/widowed state and `dissolution_year`
(`src/populace_dynamics/engine/marital.py:82-128`) but serializes only open or
in-window emitted episodes before reassembly
(`src/populace_dynamics/engine/marital.py:133-149` and
`src/populace_dynamics/engine/marital.py:252-267`). A widow-only
post-assembly patch changes person-years but explicitly creates no episode/event
(`src/populace_dynamics/models/family_transitions/components/initial_states.py:206-242`).
Reassembly defaults a person with no prior emitted
change-point to `never_married`
(`src/populace_dynamics/data/transitions.py:377-420`), recomputes episode order by
`cumcount` (`src/populace_dynamics/data/transitions.py:426-440`), calls emitted
rank 0 a first marriage (`src/populace_dynamics/data/transitions.py:453-470`),
and requires rank ≥1 plus a prior emitted dissolution for remarriage
(`src/populace_dynamics/data/transitions.py:473-509`).

That contradicts the existing seed/duration/change-point pins at
`docs/design/m6_projection_engine.md:943-994`. The separate conformance lane must,
before registration 8:

1. preserve every entry-divorced and entry-widowed person's realized entry state,
   dissolution year, and prior-order information through `_assemble_panel`;
2. preserve the existing `allow_exact_matches=False` dating and event law, so the
   first in-window marriage of an entry-dissolved person is a remarriage;
3. reproduce unchanged support, F6 weights, cells, truth rates, tolerances, RNG
   addresses, and non-remarriage histories;
4. add discriminating synthetic fixtures for entry-divorced, entry-widowed,
   in-window remarriage, no-remarriage exposure, and same-year boundary cases; and
5. publish a pre-score conformance reconciliation on the exact forensic classes.

This document implements none of it. The referee should classify a repair meeting
those conditions as current-law conformance. If a proposed patch changes event
semantics, support, a truth reducer, or a cell, it leaves that classification and
requires a narrow prospective §2.8 amendment in its own ceremony. Only after the
repair lands may the residual remarriage hazard be measured; candidate 2 does not
pre-tune it from the contaminated score.

## 8. Candidate-2 must-not-regress constraints

The five clean candidate-1 passes become an additional candidate-2 acceptance
block. They retain the candidate-1 tolerances even if the referee chooses a new
closed-domain earnings floor:

| Cell | Regression tolerance | Candidate-1 score range | Scope of evidence—not more |
|---|---:|---:|---|
| `divorce.18-44` | 0.379 | 0.0118–0.1715 | duration×order divorce on this pooled cell; not all marital dynamics |
| `incidence.20-66` | 0.404 | 0.0189–0.1763 | M4 reproduction incidence; not prevalence or production disability |
| `recovery.20-66` | 0.314 | 0.0612–0.1388 | M4 reproduction recovery; same limitation |
| `earn_dlog_sd.older` | 0.269 | 0.1833–0.2149 | older change dispersion; not prime growth or persistence |
| `earn_zero_rate.older` | 0.163 | 0.0576–0.0830 | older participation mass; not the positive-rank law |

The regression block passes iff, on at least 4 of the 5 fresh gate seeds, **all
five** cells clear those thresholds. This is the same seed-level conjunction used
by the gate (`src/populace_dynamics/harness/m6_scoring.py:624-751`), applied to
fresh candidate-2 values. A valid overall gate pass implies the block, but the
artifact reports it separately so a candidate-2 FAIL cannot hide a regression in
an otherwise passed component.

The first-marriage implementation must not mutate the separate divorce estimator
(`src/populace_dynamics/models/family_transitions/components/divorce.py:67-124`).
The earnings refresh leaves participation unchanged by construction; the older
dlog-SD threshold remains binding on any rank-dispersion side effect.

## 9. Candidate-2 protocol

### 9.1 Byte-carried contract

Unless this program names a delta or prerequisite, candidate 2 byte-carries the
registered candidate-1 protocol:

- boundary `T*=2014`, projection end 2022, gated flow years 2015–2019 and earnings
  reference years 2016/2018;
- gate seeds `0…4`, K=20 draws at `5200…5219`, 50/50 splits, household-disjoint
  marital and person-disjoint disability/earnings floors;
- `START_OF_INTERVAL` support for marital/disability and `EXACT_WAVE` earnings
  support, with earnings scored on realized support intersected with the 2014
  domain, symmetrically on both sides;
- an undefined gated draw invalidates the run; all scored surfaces are regenerated;
- every seed passes only if every gated cell clears, and the gate passes at
  `≥4/5` seeds; and
- artifact publication regardless of PASS/FAIL, plus the standing independent
  bit-exact verification requirement for any PASS.

The fresh primary path is exactly `runs/gate_m6_candidate2_v1.json`, accompanied
by its normal environment sidecar and written through the exclusive new-file
guard. `runs/gate_m6_candidate1_v1.json`, its sidecar, and every floor artifact
remain immutable.

### 9.2 Ceremony matrix

| Proposed change | Required governance before implementation/run |
|---|---|
| Support-aware first-marriage estimator/new candidate spec | Referee ratifies this candidate-family law; add a new immutable registry spec; register selected train-only fit strength and support behavior before score. No §2.8 edit. |
| Rank-refresh persistence law and substreams | Prospective §2.7 amendment, adversarial review, verification, and ratification before candidate-2 code or registration. Keep §2.7.6.2/.3/.4 unchanged. |
| Entry-dissolved remarriage repair preserving current event/support law | Separate conformance implementation and proof, merged before registration 8; no threshold/floor change. If semantics move, stop for a narrow §2.8 amendment. |
| Retain frozen tolerances | Referee records resolution A and mandates the `0.8809` / 11.91% disclosure; no floor or lock edit. |
| Re-derive closed-domain floors | Separate truth-only v4 derivation, adversarial reproduction, full floors/lock ceremony, then registration 8 against the new locked SHA. |
| Any cell, reducer, truth support, tolerance outside the selected floors resolution, seed, K, or 4-of-5 change | New gate amendment/lock ceremony; prohibited in the candidate implementation lane. |

### 9.3 Registration-8 preconditions

Registration 8 may be posted only when all applicable boxes are satisfied:

1. the fable referee has adjudicated the open decisions in §10 and the resulting
   program/amendment text is ratified;
2. the new first-marriage spec is immutable, its train-only selection ledger is
   public, its full fit converges, and the support/convergence preflight abort is
   exercised on a failing synthetic case;
3. the §2.7 amendment is merged, `q*` and the full Q-grid ledger are frozen, and
   tests prove participation, `CellMarginal`, inverse rank, `I_proj`, odd-year
   behavior, and old substreams remain unchanged;
4. the separate remarriage conformance repair is merged and its entry-dissolved
   reconciliation passes without changing gate/floor bytes;
5. resolution A's disclosure is frozen **or** resolution B's new floor artifact
   has completed its full ceremony and the live lock points to its verified SHA;
6. the runner identifies candidate number 2, refuses the candidate-1 output path,
   writes `runs/gate_m6_candidate2_v1.json` exclusively, and binds exact source,
   design, floor, spec, dependency, and environment hashes;
7. the registration enumerates exactly the two model deltas, the conformance
   prerequisite, everything byte-carried, the five regression constraints, the
   effective train-side search sizes, a candid result forecast/modal failure
   shape, the one-shot rule, and “publish regardless”; and
8. no candidate-2 score, post-2014 outcome, or realized post-2014 macro value has
   been read during build, fit selection, amendment, or registration.

After registration, no parameter, support rule, floor, or implementation byte may
move. A pre-score designed abort is reported and requires a reviewed fix plus a
fresh registration; a completed run writes and publishes the fresh artifact once.

## 10. Open decisions for the fable referee

1. **Design home/classification:** approve this sibling program, the
   first-marriage change as a new candidate spec, the earnings change as a
   prospective §2.7 amendment, and a support/event-preserving remarriage patch as
   current-§2.8 conformance.
2. **First-marriage transport:** adopt the global smooth + boundary-flat cohort
   deviation and the stated train-only `C` selection, or name a different
   support-preserving rule before any fit. In particular, decide whether nearest
   fitted-cohort behavior for unseen decades is sufficiently conservative.
3. **Fit abort:** ratify `max_iter=10_000`, `tol=1e-8`, the warning/success checks,
   and the rule that a nonconverged full fit consumes no score and requires a new
   registration after repair.
4. **Earnings mechanism:** approve the single global positive-continuation
   refresh share, its fresh-rank pool, train-only objective, and new substream
   codes; or require age-specific persistence at the cost of a larger registered
   search. Participation-rank recoupling is not in candidate 2.
5. **Mandatory floors:** choose resolution A (frozen conservative tolerances with
   explicit 11.91% faithful false-fail risk) or resolution B (recommended full
   closed-domain floors ceremony). If B, approve the proposed v4 path only after
   the combined OC is freshly derived; `0.9575` is the provisional earnings-only
   result, not a fabricated combined result.
6. **Remarriage prerequisite:** confirm the separate lane's proof standard and
   whether any implementation detail crosses from conformance into a narrow §2.8
   amendment. Do not adjudicate the residual hazard until the assembly repair is
   measured.
7. **Regression block:** approve the additional all-five-on-≥4-of-5 constraint at
   the original candidate-1 tolerances even if a v4 floor loosens the two older
   earnings tolerances.
8. **Abort/registration accounting:** decide whether a train-fit designed abort
   consumes registration 8 or is recorded as a registration error; either way it
   cannot be silently retried under changed bytes.

## 11. What this proposal does not change or certify

This proposal does **not**:

- alter candidate 1's valid FAIL, its artifact, its 0/5 seed result, or its
  candidate-1 floor contract;
- edit any frozen v1/v2/v3 floor, gated cell, tolerance, reducer, seed, K,
  conjunction, truth support, F6 weight, or shock-window disposition;
- exclude the amendment-3g class, change the first-marriage truth side, or demote
  either formation cell;
- authorize realized post-2014 NAWI, change the pooled `CellMarginal`, its inverse,
  or the current projected-index formula;
- implement or silently absorb the remarriage conformance repair;
- tune the remarriage hazard on an assembly-contaminated result;
- weaken disability, divorce, or older-worker regression obligations; or
- claim that a candidate-2 PASS certifies mortality drift, widowhood, the
  2020–2022 shock window, entrants/open-panel dynamics, lag-5 autocorrelation,
  stocks, report-only AIME/PIA/claiming levels, or projection to 2100.

A candidate-2 PASS would certify only the unchanged locked family-A statement on
the registered temporal-holdout surface, under the selected and publicly locked
floors resolution, after an audited iterative search whose structural choices
have now observed candidate 1. That search-size limitation must accompany every
PASS claim.

## 12. Proposed registration ledger (non-operative)

The following block is a review aid. It becomes operative only if the referee
ratifies it and the required amendment/ceremonies land.

```json
{
  "program_id": "2026-07-16-m6-candidate2",
  "status": "PROPOSAL_NOT_REGISTERED",
  "candidate": 2,
  "artifact": "runs/gate_m6_candidate2_v1.json",
  "authority_comment": "4997635883",
  "deltas": {
    "first_marriage": "regularized global age/sex spline plus boundary-flat cohort deviation; train-only C selection; nonconvergence hard-aborts",
    "earnings": "prospective section-2.7 positive-continuation rank-refresh mixture with train-only q; participation and section-2.7.6 rank-level/index laws unchanged"
  },
  "prerequisite": {
    "remarriage": "entry-dissolved history/episode/event conformance under existing section 2.8; separate lane merged before registration"
  },
  "floors_resolution": "OPEN_REFEREE_DECISION",
  "recommended_floors_resolution": "new_closed_domain_floor_and_full_lock_ceremony",
  "frozen_locked_domain_earnings_oc": {
    "p_seed": 0.8760,
    "p_gate_4_of_5": 0.8809,
    "weak_power_floor": 0.90,
    "faithful_false_fail_probability": 0.1191
  },
  "regression_cells": {
    "rule": "all five clear original thresholds on at least 4 of 5 fresh gate seeds",
    "tolerances": {
      "divorce.18-44": 0.379,
      "incidence.20-66": 0.404,
      "recovery.20-66": 0.314,
      "earn_dlog_sd.older": 0.269,
      "earn_zero_rate.older": 0.163
    }
  },
  "prohibitions": [
    "no candidate-1 rescore or overwrite",
    "no realized post-2014 NAWI on scored path",
    "no truth/support/cell change",
    "no score before registration 8",
    "no silent retry after a designed abort"
  ]
}
```

[forensics-4]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4997635883
[forensics-reg]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4997460916
[grading]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4997458179
[candidate1-pr]: https://github.com/PolicyEngine/populace-dynamics/pull/225
[candidate1-artifact]: https://github.com/PolicyEngine/populace-dynamics/blob/8ff7b14fa89f021c4951ddfbd2102f795ff4de21/runs/gate_m6_candidate1_v1.json
[search-rule]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4948637741
[w1-c2-reg]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4952253568
[w1-c2-grade]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4953064479
[bartley]: https://doi.org/10.1371/journal.pone.0225715
[nist]: https://www.itl.nist.gov/div898/handbook/pri/section5/pri5998.htm
[dantony]: https://doi.org/10.1093/ije/dyae033
[eilers-marx]: https://doi.org/10.1214/ss/1038425655
[pya-wood]: https://doi.org/10.1007/s11222-013-9448-7
[sklearn-logit]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
[scipy-minimize]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
[cbo-earnings]: https://www.govinfo.gov/content/pkg/GOVPUB-Y10-PURL-LPS77496/pdf/GOVPUB-Y10-PURL-LPS77496.pdf
[karahan-ozkan]: https://doi.org/10.1016/j.red.2012.08.001
