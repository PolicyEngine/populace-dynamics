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

The proposed candidate-2 package has two model deltas and one landed conformance
foundation with outstanding pre-score proof:

1. replace the unbounded first-marriage age-by-cohort extrapolation with a
   regularized, support-aware interaction whose cohort deviation is flat beyond
   observed cohort-age support; make non-convergence a hard pre-scoring abort;
2. amend §2.7 prospectively to add a train-selected stable-coordinate
   rank-refresh mixture to
   positive-to-positive earnings transitions, damping conditional-rank memory
   while preserving the fitted `CellMarginal` object, its §2.7.6 rank-to-level
   map and inverse, the participation-gate architecture and coefficients, and
   `I_proj`; and
3. bind candidate 2 to the entry-dissolved remarriage conformance repair landed
   as [#226][repair-pr] at
   `c16cb9d563bd573ce2b537b19e403fbddec3cba6`, while closing the remaining
   fixture audit and pre-score reconciliation before registration or score.

The §2.8.3a power finding is orthogonal to model quality. The referee must choose
whether registration 8 keeps the conservative frozen tolerances with the power
gap disclosed or follows a new closed-domain floors ceremony. Section 6
recommends the ceremony.

### 1.1 Why this is a sibling program, not a §2.8 edit

`docs/design/m6_projection_engine.md` is ratified domain law. Its §2.8 harness
already requires the realized entry state and dissolution duration to survive
projection assembly (`docs/design/m6_projection_engine.md:943-994`); the
pre-#226 remarriage defect violated that existing law. #226 repaired it without
changing support, events, cells, or tolerances, so it is landed conformance, not
a new harness design.
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
remaining fixture audit or reconciliation finds that #226 changed the current
event/support law, a **narrow §2.8 amendment must be proposed in a different PR
before candidate-2 registration**; this document does not pre-authorize one.
Independently, selecting floors resolution B necessarily triggers its own narrow
prospective §2.8.4/gate-contract amendment because §2.8.4 pins v3; this sibling
program names that ceremony but does not perform it.

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
(`src/populace_dynamics/models/family_transitions/components/first_marriage.py:113-163`);
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

The frozen-v3 **six-cell earnings-subfamily** OC is `p_seed=0.9347`,
`p_gate=0.9626`; applying its locked tolerances on the actual closed domain yields
earnings-only `p_seed=0.8760`, `p_gate(≥4/5)=0.8809`. That is 1.91 percentage
points below the ratified 0.90 weak-power floor. Under the artifact's registered
independence-approximation OC, its complement is an 11.91% earnings-subfamily
faithful-candidate false-fail probability—the disclosed “about 12% under-power”
finding. It is **not** the full 11-cell false-fail rate.

The frozen flow subfamily has `p_seed=0.9559`. Extending the same registered
independence approximation gives an informative, non-operative combined estimate:
`0.9559 × 0.8760 = 0.8374` per seed and `p_gate≈0.8115` (about 18.85% false-fail).
That arithmetic is an inference from the two subfamily OCs, not a field in the
candidate-1 self-check; either floors resolution must freshly publish the combined
11-cell OC. Closed-domain-derived tolerances yield earnings-only
`p_seed=0.9300`, `p_gate=0.9575`; carrying the same flow approximation would yield
combined `p_gate≈0.9019`. No domain-derived tolerance reaches a metric cap and
neither vacuity nor near-tautology fires. The §2.8.3a near-unpassable trigger does.

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
| Clamp the final fitted hazard into the minimum/maximum hazard observed on a sex×cohort training envelope | reject | The envelope depends on an arbitrary aggregation and can hide separation or non-convergence. It treats the symptom after estimation instead of defining how the unsupported interaction transports. |
| Clamp the **whole hazard** to the nearest cohort-age boundary | reject as the primary law | It prevents the explosion but freezes the global age profile too; supported information from older cohorts at ages 24–29 would be discarded. |
| Penalized smooth global age/sex curve plus a regularized cohort deviation, with the cohort deviation held flat beyond that cohort's support | **adopt** | It borrows the age pattern from supported cohorts while preventing an unsupported age×cohort interaction from running away. |
| Global monotonicity over all ages | reject | First-marriage hazards rise and then fall; a global sign restriction is substantively wrong. The adopted boundary-flat evaluation applies only to the unsupported cohort deviation; it does not force the derivative inside support to zero. |
| Exclude out-of-support people from projection or scoring | reject | The cell is reachable, 3g is not causal, and exclusion would change truth support/floors after seeing a candidate failure. §2.7.6.5's principle applies by analogy: the model must cover the scored surface. |

This choice follows the campaign's estimation-versus-determinism norm. Novel
age×cohort combinations are explicitly diagnosed rather than mistaken for
interpolation ([Bartley et al. 2019][bartley]; [NIST extrapolation
guidance][nist]). Penalized multidimensional hazard surfaces are an established
way to borrow strength while controlling unstable interactions ([Dantony et al.
2024][dantony]); [Eilers and Marx 1996][eilers-marx] motivates smoothness
penalties generally, not this exact fixed-RCS ridge. [Pya and Wood
2015][pya-wood] show that derivative constraints can be imposed on smooth terms;
candidate 2 instead uses the narrower deterministic boundary-flat rule and does
not impose global monotonicity.

### 4.2 Proposed fitted law

Candidate 2 keeps the current outcome, never-married risk set, F6 weights, fixed
knots `{20,22,25,30,40}`, sex definition, birth-decade definition, and `≤2014`
truncation. It replaces only the unguarded cohort interaction and its fit-strength
selection.

Let `s=0` denote female and `s=1` male. Let `[a_s^-,a_s^+]` be the positive-fit-
weight age support for sex `s`, and `[a_c^-,a_c^+]` the support for fitted birth
decade `c`, pooled over sex because the fitted cohort deviation is shared. Let
`B_raw(a)` be the current restricted-cubic-spline basis. With the sorted oldest
birth decade as the reference (`γ_c=b_c=0` there), one dummy and one spline block
are included for each later fitted decade, exactly preserving the current
reference-cell coding. Define

```text
a_s* = clip(a, a_s^-, a_s^+)
a_c* = clip(a, a_c^-, a_c^+)
X_raw(a,s,c) = [B_raw(a_s*), s, s B_raw(a_s*), D_c, D_c B_raw(a_c*)]
logit h(a,s,c) = α + standardize(X_raw; μ_fit, σ_fit) θ.
```

The sex-specific global curve uses the actual target age inside that sex's pooled
training support. Only the cohort-specific deviation uses `a_c*`.
Thus a female in the 1990 cohort at age 24 receives the supported global
female-age-24 curve plus the 1990 deviation evaluated at its age-23 boundary; the
1990 interaction cannot extrapolate from `0.0538753` to nearly 1. If target age is
outside sex-specific pooled support, the global sex curve uses its nearest
boundary. If a
target birth decade is newer or older than every fitted decade, its cohort main
effect, support interval, and deviation come from the nearest fitted decade. No
row is excluded and no truth-side value enters this rule.

For every fit, compute `μ_fit` and population `σ_fit` (`ddof=0`) unweighted on the
fit rows' raw columns, in canonical `(person_id,year)` order; replace a zero SD by
1.0. The intercept is unpenalized and every other coefficient has the same L2
penalty. Normalize positive F6 sample weights to mean 1 within the fit,
`w_i*=n w_i/Σw_i`, so `C` has the same scale at each expanding pseudo-boundary.
The explicit objective is the normalized weighted Bernoulli loss plus
`||θ||²/(2 C n)`; reported pseudo-holdout deviance is the raw-F6-weighted mean
Bernoulli deviance. Penalty strength is selected before registration from

```text
C_GRID = {0.0001, 0.0003, 0.001, 0.003, 0.01,
          0.03, 0.1, 0.3, 1.0}
PSEUDO_BOUNDARIES = {2006, 2008, 2010}
```

At pseudo-boundary `b`, fit only never-married risk rows and events dated `≤b`,
then evaluate raw-F6-weighted Bernoulli deviance on dated rows in `b+1…b+4`,
applying the same support-extension rule. Average first within each boundary,
then give the three boundary deviances equal weight. The windows overlap by
design—2009–2012 receive repeated stress as recent pseudo-holdout evidence; they
are not described as independent. Publish each calendar year's multiplicity and
each sex×cohort support/event count. Choose the eligible `C` with the lowest mean
pseudo-holdout deviance; ties within `1e-12` choose the smaller `C` (stronger
regularization). Publish every rung, convergence certificate, support count,
deviance, and selected value before registration 8. No 2015+ row or
candidate-1 cell score may enter numerical selection.

### 4.3 Convergence is a hard designed abort

Every candidate-grid fit and the final `≤2014` fit uses LBFGS with
`max_iter=10_000`, `tol=1e-8`, deterministic row ordering, and the registered
standardization/weight normalization. Eligibility requires successful solver
termination, no convergence warning, and an independently recomputed infinity
norm of the explicit penalized-objective gradient `≤1e-6`. A grid rung failing
one condition is ineligible and remains in the published selection ledger.

If every `C` is ineligible, or the selected `C` cannot produce an eligible final
`≤2014` fit, selection ends `NO_REGISTERABLE_FIRST_MARRIAGE_FIT`: publish the
train-only ledger and do not post registration 8. That pre-registration outcome
is not a run abort and cannot consume a registration or score.

After registration, the locked runner refits and **must abort before any
projection, holdout-truth read, score, or candidate-artifact write** if the
selected full fit:

- reaches `max_iter`, reports unsuccessful termination, or emits a convergence
  warning;
- has a non-finite coefficient, linear predictor, or probability, or emits a
  probability outside the strict floating-point interval `(0,1)`;
- exceeds the gradient threshold; or
- cannot reproduce its registered support, standardization, weight, coefficient,
  and selected-`C` checksums.

An iteration ceiling is not evidence of successful termination; the optimizer
reports those separately ([scikit-learn `LogisticRegression`][sklearn-logit];
[SciPy optimization contract][scipy-minimize]). The current `converged=False`
state therefore becomes a designed abort, not metadata attached to a model that
is allowed to score. A registered-run abort is recorded publicly, writes no
candidate artifact, and is not converted to a FAIL or silently refit with a new
law. Any repaired byte or changed environment lock requires a fresh registration;
§10 leaves the exact-byte retry/accounting question to the referee.

### 4.4 Required disclosures and acceptance

Before score assembly the runner must verify and retain for the eventual fresh
artifact, for each sex, cohort, and target age:

- fitted support endpoints, in-support count/event weight, and whether the global
  or cohort term was boundary-evaluated;
- convergence status, iterations, warning count, selected `C`, all grid results,
  and deterministic design-matrix checksum;
- the age-18–29 predicted hazard table, including female-1990 ages 23–29; and
- counts and F6 weight of in-support, age-out-of-support, and unseen-cohort rows in
  each gated first-marriage cell.

The public train-selection ledger exists before registration; a completed run
publishes the retained pre-score diagnostics in its fresh artifact, while a
designed abort publishes them in the abort record. They are transport diagnostics,
not alternate scores. Candidate 2 still clears
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
participation-gate formula/fit and its RNG address unchanged. It amends the
positive-continuation rank transition and RNG registry only. A changed 2016 level
can nevertheless change the unchanged gate's 2018 participation probability;
unchanged architecture is not bit-identical downstream participation. Realized
post-2014 NAWI stays **PROHIBITED**; the static decomposition shows it would not
solve persistence anyway.

### 5.2 Options adjudicated

| Option | Disposition | Reason |
|---|---|---|
| Substitute realized 2016/2018 NAWI | prohibited | It leaks the holdout, explains at most 12.3% of p10, and cannot move the static persistence measures. |
| Change `CellMarginal` rank↔level mapping | reject | The inverse is interior-exact and forensics found no implementation error; this would disturb a pinned mechanism without targeting the dominant channel. |
| Re-estimate `W_CURRENT/W_PRIOR/W_ANCHOR` and `λ` jointly | reject for candidate 2 | A four-dimensional search changes short-run, prior, anchor, and permanent-coordinate memory at once. The forensic record localizes excessive persistence but does not identify four parameters; use one train-selected damping degree first. |
| Refit or recouple participation and rank in candidate 2 | defer | Participation can contribute to level/growth, but the hard localization is conditional-rank memory and the older zero-rate cell passed. The refresh's endogenous 2018 participation effect is measured; coefficients/features are not another search dimension. |
| Remove all prior/anchor conditioning | reject | It is a large uncalibrated discontinuity and would discard useful persistence rather than estimate how much to retain. |
| Fully unconditional positive-rank refresh | reject | It would erase `u_A`/`u_w` association as well as recent-rank memory, although forensics did not isolate the permanent coordinate as the defect. |
| Train-selected stable-coordinate refresh mixture for positive continuers | **adopt** | It directly attenuates current/prior-rank memory while retaining the anchor/permanent coordinate, the existing conditional draw when not refreshed, and the fitted rank-to-level object. |

Longitudinal microsimulation commonly separates persistent and transitory earnings
components and estimates them from observed histories ([CBO 2006, Appendix
D][cbo-earnings]). PSID evidence also warns that persistence varies with age
([Karahan and Ozkan 2013][karahan-ozkan]). Candidate 2 nevertheless uses one
global refresh share rather than adding age-specific parameters: the limited
pre-2014 pseudo-holdouts should estimate one new degree of freedom first.

### 5.3 Proposed rank-refresh mechanism

The same-step participation draw and re-entry draw are exactly as now. For a
person who is positive at `t`, participates at `t+2`, and would receive the
current conditional donor rank `u_cond`, candidate 2 adds:

```text
B_refresh ~ Bernoulli(q*)
u_out = u_stable  if B_refresh = 1
        u_cond    otherwise.
```

`u_stable` is a weighted `k=25` single-record draw from positive-to-positive
forward-pair donors in the same target five-year age bin. The age-bin restriction
is a fixed **new** conditioning choice in the amendment—not incumbent pool
behavior—so damping recent-rank memory cannot also change target-age composition.
The pool is sorted by `(person_id,period_tp2)` and uses `weight_tp2`. Distance
drops `u_t` and `u_tm2` but retains the pinned asymmetric stable-coordinate branch:
`|u_A(d)-u_A(i)|` for Q0, and
`|0.1u_w(d)+0.9u_A(d)-u_A(i)|` otherwise. In particular, target `u_w(i)` remains
omitted, exactly as current law; weighted neighbor and record selection otherwise
use the current no-jitter donor law. The refresh thus removes recent-rank
conditioning without erasing donor anchor/permanent-rank association. As in the
current helper, use `min(25,n_pool)` donors with its stable tie-break; an empty
target-age pool is a non-registerable fit, not an unregistered adjacent-bin
fallback.

The unchanged
`rank_to_level` then maps `u_out` through the same age-bin `CellMarginal` and
`I_proj`; the next step re-ranks that level through the unchanged inverse. This
preserves those fitted objects, **not** the realized positive output marginal:
changing the output-rank mixture intentionally changes projected moments.

The §2.7 amendment must add two isolated per-person/per-period substreams:

```text
4: memory-refresh-gate
5: memory-refresh-rank
```

The existing gate/donor/re-entry streams `{1,2,3}` and their draw order remain
unchanged. A refresh therefore cannot perturb the same-step participation uniform
or incumbent conditional donor draw. It can change a later participation
probability through the carried level, so that feedback is scored and constrained.
For every eligible positive continuer, code 4 draws one refresh uniform and code 5
draws one stable-donor uniform in canonical person order regardless of `q` or the
refresh outcome; `q` only thresholds the code-4 uniform and switches the selected
rank. This nests all grid rungs under common random numbers. Odd years still
consume no RNG.

### 5.4 Train-only estimation of `q*`

Estimate, disclose, and freeze `q*` before registration 8:

```text
Q_GRID = {0.00, 0.05, ..., 0.95, 1.00}
PSEUDO_BOUNDARIES = {2006, 2008, 2010}
FIT_SEED = 5200
SELECTION_DRAW_SEEDS = {6200, ..., 6219}
```

For pseudo-boundary `b`, use only earnings rows dated `≤b` to refit the complete
forward law and construct the `b`-anchored domain. Refit `I_proj` by shifting the
pinned trailing-decade rule to `[b-9,b]`; realized `NAWI_{b+2}` and
`NAWI_{b+4}` are forbidden. Project `b→b+2→b+4`. The scored support is realized
positive-weight row support at `{b+2,b+4}` intersected with persons carrying the
required `b` anchor and `u_w`, symmetrically for truth and projection, ages
25–64. Call the existing `earnings_cells` reducer with
`level_years=(b+2,b+4)` and `change_years=(b,b+2,b+4)` and retain exactly:
`earn_p10.prime`, `earn_dlog_mean.prime`, `earn_dlog_sd.older`,
`earn_mob_h1_diag`, `earn_autocorr_lag2`, and `earn_zero_rate.older`.
Those parameterized reducers are at
`src/populace_dynamics/harness/m6_cells.py:476-586`.

For each boundary, split the full `b` anchor by person at the exact floor seeds
`0…99`, then intersect each half with the `b` domain—the F7 order—and run the
existing `run_floor`/`earnings_cells` machinery. The standardizer `σ_{j,b}` is
that machinery's `realized_sigma`, not its folded-score SD or tolerance. For each
`q`, common random numbers bind all 20 selection draws: identical person/period
addresses and identical streams 1–5 across every grid rung. Average each projected
moment across the 20 draws, compare it with the full-support truth moment using
the gated cell's exact log-ratio/absolute-gap metric, and set

```text
J(q) = Σ_b Σ_j∈{p10, mean, mobility, autocorr} [score(j,b,q)/σ(j,b)]².
```

The exact 100-seed split and `realized_sigma` construction are at
`src/populace_dynamics/harness/m6_cells.py:42` and
`src/populace_dynamics/harness/m6_cells.py:613-674`.

The three overlapping boundary windows receive equal weight; this is deliberate
recent-history stress, not three independent samples. Any undefined truth cell,
non-positive standardizer, undefined projected draw, support mismatch, or
non-regenerated surface makes that `q` ineligible; the same defect at `q=0`
invalidates the entire selector. A `q` is feasible only if, at every boundary,
both older-worker standardized scores satisfy

```text
score(dlog_sd,b,q)/σ(dlog_sd,b) <= score(dlog_sd,b,0)/σ(dlog_sd,b) + 1
score(zero_rate,b,q)/σ(zero_rate,b) <= score(zero_rate,b,0)/σ(zero_rate,b) + 1.
```

Simulation-noise protection is deterministic. Compute `J(q)` on all 20 draws,
on the fixed halves `6200…6209` and `6210…6219`, and in 20 delete-one-draw
replicates. A nonzero `q` must improve on `q=0` in the all-draw and both half-draw
objectives. Among feasible values, let `q_min` minimize all-draw `J`; use the
delete-one jackknife
`SE=sqrt[(19/20)Σ_r(J_{-r}-mean_r J_{-r})²]` for `J(q_min)` and select the
**smallest** `q` with `J(q)≤J(q_min)+SE[J(q_min)]`. Exact ties choose the smaller
`q`. This one-SE rule, not a `1e-12` comparison alone, limits departure under
Monte Carlo noise.

As a hard equivalence preflight, `q=0` must reproduce the current generator
bit-for-bit at every generated person-period level and participation state for all
pseudo-boundaries and all 20 draws; the six reduced moments must therefore also
match, and streams 1–3 must retain their exact states. Failure invalidates the
mechanism rather than selecting around it.

The selection ledger must publish every `q`, cutoff, support, simulated/truth
moment, standardizer, feasibility result, full/half/jackknife objective, selected
value, effective search size, and checksums of fit rows, pools, support IDs, and
RNG registry. Governance is two-stage: first a prospective §2.7 amendment ratifies
the generic mechanism and authorizes this **train-only, non-scoring prototype**;
then a reviewed lock addendum freezes `q*`, its ledger SHA, pool law, and substream
codes before registration 8. No 2015+ row, realized post-2014 macro value,
candidate-1 seed score, or candidate-2 score may enter numerical estimation.

### 5.5 What a result would mean

This is not a guarantee that candidate 2 passes. If the train-only rule selects
`q*=0`, the honest conclusion is that pre-2014 pseudo-holdouts do not support the
proposed damping; the program recommends pausing registration 8 and returning to
the referee rather than forcing a nonzero value or registering a knowingly no-op
earnings delta. The referee must ratify that disposition in §10. If a nonzero
value is selected and candidate 2 still fails, the fresh
artifact becomes the next forensic record. A candidate-2 pass first-certifies
this amended forward law only on the registered 2016/2018 cells; it does not
transfer gate-1 certification or validate a 2100 earnings projection.

## 6. The mandatory floors finding

### 6.1 The two registered resolutions

| Resolution | What stays/moves | Honest benefit | Honest cost |
|---|---|---|---|
| **A. Retain frozen v3 tolerances** | Keep `runs/m6_holdout_floors_v3.json`, SHA-256 `e931c88622fad84e8f8b2cf18940cbe27da1c93e0d009dfbaa3d6c6cae050c77`, all six tolerances, §2.8.4, and `gates.yaml` byte-identical. Registration 8 and every candidate-2 report disclose earnings-only `p_seed=0.8760`, `p_gate=0.8809`, the 1.91pp floor deficit, and the independence-OC's 11.91% subfamily false-fail probability, plus a freshly computed combined OC. | Preserves the ratified artifact and is conservative: harder to pass, no false-PASS direction. | Knowingly scores the closed domain with a floor derived on a larger domain; fails the ratified 0.90 weak-power floor. The same registered approximation provisionally puts the combined gate at only `≈0.8115`, worse than the six-cell “about 12%” headline. |
| **B. Registered closed-domain floors ceremony** | Build a new, never-overwritten truth-only artifact (proposed `runs/m6_holdout_floors_v4.json`) on exactly `realized support ∩ 2014 earnings domain`; rederive the six earnings tolerances, recompute earnings and combined OC/vacuity, adversarially reproduce it, SHA-pin it, then ratify a narrow prospective §2.8.4/gate-contract amendment and full lock before registration 8. The v3 artifact remains immutable and historical. Flows/cell definitions/reducers remain unchanged. | Prices noise on the population actually scored; the self-check's provisional earnings OC is `0.9575`, the same-method combined arithmetic is `≈0.9019`, and no vacuity flag fires. | Moves tolerances after candidate 1 has been observed, so governance must prove truth-only/candidate-blind derivation and preserve candidate 1's historical contract. The provisional combined margin is thin; a fresh ceremony can still pause. |

Neither resolution is authorized by this proposal. Silent threshold movement,
editing v3 in place, or applying v4 retrospectively to candidate 1 is prohibited.

### 6.2 Recommendation

**Recommend resolution B, the registered closed-domain floors ceremony.** The
scorer's domain is now empirically known and §2.8.3a expressly routes the
earnings-subfamily `0.8809 < 0.90` result to a ceremony finding. Re-deriving on
that exact truth domain is
more coherent than knowingly carrying a lineage mismatch into a new candidate,
provided the change is candidate-blind, independently reproduced, and locked
before registration 8. Candidate 1 remains a published FAIL under v3; there is no
self-rescue.

The ceremony must not simply copy the six provisional tolerances. It must rebuild
the closed-domain half-splits from the registered truth-only derivation, publish
all seeds and support counts, recompute the **combined** 11-cell OC with the
unchanged flow floors, re-run both power directions and caps, and stop if any
existing floor invariant fails. Because §2.8.4 explicitly pins v3
(`docs/design/m6_projection_engine.md:1905-1908`), resolution B necessarily
includes a separate narrow prospective §2.8.4/gate-contract amendment; only its
later ratifying lock PR may edit `gates.yaml` or add the new floor artifact. If the
referee selects resolution A, both the six-cell 11.91% finding and the fresh
combined-OC disclosure become mandatory registration/artifact fields, not a
footnote.

## 7. Remarriage §2.8 conformance repair — landed foundation and live proof

Forensics on the pre-repair candidate-1 source at
`8ff7b14fa89f021c4951ddfbd2102f795ff4de21` found lost history: projected
dissolved exposure was only 12.37–13.29% of truth while the event numerator was
33.58–39.95%. The following defect reconstruction is historical, and every code
citation in this paragraph is pinned to that pre-repair tree. The adapter read
entry-divorced/widowed state and computed `dissolution_year` from
`years_since_dissolution` (`src/populace_dynamics/engine/marital.py:82-128`) but
serialized only open/in-window emitted episodes before reassembly
(`src/populace_dynamics/engine/marital.py:145-267`). A widow-only post-assembly
patch changed person-years but explicitly created no episode/event
(`src/populace_dynamics/models/family_transitions/components/initial_states.py:206-242`).
Reassembly defaulted a person with no prior emitted change-point to
`never_married` (`src/populace_dynamics/data/transitions.py:377-420`), recomputed
episode order by `cumcount` (`src/populace_dynamics/data/transitions.py:426-440`),
called emitted rank 0 a first marriage
(`src/populace_dynamics/data/transitions.py:453-470`), and required rank ≥1 plus a
prior emitted dissolution for remarriage
(`src/populace_dynamics/data/transitions.py:473-509`).

That pre-repair behavior contradicted the existing seed/duration/change-point
pins at `docs/design/m6_projection_engine.md:943-994`. #226 landed the conformance
repair at `c16cb9d563bd573ce2b537b19e403fbddec3cba6`. On the merged tree, the
carrier-episode implementation now occupies
`src/populace_dynamics/engine/marital.py:260-302`; it passes realized dissolved
history through `_assemble_panel` rather than exhibiting the historical omission.
The five original proof conditions now map to the landed [#226 referee
record][repair-referee] as follows:

1. **Discharged:** carrier episodes preserve every entry-divorced and
   entry-widowed person's realized entry state, dissolution year, and prior-order
   information through `_assemble_panel`; the person-year test is at
   `tests/test_m6_panel_builders.py:363-394`.
2. **Discharged:** the existing `allow_exact_matches=False` dating/event law is
   unchanged, and the first in-window marriage of an entry-dissolved person is
   proved to remain a remarriage at
   `tests/test_m6_panel_builders.py:397-421`.
3. **Discharged on #226's record:** its real-frame reproduction and draw-identity
   witnesses preserved support, F6 weights, cells, truth rates, tolerances, RNG
   addresses, non-remarriage histories, and the pooled marriage-event count and
   weight; the truth side and all gate/floor/run bytes were untouched.
4. **Partially discharged:** the merged fixtures discriminate entry-divorced,
   entry-widowed, in-window remarriage, and sustained no-remarriage exposure. A
   fixture-by-fixture registration audit must confirm that coverage and add or
   identify a discriminating same-year-boundary case.
5. **Outstanding registration precondition:** publish a candidate-2 pre-score
   conformance reconciliation on the exact forensic classes.

The live reconciliation must separately publish the count and F6 weight of
emitted marriages relabeled from projected first marriage to remarriage by
restored prior history, by gated year/sex/age cell. #226's referee record measured
the aggregate assembly/relabel effect, but it did not supply that candidate-2
gated-cell ledger. Candidate 2 therefore cannot attribute any first-marriage
score movement solely to the new hazard law; the artifact must decompose the
landed conformance relabel from the estimator delta.

This docs-only proposal does not modify #226. If the remaining audit or
reconciliation reveals changed event semantics, support, a truth reducer, or a
cell, the repair leaves its conformance classification and requires a narrow
prospective §2.8 amendment before registration. With assembly conformance landed,
the residual remarriage hazard may be measured, but it must never be tuned from
the 2015–2019 holdout result.

### 7.1 Measured post-repair residual and disposition

The residual is now measured, not hypothetical. Direct-standardizing the fitted
remarriage law on truth dissolved support gives projected hazards
`0.05110–0.05457` against truth `0.03562–0.03927`: a `1.30–1.53` ratio and
`|ln|=0.263–0.427` against the `0.403` tolerance, or 65.3–105.8% of the bar.
Independently, #226's repaired real-frame reconciliation puts exposure at
97.4020–100.9204% of truth and the event numerator at 129.8756–149.9906%, whose
numerator-only log band is `0.262–0.405`. These agree on a directional overshoot
that straddles the gate threshold. The pre-repair candidate-1 result was a 2.75×
FAIL; after conformance, `remarriage.18-64` is a known coin-flip cell.

The referee must choose among three honest dispositions:

| Disposition | Honest case | Honest cost |
|---|---|---|
| **Train-only delta through a follow-up proposal amendment** | Define and select a remarriage transport delta solely on `≤2014` expanding pseudo-holdouts, then ratify and lock it before registration 8. This addresses a systematic miss without reading the scored residual as a tuning target. | Adds a third model delta, another search ledger, and another amendment/review surface; the training evidence may select no change. |
| **Register with a candid modal-failure forecast** | Keep this program's two model deltas and make the registration forecast explicitly name `remarriage.18-64`, its measured band, and modal FAIL posture. | Knowingly spends the candidate-2 one-shot with a cell whose evidence reaches 105.8% of tolerance and provides no mechanism for it. |
| **Designed pause** | Do not post registration 8 until a separately ratified learning plan resolves the residual. | Avoids a knowingly fragile run but delays all candidate-2 evidence and, without a train-only work item, learns nothing new. |

**Recommendation: the train-only delta through a follow-up proposal amendment.**
The overshoot is same-direction across the repaired real-frame seeds and agrees
with the assembly-independent direct standardization, so a forecast-only run
would knowingly accept nearly zero remarriage margin. A pause alone has no
learning mechanism. The follow-up must pre-register its law, pseudo-boundaries,
search size, no-op rule, and acceptance diagnostics before fitting; it may use no
2015–2019 row, score, residual, or truth moment numerically. If admissible
train-only evidence selects a no-op or cannot support a ratifiable law, the
required disposition is pause, not a forced delta.

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
fresh candidate-2 values. Candidate-2 acceptance requires **both** the live
11-cell gate result and this original-threshold regression block to pass. Under
resolution A a gate pass implies the block; under resolution B it does not,
because live v4 could use `0.279/0.168` while the block retains `0.269/0.163`.
The artifact therefore records `gate_contract_result`,
`must_not_regress_result`, and their conjunction. A live gate-PASS plus
regression-FAIL is published as `GATE_PASS_REGRESSION_FAIL`, never advertised as
an accepted candidate-2 PASS.

The first-marriage implementation must not mutate the separate divorce estimator
(`src/populace_dynamics/models/family_transitions/components/divorce.py:67-124`).
The earnings refresh leaves the participation formula, coefficients, and RNG
address unchanged, but carried earnings can change later gate outcomes. Both
older-worker thresholds therefore remain binding on rank-dispersion and
participation feedback.

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
that already exists remain immutable.

### 9.2 Ceremony matrix

| Proposed change | Required governance before implementation/run |
|---|---|
| Support-aware first-marriage estimator/new candidate spec | Referee ratifies this candidate-family law; add a new immutable registry spec; register selected train-only fit strength and support behavior before score. No §2.8 edit. |
| Stable-coordinate refresh law and substreams | Prospective §2.7 amendment first ratifies the generic mechanism/selector and authorizes a train-only prototype; a reviewed lock addendum then freezes nonzero `q*` and the ledger before registration 8. Keep §2.7.6.2/.3/.4 unchanged. |
| Entry-dissolved remarriage repair preserving current event/support law | **Landed** as #226 at `c16cb9d`. Bind the candidate source to that commit; finish §7 condition 4's fixture audit and condition 5's exact-class/gated-cell reconciliation before registration. No threshold/floor change. If semantics moved, stop for a narrow §2.8 amendment. |
| Retain frozen tolerances | Referee records resolution A and mandates the earnings-only `0.8809` / 11.91% plus fresh combined-OC disclosure; no floor or lock edit. |
| Re-derive closed-domain floors | Separate truth-only v4 derivation, adversarial reproduction, narrow prospective §2.8.4/gate-contract amendment, and full lock ceremony; then registration 8 against the new locked SHA. |
| Add the must-not-regress acceptance block | Program/registration ratification; artifact reports live-gate, original-threshold block, and conjunction. Moving its five thresholds or 4-of-5 rule requires a new gate amendment. |
| Any cell, reducer, truth support, tolerance outside the selected floors resolution, seed, K, or 4-of-5 change | New gate amendment/lock ceremony; prohibited in the candidate implementation lane. |

### 9.3 Registration-8 preconditions

Registration 8 may be posted only when all applicable boxes are satisfied:

1. the fable referee has adjudicated the open decisions in §10 and the resulting
   program/amendment text is ratified;
2. the new first-marriage spec is immutable, its train-only selection ledger is
   public, its full fit converges, and the support/convergence preflight abort is
   exercised on a failing synthetic case;
3. the §2.7 amendment is merged, `q*` and the full Q-grid ledger are frozen, and
   tests prove `q=0` generator-level bit equivalence and that participation
   formula/coefficients, `CellMarginal`, inverse rank, `I_proj`, odd-year behavior,
   and old substreams remain unchanged; the tests must also expose changed later
   participation when a refreshed carried level crosses the unchanged gate;
4. the bound source contains landed #226 (`c16cb9d`), §7 condition 4's remaining
   fixture audit is closed, and condition 5's entry-dissolved reconciliation
   passes without changing gate/floor bytes; that reconciliation publishes the
   post-repair residual transport measurement, and §10 decision 9 is adjudicated.
   If the train-only-delta disposition is chosen, its follow-up proposal
   amendment and lock must also be complete; a chosen pause forbids registration;
5. resolution A's earnings-only and combined disclosure is frozen **or**
   resolution B's new floor artifact and prospective §2.8.4 amendment have
   completed their full ceremony and the live lock points to its verified SHA;
6. the runner identifies candidate number 2, refuses the candidate-1 output path,
   writes `runs/gate_m6_candidate2_v1.json` exclusively, and binds exact source,
   design, floor, spec, dependency, and environment hashes;
7. the registration enumerates the two model deltas proposed here plus any
   separately ratified remarriage delta selected under §10 decision 9, the landed
   conformance proof, everything byte-carried, the five regression constraints,
   the effective train-side search sizes, a candid result forecast/modal failure
   shape naming `remarriage.18-64`, the one-shot rule, and “publish regardless”;
   and
8. the published candidate-1 artifact/forensic summaries are disclosed as the
   structural-design evidence, but no unpublished or row-level post-2014 holdout
   value, candidate-2 result, or realized post-2014 macro value enters estimator
   fitting, pseudo-selection, implementation choices, or registration. Resolution
   B's separately authorized truth-only floors lane may read the exact scoring
   domain solely to derive/reproduce floors; it cannot read a candidate output.

After registration, no parameter, support rule, floor, or implementation byte may
move. A registered-run pre-score designed abort is reported and writes no candidate
artifact; any changed byte requires a reviewed fix plus a fresh registration. A
completed run writes and publishes the fresh artifact once.

## 10. Open decisions for the fable referee

1. **Design home/classification:** approve this sibling program, the
   first-marriage change as a new candidate spec, the earnings change as a
   prospective §2.7 amendment, and landed #226 as support/event-preserving
   current-§2.8 conformance.
2. **First-marriage transport:** adopt the global smooth + boundary-flat cohort
   deviation and the stated train-only `C` selection, or name a different
   support-preserving rule before any fit. In particular, decide whether nearest
   fitted-cohort behavior for unseen decades is sufficiently conservative.
3. **Fit abort:** ratify `max_iter=10_000`, `tol=1e-8`, gradient threshold,
   warning/success checks, `NO_REGISTERABLE_FIRST_MARRIAGE_FIT` before
   registration, and the registered-run abort rule. Decide whether an exact-byte
   retry after a registered abort needs a new registration number.
4. **Earnings mechanism:** approve the single global positive-continuation
   stable-coordinate refresh share, train-only objective/one-SE rule, and new
   substream codes; or require age-specific persistence at the cost of a larger
   registered search. Participation-rank recoupling and joint weight/`λ`
   re-estimation are not in candidate 2. Confirm that a train-selected `q*=0`
   pauses registration 8 and returns for a new proposal, as recommended.
5. **Mandatory floors:** choose resolution A (frozen conservative tolerances with
   explicit 11.91% earnings-subfamily false-fail risk and fresh combined OC) or
   resolution B (recommended full closed-domain floors/§2.8.4 ceremony). If B,
   approve the proposed v4 path only after the combined OC is freshly derived;
   `0.9575` is earnings-only and `≈0.9019` is provisional arithmetic, not a
   ratified combined result.
6. **Landed remarriage record:** confirm that #226 is current-§2.8 conformance,
   adjudicate the remaining fixture-by-fixture proof under §7 condition 4, and
   retain condition 5's candidate-2 reconciliation. If either reveals a semantic
   change, require a narrow §2.8 amendment.
7. **Regression block:** approve the additional all-five-on-≥4-of-5 constraint at
   the original candidate-1 tolerances even if a v4 floor loosens the two older
   earnings tolerances.
8. **Abort/registration accounting:** confirm that a pre-registration no-fit
   outcome consumes no registration; decide whether a registered exact-byte abort
   consumes the registration number. It cannot be silently retried under changed
   bytes.
9. **Remarriage residual disposition:** choose (a) a train-only-estimated
   remarriage transport delta through a follow-up proposal amendment, (b)
   registration with a candid modal-failure forecast explicitly naming
   `remarriage.18-64` and its `0.263–0.427` score band against `0.403`, or (c) a
   designed pause. The recommendation is (a): two independent post-repair
   measurements show a systematic same-direction overshoot, while (b) knowingly
   spends the one-shot with almost no margin and (c) supplies no learning plan.
   Any delta is selected only on `≤2014` pseudo-holdouts; a train-only no-op or
   unratifiable law resolves to (c), never tuning against the 2015–2019 residual.

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
- alter, revert, or claim the landed #226 conformance repair as a candidate-2
  model delta;
- authorize a remarriage model delta in this document; the recommended delta
  requires its own follow-up proposal amendment and referee review;
- tune the remarriage hazard from the pre-repair assembly-contaminated result or
  the 2015–2019 holdout residual;
- weaken disability, divorce, or older-worker regression obligations; or
- claim that a candidate-2 PASS certifies mortality drift, widowhood, the
  2020–2022 shock window, entrants/open-panel dynamics, lag-5 autocorrelation,
  stocks, report-only AIME/PIA/claiming levels, or projection to 2100.

An **accepted** candidate-2 PASS—live gate plus must-not-regress block—would
certify only the registered family-A statement on the temporal-holdout surface,
under the selected and publicly locked floors resolution, after an audited
iterative search whose structural choices have now observed candidate 1. That
search-size limitation must accompany every PASS claim.

## 12. Machine-readable consistency ledger (non-operative)

The following tagged ledger binds the proposal's evidence and immutability claims
to the frozen artifact and authority. It becomes operative only if the referee
ratifies the program and the required amendment/ceremonies land. In this docs-only
lane it is validated read-only; no `tests/`, `runs/`, or gate file is changed.

<!-- m6-candidate2-consistency-ledger: derive evidence fields only from issue
     comment 4997635883, the immutable candidate-1 artifact at 8ff7b14, the
     #226 referee record at comment 4998474459, and the frozen v3 floor. Do not
     hand-edit a value without re-running the read-only consistency check
     recorded in the PR. -->

```json m6-candidate2-consistency-ledger
{
  "program_id": "2026-07-16-m6-candidate2",
  "status": "PROPOSAL_NOT_REGISTERED",
  "candidate": 2,
  "fresh_artifact": "runs/gate_m6_candidate2_v1.json",
  "evidence_binding": {
    "authority_comment": "4997635883",
    "grading_comment": "4997458179",
    "candidate1": {
      "artifact": "runs/gate_m6_candidate1_v1.json",
      "commit": "8ff7b14fa89f021c4951ddfbd2102f795ff4de21",
      "sha256": "546a9739f8d1c7d21a91a07eb902c8af9bda92cdaa8f7917f312894f6a861b24",
      "verdict": "FAIL",
      "n_gated_cells": 11,
      "n_failed_cells": 6,
      "n_seed_pass": 0,
      "n_gate_seeds": 5,
      "all_failed_cells_fail_all_five_seeds": true,
      "worst_score_over_tolerance": {
        "first_marriage.18-29|female": 4.18,
        "earn_mob_h1_diag": 3.47,
        "remarriage.18-64": 2.75,
        "earn_autocorr_lag2": 2.13,
        "earn_dlog_mean.prime": 1.77,
        "earn_p10.prime": 1.61
      },
      "passed_cells": [
        "divorce.18-44",
        "incidence.20-66",
        "recovery.20-66",
        "earn_dlog_sd.older",
        "earn_zero_rate.older"
      ]
    },
    "first_marriage": {
      "female_1990_max_fit_age": 23,
      "hazard_age23": 0.0538753,
      "hazard_age24": 0.999999928,
      "hazard_age25_plus": 1.0,
      "fit_max_iter": 1000,
      "fit_converged": false,
      "truth_direct_standardized_range": [0.0451, 0.0543],
      "projection_direct_standardized_range": [0.359, 0.372],
      "scored_projection_range": [0.184, 0.200],
      "amendment_3g_gated_weight_share": 0.0225,
      "score_change_if_3g_removed": 0.025142780
    },
    "earnings": {
      "beyond_tolerance_of_100": {
        "earn_p10.prime": 92,
        "earn_dlog_mean.prime": 95,
        "earn_mob_h1_diag": 100,
        "earn_autocorr_lag2": 100
      },
      "projected_over_truth_ratio": {
        "earn_p10.prime": 0.739,
        "earn_dlog_mean.prime": 0.557,
        "earn_mob_h1_diag": 1.181,
        "earn_autocorr_lag2": 1.246
      },
      "I_proj_log_slope": 0.0225137847,
      "realized_nawi_max_p10_gap_share": 0.123,
      "realized_nawi_static_persistence_effect": 0.0,
      "realized_post2014_nawi_prohibited": true
    },
    "remarriage": {
      "projected_over_truth_dissolved_exposure_range": [0.1237, 0.1329],
      "projected_over_truth_event_numerator_range": [0.3358, 0.3995],
      "direct_projected_hazard_range": [0.05110, 0.05457],
      "direct_truth_hazard_range": [0.03562, 0.03927],
      "direct_projected_over_truth_ratio_range": [1.30, 1.53],
      "postrepair_projected_over_truth_exposure_range": [0.974020, 1.009204],
      "postrepair_projected_over_truth_event_numerator_range": [1.298756, 1.499906],
      "residual_log_ratio_score_range": [0.263, 0.427],
      "tolerance": 0.403,
      "max_score_over_tolerance": 1.058,
      "postrepair_posture": "KNOWN_COIN_FLIP"
    },
    "floors": {
      "v3_artifact": "runs/m6_holdout_floors_v3.json",
      "v3_sha256": "e931c88622fad84e8f8b2cf18940cbe27da1c93e0d009dfbaa3d6c6cae050c77",
      "n_closed_domain_persons": 13561,
      "n_closed_domain_earnings_rows": 45606,
      "per_cell_tolerance_locked_to_domain_rederived": {
        "earn_autocorr_lag2": [0.087, 0.087],
        "earn_dlog_mean.prime": [0.043, 0.043],
        "earn_dlog_sd.older": [0.269, 0.279],
        "earn_mob_h1_diag": [0.052, 0.054],
        "earn_p10.prime": [0.221, 0.284],
        "earn_zero_rate.older": [0.163, 0.168]
      },
      "locked_tolerances_on_closed_domain": {
        "scope": "six_cell_earnings_subfamily",
        "p_seed": 0.8760,
        "p_gate_4_of_5": 0.8809,
        "weak_power_floor": 0.90,
        "independence_oc_false_fail": 0.1191
      },
      "domain_rederived_earnings": {
        "p_seed": 0.9300,
        "p_gate_4_of_5": 0.9575
      },
      "flow_subfamily_p_seed": 0.9559,
      "combined_same_method_inference": {
        "operative": false,
        "locked_domain_p_seed": 0.8373684,
        "locked_domain_p_gate_4_of_5": 0.8115004,
        "domain_rederived_p_gate_4_of_5": 0.9019126
      }
    }
  },
  "proposed_deltas": {
    "first_marriage": "regularized age/sex spline plus boundary-flat cohort deviation; train-only C; convergence certificate",
    "earnings": "prospective section-2.7 stable-coordinate rank-refresh mixture with train-only nonzero q; fitted section-2.7.6 rank-level/index objects unchanged"
  },
  "prerequisite": {
    "remarriage": {
      "status": "LANDED_CONFORMANCE_WITH_LIVE_PRE_SCORE_PROOF",
      "pr": 226,
      "commit": "c16cb9d563bd573ce2b537b19e403fbddec3cba6",
      "condition_1_entry_history": "DISCHARGED",
      "condition_2_event_label": "DISCHARGED",
      "condition_3_invariance": "DISCHARGED_ON_226_REFEREE_RECORD",
      "condition_4_fixture_matrix": "PARTIAL_SAME_YEAR_BOUNDARY_CHECK_LIVE",
      "condition_5_candidate2_reconciliation": "LIVE_PRECONDITION"
    }
  },
  "remarriage_residual_disposition": {
    "status": "OPEN_REFEREE_DECISION_BEFORE_REGISTRATION_8",
    "options": [
      "TRAIN_ONLY_DELTA_VIA_FOLLOW_UP_PROPOSAL_AMENDMENT",
      "CANDID_MODAL_FAILURE_FORECAST_NAMING_REMARRIAGE_18_64",
      "DESIGNED_PAUSE"
    ],
    "recommended": "TRAIN_ONLY_DELTA_VIA_FOLLOW_UP_PROPOSAL_AMENDMENT",
    "train_only_noop_fallback": "DESIGNED_PAUSE",
    "post2014_residual_tuning_prohibited": true
  },
  "floors_resolution": "OPEN_REFEREE_DECISION",
  "recommended_floors_resolution": "new_closed_domain_floor_plus_section_2_8_4_and_full_lock_ceremony",
  "regression_cells": {
    "rule": "all five clear original thresholds on at least 4 of 5 fresh gate seeds; acceptance also requires live gate pass",
    "tolerances": {
      "divorce.18-44": 0.379,
      "incidence.20-66": 0.404,
      "recovery.20-66": 0.314,
      "earn_dlog_sd.older": 0.269,
      "earn_zero_rate.older": 0.163
    }
  },
  "immutability_guards": [
    "no candidate-1 rescore or overwrite",
    "no realized post-2014 NAWI on scored path",
    "no 2015-2019 remarriage residual in model selection or tuning",
    "no candidate output in train-only selection or truth-only floors derivation",
    "no truth/support/cell change outside a ratified ceremony",
    "no score before registration 8",
    "no silent retry after a designed abort"
  ]
}
```

## 13. Methodological references

- Meridith L. Bartley et al. (2019), “Identifying and characterizing
  extrapolation in multivariate response data,” *PLOS ONE* 14(12), e0225715
  ([DOI][bartley]); and NIST/SEMATECH, “Motivation: How do we Use the Model
  Beyond the Data Domain?”, §5.5.9.9.8 of the *e-Handbook of Statistical
  Methods* ([NIST][nist]).
- Emmanuelle Dantony et al. (2024), “Multidimensional penalized splines for
  survival models: illustration for net survival trend analyses,”
  *International Journal of Epidemiology* 53(2) ([DOI][dantony]); Paul H. C.
  Eilers and Brian D. Marx (1996), “Flexible smoothing with B-splines and
  penalties,” *Statistical Science* 11(2) ([DOI][eilers-marx]); and Natalya Pya
  and Simon N. Wood (2015), “Shape constrained additive models,” *Statistics
  and Computing* 25(3), 543–559 ([DOI][pya-wood]). These support the general
  estimation tools; §4 states where candidate 2 deliberately uses a different,
  narrower deterministic transport rule.
- Congressional Budget Office (2006), *Projecting Labor Force Participation and
  Earnings in CBO's Long-Term Microsimulation Model*, especially Appendix D,
  “Permanent and Transitory Earnings Shocks” ([public PDF][cbo-earnings]); and
  Fatih Karahan and Serdar Ozkan (2013), “On the persistence of income shocks
  over the life cycle: Evidence, theory, and implications,” *Review of Economic
  Dynamics* 16(3), 452–476 ([DOI][karahan-ozkan]).
- Official optimizer contracts: scikit-learn's `LogisticRegression` API
  ([documentation][sklearn-logit]) and SciPy's `minimize` API
  ([documentation][scipy-minimize]).

[forensics-4]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4997635883
[forensics-reg]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4997460916
[grading]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4997458179
[candidate1-pr]: https://github.com/PolicyEngine/populace-dynamics/pull/225
[candidate1-artifact]: https://github.com/PolicyEngine/populace-dynamics/blob/8ff7b14fa89f021c4951ddfbd2102f795ff4de21/runs/gate_m6_candidate1_v1.json
[repair-pr]: https://github.com/PolicyEngine/populace-dynamics/pull/226
[repair-referee]: https://github.com/PolicyEngine/populace-dynamics/issues/226#issuecomment-4998474459
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
