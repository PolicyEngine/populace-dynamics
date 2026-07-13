# gate_w1 amendment 2 (proposal) — the family-A concept cells and the C1 fingerprint

- **Amendment id**: `2026-07-12-w1-family-a-concept-cells`
- **Gate**: `gate_w1` (W1 representative-frame transport, roadmap #113 M5, workstream #100)
- **Surface**: three cell groups on the amended (post-amendment-1) **55-cell**
  gated surface — (a) the family-C **C1** fingerprint (`fingerprint.ppi_nra`); (b)
  the two family-A **18-24 participation** cells
  (`earnings_participation.18-24|{female,male}`); (c) the four family-A **65+
  marital/coresident** cells (`marital_share.married.65+|{female,male}` and
  `coresident_spouse.65+|{female,male}`). **7 cells total**: 1 family-C binary + 6
  family-A joints.
- **Ceremony stage**: PROPOSAL (draft), incorporating the adversarial referee
  round (PR #169 comment [4953763322][ref-round], verdict AMEND THE AMENDMENT).
  This document is a step of the amendment ceremony (proposal → adversarial
  referee → fixes → verification → ratify-by-merge → flip). **It moves no
  threshold and edits no `gates.yaml` cell.** The prospective flip happens in a
  separate ratifying PR, only after the ceremony clears.
- **Amendment class**: STRUCTURAL, prospective, ZERO threshold movement (the
  amendment-1 precedent). Unlike amendment 1 — which demoted family-B cells that
  sit OUTSIDE the family-A OC machinery, leaving the family-A characteristic
  byte-identical — this amendment removes **6 cells from family A itself**, so the
  family-A faithful-candidate OC is **recomputed** on the residual 47-cell surface
  (§4, §5). No tolerance, σ, anchor value, or ordering is edited; the demoted
  cells' tolerances/orderings are retained under report-only.
- **Fixes round**: this revision incorporates the adversarial referee round
  ([4953763322][ref-round], verdict AMEND THE AMENDMENT). The round confirmed the
  demotion boundary is principled (the referee's own five extra
  construction-attempts confirm 65+ unreachability; the whittling question is
  answered against self-interest, §3.1) and required correcting the **per-cell
  grounds** (finding 2): the blanket "no candidate can clear any of the 7" was
  **false on the committed record** — candidate 1 *passed*
  `coresident_spouse.65+|female` on 4 of 5 seeds — so §4c/§5/§6/§8 now carry the
  per-cell truth (per-seed pass records; ceiling-vs-window for the cells the
  stationary distribution cannot reach; scored-duplicate for the coresident cell
  that straddles the boundary), the retained surface carries the mandated candor
  and series forecast (finding 4), and the σ ranges are corrected (finding 5).
- **Evidence base**: W1 forensics 2 (`runs/gate_w1_forensics2_v1.json`,
  registration [4953088871][reg2], grading [4953311492][grade2]) — Q6 (the 65+
  divorce over-accumulation) and Q9 (the measured 18-24 concept gap + the C1
  consolidation); W1 forensics 1 (`runs/gate_w1_forensics1_v1.json`, registration
  [4951218279][reg1]) — Q5 (the C1 non-reversal analytic) and Q1 (the 65+
  equilibration ceiling); candidate 1 (`runs/gate_w1_candidate1_v1.json`, PR #162,
  registration 4950931131) and candidate 2 (`runs/gate_w1_candidate2_v1.json`, PR
  #167, registration 4952253568) — the committed per-seed records on every cell in
  scope; the amended gate (`gates.yaml` `gate_w1`, post-amendment-1: 53 family-A +
  2 family-C gated); the adversarial round ([4953763322][ref-round]) — the per-cell
  correction and five additional construction-attempts. The amendment-1 doc
  (`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md`, PR #164 / #165) is
  the direct structural precedent.

[reg2]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4953088871
[grade2]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4953311492
[reg1]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4951218279
[ref-round]: https://github.com/PolicyEngine/populace-dynamics/pull/169#issuecomment-4953763322

## 1. Summary

Amendment 1 demoted the 10 family-B SSA-anchor cells and restored the gate's
headline faithful-candidate OC from a structural 0 to **0.9481 × I(family C)** on
a **55-cell** surface (53 family-A joints + 2 family-C fingerprints). W1 forensics
2 (grading [4953311492][grade2], verdict "**gate_w1 remains unpassable as amended
— amendment 2 precedes candidate 3**") then established that **three cell groups
on that residual surface cannot be cleared at the gate-relevant ≥4-of-5-seed level
by any contract-consistent candidate**, for three distinct — and each
candidate-independent — reasons:

- **(a) The C1 fingerprint** (`fingerprint.ppi_nra`: progressive price indexing
  must outrank NRA→70 in 75-year payroll savings). Its required reversal **does
  not occur** under any contract-consistent transport. Forensics 1 Q5 proved this
  **analytically** on the **heaviest plausible tail** (the upper read, most
  favourable to PPI): PPI savings **0.0169** sit far below NRA **0.2023** (gap
  **0.1854**), and a realistic tail moves PPI *down* to **0.0137**. The analytic
  leg carries the demotion; both candidates' fingerprint runs agree (τ 0.667,
  swap not realised) but are **bit-identical** — one measurement run twice
  (§6), a consistency check, not two independent confirmations.
- **(b) The two 18-24 participation cells.** The miss is a **population-concept
  delta**, not a fit-support gap: the PSID family-earnings panel is a **head/spouse
  universe** (participation ~0.865 pooled), while the CPS frame counts **every**
  18-24 person — students, dependents living with parents — and participates
  ~0.644. The concept gap is **22.1pp pooled** (female 17.3, male **27.6**), above
  the 15pp threshold the registration pre-named. The male cell fails **0/5** seeds
  for both candidates (support 0.929 vs pass-window ceilings 0.750–0.814); the
  female cell caps at **2/5** (candidate 2; candidate 1 0/5), its 17.3pp gap
  admitting the head/spouse support only on the two loosest seeds.
- **(c) The four 65+ marital/coresident cells.** The deployed 65+ married share is
  **capped at the CANDIDATE_16 stationary ceiling** (~0.577 F / ~0.717 M) because
  the certified pooled hazards **over-accumulate divorce** (deployed 65+ divorced
  ~0.19–0.21 vs the frame's older-cohort ~0.06–0.09; excess **+0.120 F / +0.127
  M**) — a **cohort-vintage mismatch** between the pooled 1998–2022 hazards and
  the frame's older 65+ cohorts, **not** the pre-registered widowhood. The
  ceiling sits below the frame's pass window: the **male** married/coresident
  cells fail **0/5** (needs ≥0.77/≥0.75 vs ceiling ~0.72), and the **married-65+|
  female** cell caps at **2–3/5** by ceiling-vs-window arithmetic (§4c). The
  **coresident-65+|female** cell, however, **candidate 1 clears on 4 of 5 seeds**
  — it is demoted not on impossibility but because in every committed deployment
  `coresident_spouse` is regenerated as `(marital_state == married)` (deployed
  coresident/married ratio **1.0** on all seeds), so it is a **scored duplicate of
  the demoted married quantity** that straddles a slightly easier window; gating a
  duplicate of a demoted cell — and letting a composition-blind pass certify a 65+
  composition the same run gets +0.12 divorced wrong — is the very
  headline-vs-achievable defect this amendment exists to cure.

Because family A's seed rule is a **conjunction** (a seed passes iff every gated
cell is within tolerance) and family C is a conjunction over its fingerprints,
each group makes the amended gate **unpassable at the gate level**: the achievable
faithful-candidate OC is **0** (§3), driven by the cells that fail every seed (the
male 18-24 and 65+ cells; C1's non-reversal).

**Per group, two options are analysed (§4); this proposal recommends the
report-only disposition for all three** (the package: demote C1 + the 18-24 pair +
the 65+ quad — **7 cells** — to report-only, with per-cell machine reasons; §5).

- **Recommended package — demote 7 cells to report-only**, each on grounds
  individually true against the committed record: C1 `fingerprint_reversal_not_realized`
  (analytic); the two 18-24 cells `population_concept_delta_head_spouse_universe`;
  the married-65+ pair and the coresident-65+ **male** cell
  `cohort_vintage_hazard_frame_mismatch` (the stationary ceiling the deployment
  cannot reach); the coresident-65+ **female** cell
  `scored_duplicate_of_demoted_married_quantity` (the boundary-straddling duplicate
  candidate 1 clears 4/5). Zero threshold movement; the demoted tolerances/ordering
  stay in-contract (report-only). The gate's faithful-candidate OC moves from an
  **achievable 0** to a well-defined **p_seed 0.9344 / p_gate 0.9623 × I(C2)** on
  the residual **48-cell** surface (47 family-A joints + the C2 fingerprint).
- **Deferred alternative — re-anchor / re-specify**: each group has an honest
  path (§4), but every one presupposes an evidence-base amendment not archived or
  not contract-consistent today, plus a fresh referee round. Deferred (§10).

**The amended 48-cell surface is not itself known passable (§5, §9): it removes
proven-unreachable cells; it does not establish a passable surface.** Two
unpassability risks are already visible in committed artifacts — the five
`hh_size` cells (Q7 graded REFUTED) and C2's `cap_150k` adjacency — and are
forecast as amendment-3 risks now, not discovered one PR at a time.

## 2. The finding (W1 forensics 2 Q6/Q9; W1 forensics 1 Q5/Q1)

`runs/gate_w1_forensics2_v1.json` and `runs/gate_w1_forensics1_v1.json` are
one-shot, `reported_not_gated`, `train_frame_side_only` diagnostics that "publish
regardless of any verdict". Every re-simulated component reproduces the committed
machinery bit-for-bit (`q6/q7/q8 instrumentation_bit_identity_max_dev == 0.0`).

### 2a. C1 — the reversal that does not occur (forensics 1 Q5, forensics 2 Q9)

C1 requires the representative frame to **reverse** the compressed-career order so
that progressive price indexing (PPI) outranks NRA→70 in 75-year payroll savings.
Forensics 1 Q5 tested it on the **upper read** — a transported-career panel of
positive prime-age earners with no zero/low-earning years, which
**over**-concentrates the upper tail (`frac_payroll_above_wage_base` 0.377, the
heaviest plausible tail, **most favourable to the swap**). Even there PPI savings
are **0.0169** against NRA's **0.2023** (`ppi_minus_nra` −0.1854). The corrected
tail applies the certified per-cell p0 (a lighter, realistic tail,
`tail_lighter_than_upper_read: true`) and moves PPI to **0.0137** vs NRA **0.2022**
— **further** from reversal. `c1_robustness_answer.answer_non_reversal_is_robust`
is **true**. The **analytic** argument carries the demotion: PPI would need to
close a ~12× savings gap the mechanism does not produce. The two candidates'
fingerprint runs both report non-reversal (τ 0.667), but they are **bit-identical**
— the fingerprint machinery consumed the same byte-carried career panel in both
runs, so they are one measurement, not two independent confirmations (§6).

### 2b. The 18-24 participation concept gap (forensics 2 Q9)

`q9_concept_cells.concept_gap_18_24_participation` measures the transport target
directly by **universe restriction** on the train side:

| Universe | female | male | pooled |
|---|---:|---:|---:|
| PSID head/spouse (the generator's fit universe) | 0.815 | 0.929 | **0.865** |
| CPS all-person (the frame the cell scores against) | 0.642 | 0.653 | **0.644** |
| concept gap (PSID − CPS) | 0.173 | **0.276** | **0.221** |

The pooled gap is **22.1pp**, `exceeds_15pp_amendment_threshold: true` — the 15pp
threshold pre-registered in the forensics-2 registration before measurement. The
PSID head/spouse universe (employed, independent young adults, ~0.89) is not the
CPS all-person universe (students, dependents, ~0.64); v2's Q2 boundary extension
already fits the PSID 18-24 support, and the cells still fail by the concept gap.

### 2c. The 65+ divorce over-accumulation and the stationary ceiling (forensics 2 Q6; forensics 1 Q1)

`q6_marital_calibration_frame` seeds each frame adult's marital ENTRY state at
`BASE_ENTRY_AGE=25` from the PSID-entry model and evolves the certified
CANDIDATE_16 hazards. At 65+ the deployed married share undershoots the frame:

| 65+ | deployed married | frame married | miss (D−A) | divorced excess (D−A) |
|---|---:|---:|---:|---:|
| female | 0.573 | 0.691 | −0.118 | **+0.120** |
| male | 0.707 | 0.856 | −0.149 | **+0.127** |

The realised dissolution channel is **divorce, not widowhood**
(`widowhood_channel_realized: false` both sexes; deployed *under*-widows). The
certified pooled divorce-minus-remarriage steady state sits far above the frame's
older-cohort divorced share — a **hazard-vs-frame cohort-vintage mismatch**.

Critically, the 40-year hazard window makes the terminal married share a **near-
stationary ceiling almost independent of the entry level**: forensics 1 Q1's
entry-0 equilibration lands **0.5767** (65+ F), and Q6's entry-0.58 chain lands
**0.5728** — a contraction of only ~0.007 per unit of entry stock, so even an
entry stock of 1.0 buys ≤ +0.004 over the 0.577 ceiling. The male ceiling is
~0.717. These ceilings sit **below the frame's pass windows** (§4c), which is why
the 65+ cells cannot be cleared by any permitted entry-state lever. The grading
adjudicated Q6 "**HALF-HELD**": the 65+ channel is divorce over-accumulation of
the **hazard-evolution class — re-calibration prohibited, so the 65+ pair is
amendment territory or unfixable**; the contract adjudication is exact — a
CPS-anchored ENTRY model is permitted and fixes 25-34 but **"CANNOT fix the 65+
undershoot"**, and back-solving the entry to reproduce the terminal is **"the
identity in disguise"**.

## 3. Why gate_w1 as amended (post-amendment-1) is unpassable

The gate is a conjunction: **family A (≥4-of-5 seeds) ∧ family C (both
fingerprints reverse)**; family B gates nothing after amendment 1. A family-A seed
passes iff **every** gated cell is within tolerance on that seed. Three groups
break the conjunction at the gate level:

1. **C1 cannot reverse** (§2a). C1's non-reversal is robust, so `I(C1) = 0` for
   any contract-consistent candidate ⇒ family C = 0.
2. **The 18-24 male cell fails on every seed** (§2b, §4b): its concept gap
   (|ln(0.93/0.63)| ≈ 0.39) exceeds tolerance 0.221 on all 5 seeds (0/5 for both
   candidates) ⇒ every seed fails ⇒ family A never clears ≥4-of-5.
3. **The 65+ male cells fail on every seed** (§2c, §4c): the divorce
   over-accumulation caps deployed 65+ married/coresident at ~0.71 vs the frame's
   ~0.75–0.79 windows (0/5 for both candidates).

The female cells (18-24 F, married-65+ F, coresident-65+ F) do **not** fail every
seed — they cap at 2–4/5 (§4) — so the group is not "no candidate can clear any
cell"; it is that the **male cells alone force every seed to fail**, and the
female cells cannot lift the group to a pass either.

**The crux — the contract's headline OC prices sampling noise, not concept/vintage
bias.** The `faithful_candidate_oc` is a **draw-noise-free half-normal** model: it
assumes a faithful candidate whose per-cell score is half-normal around the
**sampling** σ, and on that model the six family-A cells in scope are trivially
passable — per-cell pass probabilities **0.997–0.998**. But the forensics
establishes that idealisation is **unreachable**: the actual misses are
**systematic** biases, not sampling excursions. Across the six cells and both
candidates the per-seed miss is **0.11–0.44 ln** against a sampling σ of
**0.027–0.072** — a **2.0σ–6.6σ** systematic gap (referee reproduction,
[4953763322][ref-round]), against a mean-of-20-draws draw-noise of 0.002–0.007.
The **sub-3σ end is exactly the female cells that sometimes clear tolerance**: for
`coresident-65+|female` the achievable deployment sits **on** the pass boundary
(4/5), so the 0.998 sampling-σ OC wildly overprices a cell that is a gate-level
coin-flip and a duplicate of a demoted quantity — the headline-vs-achievable
defect this amendment cures. So the achievable OC is:

> **achievable faithful-candidate gate OC (amended surface) = p_gate(A) × I(C1) ×
> I(C2) = (0 via the male 18-24 and 65+ cells) × (0 via C1) × I(C2) = 0,**

**over-determined** (family A *and* family C each force it) against the contract's
stated 0.9481 × I(family C) headline. This is a defect of the *contract* — it
scores a head/spouse-fit generator against an all-person frame (18-24), a
pooled-vintage hazard ceiling against an older-cohort frame (65+), and asks for a
reversal the mechanism does not produce (C1) — exactly the class amendment 1
corrected one level up.

### 3.1. The whittling question (referee finding 1), answered

This is the second accommodation-direction amendment to gate_w1 in two days, so the
referee's central question was whether the gate is being whittled to passability.
It is not: the demotion boundary tracks one principle — **demote iff the committed
forensics establishes that no contract-permitted lever reaches the cell at the
gate ≥4-of-5 level; retain every cell with a live permitted lever, however badly
the built candidates miss it.** Applied against self-interest: the retained surface
keeps cells failing **5/5 for both candidates** — `hh_size_share.{1..5plus}`,
`marital_share.married.25-34|male`, `earnings_participation.35-44|female`,
`marital_share.never_married.25-34|male` — because each maps to a permitted
candidate-3 lever (Q8 sex covariate, proven 4/4; Q6 CPS-anchored entry, adjudicated
permitted; Q7 roster/window, permitted). A whittler would have demoted these; the
proposal does not. The 15pp concept-gap threshold was pre-registered before
measurement. The referee's own five extra construction-attempts (§4c) confirm the
65+ cells are unreachable — per-band and entry-55 seeding move them *worse*.

## 4. Options — per cell group

### 4a. Group (a): the C1 fingerprint binary

**Option a1 — report-only with disclosure (RECOMMENDED for this group).** Demote
`fingerprint.ppi_nra` to report-only, tagged `fingerprint_reversal_not_realized`.
The **ordering is still recorded per run** (the deployed savings order and the
PPI−NRA gap): the reversal question is **retired as a certification claim** — it is
robustly false (§2a) — while the deployed PPI−NRA gap stays the single most
valuable published diagnostic, telling a reader how far the representative frame's
above-second-bend AIME mass sits from the reversal threshold.

**Option a2 — re-specification.** Replace C1 with a defensible alternative
fingerprint statistic the anchors support. Enumerated honestly, the set is
**empty**. Mermin's signed 75-year savings order already has PPI (−0.14) **above**
NRA→70 (−0.50); the C1 fingerprint was the #113 hypothesis that a representative
frame flips the **magnitude** order so PPI's *absolute* savings **exceed** NRA's.
Forensics 1 Q5 falsifies exactly that magnitude reversal under every
contract-consistent tail (PPI 0.017 ≪ NRA 0.202). No published anchor asserts a
representative-frame PPI-over-NRA magnitude ordering; the continuous PPI−NRA gap is
a report-only diagnostic, not a binary gate; and re-specifying to the generator's
own deployed order is scoring it against itself. Re-specification reduces to a1
(publish the ordering + gap, certify nothing).

**Recommendation (a): a1.** The reversal claim is robustly false and cannot be
certified; the ordering + PPI−NRA gap are the valuable published diagnostic.

### 4b. Group (b): the 18-24 participation pair

**Option b1 — report-only pending a concept bridge (RECOMMENDED for this group).**
Demote `earnings_participation.18-24|{female,male}` to report-only, tagged
`population_concept_delta_head_spouse_universe`. The **male** cell is
gate-unclearable (0/5 both candidates; support 0.929 vs pass-window ceilings
0.750–0.814); the **female** cell caps at 2/5 (candidate 2; candidate 1 0/5) — its
17.3pp gap admits the head/spouse support (0.8149) only on the two loosest seeds
(upper edges 0.846/0.850; the others 0.765–0.793). Both are the **same** universe
mismatch; the pair is demoted together on the shared concept-delta ground. Report-
only, zero threshold movement, pending a concept bridge — a generator that models
**non-head** young-adult participation (a re-estimation W1 does not certify) or a
re-anchor (b2).

**Option b2 — re-anchor to a head/spouse-universe target.** Deferred: no archived
published head/spouse young-adult participation anchor; restricting the frame's
18-24 universe to heads/spouses changes the family-A estimand (an amendment); the
generator's own head/spouse rate is a near-identity. Needs an evidence-base
amendment + referee round (§10).

**Recommendation (b): b1.** The concept gap is measured (22.1pp ≥ 15pp); the pair
is report-only until the bridge exists.

### 4c. Group (c): the 65+ marital/coresident quad — per-cell grounds and construction-attempts

**Option c1 — report-only (RECOMMENDED).** Demote all four cells to report-only,
each on the ground individually true against the committed record:

| Cell | Committed pass record (c1, c2) | Machine reason | Ground |
|---|---|---|---|
| `marital_share.married.65+\|female` | **2/5, 1/5** | `cohort_vintage_hazard_frame_mismatch` | **ceiling-vs-window** (below) |
| `marital_share.married.65+\|male` | 0/5, 0/5 | `cohort_vintage_hazard_frame_mismatch` | ceiling ~0.717 vs needed ≥0.774–0.787 |
| `coresident_spouse.65+\|male` | 0/5, 0/5 | `cohort_vintage_hazard_frame_mismatch` | ceiling ~0.717 vs needed ≥0.753–0.773 |
| `coresident_spouse.65+\|female` | **4/5, 2/5** | `scored_duplicate_of_demoted_married_quantity` | **duplicate/boundary-straddle** (below) |

**Married-65+|female — ceiling-vs-window (caps 2–3/5, not the +0.12 magnitude).**
The achievable deployment is capped at the CANDIDATE_16 stationary level ~0.577
(Q1 entry-0 lands 0.5767; Q6 entry-0.58 lands 0.5728; contraction ~0.007 per unit
entry, so any entry stock ≤ 1.0 buys ≤ +0.004). Candidate 1's deployed rbar is
0.570–0.587, against the per-seed lower window edges [0.5619, 0.5950, 0.5798,
0.5853, 0.5904]: it clears only the two seeds whose lower edge dips below the
ceiling (2/5; candidate 2 1/5). With ±~0.01 seed-half composition it caps at 2–3
of 5, never 4 — unclearable **at the gate level**, but by ceiling-vs-window
arithmetic, **not** by the +0.12 divorced-excess magnitude (which is the male
cells' 7–8pp shortfall, not the female cell's 1.4pp).

**Coresident-65+|female — scored duplicate that candidate 1 clears 4/5 (stated
plainly).** Candidate 1 **passed** this cell on 4 of 5 seeds (scores
0.129/0.164/0.142/0.156/0.173 vs tolerance 0.168; flags [T,T,T,T,F]); candidate 2
2/5. It is **not** unclearable, and this proposal does not claim it is. In every
committed deployment `coresident_spouse` is regenerated as `(marital_state ==
married)` — the deployed coresident/married ratio is **1.0 on all seeds for both
candidates** — so the cell scores the **same** deployed value as the demoted
married cell against a slightly easier window (frame coresident/married ratio
~0.97–0.99, tolerance 0.168 vs 0.163), which is exactly why the identical value
straddles this window at 2–4/5 while failing the married window at 2/5. It is
demoted because (i) retaining it would **gate a duplicate of a demoted quantity**,
and (ii) a composition-blind pass would **certify a 65+ composition the same run
gets +0.12 divorced wrong** — and, sitting on the pass boundary, its 0.998
sampling-σ OC wildly overprices it, the headline-vs-achievable defect this
amendment cures. (`coresident_spouse.65+|male` is the same regenerated duplicate,
but is additionally a hard 0/5 against its own ceiling, so it carries the ceiling
reason.)

**Option c2 — re-anchor.** Deferred (§10).

**The candidate-independence argument (the construction-attempt method).** The
prohibition is on holdout tuning and identity; hazard re-calibration to the frame
would be tuning. Every permitted lever is attempted and each fails to lift the 65+
married/coresident **ceiling** to the frame window — so the male cells stay 0/5 and
the female married cell stays ≤3/5 for **any** contract-consistent candidate:

| # | Attempted permitted lever | Effect on 65+ married ceiling | Verdict |
|---|---|---|---|
| 1 | **CPS-anchored entry-level model** (Q6's permitted lever) | Entry stock barely moves the 40-year-evolved terminal (contraction ~0.007/unit); recalibrating entry to the CPS young-adult level cannot raise the ~0.577 F / ~0.717 M ceiling to the frame window. | fails |
| 2 | **Direct 65+ entry seed** | Reads the frame's own married (identity) or back-solves the terminal (inverse-map = identity in disguise). **Prohibited.** | fails |
| 3 | **Cohort/vintage-conditioned hazards** | Hazard re-calibration of the locked pooled 1998–2022 hazards. **Prohibited.** | fails |
| 4 | **Extra permitted covariate re-weighting** | The divorced steady state is a property of the pooled hazards, not the covariate mix; can't move divorced by +0.12. | fails |
| 5 | **Widowhood channel** (pre-registered) | Not realised (deployed under-widows); wouldn't fix a *divorce* over-accumulation. | fails |
| 6 | **Deployment draw (K=20 mean)** | Stochastic noise (σ ≈ 0.02–0.05); can't shift divorced by 0.12. | fails |
| R1 | **Per-band entry** (each adult enters at their own band's lower edge, 65+ from the PSID 65+ band 0.538 F / 0.696 M) — referee [4953763322][ref-round] | lands **0.4228 F / 0.6004 M**, *worse* by −0.15 / −0.11 | fails 0/5 |
| R2 | **Entry-55 for the 65+ band** (enter at 55 from the PSID 55-64 band, 10–30y window) — referee [4953763322][ref-round] | lands **0.4951 F / 0.6090 M**, *worse* by −0.08 / −0.10 | fails 0/5 |
| R3 | **Max-entry bound** (any entry stock ≤ 1.0 at 25, long window) — referee [4953763322][ref-round] | ceiling ≤ ~0.580 F / ~0.717 M (contraction 0.007–0.010/unit) | caps married F 2–3/5; males dead |

Levers R1/R2 are the sharpest result: the intuitive "seed old cohorts from
old-cohort train data" lever the six-row table never tries makes the 65+ cells
**worse** — the certified hazards' old-age dissolution erodes any short-window entry
stock below the stationary level, and PSID's own 65+/55-64 entry stocks sit below
the frame everywhere. Over the *entire* permitted entry-state family the achievable
set is **[0.42, 0.58] F / [0.60, 0.72] M** against needs **≥0.56–0.59 F / ≥0.77–0.79
M** — the 65+ demotion is not premature, and the one cell the attempts *can* reach
(coresident F, via the stationary ceiling) is the one candidate 1 already reached,
demoted on the duplicate ground, not on impossibility.

## 5. Recommendation

**Adopt the report-only package for all three groups** (a1 + b1 + c1): demote the
**7 cells** — C1, the 18-24 pair, the 65+ quad — each with its per-cell machine
reason. It (i) removes **exactly** the cells the committed forensics establishes no
contract-permitted lever reaches at the gate ≥4-of-5 level (the male 18-24 and 65+
cells 0/5; married-65+|female capped 2–3/5 by ceiling-vs-window; C1's non-reversal),
**plus** the one scored duplicate of a demoted quantity (coresident-65+|female,
which candidate 1 clears 4/5 — demoted because it duplicates the demoted married
cell), no more and no fewer; (ii) preserves every other gated cell and every
tolerance/σ/anchor/ordering byte-identical (zero threshold movement); (iii) keeps
the C1 ordering, the 18-24 tolerances, and the 65+ tolerances in-contract as
report-only disclosure; (iv) names each defect with a per-cell-true machine reason;
and (v) restores a well-defined gate OC.

**OC consequence (recomputed — differs from amendment 1).** Amendment 1 left family
A byte-identical; here six of the seven demoted cells **are** family-A cells, so
the family-A characteristic is recomputed on the residual **47-cell** surface
(draw-noise-free half-normal, the derivations basis):

- **Before**: 53 gated family-A cells → p_seed **0.922** / p_gate **0.9481**
  (contract headline; achievable OC 0, §3).
- **After**: 47 gated family-A cells → p_seed **0.9344** / p_gate **0.9623**.

Each demoted family-A cell has a faithful pass-prob 0.997–0.998, so removing the
six **raises** p_seed slightly. The overall gate OC moves from an **achievable 0**
to **p_gate(A) 0.9623 × I(C2)** on the residual **48-cell** surface (47 family-A
joints + the C2 fingerprint).

**The 48-cell surface is not known passable (referee finding 4).** The amendment
removes proven-unreachable cells; it does **not** establish a passable surface, and
"0.9623 × I(C2)" must not be read as I(C2) being a live coin-flip. Two unpassability
risks are already committed:

- **hh_size (5 retained cells, all 5/5-fail for both candidates).** Forensics-2
  Q7's pre-registration was graded **REFUTED**: the two permitted levers
  (coresident_parent roster + fuller fertility window) **jointly clear only
  size-2**; sizes 1/3/4/5+ improve but stay out of tolerance; the artifact states
  the family "**is not closable by these two entry-state levers alone**" and needs
  "**a coresidence-composition repair beyond the initial roster**" whose contract
  status is **unadjudicated**. That is a committed insufficiency finding, not a
  candidate-quality miss.
- **C2's full ordering has never been produced and the only measured lever moves
  it away.** Both candidates deployed `[elimination, cap_150k, +2pp, +1pp]` (τ
  0.333) with **bit-identical** exhaustion deltas (elimination 41.0, cap_150k
  **16.745**, +2pp 5.705, +1pp 2.551) — one measurement run twice. The deployed
  `cap_150k` delay is **16.7 years** against Smith's rank-4 anchor (**1 year**),
  and forensics-1's corrected (realistic) tail moves the order to `[cap_150k,
  elimination, +1pp, +2pp]` — τ vs required **−1/3**, further away. Forensics-1's
  own `c2_note` asserts robustness **only** for the elim↔+2pp pair, "not the full
  C2 ordering". Nothing in the evidence base adjudicates whether the `cap_150k`
  adjacencies are reachable on a representative frame — the same question shape
  that demoted C1.

**Series forecast (referee finding 4).** The gate trajectory is lock **65 gated**
→ amendment 1 **55** → amendment 2 **48**; the committed evidence already names the
next candidates. **Amendment-3 risks, pre-named now:** the five `hh_size` cells and
C2's `cap_150k` adjacency. The standing rule: if candidate 3's co-designed
roster/window lever fails to close the coresidence-composition residual, that
failure becomes amendment-3 evidence **only** via the same forensics-then-ceremony
standard (a registered, `reported_not_gated` diagnostic, then a public proposal +
adversarial referee + verification + ratify), never a bare demotion. C2's failure
is **not** yet proven candidate-independent — no analytic leg exists and the
career-panel transport is a live candidate lever — so C2 is **retained**; but if
candidate 3's fingerprint runs byte-carry the career panel again, the "empirical
burden" framing expires and a Q5-grade `cap_150k` robustness analysis becomes the
prerequisite.

The re-anchor/re-specification alternatives (a2/b2/c2) are deferred to future
bridge amendments (§10).

## 6. No-self-rescue compliance

The standing rule (`gate_w1.governance.amendment_rules`, inherited from `gate_1`):
*no candidate's committed run verdict changes under a rule proposed after that run.*
The **rescue set is empty** — demoting the 7 cells rescues **neither** committed
candidate, shown by recomputing each candidate's verdict on the amended 48-cell
surface:

1. **Candidate 1 (PR #162) STANDS FAIL — over-determined.** On the amended surface
   (47 family-A cells + C2), candidate 1 passes **0 of 5** family-A seeds (21/20/17/22/23
   of the surviving 47 cells fail per seed — e.g. `earnings_participation.35-44|female`,
   `coresident_spouse.25-34|female`, `earnings_participation.62-69|*`), so family A
   still fails; **and** C2 does not reverse (`reversed_to_anchor: false`, τ 0.333),
   so family C still fails.
2. **Candidate 2 (PR #167) STANDS FAIL — over-determined.** On the amended surface
   candidate 2 passes **0 of 5** family-A seeds (11/9/12/12/12 fail — e.g.
   `earnings_participation.35-44|female`, `hh_size_share.{1,2,3}`); **and** C2 does
   not reverse (τ 0.333).
3. **Per-cell truth on the demoted cells (referee finding 2).** The demotion does
   not rescue via the demoted cells themselves: the **male** demoted cells
   (18-24 M, married-65+ M, coresident-65+ M) are **0/5 for both candidates**;
   the **female** demoted cells sometimes passed — 18-24 F (0/5 c1, **2/5** c2),
   married-65+ F (**2/5** c1, 1/5 c2), coresident-65+ F (**4/5** c1, 2/5 c2) — but
   removing cells a candidate sometimes *passes* cannot turn its family-A 0/5 into
   a ≥4/5 (many *other* retained cells fail every seed, §6.1–2). C1 never reverses
   (τ 0.667). The family-C leg is **one bit-identical measurement** for both
   candidates (§2a); the family-A leg is genuinely per-candidate (distinct rbar).
4. **Prospective only.** Candidates 1 and 2 remain FAIL as committed; the amendment
   binds only candidates registered after ratification.
5. **The evidence is train/frame-side** (`reported_not_gated` /
   `train_frame_side_only` / `publishes_regardless`, verified) — not scored
   candidate holdouts the amendment could retroactively rehabilitate.
6. **Description-claims-exactly.** After demotion, `covers` / `certification_scope`
   claim EXACTLY the amended 48-cell surface (47 family-A + 1 family-C); the 7
   demoted cells are report-only with reasons named.

Each candidate fails via **both** surviving families independently of the 7 demoted
cells — the rescue set is verifiably empty, over-determined.

## 7. The exact prospective flip (post-ratification; NOT in this PR)

The ratifying flip PR — a separate PR, after this ceremony clears — will make
exactly these edits to `gates.yaml` `gates.gate_w1`. Every edit is a re-scope or a
text narrowing; **no tolerance, σ, anchor value, or ordering changes**, and the
demoted cells' tolerances/ordering are retained under report-only.

1. **`thresholds.family_a`** — move the six cells from their view `tolerances`
   (gate-eligible) to `report_only`, and add a parallel `report_reasons` mapping:
   - `earnings_participation.18-24|female` → `population_concept_delta_head_spouse_universe`
   - `earnings_participation.18-24|male` → `population_concept_delta_head_spouse_universe`
   - `marital_share.married.65+|female` → `cohort_vintage_hazard_frame_mismatch`
   - `marital_share.married.65+|male` → `cohort_vintage_hazard_frame_mismatch`
   - `coresident_spouse.65+|male` → `cohort_vintage_hazard_frame_mismatch`
   - `coresident_spouse.65+|female` → `scored_duplicate_of_demoted_married_quantity`

   The six tolerances are retained verbatim under the report-only disclosure. The
   `faithful_candidate_oc` block updates: `n_gated_cells` 53 → **47**,
   `p_seed_pass` 0.922 → **0.9344**, `p_gate_pass_4_of_5` 0.9481 → **0.9623**
   (recomputed on the 47-cell surface; a CHANGE, not the amendment-1 invariance).
2. **`thresholds.protocol.fresh_run_artifact_schema.per_draw_per_cell_rates`** —
   the committed-cube shape `[20, 53, 5]` → `[20, 47, 5]` (K_draws × gated_cells ×
   gate_seeds); `undefined_draw_rule` and the regenerated-surface rule unchanged.
3. **`thresholds.family_c`** — demote `fingerprint.ppi_nra` (C1): move it from
   `gate_partition.gate_eligible` to a report-only list, tag it
   `fingerprint_reversal_not_realized`, and set `gate_partition` to
   `{gate_eligible: [fingerprint.elimination_plus2pp], n_gate_eligible: 1,
   n_report_only: 1}`. The C1 `anchor_values`, `required_representative_order`, and
   `mechanism` are retained under report-only; the `pass_rule` is annotated that
   family C now gates the C2 fingerprint only.
4. **`thresholds.family_c.check`** — "BOTH fingerprints reverse" → "the C2
   fingerprint reverses (C1 demoted to report-only by amendment 2,
   `fingerprint_reversal_not_realized`; its ordering + PPI−NRA gap publish)".
5. **`gate_w1.covers`** — family A "FLOOR-priced, 53 gated / 52 report-only" →
   "**47 gated / 58 report-only**"; family C "BINARY, 2 gated" → "**1 gated / 1
   report-only** (C1 demoted by amendment 2)"; the overall roll-up
   "55 gated / 77 report-only" → "**48 gated / 84 report-only**".
6. **`thresholds.certification_scope.certifies`** — "the 55 gated cells (53
   CPS-observable family-A joints + 2 family-C compression-fingerprint reversals)"
   → "the **48** gated cells (47 family-A joints + 1 family-C fingerprint — C2, the
   taxable-max elimination reversal); the 18-24 participation pair, the 65+
   marital/coresident quad, and the C1 (PPI-over-NRA) fingerprint are report-only
   pending concept/cohort-vintage/re-specification bridges"; and narrow the
   "reverses BOTH compression fingerprints" clause to "reverses the C2 fingerprint".
7. **`gate_w1.history`** — append the amendment-2 entry (§8 draft).

`thresholds.family_b`, `thresholds.protocol` (statistic, pass_rule, estimator,
floor bindings), `governance.amendment_rules`, `frame_pin`, `holdout_basis`, and
every retained tolerance/anchor/ordering are **untouched**. A subset master-compare
must show the flip changes only `gate_w1` and moves no locked sibling.

## 8. `amendment_history` entry draft (for `gate_w1.history`)

```yaml
- id: 2026-07-12-w1-family-a-concept-cells
  proposed: '2026-07-12'
  referee_round:
    review: 'PR #169 comment 4953763322 (verdict AMEND THE AMENDMENT)'
    fixes: '<sha>'                                          # filled at ceremony
    verification: '<PR> comment <id> (verdict TBD)'         # filled at ceremony
  ratified: '<date> by merge of PR <n> (merge commit <sha>) under the
    maintainer''s standing campaign directive of 2026-07-07, exercised only
    after the full ceremony'                                # filled at ratification
  flipped_live: this pull request
  content: >-
    STRUCTURAL amendment, ZERO THRESHOLD MOVEMENT. 7 cells are DEMOTED from gated
    to report-only: the C1 fingerprint (fingerprint.ppi_nra), the two 18-24
    participation cells (earnings_participation.18-24|{female,male}), and the four
    65+ marital/coresident cells (marital_share.married.65+|{female,male},
    coresident_spouse.65+|{female,male}). Per-cell machine reasons, each true on
    the committed record: C1 fingerprint_reversal_not_realized; the two 18-24 cells
    population_concept_delta_head_spouse_universe; the married-65+ pair and
    coresident-65+|male cohort_vintage_hazard_frame_mismatch (the stationary
    ceiling the deployment cannot reach); coresident-65+|female
    scored_duplicate_of_demoted_married_quantity. All 7 tolerances/orderings
    publish report-only (retained, not deleted). Motivated by W1 forensics 2
    (runs/gate_w1_forensics2_v1.json, registration 4953088871, grading 4953311492)
    Q6/Q9, W1 forensics 1 Q5/Q1 (runs/gate_w1_forensics1_v1.json), and the
    adversarial round (PR #169 comment 4953763322). C1's required PPI-over-NRA
    reversal does NOT occur under any contract-consistent transport: on the
    heaviest (upper-read) tail PPI savings 0.0169 sit far below NRA 0.2023 (gap
    0.1854); non-reversal proven analytically (Q5); the two candidate fingerprint
    runs are bit-identical (one measurement). The 18-24 participation miss is a
    POPULATION-CONCEPT delta of 22.1pp pooled (PSID head/spouse 0.865 vs CPS
    all-person 0.644; male 27.6pp): the male cell fails 0/5 both candidates, the
    female cell caps 2/5. The 65+ undershoot is DIVORCE over-accumulation (deployed
    divorced 0.209/0.190 vs frame 0.090/0.063, excess +0.120/+0.127), a
    cohort-vintage hazard/frame mismatch NOT the pre-registered widowhood; the
    deployed married is capped at the CANDIDATE_16 stationary ceiling ~0.577 F /
    ~0.717 M (Q1 entry-0 0.5767, Q6 entry-0.58 0.5728; contraction ~0.007/unit), so
    the male 65+ cells fail 0/5 and married-65+|female caps 2-3/5 by
    ceiling-vs-window; nine permitted-lever construction-attempts (six in the
    proposal + the referee's per-band, entry-55, max-entry-bound) each fail, moving
    the cells worse. PER-CELL TRUTH (no_self_rescue, referee finding 2): candidate
    1 PASSED coresident_spouse.65+|female 4/5 seeds and marital_share.married.65+|
    female 2/5 -- the female 65+ cells are NOT unclearable; the coresident-female
    cell is a scored duplicate of the demoted married quantity (deployed
    coresident/married ratio 1.0) that straddles a slightly easier window, demoted
    on duplicate/composition-blind grounds not impossibility. Candidate 1 (PR #162)
    and candidate 2 (PR #167) both STAND FAIL, over-determined -- on the amended
    47-cell family-A surface each passes 0/5 seeds AND C2 does not reverse (tau
    0.333, bit-identical between the two runs); the demotion rescues nothing.
    Unlike amendment 1, family A CHANGES: the faithful-candidate OC recomputes on
    the residual 47 cells to p_seed 0.9344 / p_gate 0.9623 (from 53 cells / 0.922 /
    0.9481). NOT KNOWN PASSABLE: the amended 48-cell gate removes proven-unreachable
    cells; it does not establish a passable surface. Committed unpassability risks,
    forecast as amendment-3 candidates: the five hh_size cells (Q7 graded REFUTED,
    "not closable by these two entry-state levers alone", all 5/5-fail both
    candidates) and C2's cap_150k adjacency (deployed 16.7 years vs Smith rank-4 1
    year; corrected tail tau -1/3; robustness asserted only for the elim<->+2pp
    pair) -- each reopenable only via the same forensics-then-ceremony standard.
    Certification scope narrows: the C1 reversal, the 18-24 participation, and the
    65+ marital/coresident composition are report-only, published-not-certified.
    The gate_w1 faithful-candidate OC returns from an achievable 0 to 0.9623 x
    I(C2) on the residual 48-cell gated surface (55 -> 48 gated, 77 -> 84
    report-only). Evidence chain: candidate 1 (PR #162), candidate 2 (PR #167), W1
    forensics 1 (PR #163), W1 forensics 2 (PR #168), referee round (PR #169
    comment 4953763322).
```

## 9. Certification-scope language after amendment 2

What the gate **claims** after the flip:

- **Certifies (gated)**: the 47 family-A CPS-observable joints and the C2
  compression-fingerprint reversal — **48 cells**. A PASS certifies that stochastic
  regeneration through the deployed generators reproduces those 47 joints within
  the frame's own sampling floor, and reverses the C2 fingerprint. **A PASS is not
  established to be achievable** — the amended surface removes proven-unreachable
  cells but is not itself known passable (§5): the five `hh_size` cells fail 5/5
  for both candidates (Q7 graded REFUTED — "not closable by these two entry-state
  levers alone"; the coresidence-composition repair is unadjudicated), and C2's
  full ordering has never been produced (both candidate runs bit-identical, deployed
  `cap_150k` 16.7 years vs Smith's 1-year rank-4, corrected-tail τ −1/3, robustness
  asserted only for the elim↔+2pp pair).
- **Publishes (report-only, NOT certified)**: the C1 (PPI-over-NRA) ordering + gap
  (`fingerprint_reversal_not_realized`); the 18-24 participation pair
  (`population_concept_delta_head_spouse_universe`); the married-65+ pair and
  coresident-65+|male (`cohort_vintage_hazard_frame_mismatch`); coresident-65+|
  female (`scored_duplicate_of_demoted_married_quantity`); plus the report-only
  cells carried from the lock and amendment 1.
- **Does not support**: the C1 reversal, the 18-24 participation, or the 65+
  marital/coresident composition as certified quantities. **Named amendment-3
  risks** (forecast, not discovered one PR at a time): the five `hh_size` cells and
  C2's `cap_150k` adjacency — each reopenable only via the same
  forensics-then-ceremony standard, never a bare demotion; if candidate 3's
  fingerprint runs byte-carry the career panel again, the C2 "empirical burden"
  framing expires and a Q5-grade `cap_150k` robustness analysis is the prerequisite.
  (The dynamics remain gate-1/2a/2b/2c/M4 certified on their PSID holdouts; W1
  certifies transport, not re-estimation.)

Stated plainly: **the gate certifies the 47 reproducible family-A joints and the
C2 reversal — a surface not itself known passable; it certifies neither the C1
reversal, the 18-24 participation, nor the 65+ marital/coresident composition,
pending the bridges.**

## 10. Evidence-base prerequisites for a future re-anchor (the bridges)

- **C1 (group a) — no re-specification exists.** The required PPI-over-NRA
  magnitude reversal is robustly false under every contract-consistent tail
  (§2a/§4a). The prerequisite for ever re-gating C1 is a **new defensible binary
  fingerprint** the anchors support and a contract-consistent transport can flip —
  which the evidence base does not contain.
- **18-24 participation (group b).** Needs an archived **published head/spouse
  young-adult participation series** (not archived) + a concept-delta accounting +
  a referee round; **or** a generator extension that models **non-head** young-adult
  participation (a re-estimation W1 does not certify).
- **65+ marital/coresident (group c).** Needs either an **older-cohort
  divorce/remarriage hazard vintage** (a cohort-specific re-estimation — the locked
  pooled hazards cannot be re-calibrated) or a **re-anchored 65+ frame composition**
  target; both are evidence-base amendments outside the transport contract.
- **hh_size / C2 (the forecast amendment-3 risks, §5/§9).** The `hh_size` residual
  needs the contract status of a **coresidence-composition repair beyond the initial
  roster** adjudicated (Q7 named it, unadjudicated); C2's `cap_150k` needs a
  **Q5-grade representative-frame robustness analysis** to establish whether its
  adjacency is candidate-independent — neither exists today, so both stay in scope
  only via a future registered forensics + ceremony.

## 11. Ceremony checklist

- [x] **Proposal** (this document + this draft PR). No `gates.yaml` edit; no
      threshold moved.
- [x] **Adversarial referee round** ([4953763322][ref-round]; verdict AMEND THE
      AMENDMENT — demotion set of 7 survives; correct the per-cell grounds
      (finding 2), add retained-surface candor + series forecast (finding 4), fix
      the σ ranges (finding 5) and the §4a text (finding 9)).
- [x] **Fixes** (this revision: §4c per-cell grounds + ceiling-vs-window + the
      coresident-F 4/5 record; §5/§9 candor + series forecast; §3 σ ranges 2.0–6.6σ;
      §4a rewrite; the referee's R1–R3 construction-attempts; ledger + bindings
      extended and mutation-checked).
- [ ] **Verification round** (recompute every number from committed artifacts; the
      §7 flip text, §7 roll-ups, and §8 history figures are string-bound by
      `tests/test_gate_w1_amendment2_proposal.py`).
- [ ] **Ratify by merge** of the flip PR (after the ceremony clears).
- [ ] **Flip** `gates.yaml` per §7 and append the §8 history entry.
- [ ] Then: candidate 3 against the amended contract (CPS-anchored entry, interior
      sex covariate, the co-designed roster/window levers).

---

<!-- amendment-consistency-ledger: bound to the committed artifacts by
     tests/test_gate_w1_amendment2_proposal.py. Do not hand-edit a value without
     re-deriving it from runs/gate_w1_forensics2_v1.json,
     runs/gate_w1_forensics1_v1.json, runs/gate_w1_floors_v1.json,
     runs/gate_w1_candidate1_v1.json, runs/gate_w1_candidate2_v1.json, or
     gates.yaml. -->

```json amendment-consistency-ledger
{
  "amendment_id": "2026-07-12-w1-family-a-concept-cells",
  "recommendation": "report_only_package_all_three_groups",
  "n_cells_demoted": 7,
  "referee_round_comment": "4953763322",
  "cell_reasons": {
    "fingerprint.ppi_nra": "fingerprint_reversal_not_realized",
    "earnings_participation.18-24|female": "population_concept_delta_head_spouse_universe",
    "earnings_participation.18-24|male": "population_concept_delta_head_spouse_universe",
    "marital_share.married.65+|female": "cohort_vintage_hazard_frame_mismatch",
    "marital_share.married.65+|male": "cohort_vintage_hazard_frame_mismatch",
    "coresident_spouse.65+|male": "cohort_vintage_hazard_frame_mismatch",
    "coresident_spouse.65+|female": "scored_duplicate_of_demoted_married_quantity"
  },
  "per_cell_pass_record": {
    "earnings_participation.18-24|female": {"candidate1": 0, "candidate2": 2},
    "earnings_participation.18-24|male": {"candidate1": 0, "candidate2": 0},
    "marital_share.married.65+|female": {"candidate1": 2, "candidate2": 1},
    "marital_share.married.65+|male": {"candidate1": 0, "candidate2": 0},
    "coresident_spouse.65+|female": {"candidate1": 4, "candidate2": 2},
    "coresident_spouse.65+|male": {"candidate1": 0, "candidate2": 0}
  },
  "ceiling_vs_window_65plus_female": {
    "q1_entry0_stationary_round4": 0.5767,
    "q6_entry058_terminal_round4": 0.5728,
    "married_lower_window_edges_candidate1": [0.5619, 0.595, 0.5798, 0.5853, 0.5904],
    "married_pass_flags_candidate1": ["T", "F", "T", "F", "F"],
    "coresident_pass_flags_candidate1": ["T", "T", "T", "T", "F"],
    "coresident_married_deployed_ratio": 1.0,
    "coresident_clears_4_of_5_candidate1": true
  },
  "forensics2": {
    "artifact": "runs/gate_w1_forensics2_v1.json",
    "registration": "4953088871",
    "grading": "4953311492",
    "reported_not_gated": true,
    "train_frame_side_only": true,
    "concept_gap_18_24_pooled_pp_round1": 22.1,
    "concept_gap_18_24_female_round3": 0.173,
    "concept_gap_18_24_male_round3": 0.276,
    "psid_head_spouse_pooled_round3": 0.865,
    "cps_all_person_pooled_round3": 0.644,
    "exceeds_15pp_amendment_threshold": true,
    "divorced_excess_65plus_female_round3": 0.12,
    "divorced_excess_65plus_male_round3": 0.127,
    "married_65plus_deployed_female_round3": 0.573,
    "married_65plus_frame_female_round3": 0.691,
    "married_65plus_deployed_male_round3": 0.707,
    "married_65plus_frame_male_round3": 0.856,
    "widowhood_channel_realized_65plus": false,
    "realized_65plus_channel": "divorce",
    "q7_pre_registration_proves_3_4_5plus": false,
    "q7_joint_cells_cleared": ["2"]
  },
  "forensics1_q5_c1": {
    "artifact": "runs/gate_w1_forensics1_v1.json",
    "registration": "4951218279",
    "upper_read_ppi_savings_round4": 0.0169,
    "upper_read_nra_savings_round4": 0.2023,
    "upper_read_ppi_minus_nra_round4": -0.1854,
    "corrected_ppi_savings_round4": 0.0137,
    "corrected_nra_savings_round4": 0.2022,
    "non_reversal_is_robust": true,
    "candidate_fingerprint_runs_bit_identical": true,
    "candidate2_c1_kendall_tau_vs_required_round4": 0.6667
  },
  "sigma_range": {
    "ln_miss_min_round2": 0.11,
    "ln_miss_max_round2": 0.44,
    "sigma_min_round3": 0.027,
    "sigma_max_round3": 0.072,
    "sigma_multiple_min_round1": 2.0,
    "sigma_multiple_stated_max": 6.6,
    "draw_noise_max_round3": 0.007
  },
  "c2_record": {
    "deployed_order": ["elimination", "cap_150k", "payroll_plus_2pp", "payroll_plus_1pp"],
    "kendall_tau_vs_required_round4": 0.3333,
    "candidate_runs_bit_identical": true,
    "cap_150k_deployed_years_round1": 16.7,
    "cap_150k_smith_rank4_years": 1,
    "corrected_tail_tau_vs_required_round4": -0.3333,
    "robust_only_for_elim_plus2pp_pair": true
  },
  "family_a_oc": {
    "n_gated_before": 53,
    "p_seed_before": 0.922,
    "p_gate_before": 0.9481,
    "n_gated_after": 47,
    "p_seed_after": 0.9344,
    "p_gate_after": 0.9623,
    "invariant_under_amendment": false,
    "demoted_family_a_cell_faithful_passprob_min_round3": 0.997,
    "demoted_family_a_cell_faithful_passprob_max_round3": 0.998
  },
  "family_a_partition": {"gated_now": 53, "gated_after": 47, "report_only_now": 52, "report_only_after": 58},
  "family_c_partition": {"gated_now": 2, "gated_after": 1, "report_only_now": 0, "report_only_after": 1},
  "overall_partition": {"gated_now": 55, "gated_after": 48, "report_only_now": 77, "report_only_after": 84},
  "oc_statement_after": "0.9623 * I(c2)",
  "surface_not_known_passable": true,
  "amendment3_risks": ["hh_size_share.1", "hh_size_share.3", "hh_size_share.4", "hh_size_share.5plus", "c2_cap_150k_adjacency"],
  "no_self_rescue": {
    "candidate1_run": "gate_w1_candidate1_v1",
    "candidate1_pr": 162,
    "candidate1_gate_pass": false,
    "candidate1_family_a_pass_47_surface": false,
    "candidate1_seeds_pass_47_surface": 0,
    "candidate1_c2_reversed": false,
    "candidate2_run": "gate_w1_candidate2_v1",
    "candidate2_pr": 167,
    "candidate2_gate_pass": false,
    "candidate2_family_a_pass_47_surface": false,
    "candidate2_seeds_pass_47_surface": 0,
    "candidate2_c2_reversed": false,
    "rescue_set_empty": true,
    "over_determined_both_families": true
  },
  "gates_yaml_untouched_by_this_proposal": true
}
```
