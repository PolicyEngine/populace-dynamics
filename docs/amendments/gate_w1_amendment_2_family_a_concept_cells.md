# gate_w1 amendment 2 (proposal) — the family-A concept cells and the C1 fingerprint

- **Amendment id**: `2026-07-12-w1-family-a-concept-cells`
- **Gate**: `gate_w1` (W1 representative-frame transport, roadmap #113 M5, workstream #100)
- **Surface**: three cell groups on the amended (post-amendment-1) **55-cell** gated
  surface — (a) the family-C **C1** fingerprint (`fingerprint.ppi_nra`); (b) the
  two family-A **18-24 participation** cells
  (`earnings_participation.18-24|{female,male}`); (c) the four family-A **65+
  marital/coresident** cells (`marital_share.married.65+|{female,male}` and
  `coresident_spouse.65+|{female,male}`). **7 cells total**: 1 family-C binary + 6
  family-A joints.
- **Ceremony stage**: PROPOSAL (draft). This document is the first step of the
  amendment ceremony (proposal → adversarial referee → fixes → verification →
  ratify-by-merge → flip). **It moves no threshold and edits no `gates.yaml`
  cell.** The prospective flip happens in a separate ratifying PR, only after the
  ceremony clears.
- **Amendment class**: STRUCTURAL, prospective, ZERO threshold movement (the
  amendment-1 precedent). Unlike amendment 1 — which demoted family-B cells that
  sit OUTSIDE the family-A OC machinery, leaving the family-A characteristic
  byte-identical — this amendment removes **6 cells from family A itself**, so the
  family-A faithful-candidate OC is **recomputed** on the residual 47-cell surface
  (§4, §5). No tolerance, σ, anchor value, or ordering is edited; the demoted
  cells' tolerances/orderings are retained under report-only.
- **Evidence base**: W1 forensics 2 (`runs/gate_w1_forensics2_v1.json`,
  registration [4953088871][reg2], grading [4953311492][grade2]) — Q6 (the 65+
  divorce over-accumulation) and Q9 (the measured 18-24 concept gap + the C1
  consolidation); W1 forensics 1 Q5 (`runs/gate_w1_forensics1_v1.json`,
  registration [4951218279][reg1]) — the C1 non-reversal analytic proof; candidate
  1 (`runs/gate_w1_candidate1_v1.json`, PR #162, registration 4950931131) and
  candidate 2 (`runs/gate_w1_candidate2_v1.json`, PR #167, registration
  4952253568) — the committed misses on every cell in scope; the amended gate
  (`gates.yaml` `gate_w1`, post-amendment-1: 53 family-A + 2 family-C gated). The
  amendment-1 doc (`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md`, PR
  #164 proposal / #165 flip) is the direct structural precedent.

[reg2]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4953088871
[grade2]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4953311492
[reg1]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4951218279

## 1. Summary

Amendment 1 demoted the 10 family-B SSA-anchor cells and restored the gate's
headline faithful-candidate OC from a structural 0 to **0.9481 × I(family C)** on
a **55-cell** surface (53 family-A joints + 2 family-C fingerprints). W1 forensics
2 (grading [4953311492][grade2], verdict "**gate_w1 remains unpassable as amended
— amendment 2 precedes candidate 3**") then established that **three cell groups
on that residual surface are themselves unclearable by any contract-consistent
candidate**, for three distinct — and each candidate-independent — reasons:

- **(a) The C1 fingerprint** (`fingerprint.ppi_nra`: progressive price indexing
  must outrank NRA→70 in 75-year payroll savings). Its required reversal **does
  not occur** under any contract-consistent transport. Forensics 1 Q5 proved this
  analytically on the **heaviest plausible tail** (the upper read, most favourable
  to PPI): PPI savings **0.0169** sit far below NRA **0.2023** (gap **0.1854**),
  and a realistic tail moves PPI *down* to **0.0137**. Both candidate 1 and
  candidate 2 confirm it empirically (Kendall τ 0.667 vs the required order, swap
  not realised). C1 is a structural 0.
- **(b) The two 18-24 participation cells.** The miss is a **population-concept
  delta**, not a fit-support gap: the PSID family-earnings panel is a **head/spouse
  universe** (participation ~0.865 pooled), while the CPS frame counts **every**
  18-24 person — students, dependents living with parents — and participates
  ~0.644. The concept gap is **22.1pp pooled** (female 17.3, male **27.6**), above
  the 15pp threshold the registration named. v2's Q2 boundary extension already
  fits the PSID 18-24 support, and the cells **still** fail by the concept gap.
- **(c) The four 65+ marital/coresident cells.** The deployed 65+ married share
  undershoots the frame (female 0.573 vs 0.691; male 0.707 vs 0.856) because the
  certified pooled divorce-minus-remarriage hazards **over-accumulate divorce**:
  deployed 65+ divorced ~0.19–0.21 vs the frame's older-cohort ~0.06–0.09 (excess
  **+0.120 female / +0.127 male**). This is a **cohort-vintage mismatch** between
  the pooled 1998–2022 certified hazards and the frame's older 65+ cohorts — a
  hazard-evolution defect, **not** the pre-registered widowhood (deployed
  *under*-widows). Hazard re-calibration is contract-prohibited (W1 certifies
  transport, not re-estimation), so **no candidate lever exists** (§4c). The
  coresident-spouse pair fails by the identical mechanism (spouse presence tracks
  married share); the grading names the "marital pair", but the evidence extends
  the demotion to the full quad, exactly as amendment 1's referee extended 8→10.

Because family A's seed rule is a **conjunction** (a seed passes iff every gated
cell is within tolerance) and family C is a conjunction over its fingerprints,
each of these groups makes the amended gate **unpassable**: the achievable
faithful-candidate OC is **0** (§3), no matter how well a candidate transports the
other 47 family-A joints or reverses C2.

**Per group, two options are analysed below (§4); this proposal recommends the
report-only disposition for all three** (the package: demote C1 + the 18-24 pair +
the 65+ quad — **7 cells** — to report-only, with per-cell machine reasons; §5).

- **Recommended package — demote 7 cells to report-only**: C1 tagged
  `fingerprint_reversal_not_realized`; the two 18-24 cells tagged
  `population_concept_delta_head_spouse_universe`; the four 65+ cells tagged
  `cohort_vintage_hazard_frame_mismatch`. Zero threshold movement; the C1 ordering,
  the 18-24 tolerances, and the 65+ tolerances stay in-contract (report-only),
  published-not-certified, pending the bridges §10 names. The gate's
  faithful-candidate OC moves from an **achievable 0** to a well-defined
  **p_seed 0.9344 / p_gate 0.9623 × I(C2)** on the residual **48-cell** surface
  (47 family-A joints + the C2 fingerprint) — the recomputed family-A
  characteristic on the cells that carry no concept/vintage/reversal defect.
- **Deferred alternative — re-anchor / re-specify**: each group has an honest
  re-anchor path (§4), but every one presupposes an evidence-base amendment (a
  head/spouse young-adult participation anchor; an older-cohort divorce/remarriage
  hazard vintage; a defensible alternative fingerprint statistic) that is not
  archived or not contract-consistent today, and a fresh referee round. Deferred,
  not chosen now (§10).

## 2. The finding (W1 forensics 2 Q6/Q9; W1 forensics 1 Q5)

`runs/gate_w1_forensics2_v1.json` and `runs/gate_w1_forensics1_v1.json` are
one-shot, `reported_not_gated`, `train_frame_side_only` diagnostics that "publish
regardless of any verdict" — they measure the transport mechanisms on the frame
side before designing the next candidate. Every re-simulated component reproduces
the committed machinery bit-for-bit (`q6/q7/q8 instrumentation_bit_identity_max_dev
== 0.0`).

### 2a. C1 — the reversal that does not occur (forensics 1 Q5, forensics 2 Q9)

C1 requires the representative frame to **reverse** the compressed-career order so
that progressive price indexing (PPI) outranks NRA→70 in 75-year payroll savings.
The mechanism hypothesised at #113: a representative frame carries more AIME above
the second PIA bend, where PPI bites, so PPI's savings exceed NRA→70's. Forensics
1 Q5 tested it on the **upper read** — a transported-career panel of positive
prime-age earners with no zero/low-earning years, which **over**-concentrates the
upper tail (`frac_payroll_above_wage_base` 0.377, the heaviest plausible tail,
**most favourable to the swap**). Even there PPI savings are **0.0169** against
NRA's **0.2023** (`ppi_minus_nra` −0.1854): PPI does not come close. The corrected
tail applies the certified per-cell p0 (a lighter, realistic tail,
`tail_lighter_than_upper_read: true`) and moves PPI to **0.0137** vs NRA **0.2022**
— **further** from reversal. `c1_robustness_answer.answer_non_reversal_is_robust`
is **true**. Forensics 2 Q9 consolidates: C1 is "**NOT reversed by any
contract-consistent deployment built**" — analytic (Q5) plus two empirical
confirmations (candidate 1 `c1_reversed: false`; candidate 2 Kendall τ 0.667 vs
required, swap not realised). The transported AIME does not lift PPI past NRA.

C2 (the taxable-max elimination reversal) holds under both tails and is untouched
by this amendment.

### 2b. The 18-24 participation concept gap (forensics 2 Q9)

`q9_concept_cells.concept_gap_18_24_participation` measures the transport target
directly by **universe restriction** on the train side:

| Universe | female | male | pooled |
|---|---:|---:|---:|
| PSID head/spouse (the generator's fit universe) | 0.815 | 0.929 | **0.865** |
| CPS all-person (the frame the cell scores against) | 0.642 | 0.653 | **0.644** |
| concept gap (PSID − CPS) | 0.173 | **0.276** | **0.221** |

The pooled gap is **22.1pp**, `exceeds_15pp_amendment_threshold: true`. The
mechanism (`q9…mechanism`): the PSID family-earnings panel is a head/spouse
universe — at 18-24 disproportionately the employed, independent young adults — so
its participation is ~0.89; the CPS all-person frame counts every 18-24 person
(students, dependents living with parents) and participates ~0.64. This is a
**population-concept delta, not a fit-support gap**: v2's Q2 boundary extension
already fits the PSID 18-24 support, and the cell still fails by the concept gap.
The male gap (27.6pp) is the larger — the male 18-24 cell is the group's hard fail
(§4b).

### 2c. The 65+ divorce over-accumulation (forensics 2 Q6)

`q6_marital_calibration_frame` seeds each frame adult's marital ENTRY state at
`BASE_ENTRY_AGE=25` from the PSID-entry model (~0.58 F / 0.54 M married at 25) and
evolves the certified CANDIDATE_16 hazards to the current age. At 65+ the deployed
married share undershoots the frame:

| 65+ | deployed married | frame married | miss (D−A) | entry component (L−A) | hazard component (D−L) |
|---|---:|---:|---:|---:|---:|
| female | 0.573 | 0.691 | −0.118 | −0.110 | −0.008 |
| male | 0.707 | 0.856 | −0.149 | −0.319 | +0.170 |

The realised dissolution channel is **divorce, not widowhood**
(`dissolution_channel_65plus.widowhood_channel_realized: false` for both sexes):

| 65+ divorced | deployed | frame | excess (deployed − frame) |
|---|---:|---:|---:|
| female | 0.209 | 0.090 | **+0.120** |
| male | 0.190 | 0.063 | **+0.127** |

The certified pooled divorce-minus-remarriage steady state sits far above the
frame's older-cohort divorced share — a **hazard-vs-frame cohort-vintage
mismatch**: the frame's 65+ are from older marriage cohorts with lower lifetime
divorce than the pooled 1998–2022 hazards imply. The grading adjudicated Q6
"**HALF-HELD, and the refinement matters**": entry-level confirmed at 25-34 (a
CPS-anchored entry lever fixes it — a candidate-3 lever, not this amendment), but
the 65+ channel is divorce over-accumulation of the **hazard-evolution class —
re-calibration prohibited, so the 65+ pair is amendment territory or unfixable**.
The `contract_adjudication.determination` is the load-bearing finding: a
CPS-anchored ENTRY model is contract-permitted and fixes 25-34, but it
**"CANNOT fix the 65+ undershoot, which is a HAZARD-evolution miss … closable
only by re-calibrating the certified hazards (PROHIBITED …) or re-anchoring the
65+ frame composition (an amendment)."** §4c walks every permitted lever and shows
each fails.

## 3. Why gate_w1 as amended (post-amendment-1) is unpassable

The gate is a conjunction across families: **family A (≥4-of-5 seeds) ∧ family C
(both fingerprints reverse)**; family B gates nothing after amendment 1. A family-A
seed passes iff **every** gated cell is within tolerance on that seed. Three
groups break the conjunction:

1. **C1 cannot reverse** (§2a). Family C is `I(C1 reverses) × I(C2 reverses)`;
   C1's non-reversal is robust, so `I(C1) = 0` for any contract-consistent
   candidate ⇒ family C = 0.
2. **The 18-24 male cell fails on every seed** (§2b, §4b). Its concept gap
   (|ln(0.93/0.63)| ≈ 0.39) exceeds tolerance 0.221 by a systematic ~3σ, so every
   seed fails on this cell alone ⇒ family A ≤ (the ≥4-of-5 rule) never clears.
3. **The 65+ male cells fail on every seed** (§2c, §4c). The divorce
   over-accumulation drives the deployed 65+ married/coresident to ~0.71 vs the
   frame's ~0.85, a systematic miss beyond tolerance on every seed.

**The crux — the contract's headline OC prices sampling noise, not concept/vintage
bias.** The `faithful_candidate_oc` is a **draw-noise-free half-normal** model: it
assumes a faithful candidate whose per-cell score is half-normal around the
**sampling** σ, and on that model the six family-A cells in scope are trivially
passable — their per-cell pass probabilities are **0.997–0.998** (each σ ≈
0.03–0.07, tolerance 0.08–0.22). But the forensics establishes that the faithful
idealisation the model assumes is **unreachable** for these cells: the actual miss
is a **systematic** concept/vintage bias (head/spouse vs all-person; pooled
hazards vs older cohorts), many σ beyond tolerance, that no contract-consistent
candidate can remove — candidate 1 and candidate 2 miss the hard cells by 0.23–0.44
ln against a sampling σ of ~0.05–0.07 (3–8σ, systematic, not stochastic). The
half-normal model does not apply; the achievable score is bounded below by the
bias. So:

> **achievable faithful-candidate gate OC (amended surface) = p_gate(A) × I(C1) ×
> I(C2) = (0 via the 18-24 and 65+ cells) × (0 via C1) × I(C2) = 0.**

The contract's stated **0.9481 × I(family C)** headline is the idealised-candidate
figure; the **achievable** OC is 0, **over-determined** (family A *and* family C
each force it). This is a defect of the *contract* — it scores a head/spouse-fit
generator against an all-person frame (18-24), a pooled-vintage hazard against an
older-cohort frame (65+), and asks for a reversal the mechanism does not produce
(C1) — exactly the class amendment 1 corrected one level up. It is not a defect of
any candidate.

## 4. Options — per cell group

### 4a. Group (a): the C1 fingerprint binary

**Option a1 — report-only with disclosure (RECOMMENDED for this group).** Demote
`fingerprint.ppi_nra` from gate-eligible to report-only, tagged
`fingerprint_reversal_not_realized`. The **ordering is still recorded per run**
(the deployed {PI, PPI, NRA→70, reduced-COLA} savings order and the PPI−NRA gap):
the reversal question is **retired as a certification claim** — it is robustly
false (§2a) — while the diagnostic stays the single most valuable published output
of the fingerprint, because the deployed PPI−NRA gap tells a reader exactly how far
the representative frame's above-second-bend AIME mass sits from the reversal
threshold. Honest, zero threshold movement; C1 certifies nothing, but publishes
the ordering + gap.

**Option a2 — re-specification.** Replace C1 with a defensible alternative
fingerprint statistic the anchors support. Enumerated honestly, the set is
**empty**: the C1 anchor is the Mermin Table 1 payroll-savings ordering, in which
PPI (−0.14) already ranks **below** NRA→70 (−0.50) is false — Mermin has PPI
(−0.14) *above* NRA (−0.50) in signed savings, and the "representative reversal"
was the #113 hypothesis that the frame flips the *magnitude* order so PPI's
**absolute** savings exceed NRA's. Forensics 1 Q5 falsifies exactly that magnitude
reversal under every contract-consistent tail (PPI 0.017 ≪ NRA 0.202). No
published anchor asserts a representative-frame PPI-over-NRA magnitude ordering; a
re-specification to the **continuous** PPI−NRA gap is a report-only diagnostic, not
a binary gate; and re-specifying to the generator's own deployed order would be
scoring the generator against itself. There is no binary fingerprint the anchors
support that a contract-consistent transport can pass — so re-specification reduces
to option a1 (publish the ordering + gap, certify nothing).

**Recommendation (a): a1.** The reversal claim is robustly false and cannot be
certified; the ordering + PPI−NRA gap are the valuable published diagnostic and
stay report-only.

### 4b. Group (b): the 18-24 participation pair

**Option b1 — report-only pending a concept bridge (RECOMMENDED for this group).**
Demote `earnings_participation.18-24|{female,male}` to report-only, tagged
`population_concept_delta_head_spouse_universe`. The cells stay published (the
deployed head/spouse participation vs the all-person frame is an honest disclosure
of the universe mismatch), certified by nothing, pending a concept bridge — a
generator that models **non-head** young-adult participation (students, dependents),
which is a re-estimation W1 does not certify, or a re-anchor to a head/spouse
target (option b2). Zero threshold movement.

**Option b2 — re-anchor to a head/spouse-universe target.** Score the 18-24
participation cells against a head/spouse young-adult participation anchor rather
than the frame's all-person rate. Adjudicated against the identity prohibition and
availability: (i) restricting the **frame's** 18-24 universe to heads/spouses
changes the family-A estimand (family A is defined over all persons) and is itself
an amendment, not a re-scope; (ii) a **published** head/spouse young-adult
participation series that M4/gate-1 could transport onto is **not archived**;
(iii) re-anchoring to the generator's **own** head/spouse rate is a near-identity
(scoring the generator against itself) — prohibited. So b2 requires archiving an
external head/spouse young-adult participation anchor **plus** a concept-delta
accounting of the residual head/spouse-instrument deltas **plus** a fresh referee
round. Deferred (§10).

**Recommendation (b): b1.** The concept gap is measured (22.1pp ≥ 15pp); the cells
are report-only until the bridge exists.

### 4c. Group (c): the 65+ marital/coresident quad — construction-attempts

**Option c1 — report-only pending a cohort-vintage bridge (RECOMMENDED for this
group).** Demote `marital_share.married.65+|{female,male}` and
`coresident_spouse.65+|{female,male}` to report-only, tagged
`cohort_vintage_hazard_frame_mismatch`. The four cells stay published (the deployed
65+ married/coresident vs the frame's older-cohort composition is an honest
disclosure), certified by nothing, pending a cohort-vintage bridge (an
older-cohort divorce/remarriage hazard vintage, or a re-anchored 65+ frame
composition). Zero threshold movement.

**Option c2 — re-anchor.** Deferred for the same reason as b2 and §10.

**The candidate-independence argument (the amendment-1 referee's
construction-attempt method).** The prohibition is on holdout tuning and identity;
hazard re-**calibration** to the frame would be tuning. The question the referee
standard demands: *is there ANY permitted conditioning that fixes the divorce
accumulation for the 65+ quad?* Every permitted lever is attempted below and each
fails — the argument holds for **any** contract-consistent candidate, not only the
two built.

| # | Attempted permitted lever | Why it fails on the 65+ quad |
|---|---|---|
| 1 | **CPS-anchored entry-level model** (Q6's permitted lever — condition the age-25 entry state on {band, sex} from CPS aggregate rates, regenerate via certified hazards) | Recalibrates the **entry level at 25** to the CPS young-adult cross-section (0.49 F / 0.42 M), which is **below** the current PSID entry seed (0.58 F / 0.54 M), which is already **far below** the 65+ frame married (0.69 F / 0.86 M). Lowering the entry seed **widens** the 65+ undershoot; it cannot raise a 40-year-evolved terminal to a level above its own entry. Fixes 25-34 (an entry-level miss), moves 65+ the wrong way. |
| 2 | **Direct 65+ entry seed** (seed the 65+ band's entry state from a higher married level) | Either reads the 65+ frame's own married share (the **identity** — score 0, `across_draw_sd` 0) or back-solves the entry so the 65+ terminal reproduces its own `rate_a` (the **inverse map — the identity in disguise**, `contract_adjudication`). Both prohibited. |
| 3 | **Cohort/vintage-conditioned hazards** (apply older-cohort divorce/remarriage hazards to the 65+ cohorts) | This is hazard **re-calibration**: the certified CANDIDATE_16 hazards are pooled 1998–2022 and **locked** (W1 certifies transport, not re-estimation). Prohibited. |
| 4 | **Extra permitted covariate re-weighting** (condition on an already-supported covariate, e.g. education/region, to reshape the 65+ population mix) | The over-accumulated **divorced steady state** is a property of the pooled **hazards**, not the covariate mix; re-weighting the population within the same hazards cannot move the terminal divorced share by +0.12 to the frame's older-cohort level. A new conditioning covariate on the **hazard** is re-estimation (prohibited). |
| 5 | **The widowhood channel** (the pre-registered lever) | Not realised: the deployed **under**-widows (deployed widowed ≤ frame, `widowhood_channel_realized: false`). There is no widowhood surplus to redistribute, and correcting widowhood would not close a **divorce** over-accumulation. |
| 6 | **Deployment draw (K=20 mean)** | The draw is stochastic noise around the deployed steady state (across-draw σ ≈ 0.02–0.05 on these cells); it cannot systematically shift the divorced share by 0.12. |

Every permitted lever either (a) does not touch the pooled-hazard divorced steady
state (levers 1, 4, 6), (b) is the prohibited identity/inverse-map (lever 2), or
(c) is prohibited hazard re-calibration (levers 3, 5). **No contract-consistent
candidate can clear the 65+ quad.** The coresident-spouse pair fails by the
identical mechanism (spouse presence is the married-with-spouse-present quantity;
the same divorce over-accumulation lowers it) — both 65+ **male** cells are a hard
0/5-seed fail for **both** candidates (§6), so demoting only the marital pair while
leaving the coresident pair gated would leave the gate unpassable. The demotion is
the four-cell quad.

## 5. Recommendation

**Adopt the report-only package for all three groups** (a1 + b1 + c1): demote the
**7 cells** — C1, the 18-24 pair, the 65+ quad — to report-only, each with its
machine reason. It is the minimal contract-clean move that (i) removes **exactly**
the structurally-unclearable cells — 1 family-C binary + 6 family-A joints, no more
and no fewer; (ii) preserves every other gated cell and every tolerance/σ/anchor/
ordering byte-identical (zero threshold movement); (iii) keeps the C1 ordering, the
18-24 tolerances, and the 65+ tolerances in-contract as honest report-only
disclosure; (iv) names each defect with a machine reason
(`fingerprint_reversal_not_realized`, `population_concept_delta_head_spouse_universe`,
`cohort_vintage_hazard_frame_mismatch`); and (v) restores a well-defined,
achievable gate OC **without** asserting any demoted cell is clearable.

**OC consequence (recomputed — this is where amendment 2 differs from amendment
1).** Amendment 1 left family A byte-identical because the demoted cells were
outside the family-A OC machinery. Here six of the seven demoted cells **are**
family-A cells, so the family-A characteristic is recomputed on the residual
**47-cell** surface (draw-noise-free half-normal, the identical basis
`tests/test_gate_w1_derivations.py` uses):

- **Before**: 53 gated family-A cells → p_seed **0.922** / p_gate **0.9481**
  (the contract's stated figure; achievable OC 0, §3).
- **After**: 47 gated family-A cells → p_seed **0.9344** / p_gate **0.9623**.

Each demoted family-A cell has a faithful pass-prob of 0.997–0.998, so removing the
six **raises** p_seed slightly (fewer factors below 1) — the mechanical
consequence of removing structurally-unclearable cells from a conjunction. The
overall gate OC moves from an **achievable 0** to

> **p_gate(A) 0.9623 × I(C2),**

i.e. a well-defined characteristic on the residual **48-cell** surface (47 family-A
joints + the C2 fingerprint). This does **not** assert candidate 3 will pass — C2
is its empirical burden, and the residual 47 cells still hold candidate-quality
misses (the c3 levers: CPS-anchored entry, interior sex covariate, the co-designed
roster/window) — it asserts the gate is no longer **structurally zero**. It makes
**no** clearability claim about any demoted cell.

The re-anchor/re-specification alternatives (a2/b2/c2) are the right long-run fix
for the two concept groups and are **deferred to future bridge amendments** that
FIRST archive the missing anchors/vintages (§10); C1's reversal is robustly false
and has no re-specification (§4a).

## 6. No-self-rescue compliance

The standing rule (`gate_w1.governance.amendment_rules`, inherited from
`gate_1`): *no candidate's committed run verdict changes under a rule proposed
after that run; a gate rule amendment applies only to runs registered after its
ratification.* This proposal complies on every leg, and the **rescue set is
empty** — demoting the 7 cells rescues **neither** committed candidate, shown by
recomputing each candidate's verdict on the amended 48-cell surface:

1. **Candidate 1 (PR #162) STANDS FAIL — over-determined.** On the amended surface
   (47 family-A cells + C2), candidate 1 passes **0 of 5** family-A seeds (17–23 of
   the surviving 47 cells fail on each seed — e.g. `earnings_participation.35-44|female`,
   `coresident_spouse.25-34|female`, `earnings_participation.62-69|*`), so family A
   still fails ≥4-of-5; **and** C2 does not reverse (`reversed_to_anchor: false`,
   Kendall τ 0.333), so family C still fails. The gate fails via **both** surviving
   families independently of the demotion.
2. **Candidate 2 (PR #167) STANDS FAIL — over-determined.** On the amended surface
   candidate 2 passes **0 of 5** family-A seeds (9–12 of the 47 fail on each seed —
   e.g. `earnings_participation.35-44|female`, `hh_size_share.{1,2,3}`,
   `earnings_profile.35-44|*`); **and** C2 does not reverse (τ 0.333). Gate fails
   via both surviving families.
3. **The demoted cells were themselves failing for both candidates**, so their
   removal cannot flip a pass: the 18-24 **male** cell is 0/5 for both (rbar
   0.868 c1 / 0.930 c2 vs rate_a ~0.63); both 65+ **male** cells (married,
   coresident) are 0/5 for both (rbar ~0.71 vs rate_a ~0.84); C1 never reverses (τ
   0.667). The amendment removes cells the candidates were failing and they
   **still** fail on what remains.
4. **Prospective only.** The amendment binds only candidates registered **after**
   ratification. Candidates 1 and 2 remain FAIL as committed.
5. **The evidence is train/frame-side.** `gate_w1_forensics2_v1.json` and
   `…forensics1_v1.json` are `reported_not_gated`, `train_frame_side_only`,
   `publishes_regardless` — not scored candidate holdouts the amendment could
   retroactively rehabilitate.
6. **Description-claims-exactly.** After demotion, `covers` /
   `certification_scope` are narrowed to claim EXACTLY the amended 48-cell surface
   (47 family-A + 1 family-C); the 7 demoted cells are report-only with reasons
   named.

Like amendment 1, **no committed run flips** under either the amended-once or the
amended-twice surface — the disclosure is that the rescue set is empty, here
**over-determined** (each candidate fails via both surviving families).

## 7. The exact prospective flip (post-ratification; NOT in this PR)

The ratifying flip PR — a separate PR, after this ceremony clears — will make
exactly these edits to `gates.yaml` `gates.gate_w1`. Every edit is a re-scope or a
text narrowing; **no tolerance, σ, anchor value, or ordering changes**, and the
demoted cells' tolerances/ordering are retained under report-only (nothing is
deleted).

1. **`thresholds.family_a`** — move the six cells from their view `tolerances`
   (gate-eligible) to `report_only`, and add a parallel `report_reasons` mapping:
   - `earnings_participation.18-24|female` → `population_concept_delta_head_spouse_universe`
   - `earnings_participation.18-24|male` → `population_concept_delta_head_spouse_universe`
   - `marital_share.married.65+|female` → `cohort_vintage_hazard_frame_mismatch`
   - `marital_share.married.65+|male` → `cohort_vintage_hazard_frame_mismatch`
   - `coresident_spouse.65+|female` → `cohort_vintage_hazard_frame_mismatch`
   - `coresident_spouse.65+|male` → `cohort_vintage_hazard_frame_mismatch`

   The six tolerances are retained verbatim under the report-only disclosure. The
   `faithful_candidate_oc` block updates: `n_gated_cells` 53 → **47**,
   `p_seed_pass` 0.922 → **0.9344**, `p_gate_pass_4_of_5` 0.9481 → **0.9623**
   (recomputed on the 47-cell surface; a CHANGE, not the amendment-1 invariance).
2. **`thresholds.protocol.fresh_run_artifact_schema.per_draw_per_cell_rates`** —
   the committed-cube shape `[20, 53, 5]` → `[20, 47, 5]` (K_draws × gated_cells ×
   gate_seeds); `undefined_draw_rule` and the regenerated-surface rule are unchanged.
3. **`thresholds.family_c`** — demote `fingerprint.ppi_nra` (C1): move it from
   `gate_partition.gate_eligible` to a report-only list, tag it
   `fingerprint_reversal_not_realized`, and set `gate_partition` to
   `{gate_eligible: [fingerprint.elimination_plus2pp], n_gate_eligible: 1,
   n_report_only: 1}`. The C1 `anchor_values`, `required_representative_order`, and
   `mechanism` are retained under report-only (the ordering + PPI−NRA gap publish);
   the `pass_rule` is annotated that family C now gates the C2 fingerprint only.
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
   "reverses BOTH compression fingerprints" clause to "reverses the C2 compression
   fingerprint".
7. **`gate_w1.history`** — append the amendment-2 entry (§8 draft).

`thresholds.family_b` (already fully report-only after amendment 1),
`thresholds.protocol` (statistic, pass_rule, estimator, floor bindings),
`governance.amendment_rules`, `frame_pin`, `holdout_basis`, and every retained
tolerance/anchor/ordering are **untouched**. A subset master-compare must show the
flip changes only `gate_w1` and moves no locked sibling.

## 8. `amendment_history` entry draft (for `gate_w1.history`)

```yaml
- id: 2026-07-12-w1-family-a-concept-cells
  proposed: '2026-07-12'
  referee_round:
    review: '<PR> comment <id> (verdict TBD)'               # filled at ceremony
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
    coresident_spouse.65+|{female,male}). C1 carries machine reason
    fingerprint_reversal_not_realized; the two 18-24 cells carry
    population_concept_delta_head_spouse_universe; the four 65+ cells carry
    cohort_vintage_hazard_frame_mismatch. All 7 tolerances/orderings publish
    report-only (retained, not deleted). Motivated by W1 forensics 2
    (runs/gate_w1_forensics2_v1.json, registration 4953088871, grading 4953311492)
    Q6 (65+ divorce over-accumulation) and Q9 (the 18-24 concept gap + the C1
    consolidation), and W1 forensics 1 Q5 (runs/gate_w1_forensics1_v1.json, the C1
    non-reversal analytic). C1's required PPI-over-NRA reversal does NOT occur under
    any contract-consistent transport: on the heaviest plausible (upper-read) tail
    PPI savings 0.0169 sit far below NRA 0.2023 (gap 0.1854), and a realistic tail
    moves PPI to 0.0137; non-reversal proven analytically (Q5) plus two empirical
    candidate confirmations (Kendall tau 0.667 vs required). The 18-24
    participation miss is a POPULATION-CONCEPT delta of 22.1pp pooled (PSID
    head/spouse 0.865 vs CPS all-person 0.644; male 27.6pp), not a fit-support gap.
    The 65+ undershoot (deployed married 0.573/0.707 vs frame 0.691/0.856) is
    DIVORCE over-accumulation (deployed divorced 0.209/0.190 vs frame 0.090/0.063,
    excess +0.120/+0.127), a cohort-vintage hazard-vs-frame mismatch NOT the
    pre-registered widowhood, and hazard re-calibration is contract-prohibited, so
    no contract-consistent candidate lever exists (six permitted-lever
    construction-attempts each fail). So no contract-consistent candidate can clear
    any of the 7 cells. PROSPECTIVE ONLY (no_self_rescue): no committed run's
    verdict changes; candidate 1 (PR #162) and candidate 2 (PR #167) both STAND
    FAIL, over-determined -- on the amended 47-cell family-A surface each passes
    0/5 seeds AND C2 does not reverse (tau 0.333), so both fail via both surviving
    families; the demotion rescues nothing. Unlike amendment 1, family A CHANGES:
    the faithful-candidate OC recomputes on the residual 47 cells to p_seed 0.9344
    / p_gate 0.9623 (from 53 cells / 0.922 / 0.9481). Certification scope narrows:
    the C1 reversal, the 18-24 participation, and the 65+ marital/coresident
    composition are report-only, published-not-certified, pending future bridges.
    The gate_w1 faithful-candidate OC returns from an achievable 0 (three groups
    unclearable -> unpassable) to 0.9623 x I(C2) on the residual 48-cell gated
    surface (55 -> 48 gated, 77 -> 84 report-only). Evidence chain: candidate 1 (PR
    #162, run gate_w1_candidate1_v1), candidate 2 (PR #167, run
    gate_w1_candidate2_v1), W1 forensics 1 (PR #163,
    runs/gate_w1_forensics1_v1.json), W1 forensics 2 (PR #168,
    runs/gate_w1_forensics2_v1.json).
```

## 9. Certification-scope language after amendment 2

What the gate **claims** after the flip:

- **Certifies (gated)**: the 47 family-A CPS-observable joints and the C2
  compression-fingerprint reversal (the taxable-max elimination-over-+2pp
  ordering) — **48 cells**. A PASS certifies that stochastic regeneration through
  the deployed generators reproduces those 47 cross-sectional joints within the
  frame's own sampling floor, and reverses the C2 fingerprint.
- **Publishes (report-only, NOT certified)**: the C1 (PPI-over-NRA) ordering + the
  PPI−NRA gap, tagged `fingerprint_reversal_not_realized`; the 18-24 participation
  pair (head/spouse-vs-all-person disclosure), tagged
  `population_concept_delta_head_spouse_universe`; the 65+ marital/coresident quad
  (older-cohort-composition disclosure), tagged
  `cohort_vintage_hazard_frame_mismatch`; plus the report-only cells carried from
  the lock and amendment 1. Disclosed for transparency; a PASS does **not** claim
  them.
- **Does not support**: the **C1 compression fingerprint** as a certified reversal
  (it is robustly not realised); the **18-24 participation** margin as a certified
  transport (the head/spouse-vs-all-person concept gap); the **65+
  marital/coresident composition** as a certified transport (the cohort-vintage
  divorce over-accumulation). Any downstream use that rides on a certified 18-24
  participation, a certified 65+ marital composition, or a certified C1 reversal
  must await the relevant bridge amendment (§10). (The dynamics — earnings,
  marital, household, marriage × earnings, disability — remain gate-1/2a/2b/2c/M4
  certified on their PSID holdouts; W1 certifies transport, not re-estimation.)

Stated plainly: **the gate certifies the 47 reproducible family-A joints and the
C2 reversal; it certifies neither the C1 reversal, the 18-24 participation, nor
the 65+ marital/coresident composition, pending the bridges.** This is the
certification-scope cost of the report-only package, owned here rather than
softened.

## 10. Evidence-base prerequisites for a future re-anchor (the bridges)

Each deferred re-anchor/re-specification (a2/b2/c2) presupposes an evidence-base
amendment that does not exist today. Recorded so the future amendment's dependency
is explicit; this proposal does not archive them.

- **C1 (group a) — no re-specification exists.** The required PPI-over-NRA
  magnitude reversal is robustly false under every contract-consistent tail
  (§2a/§4a); no published anchor asserts a representative-frame PPI-over-NRA
  ordering, and the continuous PPI−NRA gap is a report-only diagnostic, not a
  binary. The prerequisite for ever re-gating C1 is a **new, defensible binary
  fingerprint** the anchors support and a contract-consistent transport can flip —
  which the evidence base does not contain. Absent that, C1 stays report-only.
- **18-24 participation (group b) — a head/spouse anchor or a non-head model.**
  Re-anchoring needs, at minimum, an archived **published head/spouse young-adult
  participation series** M4/gate-1 can transport onto (not archived), a
  concept-delta accounting of the residual head/spouse-instrument deltas, and a
  fresh referee round; **or** a generator extension that models **non-head**
  young-adult participation (a re-estimation W1 does not certify). Either is a
  future modelling amendment.
- **65+ marital/coresident (group c) — a cohort-vintage hazard or a re-anchored
  frame.** Re-anchoring needs either an **older-cohort divorce/remarriage hazard
  vintage** (a cohort-specific re-estimation — the certified pooled 1998–2022
  hazards are locked) or a **re-anchored 65+ frame composition** target; both are
  evidence-base amendments outside the transport contract. Archiving a cohort-vintage
  marital-transition series is the prerequisite for any future 65+ marital bridge.

## 11. Ceremony checklist

- [x] **Proposal** (this document + this draft PR). No `gates.yaml` edit; no
      threshold moved.
- [ ] **Adversarial referee round** (construction-attempt method per group; the
      candidate-independence standard applied to any contract-consistent
      candidate, not only the two built).
- [ ] **Fixes**.
- [ ] **Verification round** (recompute every number from committed artifacts; the
      §7 flip text, §7 roll-up, and §8 history figures are string-bound by
      `tests/test_gate_w1_amendment2_proposal.py` from the start — the amendment-1
      fix-B finding is not repeated).
- [ ] **Ratify by merge** of the flip PR (under the standing campaign directive,
      only after the ceremony clears).
- [ ] **Flip** `gates.yaml` per §7 and append the §8 history entry.
- [ ] Then: candidate 3 against the amended contract (the CPS-anchored entry, the
      interior sex covariate, the co-designed roster/window levers).

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
  "groups": {
    "c1_fingerprint": {
      "cells": ["fingerprint.ppi_nra"],
      "machine_reason": "fingerprint_reversal_not_realized",
      "family": "c"
    },
    "participation_18_24": {
      "cells": [
        "earnings_participation.18-24|female",
        "earnings_participation.18-24|male"
      ],
      "machine_reason": "population_concept_delta_head_spouse_universe",
      "family": "a"
    },
    "marital_coresident_65plus": {
      "cells": [
        "marital_share.married.65+|female",
        "marital_share.married.65+|male",
        "coresident_spouse.65+|female",
        "coresident_spouse.65+|male"
      ],
      "machine_reason": "cohort_vintage_hazard_frame_mismatch",
      "family": "a"
    }
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
    "realized_65plus_channel": "divorce"
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
    "candidate1_c1_reversed": false,
    "candidate2_c1_kendall_tau_vs_required_round4": 0.6667,
    "candidate2_c1_reversed": false
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
  "family_a_partition": {
    "gated_now": 53,
    "gated_after": 47,
    "report_only_now": 52,
    "report_only_after": 58
  },
  "family_c_partition": {
    "gated_now": 2,
    "gated_after": 1,
    "report_only_now": 0,
    "report_only_after": 1
  },
  "overall_partition": {
    "gated_now": 55,
    "gated_after": 48,
    "report_only_now": 77,
    "report_only_after": 84
  },
  "oc_statement_after": "0.9623 * I(c2)",
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
