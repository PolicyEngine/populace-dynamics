# gate_w1 amendment 1 (proposal) — the family-B DI bands and conversion margins

- **Amendment id**: `2026-07-12-w1-family-b-di-bands`
- **Gate**: `gate_w1` (W1 representative-frame transport, roadmap #113 M5, workstream #100)
- **Surface**: `thresholds.family_b` — all 10 gated cells (the 8 DI age-composition
  prevalence bands + the 2 disability-to-retirement conversion margins)
- **Ceremony stage**: PROPOSAL (draft). This document is the first step of the
  amendment ceremony (proposal → adversarial referee → fixes → verification →
  ratify-by-merge → flip). **It moves no threshold and edits no `gates.yaml`
  cell.** The prospective flip happens in a separate ratifying PR, only after
  the ceremony clears.
- **Amendment class**: STRUCTURAL, prospective, ZERO threshold movement (the
  gate-2 amendment-2 tranche-split pattern).
- **Fixes round**: this document incorporates the adversarial referee round
  (PR #164 comment [4951701300][ref-round], verdict AMEND THE AMENDMENT). The
  round confirmed Q4's determination on the 8 bands (over-determined three ways
  on the pinned frame) and required extending the demotion to the 2 conversion
  margins, which fail the identical candidate-independence test on committed
  evidence (§4a). The surface, OC statement, option table, ledger, and bindings
  are corrected accordingly.
- **Evidence base**: W1 forensics 1, Q4 (`runs/gate_w1_forensics1_v1.json`,
  registration [4951218279][reg], grading [4951430002][grade]); candidate 1
  (`runs/gate_w1_candidate1_v1.json`, PR #162, registration 4950931131); the M4
  disability evidence base (`runs/m4_disability_v1.json`, `conversion_validation`);
  the locked gate (`gates.yaml` blob `cd6411d9`, ratified PR #160 over floors #154).

[reg]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4951218279
[grade]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4951430002
[ref-round]: https://github.com/PolicyEngine/populace-dynamics/pull/164#issuecomment-4951701300

## 1. Summary

Family B of `gate_w1` gates 10 SSA administrative margins: 2 disability-to-
retirement **conversion** cells (an M4-simulated flow scored over retired-worker
awards) and 8 **DI age-composition prevalence bands** (the age composition of
the SSA disabled-worker beneficiary stock, DI ASR Table 19). W1 forensics 1 Q4
established — and grading [4951430002][grade] adjudicated as a **BLOCKER** — that
the 8 DI bands are **unclearable by any contract-consistent candidate**. The
gate scores a quantity the deployed model *can* produce (the M4 work-disability
**point-prevalence** among person-years) against an SSA quantity it *never
modelled* (the duration-accumulated **insured-beneficiary stock**), across a
concept bridge the gate does not define and whose third leg (the Supplement 4.C2
insured denominator) the evidence base does not even archive. Every one of the 8
bands misses its tolerance by **2.9x–21.9x**; the concept delta is the aggregate
dominant share (**0.595**, per-band range 0.023–0.891) of the level/steepness
gap, the M4 hazard shape the rest.

The adversarial round ([4951701300][ref-round]) then established that **the 2
conversion cells fail the identical candidate-independence test** (§4a). Their
numerator is the *same* undefined DI stock, read at the FRA boundary: SSA 6.B5.1
conversions are the accumulated disabled-worker stock arriving at FRA, while the
contract's deployed numerator is the M4 point-prevalence, constant at **5.79 (f)
/ 6.55 (m)** across the whole 60–66 coarse band. The M4 evidence base itself
already ruled this concept a level mismatch — `conversion_validation` reports a
PSID-analog/admin ratio of **0.267/0.322** and states the anchor "certifies
co-movement and ordering, **never a level match**" — yet the gate locked a level
match at ±1.89/±1.91pp. Every honest stock-to-flow reduction caps at **6.74/7.62**
against the required pass-window floor of **≥12.21/12.59** (anchor 14.1/14.5 −
tolerance 1.89/1.91). All **10** gated cells are therefore unclearable.

Because family B is a conjunction, its 10 unclearable cells make **gate_w1 as
locked unpassable**: the faithful-candidate operating characteristic is
structurally **0**, no matter how well a candidate transports the 53 family-A
joints or the 2 family-C fingerprints.

**Two options are analysed below; this proposal recommends option (a).**

- **Option (a) — demote (RECOMMENDED)**: move **all 10** family-B gated cells to
  report-only. The 8 DI bands and the 2 conversion cells take machine reason
  `concept_bridge_undefined_di_stock` (their common defect — the undefined
  DI-stock numerator), and the 2 conversion cells additionally carry
  `conversion_level_match_never_certified` (the M4 evidence base pre-ruled the
  conversion concept "never a level match"). Family B then gates **nothing**;
  it is fully report-only. Zero threshold movement; the SSA anchors stay
  in-contract, published-not-certified, pending a future bridge amendment. The
  gate's faithful-candidate OC returns from a structural 0 to a well-defined
  **0.9481 × I(family C)** on the residual **55-cell** gated surface — family B
  no longer contributes a structurally-zero factor because it no longer gates,
  and it certifies **no** SSA margin until the bridge amendment.
- **Option (b) — re-anchor**: replace the SSA-stock anchors with published
  work-disability **prevalence** / **incidence** targets M4 can transport.
  Deferred: no defensible transportable anchor is archived today, and the
  SSA-native re-anchor (for both the bands and the conversion flow) is blocked by
  the same unarchived Supplement 4.C2 plus the missing incidence/duration inputs.

## 2. The finding (W1 forensics 1, Q4)

`runs/gate_w1_forensics1_v1.json` is a one-shot, `reported_not_gated`,
`train_frame_side_only` diagnostic that "publishes regardless of any verdict" —
it measured the five transport mechanisms on the frame side before designing
candidate 2. Its Q4 block (`q4_di_level_bridge`) decomposes the DI level/
steepness gap.

**Mechanism.** Family B derives DI status from the M4 work-disability prevalence
(the `no_frame_di_column_rule` + `ss_proxy_laundering_rule` forbid reading the
frame's own DI column), a **point-prevalence among PSID person-years**, and
scores its age composition against the **SSA disabled-worker beneficiary STOCK**
(Table 19). The M4 prevalence peaks at 50–59 and *drops* at 60–66; the SSA stock
keeps climbing to 45.4% at 60-FRA because it is **duration-accumulated** —
entrants stay on the rolls (DI recovery ≈ 1%/yr) until they auto-convert to
retirement at FRA. A point-prevalence structurally cannot concentrate at the
stock's 60-FRA mass.

**Per-band result (all 8 fail, by wide margins).** Deployed = M4 stock
point-prevalence; anchor = SSA stock; tolerance = reference-period vintage
tolerance; miss = |anchor − deployed| ÷ tolerance.

| Band | deployed pp | anchor pp | \|gap\| pp | tolerance pp | miss | clears? |
|---|---:|---:|---:|---:|---:|:---:|
| under30 | 8.19 | 1.4 | 6.79 | 0.31 | 21.9x | no |
| 30-34 | 7.09 | 2.0 | 5.09 | 0.33 | 15.4x | no |
| 35-39 | 6.79 | 3.5 | 3.29 | 0.40 | 8.2x | no |
| 40-44 | 12.34 | 5.8 | 6.54 | 0.50 | 13.1x | no |
| 45-49 | 10.61 | 7.8 | 2.81 | 0.49 | 5.7x | no |
| 50-54 | 16.84 | 12.7 | 4.14 | 0.94 | 4.4x | no |
| 55-59 | 16.61 | 21.4 | 4.79 | 1.68 | 2.9x | no |
| 60-fra | 21.53 | 45.4 | 23.87 | 2.14 | 11.2x | no |

Even the **closest** band (55-59) misses by 2.9x its tolerance; the worst
(under30) by 21.9x. **Both** components of every band's gap sit outside the
candidate's lever set: the concept bridge is undefined (no DI-entry hazard, no
duration-to-conversion model, no insured denominator) **and** the M4 generator
is frozen ("W1 certifies transport, not re-estimation") — a candidate can neither
build the bridge nor re-estimate the hazard. This is the candidate-independent
statement the demotion rests on; it does not require the per-band gap to be
*all* concept (§ correction below).

**Decomposition.** At the worst band (60-FRA) the anchor-minus-deployed gap of
+23.87pp splits into a **duration (stock-vs-flow) component of +21.3pp** and an
**M4-hazard-shape component of +2.6pp**: even the *correct flow concept* (true
disabled-worker **awards** flow, +24.1pp) falls far short of the accumulated
**stock** (45.4pp). Across the 8 bands the duration-concept delta is the
**aggregate** dominant share (`concept_delta_dominant_share` = **0.595** =
Σ|dur| ÷ (Σ|dur| + Σ|shape|)); the M4 hazard shape is the aggregate minority.
This is an aggregate, **not** a per-band claim: the duration-concept share of
|components| spans **0.023 (40-44) to 0.891 (60-fra)** — at 40-44 the miss is
nearly all frozen-M4 shape, at 60-FRA nearly all duration concept. The
forensics enumerates **7** standing concept deltas between the PSID self-report
and a DI award (definition: self-report vs medical-vocational adjudication;
population: all adults vs insured workers; severity threshold; recovery churn
25–50%/interval vs ≈1%/yr; conversion denominator; biennial timing censoring;
period pooling 1982–2023 vs a single-era column) — most of which persist against
*any* external DI anchor, not just the stock. Delta #5 (the **conversion
denominator**) is the one the retained conversion cells sit on (§4a).

**Gate-design determination.** The forensics records
`is_gate_design_finding: true` and `insured_denominator_available: false`:
matching the stock composition requires (1) a **DI-entry hazard** (incidence, not
prevalence), (2) a **duration-to-conversion** model (stock = ∫ entries ×
on-rolls survival), and (3) an **insured denominator** (the 20/40 recent-work
test) — none defined by the gate, and the third (**Supplement 4.C2**) not even
archived (`data/external/di_asr_2023/provenance.md` lists it as *still wanted*).
With only `{age, is_female}` conditioning on a recovery-churned self-report
prevalence and the frame DI column forbidden, the bands are unreachable by a
candidate **lever**. The determination itself names the three dispositions:
*define the concept bridge, re-anchor to a work-disability prevalence M4 can
transport, or demote the DI bands to report-only until the bridge exists.*

Grading [4951430002][grade] adjudicated Q4 **HELD, and a BLOCKER**: "With 8 of 10
family-B cells unclearable, gate_w1 as locked is unpassable — an amendment
ceremony is REQUIRED before candidate 2." The adversarial round
([4951701300][ref-round]) extended the count to **10 of 10** (§4a).

## 3. Why gate_w1 as locked is unpassable

The family-B `pass_rule` is a **conjunction** over its gate-eligible cells (a run
passes family B iff every gated cell is within tolerance). **All 10** gated cells
are unclearable by any contract-consistent candidate: the 8 DI bands (§2) and the
2 conversion margins (§4a), the latter over-determined by the same undefined
DI-stock numerator, the M4 "never a level match" adjudication, and an achievable
range that excludes the pass window under every honest stock-to-flow reduction.
The overall gate is a conjunction across families (family A 4-of-5 seeds ∧ family
B ∧ family C). Therefore, no matter the family-A or family-C outcome, the
family-B factor is **deterministically 0**, and:

> **faithful-candidate gate OC (locked surface) = p_gate(A) × 0 × I(C) = 0.**

The gate is not "hard"; it is **structurally impossible**, and over-determined
(10/10, not 8/10). This is a defect of the *contract*, not of any candidate —
exactly the class the standing ceremony exists to correct.

## 4. Options

### Option (a) — demote all 10 family-B gated cells to report-only (RECOMMENDED)

Move all 10 gated cells — the 8 `di_prevalence.*` bands and the 2
`claim_age.disability_conversion|{female,male}` margins — from
`family_b.gated_cells` to `family_b.report_only`. The 8 bands are each tagged
`report_reason: concept_bridge_undefined_di_stock`; the 2 conversion cells are
tagged with **both** `concept_bridge_undefined_di_stock` (their common
undefined-stock numerator) and `conversion_level_match_never_certified` (the M4
evidence base pre-ruled the conversion concept a level mismatch). Family B then
gates **nothing** — it is fully report-only. The 10 SSA anchor values and their
tolerances **stay in the contract** (report-only): they are published for
disclosure, not deleted, and not gated. This is a naming/scope amendment with
**zero threshold movement** — no family-A cell, tolerance, σ, or family-C
ordering changes.

**Why the 2 conversion cells must also be demoted (adversarial round, finding 2).**
The proposal originally retained the conversion cells as "a flow over awards … a
quantity the deployed M4 dynamics *can* represent," calling c1's miss a
"candidate-quality" miss the candidate could improve. The referee round proved,
from committed artifacts, that no contract-consistent candidate can clear them
either — the identical candidate-independence test the proposal correctly applies
to the bands:

- **The numerator is the same undefined stock.** SSA 6.B5.1 conversions are the
  accumulated DI stock arriving at FRA. The deployed numerator under the contract
  is the M4-derived point-prevalence, constant at **5.79 (f) / 6.55 (m)** across
  the whole 60–66 coarse band (`runs/m4_disability_v1.json`,
  `psid_disabled_prevalence_60_66_pct`). A flow that equals a stock crossing a
  boundary is still duration-accumulated: landing it needs the same three bridge
  legs §2 says the gate never defined. `concept_bridge_undefined_di_stock`
  applies to these two cells verbatim (this is Q4 concept delta #5, the
  conversion denominator).
- **The M4 evidence base already adjudicated this concept — before the W1 lock.**
  `runs/m4_disability_v1.json` `conversion_validation` reports a PSID-analog/admin
  ratio of **0.267/0.322** and states in `interpretation` that "the anchor
  certifies co-movement and ordering, **never a level match**." gate_w1 then
  locked a **level** match at ±1.89/±1.91pp on exactly this margin. That is the
  contradiction the `conversion_level_match_never_certified` reason names.
- **The achievable range excludes the tolerance window under every honest
  stock-to-flow reduction.** c1's registration disclosed that the awards-cohort
  reduction is unpinned ("the stock-to-flow reduction the locked text does not
  pin"); it cannot matter, because the point-prevalence is constant on 60–66, so
  every cohort built from permitted `(age, sex)` columns yields the same deployed
  value in expectation. The referee round recomputed on the sha-verified pinned
  frame: the prevalence read is 5.79/6.55; the honest M4 forward-chain gives
  5.23/5.78; the strongest honest denominator lever (all conversions count, only
  insured non-DI persons generate awards, f = 0.85) reaches **6.74/7.62** — all
  far below the required pass-window floor of **≥12.21/12.59** (anchor 14.1/14.5 −
  tolerance 1.89/1.91). Even an indefensible f = 0.5 reaches only 10.9/12.3 —
  still 0/2. K=20-mean draw noise is 0.48/0.31pp (c1's committed across-draw sd
  2.132/1.395 ÷ √20) and the pass rule scores the mean once, so +3σ closes
  nothing; the contract-excluded absorbing overlay overshoots to ~31. The anchor
  sits in territory only a *calibrated* duration model plus the insured
  denominator can reach — the very bridge that is undefined and unarchived.

c1's committed miss on these cells (deployed 5.80/6.43 vs anchor 14.1/14.5;
abs-dev **8.30/8.07** ≫ tol 1.89/1.91) is therefore not a candidate-quality miss
but the signature of the same structural concept mismatch. Retaining the 2 cells
would leave family B gating a pair whose faithful-candidate pass probability is
0 — i.e. it would ratify a contract that *looks* passable (a 0.9481 headline)
while remaining structurally zero. Demoting all 10 is the only disposition that
makes the amended gate actually passable-in-principle.

**Cost.** Family B no longer certifies **any** SSA margin — neither the DI age
composition nor the conversion share. Certification scope narrows accordingly
(§9): family B becomes a fully report-only disclosure block, published-not-
certified, pending the concept-bridge amendment.

**OC consequence.** Family A is untouched, so its stochastic characteristic is
**byte-identical** — recomputing the draw-noise-free half-normal OC from the 53
locked family-A tolerances and the frozen floor σ gives **p_seed 0.922 /
p_gate 0.9481**, and none of the 10 family-B cells appears in the family-A OC
machinery (empty intersection), so demotion cannot move it. The overall
faithful-candidate gate OC moves from a **structural 0** (unpassable) to

> **p_gate(A) 0.9481 × I(family C),**

i.e. it returns to the well-defined family-A characteristic on the residual
**55-cell** gated surface (53 family-A + 2 family-C fingerprints; family B
contributes **nothing gated**). This does **not** assert candidate 2 will pass —
the family-C indicator is candidate 2's empirical burden — it asserts the gate is
no longer **structurally zero**. It makes **no** clearability claim about any
family-B cell: family B certifies nothing until the bridge amendment.

### Option (b) — re-anchor the family-B cells to transportable work-disability concepts

Replace the SSA disabled-worker **stock** anchor (DI ASR Table 19, the 8 bands)
and the SSA **conversion-flow** anchor (6.B5.1, the 2 cells) with published
work-disability **prevalence / incidence** targets M4's PSID point-prevalence and
transition dynamics can transport onto. This attacks the stock-vs-flow delta at
its root rather than removing the cells.

**Candidate anchors, assessed honestly:**

| Candidate anchor | Concept vs M4 | Availability | Verdict |
|---|---|---|---|
| SSA DI **insured** prevalence (beneficiaries ÷ **4.C2 insured denominator**, by age) — for the 8 bands | The SSA-native fix; still a beneficiary count, but on the correct denominator | **Not archived** — Supplement 4.C2 is *still wanted* (`di_asr_2023/provenance.md`) | **Blocked** (same evidence-base gap as the bridge) |
| SSA DI **incidence / duration** inputs (age-sex incidence-rate series + termination assumptions) — for the 2 **conversion** cells | The SSA-native fix for the conversion flow: an entry hazard + duration-to-conversion survival M4 can simulate, not a stock level | **Not archived** — Trustees V.C5/V.C6 and the age-sex incidence series are *still wanted* (`provenance.md`), same gap as 4.C2 | **Blocked** (conversion anchor deferral — same evidence-base gap) |
| **ACS** disability (6-type / work-limitation), B18120 / S1810 by age | Self-reported functional/work limitation point-prevalence — closer, but a different instrument and threshold than the PSID self-report; the definition/severity/population deltas persist | Public (PUMS + published tables), annual, age-banded | Transportable **in principle**, but needs archiving + a fresh concept-delta accounting + new vintage tolerances |
| **SIPP** work-disability / functional-limitation module | Work-limitation prevalence, closer to M4's construct | Public-use; age composition needs extraction; pooling/vintage issues | Transportable in principle; not archived; larger lift |
| **CPS-ASEC** disability / work-limitation | Nearest instrument to the certified frame | **Built-from-frame** — the certified frame is CPS-derived and calibrated, so anchoring family B on the frame's own source survey re-tests populace calibration rather than external transport | **Out** (re-tests calibration, not transport) |

**Why option (b) is deferred, not chosen now.** Every transportable candidate
requires, at minimum: (i) **archiving** a new published anchor into the evidence
base; (ii) a **concept-delta accounting** of the residual PSID-self-report vs
external-instrument deltas (the 7 named deltas do not vanish under re-anchoring —
they shrink); (iii) **re-deriving** the reference-period vintage tolerances from
the new source's vintages; and (iv) an **adversarial referee round** on the new
anchor's transportability. The SSA-native re-anchor is blocked outright for
**both** the bands (unarchived 4.C2) and the conversion cells (unarchived
incidence/duration inputs). So option (b) is a larger, multi-step **redesign**
that itself presupposes a prior evidence-base amendment — it is not a prospective
re-scope and cannot be executed inside this ceremony.

**OC consequence.** Indeterminate. The family-B gated set would stay at 10, but
the re-anchored cells acquire new anchors and new tolerances, so the faithful-
candidate OC over those cells is unknown until the anchors are chosen, archived,
and their residual concept deltas are refereed. Stating an OC today would be
fabrication.

## 5. Recommendation

**Adopt option (a).** It is the minimal contract-clean move that (i) removes
**exactly** the structurally-unclearable cells — all 10 of family B, no more and
no fewer; (ii) preserves every other gated cell and every family-A/C threshold
byte-identical (zero threshold movement); (iii) keeps the SSA anchors in-contract
as honest report-only disclosure; (iv) names the defect with machine reasons
(`concept_bridge_undefined_di_stock` on all 10, plus
`conversion_level_match_never_certified` on the 2 conversion cells); and (v)
restores a well-defined, non-zero gate OC (**0.9481 × I(family C)** on the
residual 55-cell surface) **without** asserting any family-B cell is clearable —
family B certifies nothing. It matches the gate-2 amendment-2 precedent (a
structural re-scope with no verdict to rescue and nothing re-scored) and the
grading directive, which lists demotion first.

Option (b) is the **right long-run fix** and is explicitly **deferred to a
future bridge amendment** that FIRST archives Supplement 4.C2 (for the bands) and
the incidence/termination inputs (for the conversion flow), and defines the
DI-entry + duration-to-conversion model — the very machinery the Q4 determination
enumerates (§10).

## 6. No-self-rescue compliance

The standing rule (`gate_1.amendment_rules`, inherited verbatim by
`gate_w1.governance.amendment_rules`): *no candidate's committed run verdict
changes under a rule proposed after that run; a gate rule amendment applies only
to runs registered after its ratification.* This proposal complies on every leg:

1. **c1's verdict STANDS regardless.** Candidate 1
   (`runs/gate_w1_candidate1_v1.json`, PR #162) fails **all three families
   independently**: family A 0/5 seeds, family B **0/10** cells (all 8 bands AND
   both conversion cells fail), family C false. The demotion cannot flip it: with
   family B reduced to **zero** gated cells, the gate is family A ∧ family C, and
   c1 **still** fails family A **and** family C — the two families the amendment
   does not touch. There is no committed verdict to rescue; the amendment rescues
   nothing. (Extending the demotion from 8 cells to all 10 does not change this:
   c1's failure is over-determined on families A and C alone.)
2. **Prospective only.** The amendment binds only candidates registered **after**
   ratification. Candidate 1 remains FAIL as committed.
3. **The evidence is train/frame-side.** `runs/gate_w1_forensics1_v1.json` is
   `reported_not_gated: true`, `protocol.train_frame_side_only: true`,
   `publishes_regardless: true` — a frame-side diagnostic, not a scored candidate
   holdout verdict. The evidence motivating the amendment is not itself a gated
   result the amendment could retroactively rehabilitate.
4. **Description-claims-exactly.** After demotion, `covers` and
   `certification_scope` are narrowed to claim EXACTLY the amended 55-cell gated
   surface (53 family-A + 2 family-C); all 10 family-B cells are report-only with
   their reasons named (per `description_claims_exactly_the_scored_surface`).

Unlike gate-2 amendment 1 (where retroactive application *would* have flipped one
run, disclosed and adjudicated in the record), here **no committed run flips
under either the locked or the amended surface** — the disclosure is that the
rescue set is empty.

## 7. The exact prospective flip (post-ratification; NOT in this PR)

The ratifying flip PR — a separate PR, after this ceremony clears — will make
exactly these edits to `gates.yaml` `gates.gate_w1`. Every edit is a re-scope or
a text narrowing; **no family-A/C threshold, tolerance, σ, anchor value, or
ordering changes**, and the family-B *anchor values and tolerances* for all 10
cells are retained under report-only (nothing is deleted).

1. **`thresholds.family_b.gated_cells`** — remove **all 10** keys: the 8
   `di_prevalence.*` bands
   (`di_prevalence.under30, di_prevalence.30-34, di_prevalence.35-39,
   di_prevalence.40-44, di_prevalence.45-49, di_prevalence.50-54,
   di_prevalence.55-59, di_prevalence.60-fra`) and the 2 conversion margins
   (`claim_age.disability_conversion|female, claim_age.disability_conversion|male`).
   `gated_cells` becomes **empty**.
2. **`thresholds.family_b.report_only`** — append all 10 cells, and add a parallel
   `report_reasons` mapping: each of the 8 `di_prevalence.*` bands tagged
   `concept_bridge_undefined_di_stock`; each of the 2
   `claim_age.disability_conversion|{female,male}` cells tagged with both
   `concept_bridge_undefined_di_stock` and `conversion_level_match_never_certified`.
   Retain all 10 anchor values + tolerances in a report-only disclosure block.
3. **`thresholds.family_b.gated_composition`** — "2 disability_conversion
   (M4-simulated margin) + 8 DI age-composition bands" → "**none** — all 10 cells
   (2 disability_conversion + 8 DI age-composition bands) demoted to report-only
   by amendment 1 (`concept_bridge_undefined_di_stock`; the conversion cells also
   `conversion_level_match_never_certified`)".
4. **`thresholds.family_b.candidate_protocol.pass_rule`** and
   `simulated_object.*` — annotate that family B has **no gated cells**; the
   family-B factor drops out of the gate conjunction (report-only). The
   `di_prevalence` and `disability_conversion` descriptions remain for the
   report-only publication.
5. **`gate_w1.covers`** — "10 gated (2 … + 8 DI …) / 15 report-only" → "**0 gated**
   (family B gates nothing) / **25 report-only** (2 disability-conversion
   M4-simulated margins + 8 DI age-composition bands, both demoted by amendment 1;
   14 circular claim-age + 1 benefit level)"; and the overall roll-up
   "65 gated / 67 report-only" → "**55 gated / 77 report-only**".
6. **`thresholds.certification_scope.certifies`** — "the 65 gated cells (53 … +
   10 SSA family-B … + 2 …)" → "the **55** gated cells (53 family-A + 2 family-C);
   family B certifies **no** SSA margin"; and delete the "lands the deployed +
   simulated DI/conversion margins on the SSA anchors" clause (the DI
   age-composition bands **and** the DI-to-retirement conversion share are
   report-only pending a concept-bridge amendment).
7. **`thresholds.certification_scope.supports`** — the bullet "the deployed +
   simulated DI composition (conversion share + DI age bands) …" → "the deployed
   + simulated DI margins (conversion share **and** DI age-composition bands) are
   **report-only** pending the concept-bridge amendment; see amendment 1".
8. **`gate_w1.history`** — append the amendment-1 entry (§8 draft).

`thresholds.family_a` (all views, tolerances, `power_cap`,
`faithful_candidate_oc`, `heavy_tail_boundary_bootstrap`, `family_a_prime`),
`thresholds.family_c`, `thresholds.protocol`, `governance.amendment_rules`,
`frame_pin`, and `holdout_basis` are **untouched** (`di_asr_2023` and the 6.B5.1
conversion source stay in `holdout_basis`: the anchors are still published,
report-only). A subset master-compare must show the flip changes only `gate_w1`
and moves no locked sibling.

## 8. `amendment_history` entry draft (for `gate_w1.history`)

```yaml
- id: 2026-07-12-w1-family-b-di-bands
  proposed: '2026-07-12'
  referee_round:
    review: 'PR #164 comment 4951701300 (verdict AMEND THE AMENDMENT)'
    fixes: '<sha>'                                          # filled at ceremony
    verification: '<PR> comment <id> (verdict TBD)'         # filled at ceremony
  ratified: '<date> by merge of PR <n> (merge commit <sha>) under the
    maintainer''s standing campaign directive of 2026-07-07, exercised only
    after the full ceremony'                                # filled at ratification
  flipped_live: this pull request
  content: >-
    STRUCTURAL amendment, ZERO THRESHOLD MOVEMENT. ALL 10 family-B gated cells
    are DEMOTED from gated to report-only: the 8 DI age-composition prevalence
    bands (di_prevalence.under30 .. .60-fra) AND the 2 disability-conversion
    M4-simulated margins (claim_age.disability_conversion|{female,male}). The 8
    bands and the 2 conversion cells carry machine reason
    concept_bridge_undefined_di_stock (their common undefined-DI-stock
    numerator); the 2 conversion cells additionally carry
    conversion_level_match_never_certified. Family B now gates NOTHING (fully
    report-only); it certifies no SSA margin. All 10 SSA anchors publish
    report-only (values + tolerances retained, not deleted). Motivated by W1
    forensics 1 Q4 (runs/gate_w1_forensics1_v1.json, registration 4951218279,
    grading 4951430002) for the 8 bands and by the adversarial round (PR #164
    comment 4951701300) for the 2 conversion cells: the gate scored an M4
    WORK-DISABILITY point-prevalence against the SSA DISABLED-WORKER BENEFICIARY
    STOCK (DI ASR Table 19, bands) and against the SSA 6.B5.1 conversion flow
    (conversion cells) across a concept bridge it never defined -- a
    duration-accumulated insured-beneficiary stock needs a DI-entry hazard, a
    duration-to-conversion survival model, and the Supplement 4.C2 insured
    denominator, none defined by the gate and the third not archived. For the
    bands the concept delta dominates in aggregate (share 0.595, per-band range
    0.023-0.891; +21.3pp duration-accumulation vs +2.6pp M4 hazard shape at
    60-FRA); all 8 bands fail by 2.9x-21.9x their tolerances (passes:false, gaps
    2.8-23.9pp vs tolerances 0.31-2.14pp). For the conversion cells the
    numerator is the same undefined stock (point-prevalence 5.79/6.55 constant
    on 60-66); the M4 evidence base (runs/m4_disability_v1.json
    conversion_validation) pre-ruled the concept "never a level match" (ratio
    0.267/0.322); every honest stock-to-flow reduction caps at 6.74/7.62 vs the
    required pass-window floor >=12.21/12.59 (anchor 14.1/14.5 - tol 1.89/1.91).
    So no contract-consistent candidate can clear any of the 10 cells.
    PROSPECTIVE ONLY (no_self_rescue): no committed run's verdict changes;
    candidate 1 (runs/gate_w1_candidate1_v1.json, PR #162) stands FAIL -- it
    fails family A (0/5 seeds) AND family C independently of the amendment (and
    all 10 family-B cells, dev 8.30/8.07 vs tol 1.89/1.91 on the conversion
    pair), so the demotion rescues nothing. Family A UNCHANGED (53 cells,
    faithful OC p_seed 0.922 / p_gate 0.9481, byte-identical -- none of the 10
    family-B cells is in the family-A OC machinery); family C UNCHANGED; ZERO
    family-A/C threshold movement. Certification scope narrows: family B
    certifies NOTHING; all 10 cells are report-only, published-not-certified,
    pending a future concept-bridge amendment that FIRST archives Supplement
    4.C2 and the incidence/duration inputs. The gate_w1 faithful-candidate OC
    returns from a structural 0 (10/10 family-B cells unclearable -> unpassable)
    to 0.9481 x I(family C) on the residual 55-cell gated surface (65 -> 55
    gated, 67 -> 77 report-only). Evidence chain: candidate 1 (PR #162, run
    gate_w1_candidate1_v1), W1 forensics 1 (PR #163,
    runs/gate_w1_forensics1_v1.json), M4 disability (runs/m4_disability_v1.json).
```

## 9. Certification-scope language after amendment 1

What family B **claims** after the flip:

- **Certifies (gated)**: **nothing**. Family B has no gated cell after amendment
  1; it contributes no factor to the gate conjunction. The gate certifies exactly
  the 53 family-A joints and the 2 family-C fingerprints (55 cells).
- **Publishes (report-only, NOT certified)**: the DI age-composition prevalence
  bands (8 cells, DI ASR Table 19 anchors retained), tagged
  `concept_bridge_undefined_di_stock`; the 2 DI-to-retirement conversion margins
  (6.B5.1 anchors retained), tagged both `concept_bridge_undefined_di_stock` and
  `conversion_level_match_never_certified`; the 14 circular claim-age cells; the
  benefit level. These are disclosed for transparency; a PASS does **not** claim
  them.
- **Does not support**: **any** DI-derived SSA margin as a certified quantity —
  neither the DI age composition nor the DI-to-retirement conversion share. Any
  payable-benefit baseline that rides on a DI margin must await the future
  concept-bridge amendment. (The dynamics — earnings, marital, household,
  marriage × earnings, disability — remain gate-1/2a/2b/2c/M4-certified on their
  PSID holdouts; W1 certifies transport, not re-estimation.)

Stated plainly: **family B certifies no SSA margin pending the concept-bridge
amendment.** This is the certification-scope cost of option (a), owned here
rather than softened.

## 10. Evidence-base gap (Q4-adjacent): Supplement 4.C2 and the incidence inputs

`data/external/di_asr_2023/provenance.md` records, under *still wanted*:
Trustees V.C5/V.C6 (incidence/termination assumptions), **Supplement 4.C2 (the
insured denominator)**, and the age-sex-adjusted incidence-rate series. The
insured denominator is the **third leg** of the concept bridge (§2): without it
the DI-beneficiary stock cannot be normalised to an insured-population rate M4
could target. The incidence/termination assumptions are the **entry-hazard and
duration** legs the conversion flow needs (§4a). **Archiving Supplement 4.C2 plus
the incidence/duration inputs is a prerequisite for any future DI-stock or
DI-conversion bridge amendment** (option (b) or the "define the bridge"
disposition). This proposal records the gap so the future amendment's evidence
dependency is explicit; it does not itself archive them.

## 11. Ceremony checklist

- [x] **Proposal** (this document + this draft PR). No `gates.yaml` edit; no
      threshold moved.
- [x] **Adversarial referee round** ([4951701300][ref-round]; verdict AMEND THE
      AMENDMENT — extend the demotion to all 10 family-B cells; bind the flip
      text; text-precision corrections).
- [x] **Fixes** (this revision: all 10 demoted; §7 flip text and §8 history bound
      by tests; §2/§4b/§5/PR-body precision corrections).
- [ ] **Verification round** (recompute every number from committed artifacts).
- [ ] **Ratify by merge** of the flip PR (under the standing campaign directive,
      only after the ceremony clears).
- [ ] **Flip** `gates.yaml` per §7 and append the §8 history entry.
- [ ] Then: candidate 2 against the amended contract (Q1/Q2/Q3 levers).

---

<!-- amendment-consistency-ledger: bound to the committed artifacts by
     tests/test_gate_w1_amendment1_proposal.py. Do not hand-edit a value without
     re-deriving it from runs/gate_w1_forensics1_v1.json,
     runs/gate_w1_floors_v1.json, runs/gate_w1_candidate1_v1.json,
     runs/m4_disability_v1.json, or gates.yaml. -->

```json amendment-consistency-ledger
{
  "amendment_id": "2026-07-12-w1-family-b-di-bands",
  "recommendation": "option_a_demote_all_family_b",
  "machine_reason": "concept_bridge_undefined_di_stock",
  "conversion_machine_reasons": [
    "concept_bridge_undefined_di_stock",
    "conversion_level_match_never_certified"
  ],
  "forensics": {
    "artifact": "runs/gate_w1_forensics1_v1.json",
    "registration": "4951218279",
    "grading": "4951430002",
    "reported_not_gated": true,
    "train_frame_side_only": true,
    "concept_delta_dominant_share_round3": 0.595,
    "concept_share_min_band": "40-44",
    "concept_share_min_round3": 0.023,
    "concept_share_max_band": "60-fra",
    "concept_share_max_round3": 0.891,
    "n_di_bands": 8,
    "all_di_bands_fail": true,
    "miss_ratio_min_round1": 2.9,
    "miss_ratio_max_round1": 21.9,
    "worst_band": "60-fra",
    "worst_band_duration_component_pp_round1": 21.3,
    "worst_band_m4_shape_component_pp_round1": 2.6,
    "insured_denominator_available": false,
    "n_concept_deltas": 7
  },
  "conversion_cells": {
    "referee_round_comment": "4951701300",
    "m4_artifact": "runs/m4_disability_v1.json",
    "n": 2,
    "ratio_psid_analog_to_admin_female_round3": 0.267,
    "ratio_psid_analog_to_admin_male_round3": 0.322,
    "never_a_level_match": true,
    "prevalence_60_66_female_pp": 5.79,
    "prevalence_60_66_male_pp_round2": 6.55,
    "anchor_female_pp": 14.1,
    "anchor_male_pp": 14.5,
    "tolerance_female_pp": 1.89,
    "tolerance_male_pp": 1.91,
    "required_window_floor_female_pp_round2": 12.21,
    "required_window_floor_male_pp_round2": 12.59,
    "candidate1_abs_dev_female_pp_round2": 8.3,
    "candidate1_abs_dev_male_pp_round2": 8.07,
    "candidate1_conversion_cells_fail": true
  },
  "family_a_oc": {
    "n_gated": 53,
    "p_seed": 0.922,
    "p_gate": 0.9481,
    "invariant_under_amendment": true
  },
  "family_b_partition": {
    "gated_now": 10,
    "conversion": 2,
    "di_bands": 8,
    "report_only_now": 15,
    "gated_after_option_a": 0,
    "report_only_after_option_a": 25
  },
  "overall_partition": {
    "gated_now": 65,
    "gated_after_option_a": 55,
    "report_only_now": 67,
    "report_only_after_option_a": 77
  },
  "oc_statement_after_option_a": "0.9481 * I(family_c)",
  "no_self_rescue": {
    "candidate1_run": "gate_w1_candidate1_v1",
    "candidate1_pr": 162,
    "gate_pass": false,
    "family_a_pass": false,
    "family_b_pass": false,
    "family_c_pass": false,
    "fails_family_a_and_c_independently_of_amendment": true,
    "all_ten_family_b_cells_fail_for_candidate1": true
  },
  "gates_yaml_untouched_by_this_proposal": true
}
```
