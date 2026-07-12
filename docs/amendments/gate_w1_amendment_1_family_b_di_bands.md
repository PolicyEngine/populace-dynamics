# gate_w1 amendment 1 (proposal) — the family-B DI prevalence bands

- **Amendment id**: `2026-07-12-w1-family-b-di-bands`
- **Gate**: `gate_w1` (W1 representative-frame transport, roadmap #113 M5, workstream #100)
- **Surface**: `thresholds.family_b` — the 8 DI age-composition prevalence bands
- **Ceremony stage**: PROPOSAL (draft). This document is the first step of the
  amendment ceremony (proposal → adversarial referee → fixes → verification →
  ratify-by-merge → flip). **It moves no threshold and edits no `gates.yaml`
  cell.** The prospective flip happens in a separate ratifying PR, only after
  the ceremony clears.
- **Amendment class**: STRUCTURAL, prospective, ZERO threshold movement (the
  gate-2 amendment-2 tranche-split pattern).
- **Evidence base**: W1 forensics 1, Q4 (`runs/gate_w1_forensics1_v1.json`,
  registration [4951218279][reg], grading [4951430002][grade]); candidate 1
  (`runs/gate_w1_candidate1_v1.json`, PR #162, registration 4950931131); the
  locked gate (`gates.yaml` blob `cd6411d9`, ratified PR #160 over floors #154).

[reg]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4951218279
[grade]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4951430002

## 1. Summary

Family B of `gate_w1` gates 10 SSA administrative margins: 2 disability-to-
retirement **conversion** cells (an M4-simulated flow over retired-worker
awards) and 8 **DI age-composition prevalence bands** (the age composition of
the SSA disabled-worker beneficiary stock, DI ASR Table 19). W1 forensics 1 Q4
established — and grading [4951430002][grade] adjudicated as a **BLOCKER** — that
the 8 DI bands are **unclearable by any contract-consistent candidate**. The
gate scores a quantity the deployed model *can* produce (the M4 work-disability
**point-prevalence** among person-years) against an SSA quantity it *never
modelled* (the duration-accumulated **insured-beneficiary stock**), across a
concept bridge the gate does not define and whose third leg (the Supplement 4.C2
insured denominator) the evidence base does not even archive. Every one of the 8
bands misses its tolerance by **2.9x–21.9x**; the concept delta is the dominant
share (**0.595**) of the level/steepness gap, the M4 hazard shape the minority.

Because family B is a conjunction, 8 unclearable cells make **gate_w1 as locked
unpassable**: the faithful-candidate operating characteristic is structurally
**0**, no matter how well a candidate transports the 53 family-A joints or the 2
family-C fingerprints. This proposal amends family B **prospectively** so the
gate is passable-in-principle again.

**Two options are analysed below; this proposal recommends option (a).**

- **Option (a) — demote (recommended)**: move the 8 DI bands from gated to
  report-only with machine reason `concept_bridge_undefined_di_stock`, leaving
  family B gating exactly the 2 conversion cells. Zero threshold movement; the
  DI anchors stay in-contract, published-not-certified, pending a future bridge
  amendment.
- **Option (b) — re-anchor**: replace the SSA-stock anchor with a published
  work-disability **prevalence** target M4 can transport. Deferred: no defensible
  transportable anchor is archived today, and the SSA-native re-anchor is blocked
  by the same unarchived Supplement 4.C2.

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
(under30) by 21.9x. No candidate lever narrows a 2.9x–21.9x gap when the gap is a
concept difference, not a fitting error.

**Decomposition.** At the worst band (60-FRA) the anchor-minus-deployed gap of
+23.87pp splits into a **duration (stock-vs-flow) component of +21.3pp** and an
**M4-hazard-shape component of +2.6pp**: even the *correct flow concept* (true
disabled-worker **awards** flow, +24.1pp) falls far short of the accumulated
**stock** (45.4pp). Across the 8 bands the concept delta is the dominant share
(`concept_delta_dominant_share` = **0.595**); the M4 hazard level is the minority.
The forensics enumerates **7** standing concept deltas between the PSID
self-report and a DI award (definition: self-report vs medical-vocational
adjudication; population: all adults vs insured workers; severity threshold;
recovery churn 25–50%/interval vs ≈1%/yr; conversion denominator; biennial
timing censoring; period pooling 1982–2023 vs a single-era column) — most of
which persist against *any* external DI anchor, not just the stock.

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
ceremony is REQUIRED before candidate 2."

## 3. Why gate_w1 as locked is unpassable

The family-B `pass_rule` is a **conjunction** over its gate-eligible cells (a run
passes family B iff every gated cell is within tolerance). 8 of the 10 gated
cells are unclearable by any contract-consistent candidate (§2). The overall
gate is a conjunction across families (family A 4-of-5 seeds ∧ family B ∧ family
C). Therefore, no matter the family-A or family-C outcome, the family-B factor is
**deterministically 0**, and:

> **faithful-candidate gate OC (locked surface) = p_gate(A) × 0 × I(C) = 0.**

The gate is not "hard"; it is **structurally impossible**. This is a defect of
the *contract*, not of any candidate — exactly the class the standing ceremony
exists to correct.

## 4. Options

### Option (a) — demote the 8 DI bands to report-only (RECOMMENDED)

Move the 8 `di_prevalence.*` cells from `family_b.gated_cells` to
`family_b.report_only`, each tagged `report_reason: concept_bridge_undefined_di_stock`.
Family B then gates exactly the 2 `claim_age.disability_conversion|{female,male}`
conversion margins. The 8 DI anchor values and their tolerances **stay in the
contract** (report-only): they are published for disclosure, not deleted, and not
gated. This is a naming/scope amendment with **zero threshold movement** — no
family-A cell, tolerance, σ, or family-C ordering changes.

**Why the 2 conversion cells legitimately stay gated.** The conversion margin is
the share of retired-worker awards that are DI-auto-conversions at FRA — a
**flow over awards**, a quantity the deployed M4 dynamics *can* represent. Q4
isolates the DI **bands** (the stock composition); it does not implicate the
conversion margin. Candidate 1 missed the conversion cells too (deployed 5.8/6.4
vs anchor 14.1/14.5), but that is a **candidate-quality** miss of an M4 flow the
candidate can improve — not a structural concept mismatch. Retaining them keeps
family B certifying the one DI margin M4 can honestly transport.

**Cost.** Family B no longer certifies the DI **age composition**. Certification
scope narrows accordingly (§7): the "deployed + simulated DI composition" claim
is reduced to the conversion share; the DI age bands become
published-not-certified.

**OC consequence.** Family A is untouched, so its stochastic characteristic is
**byte-identical** — recomputing the draw-noise-free half-normal OC from the 53
locked family-A tolerances and the frozen floor σ gives **p_seed 0.922 /
p_gate 0.9481**, and the 8 DI cells are provably **absent** from the family-A OC
machinery (empty intersection), so demotion cannot move it. The overall
faithful-candidate gate OC moves from a **structural 0** (unpassable) to

> **p_gate(A) 0.9481 × I(2 conversion cells) × I(family C),**

i.e. it returns to the well-defined family-A characteristic on the residual
**57-cell** gated surface (53 family-A + 2 conversion + 2 fingerprints). This
does **not** assert candidate 2 will pass — the two deterministic indicators are
candidate 2's empirical burden — it asserts the gate is no longer **structurally
zero**. A faithful candidate (already-fit generators genuinely regenerating the
frame) *can* clear a flow margin and the fingerprints; it could never clear the
stock bands.

### Option (b) — re-anchor the 8 bands to a transportable work-disability concept

Replace the SSA disabled-worker **stock** anchor (DI ASR Table 19) with a
published work-disability **prevalence** whose age composition M4's PSID
`EMPLOYMENT STATUS == 5` point-prevalence can transport onto. This attacks the
dominant (0.595) stock-vs-flow delta at its root rather than removing the cells.

**Candidate anchors, assessed honestly:**

| Candidate anchor | Concept vs M4 | Availability | Verdict |
|---|---|---|---|
| SSA DI **insured** prevalence (beneficiaries ÷ **4.C2 insured denominator**, by age) | The SSA-native fix; still a beneficiary count, but on the correct denominator | **Not archived** — Supplement 4.C2 is *still wanted* (`di_asr_2023/provenance.md`) | **Blocked** (same evidence-base gap as the bridge) |
| **ACS** disability (6-type / work-limitation), B18120 / S1810 by age | Self-reported functional/work limitation point-prevalence — closer, but a different instrument and threshold than the PSID self-report; the definition/severity/population deltas persist | Public (PUMS + published tables), annual, age-banded | Transportable **in principle**, but needs archiving + a fresh concept-delta accounting + new vintage tolerances |
| **SIPP** work-disability / functional-limitation module | Work-limitation prevalence, closer to M4's construct | Public-use; age composition needs extraction; pooling/vintage issues | Transportable in principle; not archived; larger lift |
| **CPS-ASEC** disability / work-limitation | Nearest instrument to the certified frame | **Circular** — the certified frame is CPS-derived and carries the disability/SS-proxy columns the `no_frame_di_column_rule` / `ss_proxy_laundering_rule` forbid | **Out** (circularity) |

**Why option (b) is deferred, not chosen now.** Every transportable candidate
requires, at minimum: (i) **archiving** a new published anchor into the evidence
base; (ii) a **concept-delta accounting** of the residual PSID-self-report vs
external-instrument deltas (the 7 named deltas do not vanish under re-anchoring —
they shrink); (iii) **re-deriving** the reference-period vintage tolerances from
the new source's vintages; and (iv) an **adversarial referee round** on the new
anchor's transportability. The SSA-native re-anchor is blocked outright by the
unarchived Supplement 4.C2. So option (b) is a larger, multi-step **redesign**
that itself presupposes a prior evidence-base amendment — it is not a prospective
re-scope and cannot be executed inside this ceremony.

**OC consequence.** Indeterminate. The family-B gated set stays 10, but the 8
re-anchored cells acquire new anchors and new tolerances, so the faithful-
candidate OC over those cells is unknown until the anchor is chosen, archived,
and its residual concept deltas are refereed. Stating an OC today would be
fabrication.

## 5. Recommendation

**Adopt option (a).** It is the minimal contract-clean move that (i) removes
exactly the structurally-unclearable cells; (ii) preserves every other gated
cell and every family-A/C threshold byte-identical (zero threshold movement);
(iii) keeps the DI anchors in-contract as honest report-only disclosure; (iv)
names the defect with a machine reason; and (v) restores a well-defined,
non-zero gate OC (p_gate 0.9481 on the residual 57-cell surface). It matches the
gate-2 amendment-2 precedent (a structural re-scope with no verdict to rescue and
nothing re-scored) and the grading directive, which lists demotion first.

Option (b) is the **right long-run fix** and is explicitly **deferred to a
future bridge amendment** that FIRST archives Supplement 4.C2 (and the incidence/
duration inputs) and defines the DI-entry + duration-to-conversion model — the
very machinery the Q4 determination enumerates (§8).

## 6. No-self-rescue compliance

The standing rule (`gate_1.amendment_rules`, inherited verbatim by
`gate_w1.governance.amendment_rules`): *no candidate's committed run verdict
changes under a rule proposed after that run; a gate rule amendment applies only
to runs registered after its ratification.* This proposal complies on every leg:

1. **c1's verdict STANDS regardless.** Candidate 1
   (`runs/gate_w1_candidate1_v1.json`, PR #162) fails **all three families
   independently**: family A 0/5 seeds, family B 0/10 cells, family C false. The
   demotion cannot flip it — even with family B reduced to the 2 conversion
   cells, candidate 1 **still** fails those retained cells (deployed 5.80/6.43 vs
   anchor 14.1/14.5, dev 8.30/8.07 ≫ tol 1.89/1.91), **and** family A, **and**
   family C. There is no committed verdict to rescue; the amendment rescues
   nothing.
2. **Prospective only.** The amendment binds only candidates registered **after**
   ratification. Candidate 1 remains FAIL as committed.
3. **The evidence is train/frame-side.** `runs/gate_w1_forensics1_v1.json` is
   `reported_not_gated: true`, `protocol.train_frame_side_only: true`,
   `publishes_regardless: true` — a frame-side diagnostic, not a scored candidate
   holdout verdict. The evidence motivating the amendment is not itself a gated
   result the amendment could retroactively rehabilitate.
4. **Description-claims-exactly.** After demotion, `covers` and
   `certification_scope` are narrowed to claim EXACTLY the amended 57-cell gated
   surface; the 8 DI bands are report-only with their reason named (per
   `description_claims_exactly_the_scored_surface`).

Unlike gate-2 amendment 1 (where retroactive application *would* have flipped one
run, disclosed and adjudicated in the record), here **no committed run flips
under either the locked or the amended surface** — the disclosure is that the
rescue set is empty.

## 7. The exact prospective flip (post-ratification; NOT in this PR)

The ratifying flip PR — a separate PR, after this ceremony clears — will make
exactly these edits to `gates.yaml` `gates.gate_w1`. Every edit is a re-scope or
a text narrowing; **no family-A/C threshold, tolerance, σ, anchor value, or
ordering changes**, and the family-B *anchor values and tolerances* for the 8
bands are retained under report-only (nothing is deleted).

1. **`thresholds.family_b.gated_cells`** — remove the 8 `di_prevalence.*` keys
   (`under30, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-fra`). Retain the 2
   `claim_age.disability_conversion|{female,male}` keys **byte-identical**.
2. **`thresholds.family_b.report_only`** — append the 8 `di_prevalence.*` cells,
   and add a parallel `report_reasons` mapping tagging each of the 8 with
   `concept_bridge_undefined_di_stock`. Retain the 8 anchor values + tolerances
   in a report-only `di_prevalence` disclosure block.
3. **`thresholds.family_b.gated_composition`** — "2 disability_conversion
   (M4-simulated margin) + 8 DI age-composition bands" → "2 disability_conversion
   (M4-simulated margin); the 8 DI age-composition bands demoted to report-only
   by amendment 1 (`concept_bridge_undefined_di_stock`)".
4. **`thresholds.family_b.candidate_protocol.pass_rule`** and
   `simulated_object.di_prevalence` — annotate that `di_prevalence` is now
   report-only; the family-B pass conjunction is over the 2 conversion cells
   only. The `di_prevalence` description remains for the report-only publication.
5. **`gate_w1.covers`** — "10 gated (2 … + 8 DI …) / 15 report-only" → "2 gated
   (2 disability-conversion M4-simulated margins) / 23 report-only (14 circular
   claim-age + 1 benefit level + 8 DI age-composition bands demoted by amendment
   1)"; and the roll-up "65 gated / 67 report-only" → "**57 gated / 75
   report-only**".
6. **`thresholds.certification_scope.certifies`** — "the 65 gated cells (53 … +
   10 SSA family-B … + 2 …)" → "the **57** gated cells (53 family-A + **2 SSA
   family-B disability-conversion margins** + 2 family-C)"; and "lands the
   deployed + simulated DI margins on the SSA anchors" → "lands the deployed +
   simulated DI-to-retirement **conversion** margins on the SSA anchors (the DI
   age-composition bands are report-only pending a concept-bridge amendment)".
7. **`thresholds.certification_scope.supports`** — the bullet "the deployed +
   simulated DI composition (conversion share + DI age bands) …" → "the deployed
   + simulated DI-to-retirement **conversion share** … (the DI age-composition
   bands are report-only pending the concept-bridge amendment; see amendment 1)".
8. **`gate_w1.history`** — append the amendment-1 entry (§8 draft).

`thresholds.family_a` (all views, tolerances, `power_cap`,
`faithful_candidate_oc`, `heavy_tail_boundary_bootstrap`, `family_a_prime`),
`thresholds.family_c`, `thresholds.protocol`, `governance.amendment_rules`,
`frame_pin`, and `holdout_basis` are **untouched** (`di_asr_2023` stays in
`holdout_basis`: the anchors are still published, report-only). A subset
master-compare must show the flip changes only `gate_w1` and moves no locked
sibling.

## 8. `amendment_history` entry draft (for `gate_w1.history`)

```yaml
- id: 2026-07-12-w1-family-b-di-bands
  proposed: '2026-07-12'
  referee_round:
    review: '<proposal PR> comment <id> (verdict TBD)'      # filled at ceremony
    fixes: '<sha>'                                          # filled at ceremony
    verification: '<PR> comment <id> (verdict TBD)'         # filled at ceremony
  ratified: '<date> by merge of PR <n> (merge commit <sha>) under the
    maintainer''s standing campaign directive of 2026-07-07, exercised only
    after the full ceremony'                                # filled at ratification
  flipped_live: this pull request
  content: >-
    STRUCTURAL amendment, ZERO THRESHOLD MOVEMENT. The 8 family-B DI
    age-composition prevalence bands (di_prevalence.under30 .. .60-fra) are
    DEMOTED from gated to report-only with machine reason
    concept_bridge_undefined_di_stock; family B now gates EXACTLY the 2
    disability-conversion M4-simulated margins, and the 8 DI bands publish their
    SSA-stock anchors report-only (values + tolerances retained, not deleted).
    Motivated by W1 forensics 1 Q4 (runs/gate_w1_forensics1_v1.json,
    registration 4951218279, grading 4951430002): the gate scored an M4
    WORK-DISABILITY point-prevalence against the SSA DISABLED-WORKER BENEFICIARY
    STOCK (DI ASR Table 19) across a concept bridge it never defined -- a
    duration-accumulated insured-beneficiary stock needs a DI-entry hazard, a
    duration-to-conversion survival model, and the Supplement 4.C2 insured
    denominator, none defined by the gate and the third not archived. The
    concept delta dominates (share 0.595; +21.3pp duration-accumulation vs
    +2.6pp M4 hazard shape at 60-FRA); all 8 bands fail by 2.9x-21.9x their
    tolerances (passes:false, gaps 2.8-23.9pp vs tolerances 0.31-2.14pp), so no
    contract-consistent candidate can clear them. PROSPECTIVE ONLY
    (no_self_rescue): no committed run's verdict changes; candidate 1
    (runs/gate_w1_candidate1_v1.json, PR #162) stands FAIL -- it fails family A
    (0/5 seeds), the 2 RETAINED conversion cells (dev 8.30/8.07 vs tol
    1.89/1.91), AND family C, so the demotion rescues nothing. Family A
    UNCHANGED (53 cells, faithful OC p_seed 0.922 / p_gate 0.9481,
    byte-identical -- the 8 DI cells are absent from the family-A OC machinery);
    family C UNCHANGED; ZERO family-A/C threshold movement. Certification scope
    narrows: family B certifies the DI-to-retirement conversion share only; the
    DI age-composition bands are report-only, published-not-certified, pending a
    future concept-bridge amendment that FIRST archives Supplement 4.C2. The
    gate_w1 faithful-candidate OC returns from a structural 0 (8/10 family-B
    cells unclearable -> unpassable) to the family-A characteristic p_gate
    0.9481 on the residual 57-cell gated surface (65 -> 57 gated, 67 -> 75
    report-only). Evidence chain: candidate 1 (PR #162, run
    gate_w1_candidate1_v1), W1 forensics 1 (PR #163, runs/gate_w1_forensics1_v1.json).
```

## 9. Certification-scope language after amendment 1

What family B **claims** after the flip:

- **Certifies (gated)**: the deployed + simulated **DI-to-retirement conversion
  share** (2 M4-simulated conversion margins) lands on the SSA 6.B5.1 anchors
  within their reference-period vintage tolerances.
- **Publishes (report-only, NOT certified)**: the DI age-composition prevalence
  bands (8 cells, DI ASR Table 19 anchors retained), tagged
  `concept_bridge_undefined_di_stock`; the 14 circular claim-age cells; the
  benefit level. These are disclosed for transparency; a PASS does **not** claim
  them.
- **Does not support**: the deployed DI **age composition** as a certified
  margin. Any payable-benefit baseline that rides on the DI age distribution must
  await the future concept-bridge amendment. (The dynamics — earnings, marital,
  household, marriage × earnings, disability — remain gate-1/2a/2b/2c/M4-certified
  on their PSID holdouts; W1 certifies transport, not re-estimation.)

## 10. Evidence-base gap (Q4-adjacent): Supplement 4.C2

`data/external/di_asr_2023/provenance.md` records, under *still wanted*:
Trustees V.C5/V.C6 (incidence/termination assumptions), **Supplement 4.C2 (the
insured denominator)**, and the age-sex-adjusted incidence-rate series. The
insured denominator is the **third leg** of the concept bridge (§2): without it
the DI-beneficiary stock cannot be normalised to an insured-population rate M4
could target. **Archiving Supplement 4.C2 (plus the incidence/duration inputs) is
a prerequisite for any future DI-stock bridge amendment** (option (b) or the
"define the bridge" disposition). This proposal records the gap so the future
amendment's evidence dependency is explicit; it does not itself archive 4.C2.

## 11. Ceremony checklist

- [x] **Proposal** (this document + this draft PR). No `gates.yaml` edit; no
      threshold moved.
- [ ] **Adversarial referee round** (OC-argument standard: recompute the
      amended-surface OC; verify zero threshold movement; verify no-self-rescue).
- [ ] **Fixes**.
- [ ] **Verification round** (recompute every number from committed artifacts).
- [ ] **Ratify by merge** of the flip PR (under the standing campaign directive,
      only after the ceremony clears).
- [ ] **Flip** `gates.yaml` per §7 and append the §8 history entry.
- [ ] Then: candidate 2 against the amended contract (Q1/Q2/Q3 levers).

---

<!-- amendment-consistency-ledger: bound to the committed artifacts by
     tests/test_gate_w1_amendment1_proposal.py. Do not hand-edit a value without
     re-deriving it from runs/gate_w1_forensics1_v1.json,
     runs/gate_w1_floors_v1.json, runs/gate_w1_candidate1_v1.json, or gates.yaml. -->

```json amendment-consistency-ledger
{
  "amendment_id": "2026-07-12-w1-family-b-di-bands",
  "recommendation": "option_a_demote",
  "machine_reason": "concept_bridge_undefined_di_stock",
  "forensics": {
    "artifact": "runs/gate_w1_forensics1_v1.json",
    "registration": "4951218279",
    "grading": "4951430002",
    "reported_not_gated": true,
    "train_frame_side_only": true,
    "concept_delta_dominant_share_round3": 0.595,
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
    "gated_after_option_a": 2,
    "report_only_after_option_a": 23
  },
  "overall_partition": {
    "gated_now": 65,
    "gated_after_option_a": 57,
    "report_only_now": 67,
    "report_only_after_option_a": 75
  },
  "no_self_rescue": {
    "candidate1_run": "gate_w1_candidate1_v1",
    "candidate1_pr": 162,
    "gate_pass": false,
    "family_a_pass": false,
    "family_b_pass": false,
    "family_c_pass": false,
    "retained_conversion_cells_fail_for_candidate1": true
  },
  "gates_yaml_untouched_by_this_proposal": true
}
```
