# gate_w1 amendment 3 (proposal) — the hh_size residual and the pair-scoped C2

- **Amendment id**: `2026-07-13-w1-hh-size-and-c2-pair`
- **Gate**: `gate_w1` (W1 representative-frame transport, roadmap #113 M5, workstream #100)
- **Surface**: two changes on the amended (post-amendment-2) **48-cell** gated
  surface — (a) the four family-A **household-size** cells
  (`hh_size_share.{1,3,4,5plus}`); (b) a **re-specification** of the family-C
  gated binary **C2** (`fingerprint.elimination_plus2pp`) from the full 4-element
  compression ordering to the **pair-scoped elim↔+2pp adjacent swap**. **4 cells
  demoted** (family A) **+ 1 fingerprint re-scoped** (family C). `hh_size_share.2`
  stays gated (it clears, §3); the C2 fingerprint stays gated (pair-scoped, §4b).
- **Ceremony stage**: PROPOSAL (draft). This document is the first step of the
  amendment ceremony (proposal → adversarial referee → fixes → verification →
  ratify-by-merge → flip). **It moves no threshold and edits no `gates.yaml`
  cell.** The prospective flip happens in a separate ratifying PR, only after the
  ceremony clears. The two elements were **pre-named as amendment-3 risks in the
  ratified amendment-2 series forecast** (§6), which is why the forensics-then-
  ceremony standard produced W1 forensics 3 before this proposal.
- **Amendment class**: STRUCTURAL demotion **+** RE-SPECIFICATION, prospective,
  ZERO threshold movement. Like amendment 2 — which recomputed the family-A OC on
  the residual surface — this proposal removes **4 cells from family A**, so the
  family-A faithful-candidate OC is **recomputed** on the residual 43-cell surface
  (§5). Unlike amendment 2, the family-C change is **not a demotion**: C2 stays
  gated, but its check is **re-scoped** to the anchor-supported pair the
  representative frame can coherently reverse. No tolerance, σ, anchor value, or
  ordering is edited; the demoted cells' tolerances and the C2 orderings are
  retained under report-only.
- **The whittling burden, up front (§3).** The post-amendment surface (43 family-A
  cells + the pair-scoped C2) **would be PASSABLE by the already-committed
  candidate-3 model** — c3 lands 43/43 of the surviving family-A cells in-band on
  **all five seeds**, and the elim↔+2pp swap realised. This proposal states that
  plainly, first, and carries the burden it creates (§3): every demoted or
  re-scoped element is **forensics-proven unreachable by any contract-permitted
  lever** (Q10 construction enumerates empty; Q11 levers exhausted — three
  candidates deep, all named levers deployed), and **every one was pre-named in the
  ratified amendment-2 series forecast BEFORE c3 ran** — the demotion boundary is
  the permitted-lever line applied **prospectively**, not a post-hoc fit to c3.
  Candidate 3 remains a committed FAIL (§6).
- **Evidence base**: W1 forensics 3 (`runs/gate_w1_forensics3_v1.json`,
  registration [4959668253][reg3], grading [4960691167][grade3], merged 481d452 in
  PR #179) — Q10 (the cap_150k adjacency decomposition + the entailment) and Q11
  (the hh_size residual quantification + the lever-exhaustion probe); candidate 3
  (`runs/gate_w1_candidate3_v1.json`, PR #176, registration [4959017270][reg-c3],
  grading [4959658059][grade-c3], gate **FAIL 0/5**) — the committed cube whose
  per-cell records this proposal binds; W1 forensics 2
  (`runs/gate_w1_forensics2_v1.json`, registration [4953088871][reg2]) — Q7 (the
  hh_size joint-feasibility REFUTED) and the coupling caveat; W1 forensics 1
  (`runs/gate_w1_forensics1_v1.json`) — the C2 corrected-tail τ −1/3; candidates 1
  and 2 (`runs/gate_w1_candidate{1,2}_v1.json`, PRs #162 / #167) — the committed
  per-seed records that make the hh_size demotion candidate-independent; the
  amended gate (`gates.yaml` `gate_w1`, post-amendment-2: 47 family-A + C2 gated);
  and the ratified amendment-2 doc
  (`docs/amendments/gate_w1_amendment_2_family_a_concept_cells.md`, PR #169 / flip
  #174, ratified c50f8bb), whose **series forecast pre-named exactly these two
  surfaces** as amendment-3 risks before c3 ran.

[reg3]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4959668253
[grade3]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4960691167
[reg-c3]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4959017270
[grade-c3]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4959658059
[reg2]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4953088871
[a2-flip]: https://github.com/PolicyEngine/populace-dynamics/pull/174

## 1. Summary

Amendment 2 recomputed the gate's headline faithful-candidate OC to **0.9623 ×
I(C2)** on a **48-cell** surface (47 family-A joints + the C2 compression
fingerprint), and — against self-interest — **retained** the five `hh_size` cells
and the full C2 ordering as gated, naming their two open questions as **amendment-3
risks** to be settled by a registered forensics round, never a bare demotion. W1
forensics 3 (grading [4960691167][grade3]) then settled both, each on grounds that
are **candidate-independent** and were **pre-named before candidate 3 ran**:

- **(a) The four `hh_size` cells** (`hh_size_share.{1,3,4,5plus}`). Q11 quantifies
  the post-c3 residual from the committed cube. The size-1↔size-3+ **mirror** is a
  clean coresidence-composition signature (coresidence moves size-1 −0.047 and
  size-3+ +0.046, ratio **0.98**), and coresidence owns size-3 (**0.75**); the
  **fertility** window owns size-4 (**0.67**) and size-5plus (**0.78**), and size-1
  is **mixed** (fertility **0.54** / coresidence **0.46**). Both contract-permitted
  entry-state levers are **exhausted**: the roster-ceiling probe pushed to and
  beyond the permitted seed maximum reaches terminal coresidence only **0.111** —
  capped by the **certified** parental-exit hazard, not the seed — and does **not**
  close the size-1 excess (0.179 vs the frame's 0.083); the fertility window sits at
  the certified support floor. Three candidates deep, all named levers deployed, no
  untried permitted entry-state lever exists. Closing the residual needs a
  **non-entry-state coresidence/household model** — a re-estimation W1 does not
  certify.
- **(b) The C2 4-element ordering.** Q10 decomposes the 16.7-year-vs-1-year
  `cap_150k` gap: the deployed representative frame's **above-cap payroll share owns
  0.9925** of it (config + vintage own 0.0075). The **entailment holds**: the same
  above-wage-base compression correction that certifies the elim↔+2pp swap
  **mechanically lifts `cap_150k`** to rank 2 (deployed A/B **1.694** PV). The
  ledger arithmetic is exact — elim>+2pp ⇔ A/B > **0.1613**, but the Smith 4-element
  order survives only in the narrow window A/B ∈ **(0.1613, 0.184)**, which any
  swap-realising frame overshoots (deployed ≈ **10×** the window). So the 4-element
  ordering is **internally inconsistent as a representative-frame transport
  target**, and no contract-permitted lever restores the adjacency (construction
  **enumerates empty**). The **pair-scoped C2** (elim↔+2pp only) is
  **anchor-supported** (Smith 2015 p.3 orders elimination +21yr > +2pp +18yr),
  **realised 3/3** across c1/c2/c3, and certifies exactly the frame-compression
  mechanism the fingerprint exists for.

**This proposal recommends (§4, §5):** demote the four `hh_size` cells to
report-only under one uniform machine reason
`no_permitted_entry_state_lever_reaches_cell` (with a per-cell channel-attribution
record and per-cell pass records from all three cubes), and **re-scope** C2 to the
pair-scoped elim↔+2pp swap (the `cap_150k` and +1pp legs publish report-only,
tagged `anchor_ordering_internally_inconsistent_under_certified_swap`). Zero
threshold movement; the demoted tolerances and the C2 orderings stay in-contract
under report-only. The gate's faithful-candidate OC moves from **0.9623 × I(C2)** to
**p_seed 0.9403 / p_gate 0.9684 × I(pair-swap)** on the residual **44-cell** surface
(43 family-A joints + the pair-scoped C2).

**What is new about this amendment — stated first, not buried (§3).** Amendment 2
left the amended surface "not known passable". After amendment 3, the amended
surface would be **known passable by the already-committed candidate-3 model** (c3
lands 43/43 in-band 5/5 on the surviving family-A cells; the elim↔+2pp swap
realised) — **a first for this gate**. That is exactly why §3 addresses the
whittling question head-on, in its strongest form: the demotion set is bounded by
the **permitted-lever line** the committed forensics establishes, it was **pre-named
in the amendment-2 forecast before c3 ran**, and it changes **no** committed verdict
(§6). A candidate-4 registration of the **unchanged** c3 model would follow the flip,
subject to the standing pass-verification addendum (independent bit-exact
reproduction before the record admits a pass).

## 2. The finding (W1 forensics 3 Q10/Q11)

`runs/gate_w1_forensics3_v1.json` is a one-shot, `reported_not_gated`,
`train_frame_side_only` diagnostic that "publishes regardless of any verdict". It
re-runs the committed #117 ledger and `regenerate_person_frame_v3` on the pinned
frame and reproduces the committed cube's C2 exhaustion deltas and `hh_size` cells
**bit-for-bit** (`q10/q11 instrumentation_bit_identity_max_dev == 0.0`) before any
counterfactual is measured.

### 2a. Q10 — the `cap_150k` adjacency the certified swap busts

The C2 fingerprint scores the ordinal compression ordering of four solvency
provisions by 75-year exhaustion-delay. Under the Smith (2015) anchor the required
representative order is `[elimination, payroll_plus_2pp, payroll_plus_1pp,
cap_150k]` (year-deltas **21 / 18 / 5 / 1**); the deployed representative frame lands
`[elimination, cap_150k, payroll_plus_2pp, payroll_plus_1pp]` (τ **0.333**), because
`cap_150k`'s exhaustion delay is **16.7 years**, not Smith's rank-4 **1 year**.

Q10 decomposes that 15.7-year gap. The dominant component is the **frame's above-cap
payroll share**: `share_a_frame_above_cap` = **0.9925** (config + indexation +
vintage jointly own **0.0075**). The mechanism is an **entailment**, not a coincidence.
`cap_150k` taxes the near-band `[wage_base, $150k-indexed]`; the representative frame
**de-compresses the tail** (the PV-career-weighted revenue ratio A/B rises from the
anchor's implicit **0.115** to the deployed **1.694**), and because that near-band is
~40–50% of the above-base payroll, the **same correction that lifts elimination past
+2pp** (the certified swap) mechanically lifts `cap_150k` to **rank 2**.

The ledger arithmetic makes the internal inconsistency exact. `elimination` outranks
`+2pp` iff A/B > **0.1613** (`c2_breakeven_A_over_B`) — this is the swap the
fingerprint exists to certify. But the full Smith 4-element order (which additionally
requires `cap_150k` to stay rank-4, below +1pp) survives **only** in the narrow
window A/B ∈ **(0.1613, 0.184)** (width **0.0228**). Any frame compressed enough to
realise the swap (A/B just above 0.1613) and still hold the full Smith order must sit
inside that width-0.023 window; the deployed representative frame's A/B is **1.694** —
roughly an **order of magnitude (≈10×)** above the window's upper edge. **A frame
cannot both realise the certified swap and reproduce Smith's `cap_150k` rank**, so the
full 4-element ordering is **internally inconsistent as a representative-frame
transport target**: the compression correction the fingerprint certifies is exactly
what busts the cap rank. The verdict is `entailment_holds: true`,
`deployed_breaks_adjacency: true`, `deployed_cap_rank: 2`. (On a conservative SSA-anchored true frame the break is
f-dependent near a boundary ~0.37; the **certified** frame — the actual transport
target — is unambiguous.)

No contract-permitted lever restores the Smith adjacency. The construction attempt
enumerates four levers and each is impermissible or ineffective — the pinned frame
(A/B not a free parameter), the pinned permanent-variance share ρ = 0.467 (scales the
whole tail, cannot carve the near-band hole), the committed #117 `cap_150k` encoding
(raising the threshold collapses the provision to +0, a different provision), and
back-solving the frame tail (the prohibited identity-in-disguise). Because the
near-band is the **densest** part of a monotone-thinning tail, f = A_band/A stays
large (PSID 0.438, deployed 0.220) and no unimodal earnings distribution makes it
small enough while landing A/B in the window: `restoration_exists: false`.

### 2b. Q11 — the `hh_size` residual: mixed-but-structured, levers exhausted

`hh_size = 1 + spouse + child_counts + n_parents_ego + nonfamily`. The two
contract-permitted entry-state levers are Q7's: the coresident-parent **roster**
(seeds `n_parents_ego`) and the **15-entry fertility window** (drives `child_counts`).
Q11 decomposes each failing cell's post-c3 residual across these channels
(`per_cell_attribution`, verified to partition to 1.0 within `1.1e-16`):

| Cell | Dominant channel | Channel share (of moved mass) | Mirror |
|---|---|---:|---|
| `hh_size_share.1` | fertility (**mixed**) | fertility **0.54** / coresidence **0.46** | — |
| `hh_size_share.3` | coresidence | coresidence **0.75** | owns the size-1↔size-3+ mirror |
| `hh_size_share.4` | fertility | fertility **0.67** | — |
| `hh_size_share.5plus` | fertility | fertility **0.78** | — |

The size-1↔size-3+ **mirror** is a clean coresidence-composition signature: the
coresidence channel moves size-1 by **−0.047** and size-3+ by **+0.046** —
`coresidence_mirror_ratio` **0.98** — a young adult leaving a lone household to join a
size-3+ parental home. But coresidence does **not** own >50% of every failing cell:
the fertility window (also permitted, maxed at the 15-entry certified support) loads
the size-4/5plus deficits and is the larger young-adult sub-channel for size-1. So
the registered "coresidence >50% of each cell" expectation (p≈0.7) is **refined** to
**mixed-but-structured** attribution (the p≈0.2 branch, but two channels owning two
distinct regions — not featureless) — the honest reading the lane committed.

**Both permitted entry-state levers are exhausted.** The decisive probe is the
`untried_full_age_roster_probe`: the wave-0 coresident-parent seed pushed to the
**ceiling** (every adult, above the permitted train stock — the machinery reads
wave-0 only, so this is the maximal reachable coresidence) reaches terminal
coresidence only **0.111** and lands size-1 at **0.179**, still far above the frame's
**0.083**: `closes_size1: false`. The **certified parental-exit hazard**, not the
seed, caps terminal coresidence — so the roster lever is exhausted at its ceiling, and
the fertility window is already at the certified support floor. `verdict:
levers_exhausted`. Closing the residual "needs a non-entry-state coresidence/household
model (an amendment), not another entry-state lever" — the same shape as amendment 2's
65+ finding one level up: the miss is a **contract-structural** insufficiency, not a
candidate-quality shortfall.

## 3. The whittling question, in its strongest form

This is the third accommodation-direction amendment to `gate_w1`, and it is the one
that most demands scrutiny, so state the sharpest version first, plainly, before any
defense:

> **The post-amendment surface would be passable by the already-committed candidate-3
> model.** On the residual 43-cell family-A surface (the current 47 minus the four
> `hh_size` cells), candidate 3 lands **43/43 cells in-band on all five seeds**; and
> the C2 re-scope gates exactly the elim↔+2pp swap that c3 **realised**. The four
> cells this amendment demotes are **precisely the four** family-A cells c3 fails, and
> the C2 legs it re-scopes to report-only are **precisely** the legs of the ordering
> c3 missed. An amendment that demotes exactly what the latest candidate fails is the
> textbook shape of whittling a gate to passability.

The proposal does not hide behind that framing; it carries the burden. Three facts —
each machine-checked (§ ledger) — distinguish this from whittling:

**1. The demotion boundary is the permitted-lever line, and the committed forensics
proves every demoted/re-scoped element sits beyond it.** The boundary is the same one
amendments 1 and 2 drew: *demote iff the committed forensics establishes that no
contract-permitted lever reaches the cell at the gate ≥4-of-5 level; retain every
cell with a live permitted lever, however badly the built candidates miss it.* Q11
establishes the `hh_size` quad is **lever-exhausted** — the roster ceiling (0.111,
hazard-capped) does not close size-1, the fertility window is at its support floor,
and the residual needs a non-entry-state household model — quantified, three
candidates deep, all named levers deployed. Q10 establishes the `cap_150k` rank is
**frame-entailed** (the certified swap mechanically lifts it) and its restoration
**enumerates empty**. These are not "cells c3 happened to miss"; they are cells
**proven unreachable by any contract-permitted transport**. A whittler would demote by
*outcome* (what c3 failed); this demotes by *reachability* (what no permitted lever
attains) — and the two coincide here only because c3 deployed every named lever, which
is what makes the miss candidate-independent.

**2. Both surfaces were pre-named in the ratified amendment-2 forecast, before c3
ran.** The amendment-2 series forecast (ratified c50f8bb, flip merged
[d250d4b][a2-flip] in PR #174) recorded its `amendment3_risks` as **exactly**
`{hh_size_share.1, hh_size_share.3, hh_size_share.4, hh_size_share.5plus,
c2_cap_150k_adjacency}` — the identical set this proposal acts on. Candidate 3
registered [4959017270][reg-c3] **after** the amendment-2 flip merged (the c3
registration says so explicitly). The demotion boundary was therefore drawn
**prospectively** — named in a ratified record before the candidate that lands inside
it existed — so it **cannot** be a post-hoc fit to c3. c3's grading confirms the
forecast landed: its failure union across all seeds is "exactly `{1, 3, 4, 5plus}`;
size-2 clears everywhere; size-4 clears seed 1", and its C2 "full reversal failed
because `cap_150k` landed at position 2 … the ratified amendment-2 record pre-named
exactly this."

**3. It rescues no committed verdict (§6).** The no-self-rescue rule keeps c1, c2, and
c3 as committed FAILs; the amendment binds only candidates registered after
ratification. c3 was scored under the current contract (47 family-A + the 4-element
C2) and **stands FAIL 0/5**. That the *same model*, re-registered as a candidate-4
after the flip, would pass the amended surface is not a retroactive rescue — it is the
ordinary prospective consequence of a corrected contract, and it is gated by the
standing addendum requiring **independent bit-exact reproduction** before any pass
enters the record.

**What a subsequent pass would, and would not, certify (previewed here; full language
§9).** A pass on the amended 44-cell surface **would** certify that stochastic
regeneration through the deployed generators reproduces the **43 CPS-observable
family-A joints** within the frame's own sampling floor **and reverses the elim↔+2pp
compression swap** — the frame-robust, transportable content the fingerprint exists
to test. It **would not** certify the four `hh_size` shares (a coresidence-composition
residual that needs a non-entry-state household model), the `cap_150k` rank
(frame-entailed to rise) or the +1pp↔cap adjacency, and — carried from amendments 1–2
— the C1 reversal, the 18-24 participation, the 65+ composition, or any family-B SSA
margin. The pass is **not hollow**: it certifies exactly the transportable joints and
the one compression adjacency the anchor supports on a representative frame, and it
disclaims precisely the quantities that lie outside the transport contract. That is
the distinction between a whittled gate (one that drops hard-but-reachable cells to
manufacture a pass) and a **correctly scoped** one (one that gates exactly what the
transport can certify and publishes the rest).

## 4. Options — the two changes

### 4a. The four `hh_size` cells (family A)

**Option a1 — demote to report-only (RECOMMENDED).** Demote
`hh_size_share.{1,3,4,5plus}` to report-only under one uniform machine reason
`no_permitted_entry_state_lever_reaches_cell`, carrying (i) the per-cell
channel-attribution record (§2b: size-1 mixed fertility 0.54 / coresidence 0.46;
size-3 coresidence 0.75 with the 0.98 mirror; size-4 fertility 0.67; size-5plus
fertility 0.78) and (ii) the per-cell pass records from **all three** candidate cubes.
The ground is per-cell true against the committed record, and it is **not** a blanket
impossibility claim (the amendment-2 blocker lesson):

| Cell | Pass record (c1, c2, c3) | Ground |
|---|---|---|
| `hh_size_share.1` | 0/5, 0/5, **0/5** | roster ceiling (0.111, hazard-capped) does not close the 0.107 size-1 excess; fertility at support floor |
| `hh_size_share.3` | 0/5, 0/5, **0/5** | coresidence-composition mirror residual (coresidence 0.75); needs non-entry-state model |
| `hh_size_share.4` | 0/5, 0/5, **1/5** | fertility-window deficit (0.67); **clears on c3 seed 1** — not impossible, but unreachable at ≥4/5 |
| `hh_size_share.5plus` | 0/5, 0/5, **0/5** | fertility-window deficit (0.78) at the certified support floor |

Two candor points are stated plainly, **not** blanketed:

- **`hh_size_share.2` is NOT demoted — it clears.** Candidate 3 passes size-2 on
  **5 of 5** seeds (Q7's own arithmetic: the joint roster+fertility lever clears
  size-2). It stays gated. This proposal makes **no blanket
  household-size-unreachability claim**; four of five cells are unreachable at ≥4/5,
  one (size-2) is cleared.
- **`hh_size_share.4` is not impossible — it clears on one c3 seed.** Candidate 3
  passes size-4 on **1 of 5** seeds (seed 1). It is demoted because it is unreachable
  at the gate **≥4-of-5** level under every permitted entry-state lever — three
  candidates deep — not because no candidate ever clears it. The single pass is stated
  here so no blanket-impossibility claim can stand in for the per-cell truth.

The four tolerances (`0.191 / 0.191 / 0.174 / 0.184`) are retained verbatim under the
report-only disclosure — **zero threshold movement**.

**Option a2 — retain and demand a non-entry-state household model.** Honest but
deferred: closing the residual needs a coresidence/household composition model beyond
the wave-0 roster + fertility window — a re-estimation W1 does not certify, requiring
an evidence-base amendment and a fresh referee round (§10). Retaining the cells gated
in the interim keeps four cells no contract-permitted candidate can reach at ≥4/5 on
the certified surface, re-pricing the OC to a value the achievable model cannot hit —
the headline-vs-achievable defect amendments 1–2 exist to cure.

**Recommendation (4a): a1.** The residual is lever-exhausted and quantified; the four
cells are report-only until the household-composition bridge exists.

### 4b. Family C — re-scope the gated binary to the pair-scoped C2

This is a **re-specification, not a demotion**: C2 stays gated; its check is
re-scoped. Three options, carried with the amendment-2 option-analysis standard.

**Option c-a — pair-scope C2 to the elim↔+2pp adjacent swap (RECOMMENDED).** Re-scope
the C2 check from "the full 4-element order reverses" to "the elim↔+2pp adjacency
reverses (elimination outranks +2pp)". It is **anchor-supported**: Smith 2015 p.3
orders full elimination **+21yr** > payroll +2pp **+18yr**, a single anchor-reported
adjacency that does **not** depend on the compressed 4-element tail. It is **realised
3/3** — the swap reversed in candidate 1, candidate 2, **and** candidate 3
(`required_swap_realised: true` in all three cubes). It certifies exactly the
above-wage-base **frame-compression mechanism** the #113 fingerprint exists to test —
the transportable content — and **drops** the two provisions (`cap_150k`, +1pp) whose
ranks the representative tail mechanically moves (Q10). The `cap_150k` and +1pp legs
publish **report-only**, retaining the deployed ordering and the 16.7-year figure,
tagged `anchor_ordering_internally_inconsistent_under_certified_swap`.

**Option c-b — demote family C entirely.** Enumerated for completeness; **weaker**. It
would move the C2 fingerprint to report-only alongside C1, leaving family C gating
nothing. This **discards a thrice-realised, anchor-supported mechanism**: the
elim↔+2pp swap is the single most-reproduced element on the whole gated surface (3/3
candidates, deterministic), and it is exactly the compression correction #113 was
built to certify. Demoting it would under-claim the transport — throwing away a real,
certifiable result to avoid re-scoping. Rejected.

**Option c-c — keep the full 4-element ordering.** **Untenable.** Q10 proves the
4-element order is **internally inconsistent as a representative-frame transport
target**: the ledger arithmetic gives elim>+2pp ⇔ A/B > 0.1613 while the full Smith
order survives only in A/B ∈ (0.1613, 0.184), a window any swap-realising frame
overshoots (deployed A/B 1.694, ≈10× the window); the above-cap share owns **0.9925**
of the gap; contract-permitted lever restoration **enumerates empty**; and **no
published anchor supports the full 4-element ordering on a representative frame**
(Smith's full order is a 2015-vintage DYNASIM projection on a different frame). Keeping
it would gate a target the certified frame **cannot** satisfy while realising the swap
— the same impossibility-vs-headline defect the ceremony exists to cure, and (unlike
C1, whose re-spec enumerated empty) here a **non-empty, anchor-consistent** re-spec
exists, so there is no reason to keep the incoherent target.

**Recommendation (4b): c-a.** Pair-scope C2 to the anchor-supported, thrice-realised
elim↔+2pp swap; publish the `cap_150k`/+1pp legs report-only.

## 5. Recommendation

**Adopt a1 + c-a:** demote the four `hh_size` cells to report-only under
`no_permitted_entry_state_lever_reaches_cell` (with the per-cell channel attribution +
tri-cube pass records), and re-scope C2 to the pair-scoped elim↔+2pp swap (the
`cap_150k`/+1pp legs report-only under
`anchor_ordering_internally_inconsistent_under_certified_swap`). It (i) removes
**exactly** the four family-A cells the committed forensics establishes no
contract-permitted entry-state lever reaches at ≥4/5 (Q11 lever exhaustion, three
candidates deep) — no more (size-2 clears and stays gated), no fewer; (ii) re-scopes
C2 to the one adjacency the representative frame **coherently** reverses (Q10
entailment) while retaining the full ordering as report-only; (iii) preserves every
other gated cell and every tolerance/σ/anchor/ordering byte-identical (zero threshold
movement); (iv) names each defect with a per-cell-true machine reason; and (v)
recomputes a well-defined gate OC.

**OC consequence (recomputed).** Six of amendment 2's seven demoted cells were
family-A cells and moved the family-A OC; here all four demoted cells are family-A
cells, so the family-A characteristic is recomputed on the residual **43-cell**
surface from the **frozen floor σ** (`runs/gate_w1_floors_v1.json`; the
draw-noise-free half-normal basis, the same machinery amendment 2 used):

- **Before** (the committed post-amendment-2 state): 47 gated family-A cells → p_seed
  **0.9344** / p_gate **0.9623**.
- **After**: 43 gated family-A cells → p_seed **0.9403** / p_gate **0.9684**.

Each demoted `hh_size` cell has a faithful pass-prob **0.998–0.9989** (sampling-σ
model), so removing the four **raises** p_seed slightly — the same
sampling-noise-vs-systematic-bias defect as amendment 2: the contract's headline OC
prices sampling noise, while the actual `hh_size` misses are systematic
coresidence-composition biases the faithful model never sees. The overall gate OC
moves from **0.9623 × I(C2)** to **p_gate(A) 0.9684 × I(pair-swap)** on the residual
**44-cell** surface (43 family-A joints + the pair-scoped C2). The frozen floor keeps
its own 53-cell `faithful_candidate_oc` (0.922 / 0.9481); the amendment re-scopes the
**contract**, not the floor.

**The 44-cell surface — for the first time on this gate — is known passable by a
committed model (§3).** Candidate 3 lands 43/43 of the surviving family-A cells in-band
on all five seeds and realised the elim↔+2pp swap; the amended surface is not a
proven-unreachable-removal that leaves passability open (amendment 2's posture) but a
surface a committed model clears. That strengthens rather than weakens the whittling
scrutiny, which §3 carries in full.

**Series forecast.** The gate trajectory is lock **65 gated** → amendment 1 **55** →
amendment 2 **48** → amendment 3 **44**. After amendment 3, **no family-A cell remains
in pre-named jeopardy**: the amendment-2 forecast's five `amendment3_risks` are all
resolved (the four `hh_size` cells demoted here; `c2_cap_150k_adjacency` re-scoped out
of the gated check), and the surviving 43 family-A cells are c3-passable 5/5. The one
gated family-C element — the elim↔+2pp swap — is the **most-realised** quantity on the
surface (3/3 candidates, deterministic, frame-driven), so it carries no seed-robustness
jeopardy: it is not a coin-flip but a reproduced mechanism. **Honestly, nothing on the
amended gated surface remains in pre-named jeopardy.** What remains is not a gate risk
but a **verification** step and **report-only bridge** work: a candidate-4 registration
of the unchanged c3 model must clear the standing **independent bit-exact
reproduction** addendum before the record admits a pass (the gate between
"c3-passable" and "certified"); and the report-only families — the `hh_size`
household-composition model, the `cap_150k` rank, C1, the 18-24 pair, the 65+ quad, and
family B — stay amendment territory, reopenable only via the same
forensics-then-ceremony standard, never a bare re-gate (§10).

The retain-and-demand-a-model / demote-entirely alternatives (a2 / c-b) are the
deferred paths (§10).

## 6. No-self-rescue compliance

The standing rule (`gate_w1.governance.amendment_rules`, inherited from `gate_1`): *no
candidate's committed run verdict changes under a rule proposed after that run.* The
**rescue set is empty** — the amendment rescues **no** committed candidate, shown by
recomputing each verdict:

1. **Candidate 1 (PR #162) STANDS FAIL.** Committed gate FAIL; it fails every one of
   the four `hh_size` cells **0/5** and the C2 4-element reversal does not occur (and
   it fails many other family-A cells demoted by amendments 1–2). Removing the four
   `hh_size` cells cannot lift its family-A pass to ≥4/5.
2. **Candidate 2 (PR #167) STANDS FAIL.** Same structure: `hh_size` 0/5 on all four,
   C2 unreversed.
3. **Candidate 3 (PR #176) STANDS FAIL — and its FAIL was under the current contract.**
   c3 is a committed gate FAIL (0/5 seeds; `family_a_pass: false`, `family_c_pass:
   false`, `gate_pass: false`). Its family-A failure union across all seeds is
   **exactly** the four `hh_size` cells (every other gated family-A cell in-band 5/5),
   and its family-C leg **failed under the 4-element rule** (`reversed_to_anchor:
   false`, `both_reverse: false`, τ 0.333). c3 was scored on the **current** 47-cell
   family-A + 4-element-C2 surface; that verdict **stands**. The pair-swap it realised
   does **not** rescue it — the current contract gates the full ordering, which it
   missed.
4. **Prospective only.** The amendment binds only candidates registered after
   ratification. That the unchanged c3 model would pass the **amended** surface is the
   ordinary prospective consequence of a corrected contract, **not** a retroactive
   change to c3's verdict — and it is separately gated by the independent bit-exact
   reproduction addendum. A candidate-4 registration is the vehicle (§5, §10), not a
   re-grade of c3.
5. **The evidence is train/frame-side** (`reported_not_gated` / `train_frame_side_only`
   / `publishes_regardless`, verified) — not scored candidate holdouts the amendment
   could retroactively rehabilitate.
6. **Description-claims-exactly.** After the flip, `covers` / `certification_scope`
   claim **exactly** the amended 44-cell surface (43 family-A + the pair-scoped C2);
   the four `hh_size` cells and the `cap_150k`/+1pp legs are report-only with reasons
   named.

Each committed candidate fails independently of the four demoted cells and the
re-scoped legs — the rescue set is verifiably empty, and the amendment is
prospective-only.

## 7. The exact prospective flip (post-ratification; NOT in this PR)

The ratifying flip PR — a separate PR, after this ceremony clears — will make exactly
these edits to `gates.yaml` `gates.gate_w1`. Every edit is a re-scope or a text
narrowing; **no tolerance, σ, anchor value, or ordering changes**, and the demoted
cells' tolerances and the C2 orderings are retained under report-only.

1. **`thresholds.family_a`** — move the four `hh_size` cells from their view
   `tolerances` (gate-eligible) to `report_only`, and extend `report_reasons`:
   - `hh_size_share.1` → `no_permitted_entry_state_lever_reaches_cell`
   - `hh_size_share.3` → `no_permitted_entry_state_lever_reaches_cell`
   - `hh_size_share.4` → `no_permitted_entry_state_lever_reaches_cell`
   - `hh_size_share.5plus` → `no_permitted_entry_state_lever_reaches_cell`

   The four tolerances (`0.191 / 0.191 / 0.174 / 0.184`) are retained verbatim under
   `retained_tolerances`. `hh_size_share.2` (tolerance `0.123`) stays gated. The
   `faithful_candidate_oc` block updates: `n_gated_cells` 47 → **43**, `p_seed_pass`
   0.9344 → **0.9403**, `p_gate_pass_4_of_5` 0.9623 → **0.9684** (recomputed on the
   43-cell surface; a CHANGE).
2. **`thresholds.protocol.fresh_run_artifact_schema.per_draw_per_cell_rates`** — the
   committed-cube shape `[20, 47, 5]` → `[20, 43, 5]` (K_draws × gated_cells ×
   gate_seeds); `undefined_draw_rule` and the regenerated-surface rule unchanged.
3. **`thresholds.family_c`** — **re-scope** `fingerprint.elimination_plus2pp` (C2)
   from the 4-element ordering to the pair-scoped elim↔+2pp swap: keep it in
   `gate_partition.gate_eligible` (it stays gated, `n_gate_eligible: 1`), annotate the
   fingerprint that the gated check is now `required_swap_realised` on
   `[payroll_plus_2pp, elimination]` only, and publish the `cap_150k` and +1pp legs
   (the deployed 4-element ordering + the 16.7-year `cap_150k` figure) **report-only**,
   tagged `anchor_ordering_internally_inconsistent_under_certified_swap`. The C2
   `required_representative_order`, `psid_frame_order`, and `provision_deltas` are
   retained under the report-only ordering disclosure.
4. **`thresholds.family_c.check`** — "the C2 fingerprint reverses … to
   required_representative_order (Kendall τ → 1.0)" → "the C2 elim↔+2pp adjacency
   reverses (elimination outranks +2pp; anchor-supported, realised 3/3), the pair the
   representative frame can coherently reverse; the `cap_150k`/+1pp legs are
   report-only (`anchor_ordering_internally_inconsistent_under_certified_swap` — the
   certified compression correction mechanically lifts `cap_150k`, Q10)".
5. **`gate_w1.covers`** — family A "47 gated / 58 report-only" → "**43 gated / 62
   report-only**"; family C "1 gated / 1 report-only" → "**1 gated / 1 report-only**"
   (unchanged count; C2 re-scoped to the pair, `cap_150k`/+1pp legs report-only within
   C2); the overall roll-up "48 gated / 84 report-only" → "**44 gated / 88
   report-only**".
6. **`thresholds.certification_scope.certifies`** — "the 48 gated cells (47 family-A
   joints + 1 family-C fingerprint — C2, the taxable-max elimination reversal)" → "the
   **44** gated cells (43 family-A joints + 1 family-C fingerprint — the pair-scoped C2
   elim↔+2pp compression swap); the four `hh_size` cells and the `cap_150k`/+1pp
   ordering legs are report-only pending a non-entry-state household-composition model
   and the anchor-ordering re-specification"; and narrow the "reverses the C2
   fingerprint" clause to "reverses the C2 elim↔+2pp swap".
7. **`gate_w1.history`** — append the amendment-3 entry (§8 draft).

`thresholds.family_b`, the C1 report-only disposition, `thresholds.protocol`
(statistic, pass_rule, estimator, floor bindings), `governance.amendment_rules`,
`frame_pin`, `holdout_basis`, and every retained tolerance/anchor/ordering are
**untouched**. A subset master-compare must show the flip changes only `gate_w1` and
moves no locked sibling.

## 8. `amendment_history` entry draft (for `gate_w1.history`)

```yaml
- id: 2026-07-13-w1-hh-size-and-c2-pair
  proposed: '2026-07-13'
  referee_round:
    review: '<PR> comment <id> (verdict TBD)'                # filled at ceremony
    fixes: '<sha>'                                           # filled at ceremony
    verification: '<PR> comment <id> (verdict TBD)'          # filled at ceremony
  ratified: '<date> by merge of PR <n> (merge commit <sha>) under the
    maintainer''s standing campaign directive of 2026-07-07, exercised only
    after the full ceremony'                                 # filled at ratification
  flipped_live: this pull request
  content: >-
    STRUCTURAL demotion + RE-SPECIFICATION, ZERO THRESHOLD MOVEMENT. 4 family-A
    cells are DEMOTED from gated to report-only -- the household-size quad
    hh_size_share.{1,3,4,5plus} -- under one uniform machine reason
    no_permitted_entry_state_lever_reaches_cell; and the family-C gated binary C2
    (fingerprint.elimination_plus2pp) is RE-SCOPED from the full 4-element
    compression ordering to the pair-scoped elim<->+2pp adjacent swap, with the
    cap_150k and +1pp legs published report-only
    (anchor_ordering_internally_inconsistent_under_certified_swap). hh_size_share.2
    stays gated (candidate 3 clears it 5/5). All 4 hh_size tolerances (0.191/0.191/
    0.174/0.184) and the C2 orderings publish report-only (retained, not deleted).
    Motivated by W1 forensics 3 (runs/gate_w1_forensics3_v1.json, registration
    4959668253, grading 4960691167) Q10/Q11, W1 forensics 2 Q7, and the ratified
    amendment-2 series forecast that PRE-NAMED both surfaces before candidate 3 ran.
    Q11 (hh_size): the size-1<->size-3+ coresidence mirror is confirmed (coresidence
    moves size-1 -0.047 vs size-3+ +0.046, ratio 0.98; owns size-3 at 0.75), while
    the fertility window owns size-4 (0.67) and size-5plus (0.78) and size-1 is
    mixed (fertility 0.54 / coresidence 0.46); both permitted entry-state levers are
    EXHAUSTED -- the roster-ceiling probe (seed pushed to and beyond the permitted
    maximum) reaches terminal coresidence only 0.111, capped by the CERTIFIED
    parental-exit hazard, and does NOT close the size-1 excess (0.179 vs frame
    0.083); closing it needs a non-entry-state household model (an amendment). PER-
    CELL TRUTH (no blanket impossibility): from the three committed cubes, sizes
    1/3/5plus fail 5/5 for c1/c2/c3, size-4 fails 5/5 for c1/c2 but CLEARS c3 seed 1
    (1/5), and size-2 CLEARS c3 5/5 -- so size-2 stays gated and no blanket
    household-size-unreachability claim is made. Q10 (cap_150k): the 16.7y-vs-1y gap is
    owned 0.9925 by the frame's above-cap payroll share; the entailment HOLDS -- the
    same compression correction that certifies the elim<->+2pp swap mechanically
    lifts cap_150k to rank 2 (deployed A/B 1.694 PV), so the 4-element ordering is
    internally inconsistent as a representative-frame transport target (elim>+2pp <=>
    A/B>0.1613 vs the Smith-adjacency window A/B in (0.1613, 0.184), overshot ~10x);
    contract-permitted lever restoration enumerates EMPTY. The pair-scoped C2 is
    anchor-supported (Smith 2015 p.3: elim +21yr > +2pp +18yr), realised 3/3 across
    c1/c2/c3, and certifies exactly the frame-compression mechanism the fingerprint
    exists for; no published anchor supports the full 4-element ordering on a
    representative frame. NO-SELF-RESCUE: candidate 1 (PR #162), candidate 2 (PR
    #167), and candidate 3 (PR #176) all STAND FAIL; c3's committed gate FAIL was
    scored under the current contract (47 family-A + the 4-element C2) with family-A
    0/5 (failure union exactly the four hh_size cells) AND family-C failing the
    4-element reversal -- the demotion rescues nothing. Unlike amendment 2, this is
    the first amendment after which the amended surface is KNOWN PASSABLE by a
    committed model: candidate 3 lands 43/43 of the surviving family-A cells in-band
    5/5 and realised the elim<->+2pp swap. The whittling question is therefore
    answered in its strongest form: the demotion boundary is the permitted-lever line
    (Q10 construction empty, Q11 levers exhausted, three candidates deep), it was
    pre-named in the ratified amendment-2 forecast BEFORE c3 ran (amendment3_risks =
    exactly this set), and it changes no committed verdict -- a candidate-4
    registration of the UNCHANGED c3 model follows the flip, subject to the standing
    independent bit-exact reproduction addendum before the record admits a pass. The
    gate_w1 faithful-candidate OC moves from 0.9623 x I(C2) to 0.9684 x I(pair-swap)
    on the residual 44-cell gated surface (48 -> 44 gated, 84 -> 88 report-only;
    family A 47 -> 43, cube shape [20,47,5] -> [20,43,5]). Certification scope
    narrows: the four hh_size shares and the cap_150k/+1pp ordering legs are report-
    only, published-not-certified. Evidence chain: candidate 1 (PR #162), candidate 2
    (PR #167), candidate 3 (PR #176), W1 forensics 2 (PR #168), W1 forensics 3 (PR
    #179).
```

## 9. Certification-scope language after amendment 3

What the gate **claims** after the flip:

- **Certifies (gated)**: the 43 family-A CPS-observable joints and the C2 **elim↔+2pp
  compression swap** — **44 cells**. A PASS certifies that stochastic regeneration
  through the deployed generators reproduces those 43 joints within the frame's own
  sampling floor, and reverses the elim↔+2pp adjacency (elimination outranks +2pp) —
  the anchor-supported, thrice-realised, frame-robust compression correction. **For the
  first time this surface is known passable by a committed model** (candidate 3, §3/§5),
  but a pass still enters the record only after the standing independent bit-exact
  reproduction addendum clears.
- **Publishes (report-only, NOT certified)**: the four `hh_size` shares
  (`no_permitted_entry_state_lever_reaches_cell` — the coresidence-composition residual
  Q11 shows is lever-exhausted, with the per-cell channel attribution); the `cap_150k`
  and +1pp ordering legs (`anchor_ordering_internally_inconsistent_under_certified_swap`
  — frame-entailed to move, Q10); plus the report-only cells carried from the lock and
  amendments 1–2 (C1's PPI-over-NRA ordering, the 18-24 participation pair, the 65+
  marital/coresident quad, and every family-B SSA margin).
- **Does not support**: the four `hh_size` shares, the `cap_150k` rank or the +1pp↔cap
  adjacency, the C1 reversal, the 18-24 participation, the 65+ composition, or any
  family-B margin as certified quantities. **No named amendment-4 risk remains on the
  gated surface** — the amendment-2 forecast's five `amendment3_risks` are all resolved
  here, and the surviving 43 joints are c3-passable while the gated pair-swap is the
  most-realised element on the surface. The open work is report-only bridge work
  (§10), reopenable only via the same forensics-then-ceremony standard, never a bare
  re-gate. (The dynamics remain gate-1/2a/2b/2c/M4 certified on their PSID holdouts; W1
  certifies transport, not re-estimation.)

Stated plainly: **the gate certifies the 43 reproducible family-A joints and the
elim↔+2pp compression swap — a surface now known passable by the committed c3 model; it
certifies neither the four `hh_size` shares, the `cap_150k` rank, nor (carried from
amendments 1–2) the C1 reversal, the 18-24 participation, the 65+ composition, or any
family-B margin, pending the bridges.**

## 10. Evidence-base prerequisites for the deferred paths (the bridges)

- **`hh_size` (option a2) — a non-entry-state household-composition model.** The
  residual is a coresidence-composition insufficiency the two permitted entry-state
  levers (wave-0 roster + fertility window) cannot close (Q11 lever exhaustion; the
  roster ceiling is hazard-capped at terminal coresidence 0.111). Re-gating the
  `hh_size` cells requires either a **non-entry-state coresidence/household model** (a
  re-estimation W1 does not certify — an evidence-base amendment) or a re-anchored
  household-composition target; both are outside the transport contract and need a
  fresh referee round.
- **`cap_150k` / the full C2 ordering (option c-c) — a coherent representative-frame
  target.** Q10 proves the 4-element order is internally inconsistent on a
  representative frame (the certified swap entails cap rank-2), and no published anchor
  supports the full ordering on such a frame. Re-gating the `cap_150k` rank requires a
  **new published anchor** that asserts a representative-frame `cap_150k` ordering a
  contract-consistent transport can realise — which the evidence base does not contain.
  The pair-scoped C2 recommended here is the anchor-consistent target that exists today.
- **The report-only families carried from amendments 1–2** (C1, the 18-24 pair, the 65+
  quad, family B) keep the bridge prerequisites amendment 2 and amendment 1 named — a
  new PPI-over-NRA binary, a head/spouse young-adult participation anchor or non-head
  generator extension, an older-cohort divorce/remarriage hazard vintage or re-anchored
  65+ frame, and the DI concept-bridge — each an evidence-base amendment via the same
  forensics-then-ceremony standard.

## 11. Ceremony checklist

- [x] **Proposal** (this document + this draft PR). No `gates.yaml` edit; no threshold
      moved; the machine-checkable ledger + §7 flip text are string-bound to
      `tests/test_gate_w1_amendment3_proposal.py`.
- [ ] **Adversarial referee round** (pending).
- [ ] **Fixes** (pending the referee round).
- [ ] **Verification round** (pending).
- [ ] **Ratify by merge** of the flip PR (after the ceremony clears).
- [ ] **Flip** `gates.yaml` per §7 and append the §8 history entry.
- [ ] Then: candidate 4 — the **unchanged** candidate-3 model re-registered against the
      amended contract, its pass admitted to the record only after independent bit-exact
      reproduction (the standing pass-verification addendum).

---

<!-- amendment-consistency-ledger: bound to the committed artifacts by
     tests/test_gate_w1_amendment3_proposal.py. Do not hand-edit a value without
     re-deriving it from runs/gate_w1_forensics3_v1.json,
     runs/gate_w1_candidate3_v1.json, runs/gate_w1_candidate2_v1.json,
     runs/gate_w1_candidate1_v1.json, runs/gate_w1_forensics2_v1.json,
     runs/gate_w1_forensics1_v1.json, runs/gate_w1_floors_v1.json, gates.yaml, or
     docs/amendments/gate_w1_amendment_2_family_a_concept_cells.md. -->

```json amendment-consistency-ledger
{
  "amendment_id": "2026-07-13-w1-hh-size-and-c2-pair",
  "recommendation": "demote_hh_size_quad_and_pair_scope_c2",
  "n_cells_demoted": 4,
  "family_c_rescope": "pair_scoped_c2_elim_plus2pp_swap",
  "cell_reasons": {
    "hh_size_share.1": "no_permitted_entry_state_lever_reaches_cell",
    "hh_size_share.3": "no_permitted_entry_state_lever_reaches_cell",
    "hh_size_share.4": "no_permitted_entry_state_lever_reaches_cell",
    "hh_size_share.5plus": "no_permitted_entry_state_lever_reaches_cell"
  },
  "cap_150k_leg_reason": "anchor_ordering_internally_inconsistent_under_certified_swap",
  "per_cell_pass_record": {
    "hh_size_share.1": {"c1": 0, "c2": 0, "c3": 0},
    "hh_size_share.2": {"c1": 0, "c2": 0, "c3": 5},
    "hh_size_share.3": {"c1": 0, "c2": 0, "c3": 0},
    "hh_size_share.4": {"c1": 0, "c2": 0, "c3": 1},
    "hh_size_share.5plus": {"c1": 0, "c2": 0, "c3": 0}
  },
  "hh_size_channel_attribution": {
    "1": {"fertility_share_round2": 0.54, "coresidence_complement_round2": 0.46, "coresidence_main_effect_share_round2": 0.4, "dominant": "fertility", "mixed": true},
    "3": {"coresidence_share_round2": 0.75, "dominant": "coresidence"},
    "4": {"fertility_share_round2": 0.67, "dominant": "fertility"},
    "5plus": {"fertility_share_round2": 0.78, "dominant": "fertility"}
  },
  "hh_size_mirror": {
    "coresidence_mirror_ratio_round2": 0.98,
    "coresidence_moves_size1_round3": -0.047,
    "coresidence_moves_size3plus_round3": 0.046,
    "coresidence_owns_size1_size3_mirror": true
  },
  "roster_ceiling_probe": {
    "terminal_coresidence_at_ceiling_round3": 0.111,
    "size1_at_ceiling_round3": 0.179,
    "frame_rate_a_size1_round3": 0.083,
    "closes_size1": false,
    "levers_exhausted": true,
    "three_candidates_deep": true
  },
  "forensics3": {
    "artifact": "runs/gate_w1_forensics3_v1.json",
    "registration": "4959668253",
    "grading": "4960691167",
    "reported_not_gated": true,
    "train_frame_side_only": true,
    "q10_share_a_frame_above_cap_round4": 0.9925,
    "q10_share_bc_config_vintage_round4": 0.0075,
    "q10_deployed_A_over_B_round3": 1.694,
    "q10_c2_breakeven_A_over_B_round4": 0.1613,
    "q10_smith_adjacency_window": [0.1613, 0.184],
    "q10_smith_adjacency_window_width_round4": 0.0228,
    "q10_deployed_cap_rank": 2,
    "q10_deployed_breaks_adjacency": true,
    "q10_entailment_holds": true,
    "q10_permitted_lever_restoration_exists": false,
    "q10_cap_150k_deployed_years_round1": 16.7,
    "q10_smith_cap_150k_years": 1,
    "q11_attribution": "mixed_but_structured",
    "q11_levers_exhausted": true
  },
  "pair_scoped_c2": {
    "anchor_basis_smith_elim_years": 21,
    "anchor_basis_smith_plus2pp_years": 18,
    "swap_realised_c1": true,
    "swap_realised_c2": true,
    "swap_realised_c3": true,
    "realised_3_of_3": true,
    "required_swap_pair": ["payroll_plus_2pp", "elimination"],
    "any_published_anchor_supports_full_4_element": false
  },
  "c2_record": {
    "deployed_order": ["elimination", "cap_150k", "payroll_plus_2pp", "payroll_plus_1pp"],
    "required_representative_order": ["elimination", "payroll_plus_2pp", "payroll_plus_1pp", "cap_150k"],
    "kendall_tau_vs_required_round4": 0.3333,
    "smith_year_deltas": {"elimination": 21, "payroll_plus_2pp": 18, "payroll_plus_1pp": 5, "cap_150k": 1},
    "cap_150k_deployed_years_round1": 16.7,
    "cap_150k_smith_rank4_years": 1
  },
  "family_a_oc": {
    "n_gated_before": 47,
    "p_seed_before": 0.9344,
    "p_gate_before": 0.9623,
    "n_gated_after": 43,
    "p_seed_after": 0.9403,
    "p_gate_after": 0.9684,
    "invariant_under_amendment": false,
    "demoted_hh_size_faithful_passprob_min_round3": 0.998,
    "demoted_hh_size_faithful_passprob_max_round4": 0.9989
  },
  "family_a_partition": {"gated_now": 47, "gated_after": 43, "report_only_now": 58, "report_only_after": 62},
  "family_c_partition": {"gated_now": 1, "gated_after": 1, "report_only_now": 1, "report_only_after": 1},
  "overall_partition": {"gated_now": 48, "gated_after": 44, "report_only_now": 84, "report_only_after": 88},
  "oc_statement_after": "0.9684 * I(pair_swap)",
  "cube_shape_before": [20, 47, 5],
  "cube_shape_after": [20, 43, 5],
  "whittling": {
    "post_amendment_surface_c3_passable": true,
    "c3_family_a_cells_in_band_5of5_after": 43,
    "c3_pair_swap_realised": true,
    "pre_named_in_amendment2_forecast": true,
    "prospective_only": true,
    "surface_known_passable_by_committed_model": true,
    "first_for_this_gate": true
  },
  "prenamed_amendment3_risks_from_a2": ["hh_size_share.1", "hh_size_share.3", "hh_size_share.4", "hh_size_share.5plus", "c2_cap_150k_adjacency"],
  "amendment2_doc": "docs/amendments/gate_w1_amendment_2_family_a_concept_cells.md",
  "no_self_rescue": {
    "candidate1_pr": 162,
    "candidate1_gate_pass": false,
    "candidate2_pr": 167,
    "candidate2_gate_pass": false,
    "candidate3_run": "gate_w1_candidate3_v1",
    "candidate3_pr": 176,
    "candidate3_gate_pass": false,
    "candidate3_family_a_seeds_pass_47_surface": 0,
    "candidate3_family_a_fail_union_47_surface": ["hh_size_share.1", "hh_size_share.3", "hh_size_share.4", "hh_size_share.5plus"],
    "candidate3_family_c_pass_4element": false,
    "candidate3_c2_reversed_to_anchor": false,
    "candidate3_pair_swap_realised": true,
    "rescue_set_empty": true,
    "prospective_binding_only": true
  },
  "candidate4_followup": {
    "same_model_as_c3": true,
    "registration_after_flip": true,
    "subject_to_bitexact_reproduction_addendum": true
  },
  "family_b_unchanged_report_only": true,
  "surface_known_passable_by_committed_c3": true,
  "gates_yaml_untouched_by_this_proposal": true
}
```
