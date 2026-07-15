# M6 omnibus row-coverage seam audit

Audit base: `origin/master` at `75d30dd57d71b91ee0929246b2f3cbb92263b350`<br>
Audit date: 2026-07-15<br>
Surfaces: the ordered M6 wave loop; realized-population construction; both split-side projections; truth and projected cell preparation; draw and seed aggregation; the §2.8.3a domain-floor recompute; every report-only builder; and artifact assembly.

## Result

The audit found **11 remaining seams or schema debts**:

| Severity | Count | Finding IDs |
|---|---:|---|
| Registered-path crash | 2 | F1, F2 |
| Certified-caller-conditional crash debt | 1 | F3 |
| Silent-wrong | 5 | F4–F8 |
| Benign in current M6 | 3 | F9–F11 |

Two crash findings, F1 and F2, lie on ordinary registered-run paths. F3 is a caller-conditional empty-universe group: the large certified halves are expected to make it unreachable, but the invariant is not enforced by construction. The already-known amendment-3h death-then-drawn fertility crash is catalogued separately as K1 and **is not included** in these new counts.

**The single most dangerous remaining seam is F1, the disability scoring-universe mismatch.** The realized disability reproduction panel legitimately contains adjacent pairs outside ages 20–66, projected preparation retains them, truth preparation drops them, and the exact support guard aborts on one extra selected key. This was reproduced with an in-memory age-18/age-30 fixture.

No real-data loader, scored pipeline, or real-data statistic was executed. The environment was built exactly as directed:

```text
uv venv .venv-wt
uv pip install -p .venv-wt/bin/python -e ".[dev]"
```

The five draft characterizations in `tests/test_m6_seam_audit_draft.py` are synthetic, are not registered in `tests/tier_counts.json`, and pass while asserting current defect behavior. No gate, frozen floor, run artifact, design document, or tier manifest was changed.

## Verdict language

- **BC — by construction:** code enforces the universe/key invariant for every caller admitted by the interface.
- **CC — certified caller:** the invariant depends on a named property of the M6 inputs or their size; generic code does not enforce it.
- **V — violatable:** a concrete row condition breaks the invariant, crashes, or silently changes meaning.
- **KD — known/disclosed:** the mismatch is intentional or already in the 3h record; it is retained here so the inventory has no holes.

“Schedule” below means a frame-independent cached/native-panel simulation. “Roster” means the mortality-thinned live engine frame. Keeping those universes distinct is required by §§2.8.1–2.8.4 of `docs/design/m6_projection_engine.md`; materialization into the roster must nevertheless be live-keyed and reconciled.

## Full seam inventory

| Surface | Consumed universe and keying | Removal, addition, move, or re-key exposure | Verdict |
|---|---|---|---|
| Contract/floor resolve | Locked `gate_m6` mappings and frozen artifact SHA; no person keys | No roster materialization | **BC.** Exact path/SHA checks at `src/populace_dynamics/harness/m6_runner.py:275-329`. |
| Refit + anchor construction | Earliest positive-weight 2015/2017/2019 presence, keyed by `person_id`; fixed `household_id`, F6 `weight`, `anchor_wave`; coded sex joined from death records; global fitted earnings maps | Harness admits all anchor ages and later openers; module panels may cover strict subsets | **BC** for identity/sex/domain-subset checks at `src/populace_dynamics/harness/m6_population.py:202-259`; **V** for unvalidated mortality age support (F6) and household coverage (F5). |
| Pre-flight 1 | Truncated training `MaritalPanel`, household panel, and `train_ids`; both candidate-9 arms use the same IDs | No holdout-side roster and no live mortality removal | **CC.** Same-ID comparison is structurally aligned at `src/populace_dynamics/harness/m6_runner.py:391-418`; nonempty training panels are a certified-input assumption covered by F3. |
| Pre-flight 2 | Synthetic earnings sign probe; no realized person/household keys | None | **BC**, `src/populace_dynamics/harness/m6_runner.py:421-430`. |
| Person and household side split | Full anchor split by fixed `person_id` or fixed anchor `household_id`; then anchor, initial, scheduled, presence, earnings support/domain, weights, and disability IDs are person-filtered | Later household composition never changes the split key | **BC.** Split at `src/populace_dynamics/harness/m6_scoring.py:394-425`; complete subsetting at `src/populace_dynamics/harness/m6_population.py:76-126`. |
| Initial slice + scheduled openers | 2015 initial roster plus 2017/2019 frames carrying year `entry-1`, keyed by `person_id` | Mortality may empty the active roster before a later cohort enters; scheduled entrants add rows | **BC** for duplicate/overlap checks and activation at `src/populace_dynamics/engine/loop.py:192-221,239-265`; **CC/V** for empty initial/live frames (F3). |
| RNG ordinals + synthetic ID allocation | Stable ordinals over side initial + scheduled IDs; monotone IDs begin at `max(side IDs)+1` | Births add IDs; deaths remove them; fitted maps still retain full-population IDs | Ordinals are **BC** at `src/populace_dynamics/engine/loop.py:213-237,329-331`; allocator namespace is **V** because it is side-local while earnings maps are global (F2). |
| Mortality | Current roster, `(age band, sex)` probability map, person-keyed RNG | Removes rows before every later module; openers enter immediately before mortality | Subset removal is **BC** at `src/populace_dynamics/engine/steps.py:89-134`; age coverage is **V** above 120/below 0 (F6). |
| Mortality collector/report | Pre-death roster keyed against survivor-ID set; report filters realized presence and age/sex bands | Sees scheduled entrants and previous births; later reports drop out-of-band ages | Death flagging is **BC** at `src/populace_dynamics/engine/assembly.py:252-269`; top-age report omission shares F6 (`src/populace_dynamics/harness/m6_runner.py:708-746`). |
| Aging | Current survivor roster; optional year-indexed NAWI and taxable maximum; no external person keys | Advances all live rows and no schedule rows | **BC**, `src/populace_dynamics/engine/steps.py:137-166`. |
| Marital panel builder | `marital.attrs ∩ side anchor`; realized censor; one certified entry row keyed `(person_id, year)`; 3g start is `max(anchor_wave, birth_year+15)` | Deliberately independent of simulated mortality; later openers and anchor minors enter on their certified start | **BC after 3g**, `src/populace_dynamics/engine/panel_builders.py:150-241`. Empty attrs are **CC/V** under F3. |
| Cached marital simulation | Native attrs/entry rows, fitted hazards, permanent-earnings axis; cached once on period-0 streams | Continues on realized support after simulated death | **BC by the gated-realized law**, `src/populace_dynamics/engine/assembly.py:271-295` and `src/populace_dynamics/engine/marital.py:53-273`. |
| Marital schedule → roster | Current live IDs left-joined to that year’s cached state by `person_id` | Deaths remove schedule IDs; MH gaps/censor and newborns create roster-only IDs | No resurrection is **BC**, but partial update erasure is **V** (F4), `src/populace_dynamics/engine/assembly.py:174-192,297-314`. |
| Fertility schedule | Frame-independent marital attrs and `state.marital_ids`; maternal/paternal records keyed `(parent_person_id, birth_year)` | Includes a mother after simulated mortality; excludes live people outside marital attrs; recomputed per live period | Death-then-drawn is **KD/K1**. Domain-limited coverage is **CC/benign F11**. Schedule construction: `src/populace_dynamics/engine/steps.py:185-207,440-472`. |
| Maternal birth materialization | Current roster parent lookup; sequential synthetic ID; child copies mother household/weights | Mortality can remove scheduled parent; child adds a row unknown to every cached native panel | Parent guard is correct but K1 currently reaches it (`src/populace_dynamics/engine/steps.py:381-437`). New-row schema is **benign/F10**; ID alias is **V/F2**. |
| Disability reproduction schedule | Full realized disability person-years intersected with side disability IDs; F6 weights; per-period RNG; keyed `(person_id, period)` | Independent of mortality; includes ages outside the 20–66 scored hazard domain; gaps/censor create partial years | Frame-independent reproduction is **BC**, `src/populace_dynamics/engine/assembly.py:326-369` and `src/populace_dynamics/engine/disability.py:37-202`. Scored preparation is **V/F1**. |
| Disability schedule → roster | Current live IDs left-joined to cached `disabled/retired/status/converted` updates | Deaths discard extras; openers without status, attriters, minors, and newborns are roster-only | **V/F4.** Missing `di_converted` is later treated as false at `src/populace_dynamics/engine/steps.py:363-373`. |
| Earnings initialization/live step | Side initial roster and `earnings_domain` marker, checked against the full refitted 2014 state maps; live current IDs thereafter | 2017/2019 openers and births are intended domain-false; deaths remove rows | **CC** that every true domain ID is a 2015/2014-state initial person; scored reproduction asserts it at `src/populace_dynamics/harness/m6_projection.py:178-195`. **V/F2** if a synthetic ID aliases a global map key. Otherwise wrapper behavior is **BC**, `src/populace_dynamics/engine/earnings_domain.py:156-210`. |
| Gated earnings reproduction | Side initial domain state + realized earnings support + all side IDs for stable ordinals; keyed `(person_id, period)` | Intentionally ignores simulated death; later earnings openers excluded symmetrically | **BC**, `src/populace_dynamics/harness/m6_projection.py:163-245` and exact restriction at `src/populace_dynamics/harness/m6_scoring.py:428-481`. |
| Claiming | Current roster, sex/year PMF, carried claim state, `di_converted` | Sees mortality-thinned rows, new openers, newborns, and disability overlay gaps | Person-local draws are **BC**, but unavailable disability state is silently read as non-conversion under **F4**, `src/populace_dynamics/engine/steps.py:320-378`. |
| Household panel builder | Certified household person-waves with age 15–120; exact anchor row; side anchor weights; sparse cohabitation; keyed `(person_id, year)` | All-age anchor admits children; later eligible rows do not rescue a missing exact-anchor seed | **V/F5**, `src/populace_dynamics/data/household_composition.py:382-424` and `src/populace_dynamics/engine/panel_builders.py:257-367`. Empty panel is **CC/V** under F3. |
| Candidate-9 cached household simulation | Household native support + cached marital state + separate period-0 fertility realization | Continues after live deaths and does not contain live newborn rows; no household IDs are moved | Internally **BC** with fallback for missing marital rows at `src/populace_dynamics/models/household_composition/components/marital_core_adapter.py:66-113`; separate fertility stitching is **KD**, while full live-roster reconciliation is **CC/benign F9**. |
| Household schedule → roster / household identity | Current live IDs left-joined to cached flags/counts; anchor `household_id` is carried and newborn copies mother’s ID | Deaths discard extras; births/gaps produce roster-only IDs; candidate-9 emits no ID moves | Fixed identity/no resurrection is **BC**; partial state erasure is **V/F4** and count reconciliation is **F9**, `src/populace_dynamics/engine/assembly.py:385-424`. |
| Truth cell preparation | Household-side IDs for marital; person-side IDs for disability/earnings; earnings also intersects the side domain | No live-roster join | **BC**, `src/populace_dynamics/harness/m6_runner.py:460-478`. Nonempty locked truth strata are a certified-floor assumption under F3. |
| Projected marital support | Cached marital events/person-years, side anchor weights, opening-wave realized presence, gated years, bands and coded sex | Mortality-thinned roster is intentionally irrelevant | **BC after 3g**, `src/populace_dynamics/harness/m6_projection.py:71-107`; exact projected/truth keys enforced at `src/populace_dynamics/engine/support.py:285-332`. |
| Projected disability support | Every cached adjacent 2015/2017 pair, then side-anchor weight merge | Schedule includes off-band ages that truth excludes | **V/F1**, `src/populace_dynamics/harness/m6_projection.py:110-146` versus truth `src/populace_dynamics/harness/m6_cells.py:294-334`. |
| Cell reduction and draw scoring | Prepared long frames reduced to exactly 11 locked cells; truth rate and K projected rates by cell name | No person joins remain; sparse support can omit a locked name or leave a value undefined | Exact-name enforcement is **BC** but availability is **CC/V** (`src/populace_dynamics/harness/m6_scoring.py:484-541`). Projected undefined/nonpositive values become an explicit invalid result (**BC**); truth key/undefined failures remain designed raises (`:580-688`). See F3. |
| Seed aggregation | Exact registered seed mapping and 4-of-5 conjunction | No row universe remains | **BC**, `src/populace_dynamics/harness/m6_scoring.py:691-707` and order check `src/populace_dynamics/harness/m6_runner.py:1375-1383`. |
| §2.8.3a domain-floor recompute | Truth earnings restricted to domain; currently splits the already-filtered domain anchor | Gate side membership was assigned on the full anchor before domain intersection | **V/F7**, `src/populace_dynamics/harness/m6_scoring.py:723-873`. |
| Mortality/marital/disability shock and not-certified reports | Side-specific cached panels or live earnings inner-joined to truth metadata; realized presence/band filters | Live report earnings intentionally drops deaths and synthetic additions; shock disability correctly band-filters | **BC/KD**, `src/populace_dynamics/harness/m6_runner.py:630-927`. These are explicitly non-gating. |
| Entrant report | Synthetic IDs from the household-side reference projection; scheduled opener count; global domain/open-addition counts | Reference birth count is one labeled seed/draw; later earnings entrants are inferred from anchor wave rather than earnings-row support | Reference/ensemble labeling is **BC** at `src/populace_dynamics/harness/m6_runner.py:1091-1111`; later-entrant classification is **V/F8** at `:1019-1044`. |
| Alignment/redrawn/mortality-anchor reports | Either exact before/after keys or explicit unavailable records; no silent zero substitution | No live row join in current runner (`None, None`) | **BC**, `src/populace_dynamics/harness/m6_reporting.py:321-415,418-496`; report-only enforcement at `:499-533`. |
| Artifact assembly/write | Validated seed/cell/report mappings; no person/household materialization | Sets/scalars only serialized; exclusive output write occurs last | **BC**, `src/populace_dynamics/harness/m6_runner.py:1142-1299,1349-1398`. |

## Crash-class findings

### F1 — Projected disability retains rows outside the scored hazard universe

**Severity:** would crash the registered run before scoring.<br>
**Verdict:** V; confirmed synthetically.

**Failing condition.** A selected, realized-present person has a grid-adjacent disability pair starting in 2015 or 2017 with start age below 20 or above 66. The disability reader/reproduction law validly carries status for ages 0–120 (`src/populace_dynamics/data/disability.py:368-426`), and `prepare_projected_disability` converts every adjacent pair and only anchor-inner-joins it (`src/populace_dynamics/harness/m6_projection.py:110-146`). Truth applies `band_of(age, DISABILITY_BANDS)` and coded-sex filtering (`src/populace_dynamics/harness/m6_cells.py:294-334`, especially `:321-325`). Presence conditioning does not remove an off-band row merely because it is outside the hazard domain; exact key equality then raises at `src/populace_dynamics/engine/support.py:299-315`.

The existing focused test uses only ages 30–36 (`tests/test_m6_projection_support.py:155-169`), so it certifies pair mechanics but not universe identity. The draft age-18/age-30 fixture at `tests/test_m6_seam_audit_draft.py:59-88` reproduces `ValueError: symmetric presence-conditioning requires identical projection and truth person-period support`.

**Minimal law/fix.** Apply the exact truth transform in `prepare_projected_disability`: attach the shared `band_of(age, DISABILITY_BANDS)`, retain non-null bands and coded `SEXES`, then enter the support guard. Prefer one shared helper so truth and projection cannot drift again. The coded-sex clause is symmetry hardening—the production `attach_sex` reader already drops uncoded rows at `src/populace_dynamics/data/disability.py:433-446`; off-band age is the reachable failure. Keep the reproduction schedule frame-independent and unchanged; report excluded pairs if desired. This changes no fitted hazard, cell, tolerance, or floor.

### F2 — Split-local synthetic IDs can alias full-population fitted-map IDs

**Severity:** would crash a registered draw when the allocated interval hits an omitted in-domain real ID.<br>
**Verdict:** V; the alias-to-validator failure is confirmed synthetically, while the side-local default allocation premise is established statically.

**Failing condition.** `score_m6_seed` projects person- and household-side subsets (`src/populace_dynamics/harness/m6_runner.py:930-969`). Subsetting removes out-of-side initial/scheduled IDs and domain membership (`src/populace_dynamics/harness/m6_population.py:76-126`), so the engine starts its allocator at `max(side initial + side scheduled)+1` (`src/populace_dynamics/engine/loop.py:213-228`). Assembly nevertheless wraps the full refit's earnings generator (`src/populace_dynamics/engine/assembly.py:230-250`), whose fitted domain is the global intersection of person-keyed 2014 maps (`src/populace_dynamics/engine/earnings_domain.py:68-84,138-154`).

If a newborn receives an omitted real ID, its missing `earnings_domain` marker reads false, while fitted membership reads true. `apply_earnings` validates the entire frame before applying age ≥15 eligibility (`src/populace_dynamics/engine/steps.py:218-242`), so even an age-zero child raises `earnings_domain marker disagrees with fitted 2014 state` (`src/populace_dynamics/engine/earnings_domain.py:189-196`). The draft fixture at `tests/test_m6_seam_audit_draft.py:89-123` reproduces that exact path.

**Minimal law/fix.** Before any split, compute a reserved real-person namespace containing the full anchor and every person-keyed fitted surface; pass a synthetic-ID lower bound strictly above its maximum through population metadata. Assert every allocation is disjoint from the reserved set. Scoping the earnings adapter to a side would avoid this one validator but would leave an identity alias in the engine, so global namespace reservation is the safer minimal law. Domain membership, scoring support, and floors remain unchanged.

### F3 — Empty universes have incidental exceptions instead of typed dispositions

**Severity:** caller-conditional registered-run crash.<br>
**Verdict:** CC/V; no real-data prevalence claim is made.

**Failing conditions.** Any of the following certified-caller size assumptions can fail:

1. A split contains only later openers, leaving the 2015 `initial_slice` empty; the engine requires exactly one observed initial year (`src/populace_dynamics/engine/loop.py:172-176`).
2. No valid marital attrs remain; candidate-16 calls `min()`/`max()` on empty start/end arrays (`src/populace_dynamics/engine/marital.py:150-152`).
3. No exact-anchor household seed remains; `padded_person_matrices` hits an incidental empty-array indexing failure while building `first`/`person_start` (`src/populace_dynamics/models/household_composition/common.py:62-69`).
4. Mortality removes every active row; if the intervening modules otherwise tolerate the empty roster, terminal projected-slice validation rejects the empty set of year values (`src/populace_dynamics/engine/loop.py:323-327`), even if a later scheduled cohort could enter. An empty native panel or K1 can fail earlier.
5. A sparse reduction that omits a locked cell name raises (`src/populace_dynamics/harness/m6_scoring.py:536-541`), as do truth key mismatch/undefined values (`:597-604`). By contrast, a present projected cell that is missing, undefined, or nonpositive under a log metric already becomes an explicit invalid draw (`:605-627`); that path is safe.

The full M6 population is expected to make these events remote or impossible in practice, but that is a caller assumption, not a code guarantee.

**Minimal law/fix.** Define typed empty marital/household results and permit an empty live slice while the engine retains the context year. Preserve the existing explicit projected-undefined invalidation. Empty/undefined truth is a different protocol condition whose current designed disposition is to raise; changing it would require a new registered protocol law, not a mechanical fallback. No synthetic rows or denominator changes are permitted.

## Silent-wrong findings

### F4 — A partial native-panel update erases unmatched live state

**Severity:** silent-wrong live engine state; gated marital/disability cells remain safe because they consume cached panels directly.<br>
**Verdict:** V; confirmed synthetically.

`_merge_period_columns` preserves the roster only when the entire year's update frame is empty. When at least one update exists, it drops the named columns from every roster row and left-merges the partial schedule (`src/populace_dynamics/engine/assembly.py:174-192`). Every unmatched live ID therefore becomes `NaN`, even if it had valid carried state. In the sharpest case, a nonempty update frame containing only dead/schedule-only IDs erases the named columns for every live row.

The helper is used for marital (`:297-314`), disability (`:369-377`), and household composition (`:412-424`). Violating rows include a live survivor after realized censor, an MH/household/status gap, a 2017/2019 opener without that module row, a subdomain minor, and every newborn outside the cached panels. Disability makes the consequence concrete: `apply_claiming` fills missing `di_converted` with false (`src/populace_dynamics/engine/steps.py:363-373`), collapsing “unavailable” into an observed no-conversion state. The draft fixture at `tests/test_m6_seam_audit_draft.py:126-146` demonstrates the erasure.

**Minimal law/fix.** Replace the destructive merge with a keyed patch that overwrites matching live IDs and never outer-adds schedule-only/dead IDs. For an unmatched live row, retain an existing value only under an explicit per-module availability/domain law; otherwise leave it explicitly unavailable and quarantine it from claiming/report interpretation. Carry-versus-unavailable must be pinned per field—especially for the event-like `di_converted` flag—rather than chosen generically in the helper. Publish unmatched-live/dropped-schedule counts by module and year. Cached scoring schedules and floors stay untouched.

### F5 — Household composition silently drops anchor minors and other exact-anchor gaps

**Severity:** latent silent-wrong engine state, benign to the current serialized artifact; wholly empty support is covered by F3.<br>
**Verdict:** V.

The M6 anchor admits every positive-weight person in a gated start wave, with no age filter (`src/populace_dynamics/harness/m6_cells.py:122-140`). The certified household reader retains ages 15–120 (`src/populace_dynamics/data/household_composition.py:382-424`, especially `:408`). The builder then requires an exact household row at each person's anchor and permanently restricts all later support to those seed IDs (`src/populace_dynamics/engine/panel_builders.py:293-314`). Population seeding left-merges missing household state without a coverage invariant (`src/populace_dynamics/harness/m6_population.py:273-284`).

Thus a 3g anchor minor who reaches 15 mid-window can have later certified household rows but never enter candidate-9. The same silent exclusion applies to any adult with an exact-anchor source gap. Existing synthetic coverage explicitly pins exclusion of a person whose only household row precedes their anchor (`tests/test_m6_panel_builders.py:453-482`).

This remains in the silent-wrong count because the wave loop emits household fields on `ProjectionResult.panel`, one of the requested audit surfaces, with no marker explaining the missing row. The current runner happens not to serialize those fields. F9 differs because its separate fertility stitching is explicitly disclosed and intentional; F5 is an unmarked coverage gap whose existence depends on anchor/source rows.

**Minimal law/fix.** Pin an explicit `household_state_domain` and reconcile `anchor_ids - household_seed_ids`. Because household outputs are report-only in M6, exclude-and-mark is floor-inert and safer than borrowing a realized post-anchor household outcome. If the intended law is to admit minors at age 15, that requires a separately reviewed, non-leaking initializer; unlike 3g's structurally constant marital risk-entry state, a future observed household state is not automatically an admissible seed.

### F6 — Mortality's `85+` band is closed at 120 and uncovered ages get zero risk

**Severity:** latent silent-wrong engine and mortality-report state.<br>
**Verdict:** V; confirmed synthetically.

`AgeSexMortalityModel` requires bands through age 120 and labels a band ending at 120 with `+`, but `probabilities` applies both `age >= lower` and `age <= upper` and initializes all rows to zero (`src/populace_dynamics/engine/steps.py:51-110`). Age 121 and negative/sentinel ages therefore receive death probability zero. The all-age demographic seed is cast to integer without a `[0,120]` validation (`src/populace_dynamics/harness/m6_population.py:230-247,313-320`), and a real age-120 survivor can age to 121 within the loop. The mortality report also drops ages outside its closed `(85,120)` band (`src/populace_dynamics/harness/m6_runner.py:731-746`). The draft test at `tests/test_m6_seam_audit_draft.py:160-172` pins the current `[1.0, 0.0]` probability result for ages 120/121 under a nominal `85+` probability of one.

**Minimal law/fix.** Reject nonfinite or numeric source ages outside the certified `[0,120]` initial/scheduled support, make the final mortality band genuinely open for engine-produced 121+ survivors (`age >= lower` with no upper), and use the same open-top rule in mortality reporting. Mortality is ungated in the current family-A surface, but this affects the live roster and family-B reports; no floor moves.

### F7 — The domain-floor self-check repartitions people after filtering

**Severity:** silent-wrong report/ceremony output; no direct gate or frozen-tolerance change.<br>
**Verdict:** V; confirmed synthetically.

Actual gate membership is drawn on the full anchor (`src/populace_dynamics/harness/m6_runner.py:938-947`; `src/populace_dynamics/harness/m6_scoring.py:394-425`) and only then intersected with the earnings domain (`src/populace_dynamics/harness/m6_runner.py:948-953`). The self-check instead filters to `domain_anchor` first and passes that frame to `run_floor` (`src/populace_dynamics/harness/m6_scoring.py:750-761`). `run_floor`/`split_panel_by_person` assign sequential RNG uniforms to sorted IDs (`src/populace_dynamics/harness/m6_cells.py:607-650`; `src/populace_dynamics/harness/panel.py:195-213`). Removing non-domain IDs before the draw shifts the uniforms attached to later domain IDs.

Consequently, the reported domain floor, effective OC, weak-power flag, and vacuity escalation are for a different 100-seed partition than the actual gate/frozen-floor split. The three-person fixture at `tests/test_m6_seam_audit_draft.py:149-157` shows person 3 on the left under full-anchor-then-domain at seed 1 but on the right under domain-first splitting.

**Minimal law/fix.** Call `run_floor` with the full anchor and intersect each returned half's IDs with `domain_earnings` inside `compute`, or freeze full-anchor split memberships first. Keep all tolerances, cells, and the frozen artifact untouched.

### F8 — The later-earnings-entrant report infers row existence from anchor wave

**Severity:** silent-wrong report-only classification.<br>
**Verdict:** V.

`build_report_only` labels every non-domain person with `anchor_wave > 2015` a later earnings entrant (`src/populace_dynamics/harness/m6_runner.py:1019-1044`). The §2.8.3a law defines earnings support by explicit 2016/2018 person-period row existence and explains that `family_earnings_panel` emits rows only for present heads/spouses (`docs/design/m6_projection_engine.md:1210-1245,1276-1283`; implementation at `src/populace_dynamics/data/family.py:785-857`). It therefore distinguishes later 2017/2019 head/spouse entrants with realized earnings rows from non-head/spouse people with no earnings state at all. Both are outside the 2014 domain, but only the first group is a later earnings entrant. Reporting also says counts must come from explicit runner markers (`src/populace_dynamics/harness/m6_reporting.py:277-315`). The current anchor-wave inference folds the second group into the named entrant total.

**Minimal law/fix.** Derive later entrants from explicit `phase.population.earnings_support` row existence at the gated reference years, intersect `anchor_wave > 2015`, and subtract the 2014 domain. Retain `n_closed - n_domain` as the broader `marked_no_earnings_state` count. No floor or verdict changes.

## Benign current-M6 debts

### F9 — Candidate-9 household counts are not a reconciliation of the live roster

**Severity:** benign to the current gate/artifact; latent silent-wrong if the engine's household fields are consumed as roster-consistent levels.<br>
**Verdict:** KD/CC. The separate fertility stitching is disclosed; full mortality/birth roster reconciliation is not enforced and remains a current-consumer assumption.

Live fertility recomputes a whole-window schedule on each period stream and materializes that period's maternal births (`src/populace_dynamics/engine/assembly.py:297-324`). Candidate-9 receives a separate period-0 fertility draw (`:392-408`) and runs only over its realized household support. Its `coresident_child` and `hh_size` calculations use that separate ledger (`src/populace_dynamics/engine/composition.py:365-439,457-523`), and its output contains no newborn rows (`:591-609`). Mortality likewise removes live members without removing schedule support.

The local design explicitly keeps per-period fertility stitching report-only (`docs/design/m6_projection_engine.md:1598-1605`), and the current runner does not publish household fields as gate cells (`src/populace_dynamics/harness/m6_runner.py:749-767,873-927`). A future live-level consumer should preserve candidate-9's certified frame-independent panel, then add a distinct reconciliation layer using the accepted live birth ledger and live roster, with discrepancy counts. Floors remain untouched.

### F10 — Newborn and scheduled-entry schema defaults are incomplete

**Severity:** benign in current reducers; future panel-consumer hazard.<br>
**Verdict:** V but currently inert.

Birth materialization creates a row with the roster's full columns but initializes only identity, time, age, birth year, sex, parent, `synthetic_entry`, and selected household/weight fields (`src/populace_dynamics/engine/steps.py:415-433`). Marital, disability, household, macro, and chain-state values begin unavailable/`NA`; the subsequent earnings step correctly normalizes the child to the §2.8.3a domain-false, `earnings=0` rule while leaving chain state unavailable. In addition, existing rows are set `synthetic_entry=False` only when the column is wholly absent (`:434-437`). If a birth created the column before a 2017/2019 scheduled frame is concatenated, those scheduled real entrants receive `synthetic_entry=NA`, not false (`src/populace_dynamics/engine/loop.py:242-257`).

Current gate/report reducers identify synthetic people by ID-set difference and ignore these missing fields (`src/populace_dynamics/harness/m6_runner.py:873-919`), so there is no current verdict effect. Add a single entrant-schema normalizer with explicit module-domain/unavailable defaults and fill the real/synthetic marker on every concat.

### F11 — Fertility births cover the marital domain, not every live reproductive row

**Severity:** benign/domain-limited in current M6; a disclosure requirement for interpreting family-B counts.<br>
**Verdict:** CC/benign domain-limited.

Production fertility passes `state.marital_ids`, the valid anchor/marital-panel intersection, into the frame-independent fertility kernel (`src/populace_dynamics/engine/assembly.py:293,315-324`; `src/populace_dynamics/engine/steps.py:468-472`). A live reproductive-age person absent from marital attrs cannot draw a birth, and synthetic children never join the cached marital domain. The kernel uses the same declared attrs domain as marital, but family-B birth counts have no truth-side support comparison that makes excluded live rows symmetric. There is no current gated effect; later openers and 3g minors who are in the panel cannot draw before their clamped entry (`src/populace_dynamics/engine/panel_builders.py:187-225`). Births occur only in 2015–2022, so their synthetic children cannot age into fertility eligibility before the M6 horizon, and adult immigrant cohorts are not wired. Paternal shadow births are conditioning-only and deliberately never materialize person rows (`src/populace_dynamics/engine/steps.py:185-207,381-392`).

The minimal treatment is disclosure: publish the fertility-domain denominator and excluded-live count beside synthetic-birth counts. Do not invent a fallback fertility law or expand the scoring universe.

## K1 — Known 3h baseline, not a new finding

The current master still has the death-then-drawn crash described in [issue #42 comment 4984997277](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4984997277):

1. mortality removes a mother from the live roster (`src/populace_dynamics/engine/steps.py:113-134`);
2. the frame-independent, realized-censor-bounded marital/fertility schedule can still draw her birth (`src/populace_dynamics/engine/marital.py:288-352`); and
3. materialization raises because the parent key is absent (`src/populace_dynamics/engine/steps.py:403-414`).

The 3h-consistent fix is post-draw live-parent filtering before materialization, retaining the absent-parent guard, and publishing dropped parent IDs/counts by year. Drawing, RNG consumption, scoring schedules, and floors remain unchanged. Candidate-9's separate conditioning draw remains untouched.

## Construction-safe conclusions

- The sex-source crash is closed: canonical coded sex comes from death records and full anchor coverage is checked (`src/populace_dynamics/harness/m6_population.py:154-184,222-247`).
- The former sub-15 marital crash is closed by 3g's entry clamp and exact certified entry-row check (`src/populace_dynamics/engine/panel_builders.py:187-225`).
- Projected deaths are not resurrected by marital, disability, or household overlays: every merge is roster-left. F4 is the opposite error—unmatched live state is erased.
- Gated marital, disability, and earnings schedules are intentionally frame-independent under the realized-support reproduction law. Earnings regeneration is mortality-independent and exact-domain keyed (`src/populace_dynamics/harness/m6_projection.py:163-245`).
- Start-wave weights cover every selected realized support ID because subsetting rebuilds the snapshot from the side anchor (`src/populace_dynamics/harness/m6_population.py:104-125`).
- Scheduled IDs receive stable RNG ordinals up front; newborns receive ordinals at the end of their birth wave before their next mortality exposure (`src/populace_dynamics/engine/loop.py:234-237,329-331`). Same-wave earnings and claiming skip ordinary newborns by age; F2 is the validator-before-age exception.
- Candidate-9 falls back to observed initial spouse state for household rows outside cached marital years rather than raising (`src/populace_dynamics/models/household_composition/components/marital_core_adapter.py:66-87`).
- Household composition never consumes or emits a moving `household_id`; the M6 split remains fixed at the anchor interview ID, and newborns copy the mother's ID. There is no current re-key seam, only the report-only reconciliation debt F9.
- Shock disability preparation does apply the correct band/sex filter (`src/populace_dynamics/harness/m6_runner.py:666-705`); F1 is isolated to gated projected preparation.
- Reduction, seed aggregation, report-only non-gating enforcement, and artifact assembly introduce no new person/household joins after prepared cells.

## Synthetic verification record

Only in-memory/synthetic tests were run. The focused draft command was:

```text
.venv-wt/bin/python -m pytest -q tests/test_m6_seam_audit_draft.py
```

Result: `5 passed`. The characterizations prove:

| Fixture | Finding | Current behavior asserted |
|---|---|---|
| Off-band disability pair + in-band pair | F1 | Exact-support `ValueError` |
| Side ID 1, injected next ID/global domain ID 2, one newborn | F2 | Downstream earnings-domain marker `ValueError`; default allocator premise is static |
| Two live rows, one schedule update | F4 | Unmatched carried state becomes `NaN` |
| Full IDs `{1,2,3}`, domain `{1,3}`, seed 1 | F7 | Domain member changes split side |
| Nominal `85+` probability at ages 120/121 | F6 | Age 121 receives probability zero |

The test file is explicitly marked DRAFT and is not registered in any tier manifest. A broader focused synthetic regression over loop, steps, assembly, support, population, panel builders, projected support, earnings domain, scoring, reporting, and runner also completed with `87 passed in 2.94s`; PSID schema/loader tests and the registered execution entry point were not invoked.
