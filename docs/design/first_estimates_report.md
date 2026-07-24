# The first estimates report: statutory-formula benefit and revenue estimates on the candidate-3 reproduction panel

- **Status:** `DRAFT_NOT_OPERATIVE`, revision 5 — hardened through
  four adversarial referee rounds (PR #285 record: rounds 1-4, every
  finding accepted). Submitted for round 5. Nothing here authorizes a
  run.
- **Resolves:** forecast ledger entry 8 — "end-to-end benefit and
  revenue estimates computed on projected earnings/demographic histories
  from the certified engine, published in-repo with disclosed gaps (no
  immigration, pre-alignment acceptable)"
  (`docs/forecasts/timeline_ledger.json`, p50 2026-08-02).
- **Class:** **registered estimates report** — a class this document
  charters (round-1 finding 10: the M2/W2 precedents are similar but
  not identical, so the class is defined here rather than claimed by
  reference): registered configuration before execution; execution
  governed solely by §11's canonical rule; append-only publication
  regardless of results; no gates.yaml surface, no floors, no verdict;
  every output labeled with its evidential status.
- **Evidence base:** the machinery survey and the round-1 review
  (PR #285 record), both verified against master `65695bb`. File:line
  citations are to that commit.

## 1. What this report is (relabeled per round-1 finding 1)

This report computes statutory Social Security formulas — the exact
AIME/PIA oracle and a payroll-contribution ledger — over the
**candidate-3 `GATED_REALIZED` reproduction panel**: the same
projection object the ratified gate run scored, projected unsplit. That
object seeds its 2014 state and its 2017/2019 scheduled openers from
realized PSID anchors (`m6_projection_engine.md:1562-1597`); the
genuine `FORWARD` production mode (redrawn transport state, endogenous
widowhood, synthetic disability support) is explicitly unbuilt
(`m6_projection_engine.md:1599-1608`). The gate certificate covers the
registered 2016/2018 earnings and 2015-2019 flow surfaces only;
AIME, PIA, and claiming outputs are report-only under the design's own
law (`m6_projection_engine.md:2501-2508`).

Accordingly, nothing in this report is "certified forward production."
It is the first end-to-end exercise of the statutory pipeline on
engine-projected histories — the entry-8 deliverable — with three
labels carried on every table:

- **frame-relative** (closed PSID frame; not national dollars),
- **pre-alignment** (no calibration to any administrative level), and
- **labor-income proxy** (round-1 finding 2: PSID head/spouse labor
  income is not verified OASDI-covered earnings; §3.4).

The closing claim is correspondingly narrow: this is what the statutory
formulas yield on the reproduction panel — not an estimate of what
Social Security pays and collects.

## 2. The projection object (frozen; binding per round-1 finding 9)

- Candidate-3 configuration bound **by the gate runner's own call
  sequence, frozen in full** (round-2 completion): the sibling
  family/engine specs passed explicitly as
  `m6_candidate3_runner.py:1417-1425` does — not via
  `registered_m6_candidate3_inputs.build_input_plan` alone, which wraps
  candidate-2 inputs and does not select candidate-3 specs
  (`scripts/registered_m6_candidate3_inputs.py:19-25`,
  `engine/refit.py:605-618`) — followed by the runner's own
  fit/preflight/materialization ordering and its fitted phase-lineage
  assertion (`m6_candidate3_runner.py:1879-1897,2843-2898`), which the
  implementation reproduces as the normative sequence. Both
  CandidateSpec sha256s (family `734a5b04…cec5`, engine
  `c9be28a2…acd6`) are asserted against the registration-12 pins before
  any projection draw; a mismatch aborts.
- **Unsplit population**: full-population fit and materialization via
  the m6_runner surface (`m6_runner.py:400-455`), projected with
  `_project_side`'s recipe on the unsplit `phase.population`
  (`m6_runner.py:562-589`); the split seeds are a scoring construct
  only (`m6_runner.py:1046-1085`).
- Projection 2014 → 2022, **draw indices 0-19**, where draw index k
  maps to root seed 5200+k per the RNG registry's derivation
  (`rng.py:16-17`, law at `rng.py:41-59`). Per-draw values publish
  alongside across-draw mean and SD for every aggregate.

## 3. The statutory career join (registered laws; round-1 finding 2)

### 3.1 Birth year — source precedence

1. Exact birth year where the marriage-history support carries it;
2. otherwise the inferred `median(period − age)` construction
   (`data/couple_earnings.py:303-323`).

The artifact publishes weighted and unweighted counts by source, and
the inferred-source rows are flagged person-level in the intermediate
frame. Synthetic children carry native `birth_year`
(`steps.py:485-503`).

### 3.2 Career reconstruction — the annual series

Nominal labor income by calendar year assembled as:

- **Observed 1968-2012 income years** from the label-verified
  head/spouse panel (the source begins in income year 1968,
  `data/family.py:25-28`, pinned by `test_family.py:263-272`), annual
  through the 1997 wave and biennial after (`data/family.py:51-55`).
- **The biennial gap law**: an unobserved odd income year bracketed by
  two observed years is imputed as their arithmetic mean; bracketed by
  one observed year only (edge), it carries that neighbor. Imputed
  years are flagged; the artifact publishes the imputed-year share of
  each included career.
- **The 2013/2014 seam** (corrected per rounds 2-3): the 2014 value is
  the engine's initialized state, which the registered inputs seed from
  the realized 2014 boundary anchor
  (`registered_m6_candidate2_inputs.py:9-17,235-236`;
  `forward_earnings.py:1411-1463`); its provenance class is
  `boundary_2014`. Income year 2013 follows the gap law bracketed by
  observed 2012 and that 2014 value. **Missing-2012 disposition
  (round-3 completion)**: where 2012 is unobserved, 2012 remains
  `unknown` (no value is invented for it) and 2013 carries the 2014
  neighbor with its imputation flag set. **Cutoff override (round-5)**:
  for a claim year of 2013, the §3.3 cutoff-before-imputation law
  governs — the 2014 value is outside the as-of set, so with 2012
  unobserved, 2013 is `unknown` (zero contribution, flagged); the seam
  carry applies only when 2014 is inside the as-of restriction.
- **Per-year provenance enum (round-3 completion)**: every career year
  carries exactly one class — `observed`, `gap_imputed`,
  `boundary_2014`, `projected`, or `unknown` — and the artifact
  publishes the class mix per included career. `unknown` years inside
  an included career contribute zero to AIME, disclosed.
- **Projected 2015-2022** from the projection slices, **with the
  odd-year carry law disclosed** (round-2 fresh finding 5): the engine
  draws even-year earnings and carries the prior even year into odd
  years (2015 repeats 2014, 2017 repeats 2016, and so on —
  `forward_earnings.py:1686-1717,1890-1903`). Every annual benefit and
  revenue table states this law, and the artifact publishes the
  affected-year share of each aggregate; a biennial-presentation
  companion column accompanies the annual tables.
- **Pre-career years**: years before age 22 or before 1968 with no
  observation are zero; the artifact counts persons whose top-35 window
  reaches years before 1968 (oldest cohorts, benefits understated) and
  flags them.

### 3.3 Zero semantics and the coverage rule

`ss.benefits.aime` treats absent years as zero and makes coverage the
caller's responsibility (`ss/benefits.py:100-117`). Two ordering and
domain laws frozen per round 4:

- **Cutoff before imputation** (the information-as-of principle
  extends to imputation itself): the as-of restriction to income years
  ≤ the claim year is applied FIRST; the gap law then runs on the
  restricted series only, so a gap year at the restricted edge carries
  its observed earlier neighbor and no post-claim year ever influences
  an imputed value.
- **The eligibility-era bound**: benefit tables include only persons
  with eligibility year (birth + 62) **≥ 1979** — the statutory era
  the oracle implements (its bend-point computation is the
  post-1977-amendments 1979-base formula, `ss/params.py:138`; the
  frame contains cohorts eligible as early as the 1970s, which the
  formula does not cover). Earlier-eligibility persons are
  excluded-and-counted. This bound also fixes the §7.4 COLA coverage
  span at 1979-2022.
- **Empty-span disposition**: a person whose coverage span is empty —
  the exact predicate is max(1968, birth + 22) > min(claim year, 2022)
  — is excluded-and-counted, a defined exclusion reason, not an abort.
  (The eligibility-era bound makes this vacuous for included persons;
  the rule exists as a fail-safe and sits at precedence position 4 of
  the canonical inclusion law.)

**The canonical inclusion law** (round-5 consolidation — one rule, all
predicates, evaluated in this precedence order; the FIRST failing
predicate is the person's single published exclusion reason):

1. non-DI under §5 (else excluded as `di_conversion` / `di_unknown`);
2. earnings-domain state complete;
3. eligibility era: birth + 62 ≥ 1979;
4. nonempty span: max(1968, birth + 22) ≤ min(claim year, 2022);
5. **chronology invariant** (round-5 fresh finding 1): birth + 62 ≤
   claim year, asserted with the report's own §3.1 birth year against
   the operative claim year (engine-stamped for modeled awards, imputed
   for opening stock) — violations excluded-and-counted as
   `chronology_inconsistent`, which also closes the COLA span at
   1979-2022 from both ends;
6. coverage ratio ≥ 0.80.

Restated for clarity, the earlier prose predicates are subsumed: a
person is included iff

- their earnings-domain state is complete (the no-earnings-state
  persons are excluded-and-counted; **one entrant law** per round 3:
  the operative counts are the §10 re-derived explicit-row counts —
  the candidate-3 artifact's 6,698 figure is cited as context only,
  never as an operative rule), and
- their career **coverage ratio** — years classed `observed`,
  `gap_imputed`, `boundary_2014`, or `projected` within the inclusive
  span, divided by the span from max(1968, birth+22) through
  min(claim year, 2022) — is **≥ 0.80** (frozen prospectively;
  numerator years are counted only inside the denominator interval,
  and the implementation asserts 0 ≤ ratio ≤ 1).

Excluded persons are published as weighted and unweighted counts by
exclusion reason. Revenue tables include every person with projected
in-window earnings (no career completeness needed).

### 3.4 Covered-earnings interpretation

PSID labor income is a **proxy** for OASDI-covered earnings: it
includes non-covered employment and self-employment reporting
differences and misses covered amounts PSID under-reports. All benefit
and contribution outputs carry the proxy label; the report makes no
covered-earnings claim. (The successor step that would retire this
label — a covered-earnings correction model — is out of scope and
named in §10.)

## 4. Claim origin and cohort disjointness (round-1 finding 3)

The engine draws a claim age on first exposure for every unassigned
person 50+, and anyone already past the drawn age is stamped claimed in
the **current projection year** (`steps.py:409-447`) — a mechanical
backfill, not a modeled award. The report derives a `claim_origin`
field in its own pipeline (no engine change), by a **frozen
derivation** (round-2 resolution): origin is computed from the first
post-start slice in which the person carries a non-null `claim_age`
(`steps.py:409-447`; slices per `loop.py:141-152`), comparing the drawn
claim age with the person's age in that slice:

- **`modeled_award`**: drawn claim age **≥** age at first exposure —
  the claim, whenever stamped, corresponds to an age the person
  reaches during the window (equality means the crossing occurs in the
  first exposure year and is a genuine modeled award, not a
  fabrication — round-2 correction). These and only these populate the
  new-awards flow table, with claim year = the engine's stamped year.
- **`opening_backfill`**: drawn claim age **<** age at first exposure —
  the engine's stamped claim year is fabricated. These persons join
  the opening stock only, with claim age AND claim year re-imputed
  under §6; both engine-stamped values are discarded.
- **DI-excluded persons carry no origin** (round-4 consistency fix):
  the §5 precedence law runs FIRST; `di_conversion` and `di_unknown`
  persons are excluded before origin assignment. They may carry engine
  claim state (a `di_unknown` person can still be stamped claimed,
  `steps.py:433`); that state is never consumed. The origin partition
  is asserted complete over **non-DI claimed persons only**.

The artifact asserts and publishes cohort disjointness: every non-DI
claimed person appears in exactly one origin class; the flow and stock
tables share no person.

## 5. DI conversions (round-1 finding 5)

Runtime eligibility still draws claim ages for `di_converted` persons
(`steps.py:409-447`; pinned by `tests/test_m6_engine_steps.py:179-202`),
and the certified record discloses that partial overlays can degrade a
carried `di_converted=True` to `NaN` read as false
(`m6_projection_engine.md:2337-2347`). Round 2 additionally showed that
a blanket "True or NaN anywhere" rule would empty the benefit universe:
the published 2014 starting slice carries no `di_converted` field at
all, so every bulk incumbent has a structural `NaN`
(`m6_population.py:32-49,264-350`; `loop.py:218`). The registered rule
is therefore a **precedence law over the whole trajectory** (round-3
completion — exhaustive and mutually exclusive, since partial left
overlays can also produce a post-start `NaN` for a person who never
carried a concrete value, `assembly.py:174-192,380-388`):

1. **`di_conversion`**: True in any slice — excluded and counted.
   (Ever-True takes precedence; a later True→NaN degradation does not
   demote a confirmed conversion.)
2. **`di_unknown`**: otherwise, any **post-start** missing observation
   — whether after a concrete value (overlay degradation) or without
   one ever appearing — excluded from benefit tables and counted
   separately. Excluded persons need no `claim_origin`.
3. **non-DI**: otherwise (including 2014-seed structural absence with
   concrete non-True values throughout the projection) — included.

The implementation asserts that this partition is complete: every
person maps to exactly one class, and every benefit-table person is
non-DI with exactly one §4 origin. DI benefit dollars remain out of
scope entirely.

## 6. The opening-stock imputation (round-1 finding 7; report-only table)

For `opening_backfill` persons: claim age is drawn from the pinned SSA
Table 6.B5.1 sex-specific distributions
(`data/external/ssa_claim_ages_2014supplement.json`) **through the
engine's own `ClaimingSchedule.distribution` nearest-year snapping**
(`steps.py:360-387`, which selects among its own 1998-2013 keys — not
the module-level helpers, which can `KeyError` on this vintage), keyed
by the person's **cohort-eligibility year** (birth year + 62 — a
deliberate proxy for actual entitlement year, disclosed as such;
round-2 correction) clamped to the table's 1998-2013 coverage. Two
round-2 coherence fixes are law:

- **The PMF is truncated and renormalized to ages strictly below the
  person's age at first exposure** before drawing (round-3 correction:
  `<` not `≤` — equality is a §4 modeled award, so an equality
  imputation would collide with the origin law). Opening-stock
  membership itself proves the original draw was strictly below the
  exposure age, and exposure is therefore at least 63 against a table
  whose support starts at 62 with positive age-62 mass in every pinned
  sex/year row — but the implementation still **fails closed** if the
  truncated mass is empty for any person. The imputed claim year is
  `birth_year + imputed_age`.
- **A dedicated deterministic RNG namespace**, keyed by person
  identifier under a registered stock-imputation root seed (disjoint
  from the projection's 5200-series), so the imputation is
  reproducible and independent of projection draws.

This is non-circular: the conditioning uses only birth year, sex, and
first-exposure age, all fixed before the imputation draw. The artifact
publishes lower- and upper-endpoint snap counts and weighted shares
(pre-1998 cohorts extrapolate 1998 behavior; a named limitation). The
stock table is report-only, labeled imputed, and never contributes to
a headline.

## 7. The benefit ledger (information-as-of; round-1 finding 4)

Per included claimant, the ledger is year-by-year:

1. **As-of cutoff**: the AIME history passed to `ss.benefits.aime`
   contains only income years **≤ the claim year**. No later year
   enters any award computation.
2. **PIA** at eligibility year (birth + 62) via `ss.benefits.pia`.
3. **Claim-age adjustment** via
   `benefit_factor(claim_age × 12, birth_year, params)` — months, not
   the engine's integer years (`claiming.py:338-376`).
4. **COLA**: the eligibility-year PIA is stepped to each payment year
   by the statutory COLA series. **Round-3 correction: the pinned
   PE-US 1.752.2 tree cannot supply this** — its only SSA uprating
   file begins at 2022 and is a CPI-W level series, not historical
   COLAs. The COLA source is therefore a **new committed extraction**:
   `data/external/ssa_cola_history.json`, transcribed from SSA's
   published automatic-determination COLA history, with provenance
   (URL, retrieval date, vintage), content hash, and a
   transcription-validating test — the Table 6.B5.1 committed-anchor
   pattern. Frozen application semantics: each COLA is the
   December-effective percentage of its determination year; the PIA at
   eligibility is compounded by the COLAs of the eligibility year
   through the year before the payment year, rounding down to the next
   lower dime at each step; the implementation PR's referee verifies
   this convention against 42 USC 415(i) before ratifying. Coverage is
   asserted for 1979-2022 exactly (the §3.3 eligibility-era bound
   fixes the earliest included eligibility year at 1979); preparation
   aborts if any required COLA year is absent — no silent skip, no
   fallback constant.
5. **No recomputation (registered simplification)**: post-claim
   earnings do not trigger AERO-style recomputation. The artifact
   counts included claimants with positive post-claim earnings — the
   population whose benefits this understates — and the limitation is
   in the gap block.
6. **Payment-year convention (registered simplification)**: an
   included claimant contributes 12 × the COLA-stepped adjusted PIA
   for each calendar year in the intersection of [claim year,
   last-present year], [2015, 2022], and the person's actual presence
   years (round-5 fresh finding 2: scheduled 2017/2019 openers pay
   nothing before first presence; the imputed or stamped claim year
   sets the benefit LEVEL through the claim-age adjustment, never a
   payment year outside the window or before presence; deaths are realized
   removals; no survival expectation is applied — the double-counting
   rule the round-1 review verified complete). Partial first/last-year
   months are not modeled; the convention is disclosed. Output is
   labeled **"annualized statutory benefit, eligibility-PIA with COLA,
   no recomputation"** — not administrative payment dollars.

Primary table: `modeled_award` flow — weighted award counts by year,
average monthly benefit at award, aggregate frame-relative annualized
benefits by calendar year, per-draw with mean/SD. Secondary
(report-only): the §6 stock analogue. Context ratio (report-only): the
simulated average monthly benefit at award beside the published SSA
average for the corresponding years, source pinned, explicitly not an
anchor.

## 8. Parameters: the full-actuals oracle bundle (round-1 finding 6)

The M6 input factory's parameter bundle **replaces post-2014 NAWI with
projected values and deletes post-2014 wage-base changes**
(`registered_m6_inputs.py:184-235`; by design,
`m6_projection_engine.md:3022-3063`) — correct for the fit, wrong for
statutory computation on realized-calendar years. The report registers
a **second, independent parameter load**:

- `ss.params.load_ssa_parameters` (the full loader,
  `ss/params.py:196-227`) against the pinned policyengine-us 1.752.2
  checkout, path recorded and content hash-verified;
- **rate legs named** (round-3 completion): the implementation PR adds
  a pinned loader for the employee and employer OASDI rate legs at
  `gov/irs/payroll/social_security/rate/employee.yaml` and
  `gov/irs/payroll/social_security/rate/employer.yaml` in the pinned
  1.752.2 tree, path and content-hash recorded in the artifact, with
  tests. The COLA series comes from the §7.4 committed extraction, not
  the tree;
- preparation-time assertions: actual NAWI present through 2020,
  actual wage bases through 2022, both rate legs equal to 6.2% and the
  combined rate 12.4% as read from the tree — **no fallback constant
  permitted** (M2's silent 12.4% fallback,
  `m2_pseudo_projection.py:309-343`, is explicitly forbidden here);
  COLA coverage per §7.4;
- the projection's own `phase.bundle` is never consulted by the
  statutory pipeline.

## 9. The revenue ledger

M2's per-person arithmetic (`m2_pseudo_projection.py:452-494`) with the
registered replacements:

- **nominal dollars by calendar year 2015-2022** (a CPI constant-dollar
  appendix column may accompany; no common-NAWI indexing);
- taxable payroll `min(labor_income_proxy, wage_base(year))` and
  contributions `weight × (rate legs per §8's pinned loaders) ×
  taxable` on realized projected person-years only — no survival or
  claim-PMF expectation layers (verified complete by the round-1
  review);
- the odd-year carry law of §3.2 disclosed on every annual revenue
  table, with the biennial-presentation companion;
- the proxy label of §3.4 on every revenue number.

Published: weighted taxable payroll and contributions by year, per-draw
mean/SD, weighted covered-earner counts.

## 10. Weights, labels, and the gap block (round-1 finding 8)

Weights: fixed start-wave PSID cross-sectional weights
(`m6_cells.py:113-140`, `support.py:54-133`), synthetic children
inheriting the mother's weight; every table carries the frame's
weighted count and the three §1 labels. The W1 deployment frame's
national weighting (`deployment_frame.py:371-418`) is the successor
alignment path, unused here.

The gap block is **frozen here** (round-2 correction: in-design, not
promised for later), with each item's classification:

| Disclosure (certified-record source) | Classification here |
|---|---|
| Scheduled realized 2017/2019 openers condition the object | material — the reproduction panel is anchored, not forward |
| Widowhood limitations | material — survivor composition affects presence |
| Open additions — certified sentence quoted exactly: "The gate covers the closed panel only; synthetic births, immigrant cohorts, and other open additions remain report-only." | material |
| Lag-5 persistence unscored | material context for earnings paths |
| Stock margins unscored | material context |
| 65+ remarriage tail limitation | material context — presence of older married persons |
| Earnings survivorship — certified sentence quoted exactly: "Gated earnings use realized support and do not certify mortality's effect on the earnings composition through survivorship." | material |
| Full-window model selection | material context |
| Redrawn-seed comparison unavailable | material context |
| The artifact's earnings-certification string quoted exactly: "M6-first-certified forward earnings law; no gate_1 backward-law certificate transfers" | restated verbatim |
| F4 — partial overlay: `_merge_period_columns` drops named columns before left-merging, so unmatched live state becomes `NaN` (pinned: carried `di_converted=True` read as no-conversion) | **material** — directly motivates the §5 precedence law and the `di_unknown` class |
| F5 — exact-anchor household seed gap (minors reaching 15 later and source-gap adults never enter the household domain) | inapplicable to presence (certified: household fields feed no locked cell and are not serialized; roster presence is unaffected) — material only if household-domain counts are quoted, and then the certified excluded/domain counts publish first |
| F6 — closed "85+" band (nominal 85+ ends at 120; uncovered ages get p=0) | material context — oldest-old presence in benefit-years |
| F8 — entrant classification (`anchor_wave > 2015 & ~domain` treated as row existence) | **material** — the reason §3.3/§10 re-derive the entrant count from explicit earnings rows |
| F9 — candidate-9/live-roster reconciliation (household fields do not reconcile mortality-thinned members or newborns) | inapplicable — household composition fields are not consumed by this report |
| F9 sub-item — `coresident_spouse` carried for a person whose spouse was removed by simulated mortality | inapplicable here (household column unconsumed), listed by name as the certified record requires |
| F10 — entrant schema NAs (`synthetic_entry=NA` inheritance; certified surface: future panel/schema consumers) | this report is such a consumer — it identifies synthetic persons by ID-set difference per the certified mechanism and never reads this field; classified handled-by-construction, listed |
| F11 — fertility-domain coverage (births draw over `state.marital_ids` only; certified surface: family-B birth counts, no gated cell) | inapplicable to benefit tables (no in-window newborn claims); for revenue person-years the certified fertility-domain denominator disclosure is restated, not extended |
| Certified `forward_projection_2100_extrapolation` limitation | material — restated: nothing here extends past 2022, and nothing certifies any longer horizon |
| Mortality drift uncertified | material |
| Families B/C ungated | material |
| 2020-2022 shock window report-only | material — in-window years |
| Mechanical claiming, 1998-2013 table | material |
| M4 is not DI adjudication | material — DI out of scope |
| Alignment `not_computed`; scored path unaligned | material |
| Domain and coverage exclusions (§3.3) | material; counts published |
| Odd-year earnings carry law (§3.2) | material — annual tables |
| Spouse/survivor benefits out of scope | material |
| Levels unanchored — no committed annual SSA level series | material; the registered anchor extraction is the successor step |

**Entrant-count re-derivation** (round-2 fresh finding 7): the
candidate-3 artifact's 6,698 later-entrant figure is computed as later
anchor plus outside-domain, overlaps the 16,231 no-state count, and
includes persons with no earnings row (`m6_runner.py:1144-1160`). The
report derives its own entrant count from explicit 2016/2018 earnings
rows and publishes mutually exclusive exclusion counts (or the
overlap, stated).

## 11. Artifact, tests, publication, ceremony (round-1 finding 10)

- `runs/first_estimates_v1.json` via `artifacts.write_new` with its
  **paired `.env.json` sidecar** enabled (round-2 wording fix:
  `sidecar=True` writes a separate environment/contract file and
  requires BOTH `v1` and `v1.env.json` to be absent,
  `artifacts.py:45-79`; the primary artifact records the sidecar's
  content hash so the pair is integrity-bound): identity, schema
  version, registration reference, the full configuration echo (spec
  shas, seeds, draw indices, parameter-bundle hashes), per-draw and
  aggregate tables, origin-class and exclusion counts, endpoint-snap
  counts, the §10 gap block, and `certifies_nothing` scope statements.
- **The canonical execution rule — the sole normative wording,
  document-wide** (round-3 unification): one registered run;
  `publishes_regardless`; `no_self_rescue`; at most one
  coordinator-adjudicated, report-first retry **solely for an external
  pre-output failure yielding no estimate-bearing information**. A
  published `v1`, any changed configuration byte, or a second failure
  of any kind requires **fresh registration** — there is no
  same-ceremony v2 path. Any other clause in this document or the
  eventual registration that appears to describe execution defers to
  this paragraph.
- **The INVALID/incident record** (schema frozen per round 4): a
  preparation, invariant, or compute abort publishes an append-only
  record at `runs/first_estimates_incident_<n>.json` (n = 1, 2, … in
  order) with exactly these keys —
  `schema_version` (the string `"first_estimates_incident.v1"`),
  `incident_index` (integer n), `timestamp_utc` (ISO-8601 with `Z`),
  `phase` (one of `"preparation" | "invariant" | "compute" |
  "publication"`), `reason` (a machine string), `reason_detail` (free
  text), `registration_reference` (issue/comment id), and
  `configuration_echo` (the same object the artifact would carry);
  `registration_reference` is a JSON string. A **publication abort**
  (failure after compute, before or during artifact write) also
  triggers an incident record, with `phase: "publication"` and, if a
  partial artifact exists, its path referenced. The record carries
  **no estimate-bearing value of any kind**; the schema-validation
  test enforces the exact key set, the types, and that no key outside
  the schema and no numeric array or table of any kind is present. Incident records never occupy the `v1` path and are
  cross-referenced by any later artifact or fresh registration.
- **Execution topology — the complete launcher contract** (round-4
  completion; the launchd user-domain lineage adjudicated and verified
  on the candidate-3 record, issue #42 comments 5065343857 and
  5065367143): a one-shot LaunchAgent with `RunAtLoad=true`,
  `AbandonProcessGroup=true`, `ProcessType=Interactive`, and **no
  KeepAlive of any kind**, whose program is a launcher script that
  `nohup`-spawns the runner and **exits** (the spawned process must not
  be the agent's main process, or agent removal kills it); the runner
  command wraps itself in `caffeinate -sim` so the sleep assertion
  belongs to the runner's own lifetime; the agent is bootstrapped once,
  the runner verified healthy, and the agent booted out with its plist
  deleted — persistence-object lifetime under two minutes; the runner
  ends as a launchd-domain orphan (parent pid 1) in no application
  coalition; no network-dependent parent; publication is performed by
  the coordinator after exit.
- Tests: schema/invariant validation plus a committed-fixture rebuild
  test of the join, origin classification, and ledger arithmetic
  without re-running the projection (new work modeled on, not copied
  from, the W2 tests).
- Paper: a "First estimates" section after the artifact merges,
  narrating the registered procedure, the labels, and the gap block.
- Ceremony: this design ratified (referee rounds → fixes → verify →
  merge) → implementation PR (referee-gated; no run) → registration on
  a fresh issue (configuration frozen; §11's canonical execution rule
  restated verbatim) → one registered run (hours-scale: twenty draws
  of a 2014→2022 projection) → publication regardless → entry 8
  resolves at the publication PR's merge.

## 12. What this unlocks

The first in-repo, fully labeled answer to what the statutory formulas
yield on histories the engine projected — every simplification counted
and on the page. Successor steps in leverage order: the annual
SSA/Trustees level-anchor extraction; the covered-earnings correction
that retires the proxy label; the W1 population bridge (frame-relative
→ national); the spouse/survivor entitlement adapter; behavioral
claiming; the `FORWARD` production object.
