# The first estimates report: statutory-formula benefit and revenue estimates on the candidate-3 reproduction panel

- **Status:** `DRAFT_NOT_OPERATIVE`, revision 2 — rewritten against the
  round-1 adversarial referee review (PR #285 record, ten findings, all
  accepted). Submitted for round 2. Nothing here authorizes a run.
- **Resolves:** forecast ledger entry 8 — "end-to-end benefit and
  revenue estimates computed on projected earnings/demographic histories
  from the certified engine, published in-repo with disclosed gaps (no
  immigration, pre-alignment acceptable)"
  (`docs/forecasts/timeline_ledger.json`, p50 2026-08-02).
- **Class:** **registered estimates report** — a class this document
  charters (round-1 finding 10: the M2/W2 precedents are similar but
  not identical, so the class is defined here rather than claimed by
  reference): registered configuration before execution; one run; one
  disclosed re-execution; append-only publication regardless of
  results; no gates.yaml surface, no floors, no verdict; every output
  labeled with its evidential status.
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
- **The 2013/2014 seam** (corrected per round 2): the 2014 value is the
  engine's initialized state, which the registered inputs seed from the
  realized 2014 boundary anchor
  (`registered_m6_candidate2_inputs.py:9-17,235-236`;
  `forward_earnings.py:1411-1463`). Income year 2013 follows the gap
  law bracketed by observed 2012 and that 2014 value; where 2012 is
  itself unobserved, 2013 carries the 2014 neighbor and both years'
  imputation flags are set.
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
caller's responsibility (`ss/benefits.py:100-117`). The registered
inclusion rule for benefit tables: a person is included iff

- their earnings-domain state is complete (the 16,231 no-state and
  6,698 later-entrant persons of
  `gate_m6_candidate3_v1.json:5430-5441` are excluded-and-counted), and
- their career **coverage ratio** — (observed + gap-law-imputed +
  projected years) / (span from max(1968, birth+22) through
  min(claim year, 2022)) — is **≥ 0.80** (frozen prospectively here;
  the numerator includes projected years per round-2 fresh finding 3,
  so a career is penalized only for genuinely unknown spans).

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
- **`di_conversion`**: any person conversion-classified under §5.
  Excluded from both tables.

The artifact asserts and publishes cohort disjointness: every claimed
person appears in exactly one origin class; the flow and stock tables
share no person.

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
is therefore a **three-way classification** computed per person over
the trajectory, with the pipeline persisting an `ever_di_converted`
derivation:

- **structural not-applicable**: `NaN` only in slices where the field
  is structurally absent (the 2014 seed), with no later True and no
  later degradation — treated as not converted; counted.
- **overlay-unknown**: a `NaN` appearing after the person has carried a
  concrete value (the disclosed overlay degradation) — excluded from
  benefit tables and counted separately.
- **confirmed conversion**: True in any slice — excluded and counted.

DI benefit dollars remain out of scope entirely.

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

- **The PMF is truncated and renormalized to ages ≤ the person's age
  at first exposure** before drawing — an opening-stock member cannot
  be imputed a claim age later than the age at which the engine
  observed them unclaimed-then-backfilled; the imputed claim year is
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
   by the statutory COLA series. **Round-2 correction: no COLA series
   exists in the current loader** (`ss/params.py:65-85,206-355`
   exposes NAWI, wage base, bend factors, FRA, and reduction/credit
   rates only). The implementation PR therefore adds a **pinned COLA
   loader** (extending `SSAParameters` or a sibling loader) reading
   the named COLA parameter path in the pinned policyengine-us 1.752.2
   tree, source/path/hash recorded, with tests — and with **coverage
   asserted from the earliest included eligibility year (opening-stock
   cohorts reach back well before 2015) through 2022**. Preparation
   aborts if any required COLA year is absent — no silent skip, no
   fallback constant.
5. **No recomputation (registered simplification)**: post-claim
   earnings do not trigger AERO-style recomputation. The artifact
   counts included claimants with positive post-claim earnings — the
   population whose benefits this understates — and the limitation is
   in the gap block.
6. **Payment-year convention (registered simplification)**: an
   included claimant contributes 12 × the COLA-stepped adjusted PIA
   for each calendar year from the claim year through the last year
   they are present in the survivor panel (deaths are realized
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
- **round-2 correction — `SSAParameters` exposes neither payroll-rate
  legs nor COLA** (`ss/params.py:65-85`): the implementation PR adds
  **two pinned parameter-tree loaders** — the employee and employer
  OASDI rate legs, and the COLA series of §7.4 — each reading a named
  path in the pinned 1.752.2 tree, path and content-hash recorded in
  the artifact, with tests; exact path selection is an implementation
  matter gated by that PR's own referee round;
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
| Open additions / immigration absent | material |
| Lag-5 persistence unscored | material context for earnings paths |
| Stock margins unscored | material context |
| 65+ remarriage tail limitation | material context — presence of older married persons |
| Earnings survivorship | material — earnings-conditioned mortality untested |
| Full-window model selection | material context |
| Redrawn-seed comparison unavailable | material context |
| No gate-1 backward-law transfer | restated verbatim |
| F4-F11 live-consumer ledger (`m6_projection_engine.md:2337-2353`) | each item listed; the claiming- and benefit-touching items material, others classified inapplicable with reasons |
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
- **The canonical execution rule** (round-2 prescription, adopted): one
  registered run; `publishes_regardless`; `no_self_rescue`; at most
  one coordinator-adjudicated, report-first retry **solely for an
  external pre-output failure yielding no estimate-bearing
  information**. A published `v1`, any changed configuration byte, or
  a second failure of any kind requires **fresh registration** — there
  is no same-ceremony v2 path. Preparation, invariant, or compute
  aborts publish an append-only INVALID/incident record before any
  further step.
- Tests: schema/invariant validation plus a committed-fixture rebuild
  test of the join, origin classification, and ledger arithmetic
  without re-running the projection (new work modeled on, not copied
  from, the W2 tests).
- Paper: a "First estimates" section after the artifact merges,
  narrating the registered procedure, the labels, and the gap block.
- Ceremony: this design ratified (round 2 → fixes → verify → merge) →
  implementation PR (referee-gated; no run) → registration on a fresh
  issue (configuration frozen; one-run + one-disclosed-re-execution
  terms restated verbatim; the incident-5028176439 execution topology
  mandatory) → one registered run (hours-scale: twenty draws of a
  2014→2022 projection) → publication regardless → entry 8 resolves at
  the publication PR's merge.

## 12. What this unlocks

The first in-repo, fully labeled answer to what the statutory formulas
yield on histories the engine projected — every simplification counted
and on the page. Successor steps in leverage order: the annual
SSA/Trustees level-anchor extraction; the covered-earnings correction
that retires the proxy label; the W1 population bridge (frame-relative
→ national); the spouse/survivor entitlement adapter; behavioral
claiming; the `FORWARD` production object.
