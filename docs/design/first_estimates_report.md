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
  sequence**: the sibling family/engine specs passed explicitly exactly
  as `m6_candidate3_runner.py:1417-1425` does — not via
  `registered_m6_candidate3_inputs.build_input_plan` alone, which wraps
  candidate-2 inputs and does not select candidate-3 specs
  (`scripts/registered_m6_candidate3_inputs.py:19-25`,
  `engine/refit.py:605-618`). Both CandidateSpec sha256s
  (family `734a5b04…cec5`, engine `c9be28a2…acd6`) are asserted against
  the registration-12 pins before any projection draw; a mismatch
  aborts.
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

- **Observed 1967-2012 income years** from the label-verified
  head/spouse panel (`data/family.py:785-843`), annual through the 1997
  wave and biennial after (`data/family.py:51-55`).
- **The biennial gap law**: an unobserved odd income year bracketed by
  two observed years is imputed as their arithmetic mean; bracketed by
  one observed year only (edge), it carries that neighbor. Imputed
  years are flagged; the artifact publishes the imputed-year share of
  each included career.
- **The 2013/2014 seam**: income year 2013 falls in the biennial gap
  (the 2015 wave's 2014 income is post-T\* and unused); 2013 follows
  the gap law using observed 2012 and the engine's initialized 2014
  state (`forward_earnings.py:1411-1468`) as its brackets. 2014 is the
  engine's initialized value.
- **Projected 2015-2022** from the projection slices.
- **Pre-career years**: years before age 22 or before 1967 with no
  observation are zero; the artifact counts persons whose top-35 window
  reaches years before 1967 (oldest cohorts, benefits understated) and
  flags them.

### 3.3 Zero semantics and the coverage rule

`ss.benefits.aime` treats absent years as zero and makes coverage the
caller's responsibility (`ss/benefits.py:100-117`). The registered
inclusion rule for benefit tables: a person is included iff

- their earnings-domain state is complete (the 16,231 no-state and
  6,698 later-entrant persons of
  `gate_m6_candidate3_v1.json:5430-5441` are excluded-and-counted), and
- their career **coverage ratio** — (observed + gap-law-imputed years)
  / (span from max(1967, birth+22) through min(claim year, 2022)) — is
  **≥ 0.80** (frozen prospectively here).

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
backfill, not a modeled award. The report therefore derives a
`claim_origin` field in its own pipeline (no engine change):

- **`modeled_award`**: claim year > first-exposure year and claim age
  reached during the window — a genuine age-crossing draw. These and
  only these populate the new-awards flow table.
- **`opening_backfill`**: drawn claim age ≤ age at first exposure (the
  engine's stamped claim year is fabricated). These persons join the
  opening stock only, with claim ages re-imputed under §6 — the
  engine's stamped year is discarded.
- **`di_conversion`**: any person with a DI-conversion marker (§5).
  Excluded from both tables.

The artifact asserts and publishes cohort disjointness: every claimed
person appears in exactly one origin class; the flow and stock tables
share no person.

## 5. DI conversions (round-1 finding 5)

Runtime eligibility still draws claim ages for `di_converted` persons
(`steps.py:409-447`; pinned by `tests/test_m6_engine_steps.py:179-202`),
and the certified record discloses that partial overlays can degrade a
carried `di_converted=True` to `NaN` read as false
(`m6_projection_engine.md:2337-2347`). The report does not repair the
engine. Its registered rule is conservative: a person is
conversion-excluded if `di_converted` is True **or NaN** at any point
in their trajectory. Both counts (true-flag and NaN-flag exclusions)
publish separately. DI benefit dollars remain out of scope entirely.

## 6. The opening-stock imputation (round-1 finding 7; report-only table)

For `opening_backfill` persons: claim age is drawn from the pinned SSA
Table 6.B5.1 sex-specific distributions
(`data/external/ssa_claim_ages_2014supplement.json`) **through the
engine's own `ClaimingSchedule.distribution` nearest-year snapping**
(`steps.py:360-387` — not the module-level helpers, which can
`KeyError` on this vintage), with entitlement year = birth year + 62
clamped to the table's 1998-2013 coverage. This is non-circular: the
conditioning uses only birth year and sex, both fixed before any draw.
The artifact publishes lower- and upper-endpoint snap counts and
weighted shares (pre-1998 cohorts extrapolate 1998 behavior; that
extrapolation is a named limitation of the stock table). The stock
table is report-only, labeled imputed, and never contributes to a
headline.

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
   by the statutory COLA series from the full-actuals parameter bundle
   (§8). If the loader cannot supply the actual COLA series for
   2015-2022, the run aborts at preparation — no silent skip.
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
- preparation-time assertions: actual NAWI present through 2020,
  actual wage bases through 2022, and the combined OASDI payroll rate
  equal to 12.4% with both 6.2% legs read from the parameter tree —
  **no fallback constant permitted** (M2's silent 12.4% fallback,
  `m2_pseudo_projection.py:309-343`, is explicitly forbidden here);
- the COLA series requirement of §7.4;
- the projection's own `phase.bundle` is never consulted by the
  statutory pipeline.

## 9. The revenue ledger

M2's per-person arithmetic (`m2_pseudo_projection.py:452-494`) with the
registered replacements:

- **nominal dollars by calendar year 2015-2022** (a CPI constant-dollar
  appendix column may accompany; no common-NAWI indexing);
- taxable payroll `min(labor_income_proxy, wage_base(year))` and
  contributions `weight × 0.124-from-tree × taxable` on realized
  projected person-years only — no survival or claim-PMF expectation
  layers (verified complete by the round-1 review);
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

The artifact's gap block **reproduces the certified record's own
`not_certified`/unavailable disclosures verbatim and classifies each
as material-here or inapplicable-here**, including (from the round-1
review's enumeration): scheduled realized openers (material —
conditioning of the reproduction object); widowhood limitations
(material — survivor composition affects who is present); open
additions/immigration absent (material); lag-5 persistence and stock
margins (material context for earnings paths); the 65+ remarriage tail
(classified); earnings survivorship (material); full-window model
selection and the unavailable redrawn-seed comparison (material
context); no gate-1 transfer (restated); the F4-F11 live-consumer
disclosure ledger items that touch claiming and benefits
(`m6_projection_engine.md:2337-2353` — each item listed and
classified); mortality drift uncertified; families B/C ungated;
2020-2022 shock window report-only; mechanical claiming with the
1998-2013 table; M4 not DI adjudication; alignment `not_computed`;
domain and coverage exclusions per §3.3; spouse/survivor benefits out
of scope; levels unanchored (no committed annual SSA level series —
the registered anchor extraction is the successor step).

## 11. Artifact, tests, publication, ceremony (round-1 finding 10)

- `runs/first_estimates_v1.json` via
  `artifacts.write_new(..., sidecar=True)` (the default is
  sidecar=False — set explicitly; `artifacts.py:13-17`): identity,
  schema version, registration reference, the full configuration echo
  (spec shas, seeds, draw indices, parameter-bundle hashes, environment
  sidecar), per-draw and aggregate tables, origin-class and exclusion
  counts, endpoint-snap counts, the §10 gap block with citations, and
  `certifies_nothing` scope statements.
- **Re-execution path rule**: `write_new` refuses an existing path, so
  a disclosed re-execution publishes to `v1` only if no artifact
  exists; if a defective `v1` has already been published, the
  re-execution writes `v2` with `v1` immutable and cross-linked, and
  the defect disclosed in `v2`. (New rule — this class's own, not
  inherited.)
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
