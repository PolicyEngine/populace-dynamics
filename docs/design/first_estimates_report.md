# The first estimates report: end-to-end benefit and revenue aggregates on certified projected histories

- **Status:** `DRAFT_NOT_OPERATIVE` — design proposal for the
  forecast-ledger entry-8 deliverable, submitted for adversarial referee
  review under the standing ceremony (proposal → referee → fixes →
  verification → ratify-by-merge). Nothing here authorizes a run.
- **Resolves:** forecast ledger entry 8 — "end-to-end benefit and
  revenue estimates computed on projected earnings/demographic histories
  from the certified engine, published in-repo with disclosed gaps (no
  immigration, pre-alignment acceptable)"
  (`docs/forecasts/timeline_ledger.json`, p50 2026-08-02).
- **Class:** registered, **reported-not-gated** (the M2/W2 class). No
  gates.yaml surface, no floors, no verdict. The report registers its
  configuration before execution, publishes append-only through
  `artifacts.write_new`, and discloses every named gap; it does not
  certify benefit or revenue levels.
- **Evidence base:** the post-PASS machinery survey (coordinator lane
  record, 2026-07-24), verified against master `65695bb`. File:line
  citations below are to that commit.

## 1. What this report is and is not

This is the campaign's first DynaSim-style output: statutory
retired-worker benefits and payroll-tax revenue computed person-year by
person-year on **simulated forward histories drawn from the certified
candidate-3 engine**, aggregated with the certified object's own
weights, published with uncertainty across draws. It is the first time
the ratified forward law, the statutory AIME/PIA oracle, and the
revenue ledger meet in one pipeline.

It is **not** a solvency projection, a national total, or a calibrated
estimate. Dollars are **frame-relative** (the closed PSID frame whose
weight mass is the certified object's, not the U.S. population),
**pre-alignment** (no calibration to any administrative level), and
horizon-limited (projection through 2022, of which only the registered
2016/2018 earnings and 2015-2019 flow surfaces are certified; later and
report-only surfaces are extrapolation). The entry-8 claim licenses
exactly this shape; the report's front matter repeats it.

## 2. The projection object (frozen)

**Decision: the runnable closed-PSID unsplit projection, not a
populace-frame bridge.** The design doc's production populace start
(`m6_projection_engine.md:362-376`) has no implemented builder; the
runnable certified object is the M6 population projected by the
candidate-3 configuration. Waiting for the W1→M6 bridge would gate the
first estimates on the productization stage; the entry's
"pre-alignment acceptable" clause exists precisely so it does not.
The report labels the frame on every table.

Frozen configuration:

- Candidate-3 registered engine and family specs, byte-identified by the
  registration-12 pins (family
  `734a5b04…cec5`, engine `c9be28a2…acd6` binding q*=0.55, ρ*=-0.60);
  fitted via the registered candidate-3 input adapter
  (`scripts/registered_m6_candidate3_inputs.py`) with its pinned
  PE-US 1.752.2, boundary-year SSA parameters, 2014 claiming table,
  NCHS 2010 mortality, T*=2014, and fit seed 5200.
- **Unsplit population**: the full `phase.population` object (the
  m6_runner fit/materialization path, `m6_runner.py:400-455`), NOT the
  gate's five split-seed partitions. Gate split seeds exist to score
  holdout halves; a production estimate projects everyone once per
  draw.
- Projection 2014 → 2022 via `ProjectionEngine.project`
  (`loop.py:170-266`), **draw indices 0-19** (root seeds 5200-5219) —
  the same draw-index law the candidate-3 run used (`rng.py:16-17`).
  Every aggregate publishes the across-draw mean and SD and the
  per-draw values.
- One disclosed re-execution maximum, `write_new` output
  `runs/first_estimates_v1.json`, publication regardless of how the
  numbers look — the M2/W2 discipline verbatim.

## 3. History construction (the statutory join)

The projected slices carry 2015+ annual earnings, claim state, marital
state, and death-by-removal; they do not carry incumbent birth years,
pre-2014 earnings vectors, or any benefit field (survey §1). The
statutory history for each person is therefore a registered join:

1. **Birth year**: joined from the demographic readers for incumbents
   (the same source the engine's age law consumed); synthetic children
   carry `birth_year` natively (`steps.py:485-503`).
2. **Observed pre-2014 covered earnings**: joined from the label-verified
   PSID head/spouse earnings panel (the fit's own source), nominal
   dollars by calendar year.
3. **Projected 2015-2022 earnings**: from the projection slices.
4. **Missing years**: a year absent from both the observed panel and the
   projection is treated as zero covered earnings **only** for persons
   inside the earnings domain; the AIME caller's zero-semantics
   requirement (`ss/benefits.py:105-111`) is satisfied by construction
   for that set and by exclusion otherwise.
5. **Domain exclusions, counted**: the candidate-3 artifact's own
   disclosure — 16,231 persons with no earnings state and 6,698 later
   earnings entrants whose AIME/PIA levels are understated
   (`gate_m6_candidate3_v1.json:5430-5441`) — becomes an exclusion
   rule: persons without a complete domain history for their top-35
   computation are **excluded from benefit tables and separately
   counted** (weighted and unweighted), rather than included with
   silently understated benefits. The exclusion count is a published
   field. Revenue tables (which need only in-window earnings) include
   everyone with projected earnings.

## 4. Benefit side (scope frozen)

**Own retired-worker benefits only.** The oracle's spouse/survivor
functions exist but no worker/spouse/deceased entitlement adapter does
(survey §6); DI benefit dollars are outside the certified surface (M4
is a work-limitation reproduction, not DI adjudication). Auxiliary and
DI totals are named future work, not silently approximated.

- **Primary table — the new-awards flow.** Persons whose engine-drawn
  claim occurs in 2015-2022 (`claim_age`/`claim_year`,
  `steps.py:390-448`; DI conversions excluded from the behavioral draw
  per the engine law). For each: AIME from the joined nominal history
  (`ss.benefits.aime`), PIA at eligibility year birth+62
  (`ss.benefits.pia`), claim-age adjustment via `benefit_factor`
  (`claiming.py:338-376`), annual benefit = 12 × adjusted PIA in the
  claim year and each subsequent in-window year the person survives
  (deaths are realized removals — **no external survival expectation is
  applied**; the M2 expectation machinery would double-count mortality
  and claiming, survey §3). Published: weighted award counts by year,
  average monthly benefit at award, aggregate frame-relative benefit
  dollars by calendar year 2015-2022, all per-draw with mean/SD.
- **Secondary table (report-only, labeled imputation) — the stock
  analogue.** Persons already benefit-eligible before 2015 lack
  observed claim ages. A registered imputation draws their claim age
  from the pinned SSA Table 6.B5.1 distributions
  (`data/external/ssa_claim_ages_2014supplement.json`) for their
  entitlement year (snapping to the nearest covered year, exactly the
  engine's own convention for post-2013 years), then computes benefits
  as above. This table exists to show order of magnitude for total
  in-payment benefits; it is labeled imputed and excluded from any
  headline.
- **Context ratio (report-only):** the simulated average monthly
  retired-worker benefit at award, next to the published SSA average
  for the corresponding years — a per-person concept in which the
  frame-relative weight mass partially cancels. One number with its
  source pinned; explicitly not an anchor and not a validation claim.

## 5. Revenue side (scope frozen)

M2's ledger arithmetic (`m2_pseudo_projection.py:452-494`) with four
registered replacements per the survey's warnings:

1. **Nominal dollars by calendar year 2015-2022** — not M2's
   common-2048 NAWI indexing (that convention served a 75-year PV
   comparison; an eight-year in-window ledger reports nominal, with
   the CPI series available for a constant-dollar appendix column).
2. **Actual-law parameters**: per-year wage bases and the PE-US
   combined employee+employer OASDI rate, all actual values within
   2015-2022 from the pinned PE-US revision — no future-parameter
   registration needed at this horizon (NAWI actuals cover every
   age-60 indexing year for in-window claimants).
3. **No expectation reweighting**: taxable payroll
   `min(earnings, wage_base(year))` and contributions
   `weight × rate × taxable` are computed on realized projected
   person-years only.
4. **PIA source**: nothing on the revenue side consumes M2's
   pre-415(g) `base_pia`; the benefit side above uses the exact oracle
   exclusively.

Published: weighted taxable payroll and contributions by year, per-draw
mean/SD, with the covered-earner weighted count.

## 6. Weights and labeling

Fixed start-wave PSID cross-sectional weights — the certified object's
own weighting (`m6_cells.py:113-140`, `support.py:54-133`), synthetic
children inheriting the mother's weight. Every dollar table carries the
frame's weighted person count and the words "frame-relative,
pre-alignment"; the report includes no national-total claim. The W1
deployment frame's national weighting (`deployment_frame.py:371-418`)
is named as the alignment path for the successor report, not used here.

## 7. Named gaps (the disclosure block, from the certified record)

Immigration absent (dormant path, zero immigrant cohorts in the
candidate-3 artifact); mortality drift uncertified; families B/C
ungated; 2020-2022 shock window report-only (no macro channel);
claiming mechanical (1998-2013 table, later years snap to 2013); M4
disability is not DI adjudication — DI dollars out of scope; alignment
`not_computed`, scored path unaligned; earnings-domain exclusions
counted per §3.5; validation horizon — certified surfaces are the
registered 2016/2018 earnings and 2015-2019 flows, everything later is
extrapolation; spouse/survivor benefits out of scope pending an
entitlement adapter; levels unanchored (no committed annual SSA
beneficiary-stock, aggregate-benefit, or taxable-payroll series —
survey §4; a registered anchor extraction is the successor step).
Each gap cites its in-repo evidence in the artifact.

## 8. Artifact, tests, and publication (the W2 pattern)

- `runs/first_estimates_v1.json` via `artifacts.write_new`: identity,
  schema version, registration reference, full configuration echo
  (specs, seeds, draw indices, PE-US revision, environment sidecar),
  per-draw tables, aggregate tables, exclusion counts, the named-gap
  block with citations, and `certifies_nothing` scope statements.
- Tests: schema/invariant validation plus a cheap rebuild test that
  reconstructs the aggregation from a small committed fixture without
  re-running the projection (`test_w2_seam_caregiver*.py` pattern).
- Paper: a "First estimates" section narrating the registered
  procedure, the frame-relative tables, and the gap block — after the
  artifact merges, in the interim-seam section's voice.
- Ledger: entry 8 resolves at the publication PR's merge.

## 9. Ceremony and sequence

1. This design ratified (referee → fixes → verify → merge).
2. Implementation PR: the join, the oracle adapter, the ledger, the
   artifact writer, tests (referee-gated; no run).
3. Registration on a fresh issue: configuration frozen (specs, seeds
   0-19, exclusion rules, output path, one-run + one disclosed
   re-execution terms restated verbatim), the incident-5028176439
   execution topology mandatory for the compute.
4. One registered run (estimated hours, not days: twenty draws of a
   2014→2022 projection at candidate-3 per-draw cost) → publication
   regardless → paper section → entry 8 resolves.

## 10. What this unlocks

The first in-repo answer to "what does the open DynaSim replacement
say Social Security pays and collects on histories it projected
itself" — with every simplification on the page. The successor steps it
tees up, in order of leverage: the annual SSA/Trustees level-anchor
extraction (turns frame-relative into validated), the W1 population
bridge (turns frame-relative into national), the spouse/survivor
entitlement adapter (completes the benefit side), and behavioral
claiming (replaces the mechanical draw).
