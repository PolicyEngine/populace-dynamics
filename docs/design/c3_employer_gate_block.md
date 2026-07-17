# C3 employer gate block — thresholds, partitions, and rulings

- **Design id**: `2026-07-17-c3-employer-gate-block`
- **Status**: **DRAFT FOR REFEREE — NOT RATIFIED.** Nothing in this
  document binds until the lock ceremony flips it (§12). This document
  edits no `gates.yaml` cell, moves no threshold, builds no floor, and
  writes no test. Every number labelled PROPOSED is a proposal to the
  referee round, not a lock.
- **Plan**: issue #192 (E1–E12 battery, two-workstream split, protocol
  rules: disjoint calibration/gate cells, floors before thresholds, no
  one-shot candidate runs before C3 locks).
- **Contracts**: ADR 0003 (`docs/adr/0003-employer-firm-extension.md`,
  C1 spell schema and C2 banding, **frozen**); ADR 0004
  (`docs/adr/0004-linkage-qc.md`, linkage QC, adopted-with-changes per
  the Workstream B review on PR #224).
- **Evidence base** (all DRAFT — NOT RATIFIED artifacts, cited by
  branch pending merge):
  - PR #212 (`sipp-spell-floors`): `runs/sipp_spell_floors_draft_v0.json`
    (E4/E5), `runs/sipp_e8_e9_floors_draft_v0.json` (E8/E9),
    `runs/tenure_floors_draft_v0.json` (E3).
  - PR #214 (`seam-reconciliation`):
    `runs/seam_reconciliation_draft_v0.json`.
  - PR #223 (`firm-floors-pre-c3`):
    `runs/employer_firm_floors_draft_v0.json` (E1/E2/E6/E7, E11/E12
    method findings).
  - PR #228 (`j2j-se-od-extracts`):
    `data/external/j2j_us_sexage_2015on.csv` (E2 gate axis),
    `data/external/j2jod_us_firmsize_od_2015on.csv` (E11), with the
    test-pinned detail-window finding.
- **Authors**: joint Workstream A (@daphnehanse11) / Workstream B
  (@vahid-ahmadi) per the #192 interface-contract schedule (C3 is the
  jointly-authored employer gate block).

## 1. Scope and shape of the block

This is the pre-registration draft for the `gates.employer` block
(E1–E12) that will be added to `gates.yaml` by amendment PR after the
referee round. It follows the repo's locked-gate schema conventions:
per-gate `thresholds` derived from committed floor artifacts as
`floor mean + k × floor sd` with machine-checkable derivations
(the `tests/test_gates_derivations.py` pattern), floor runs cited by
path, thin-cell exclusions pre-registered, and `locked: false` until
ratification.

Two structural rules carry over from #192 and ADR 0003 unchanged:

1. **Floors before thresholds.** Every gated cell cites a committed
   floor artifact; cells with no derivable floor cannot gate
   (consequences for E1-sector, E11-detail, E12 below).
2. **No one-shot candidate runs before C3 locks.** Registration of
   employer-block candidates (issue #42 convention) opens only after
   the amendment PR merges with `locked: true`.

## 2. Proposed threshold policy (uniform, PROPOSED)

For every gated cell:

```
threshold_cell = max( floor_mean_cell + k × floor_sd_cell,
                      substantive_tolerance_gate )
```

- `k = 4` PROPOSED, matching the gate-1 precedent band (locked gate-1
  rules use k in 4.2–8 against seed-level floor sds; the employer
  floors aggregate 5 seeds or 24–36 year-pairs, so k = 4 on the
  aggregated sd is the conservative end). The referee round may set k
  per-gate.
- The `substantive_tolerance` term prevents degenerate-floor cells
  (several tenure quantile-gap floors and the E9-stay median floor are
  exactly 0.0 — see §4) from imposing a zero tolerance no model can
  meet; it is a per-gate scientific-relevance bound, every value
  PROPOSED in the table below.
- Cells flagged `thin` in the floor artifacts (E4/E5/E8/E9:
  `THIN_CELL_PERSONS = 200`; B-side: min denominator < 10,000 jobs)
  are report-only, pre-registered now.
- Verdict rule: a gated gate passes iff **all** its non-thin cells
  pass; report-only gates publish pass/fail per cell but do not block.

## 3. Per-gate proposals

Summary table (all numbers PROPOSED; "floor basis" names the committed
statistic the threshold derives from):

| Gate | Moment | Reference | Floor artifact | Recommended floor basis | First-lock status |
|---|---|---|---|---|---|
| E1 | employment share by firm-size band (× sector) | SUSB 2022 | #223 `e1` | SUSB noise-flag CV bounds + BDS YoY margin floor (coarsened `20_99`) | **gated** on the national size margin; sector cross **report-only** (no sector floor) |
| E2 | hire/sep/J2J rates by sex × age | LEHD J2J `sa` (#228) | #223 `e2` (aggregate-side); sex×age floor to be built on the #228 extract | ex-pandemic YoY \|log ratio\| | **gated** (after the sex×age floor build) |
| E3 | tenure quantiles by age | CPS tenure supplement 2020/22/24 | #212 `tenure_floors` | **ECDF max-gap** (heaping-robust) | **gated** |
| E4 | employer-retention pairs by age × sex | SIPP holdout | #212 `e4_retention_by_age_sex` | \|log rate ratio\| | **gated** (linkage-QC prerequisite, §9) |
| E5 | multi-window attachment runs by age | SIPP holdout | #212 `e5_runs_by_age` | \|log share ratio\| | **gated** (linkage-QC prerequisite, §9) |
| E6 | hire/sep flow rates by firm-size (× sector) | QWI | #223 `e6_e7` | ex-pandemic YoY \|log ratio\| | **gated** on size margin; sector cells report-only |
| E7 | mean earnings (`EarnS`) by firm-size | QWI | #223 `e6_e7` | **aggregate-relative** EarnS floor | **gated** |
| E8 | nonemployment incidence/duration by age | SIPP holdout | #212 `e8_nonemployment_by_age` | \|log share ratio\| (any/long) | **gated** |
| E9 | earnings-change dist. by transition type | SIPP holdout | #212 `e9_transitions` | j2j: median + IQR gaps; stay: **IQR only** | **gated** (stay-median report-only) |
| E10 | regression gate: locked PSID gates still pass | existing `gates.yaml` | existing gate-1/2 floors | unchanged | **gated** (always) |
| E11 | J2J flows origin × destination firm size | LEHD J2JOD (#228) | #228 extract; margins floored via #223 `e2` ee\_\* | margins: ex-pandemic YoY; detail: none derivable | **margins gated; 5-quarter detail report-only** (§7) |
| E12 | within/between-firm variance, coworker corr. | AKM (Song et al.; KSS) | none | none buildable | **deferred — cannot lock** (§8) |

Per-gate detail follows.

### E1 — employment share by firm-size × sector (SUSB)

- **Cells**: five canonical C2 bands (`LT10`…`B500_PLUS`) × NAICS
  sector, from `data/external/susb_us_sector_size_2022.csv`;
  calibration consumes the same source's *margins*, so the E1 gate
  axis must be the residual structure (see §10).
- **Floor problem** (#223 `method_findings.e1_no_sector_replicate`):
  the committed SUSB extract is a single 2022 cross-section — no
  same-source temporal/resampling floor exists for size × sector.
  The committed composite is (a) SUSB published noise-flag CV bounds
  per cell (G ⇒ CV ≤ 2%, H ⇒ ≤ 5%, J ⇒ unbounded) and (b) a BDS
  year-over-year national size-margin stability floor, stated on the
  coarsened partition keeping `20_99` whole
  (`method_findings.e1_bds_straddle`; `bds_fsize_to_canonical` is
  inexact across the 50 edge).
- **Recommendation**: gate the **national size margin** on the
  BDS-stability + CV composite; carry the size × sector cross
  **report-only** until a sector-axis floor source is committed.
  Cells with noise flag J (unbounded CV) are report-only.
- **Alternative for the referee**: gate size × sector using the CV
  bounds alone as the floor term (pure published-noise basis, no
  temporal term). Weaker provenance; on the record.
- **Substantive tolerance**: 5% relative share error PROPOSED
  (ECPS-convention "±5% of administrative statistics").

### E2 — hire/separation/J2J rates by sex × age (LEHD J2J)

- **Reference**: the #228 `j2j_us_sexage_2015on.csv` extract
  (national, 2015Q1–2025Q1; note LEHD `se` is sex×education — sex×age
  is the `sa` tabulation, verified empirically and test-pinned).
  This is a held-out **gate axis** per ADR 0003: calibration touches
  firm-size × sector flow margins only, never the demographic axes.
- **Floor**: the committed #223 `e2` block floors the aggregate-side
  firm-size cells (36 same-quarter YoY pairs; full and ex-pandemic
  variants both committed per
  `method_findings.cycle_signal_in_floors`). The sex × age floor
  itself is a required pre-lock build on the #228 extract, same
  method (#223 `method_findings.e2_no_age_sex_axis` recorded the
  deferral explicitly).
- **Recommendation**: **ex-pandemic** YoY \|log ratio\| floors
  (e.g. firmsize1 hire rate: full 0.0565 ± 0.0549 vs ex-pandemic
  0.0273 ± 0.0211). Rationale: the full-sample floor treats the
  2020–2021 shock as noise and would roughly double every tolerance;
  a model should not get credit for pandemic-sized errors in normal
  years. The **full-sample** variant stays on the record as the
  referee alternative (argument for it: candidates will be scored on
  periods that include shocks).
- **Unit caveat** (ADR 0003): J2J counts jobs, not persons, and the
  #228 extract is ownership `oslp`; §11 carries the disposition.
- **Substantive tolerance**: 10% relative rate error PROPOSED.

### E3 — tenure distribution by age (CPS tenure supplement)

- **Cells**: BLS age bands × supplement years 2020/2022/2024
  (`PTST1TN`, `PWTENWGT`; reader per #205).
- **Degeneracy finding** (#212 `tenure_floors_draft_v0.json`,
  `heaping_caveat`): reported tenure heaps on integers, so
  half-vs-half quantile-gap floors are exactly zero in 36/63 cells —
  a degenerate threshold basis.
- **Recommendation**: state E3 on the **weighted-ECDF max-gap**
  (heaping-robust; non-degenerate in every cell, e.g. 2020 25–34:
  0.018 ± 0.0059). The quantile-gap formulation stays committed as
  the referee alternative, but adopting it would require an
  arbitrary substantive tolerance to paper over the zeros.
- **Substantive tolerance**: ECDF max-gap 0.02 PROPOSED (comparable
  to the largest observed adult-band floor means).

### E4 — employer-retention pairs by age × sex (SIPP holdout)

- **Cells**: 6 age bands × sex, monthly same-employer retention
  (#212 `e4_retention_by_age_sex`; person-disjoint sha256
  half-splits, seeds 0–4, WPFINWGT-weighted; all 12 cells non-thin,
  floors 0.0003–0.0020 \|log ratio\|).
- **Recommendation**: gate on \|log rate ratio\| with the uniform
  policy. Because retention rates sit near 1, the floor is tiny;
  the substantive tolerance term (PROPOSED: \|log ratio\| 0.005,
  ≈ 0.5% of the retention rate) is the operative bound in most cells
  — deliberately: retention is exactly the moment a chained model
  understates.
- **Prerequisites**: the §6 seam ruling (half-splits share SIPP's
  seam structure — `seam_caveat` in the artifact) and the §9
  first-lock linkage audit (E4 is in the minimal-viable-audit scope).

### E5 — multi-window attachment runs by age (SIPP holdout)

- **Cells**: 6 age bands, full-year same-employer run share
  (#212 `e5_runs_by_age`; floors 0.011–0.049 \|log ratio\|, all
  non-thin). This is the employer analogue of the gate-1 runs view:
  window-2 moments cannot see chained-persistence understatement, so
  E5 is the load-bearing persistence gate and must be **gated**, not
  report-only.
- **Recommendation**: uniform policy on \|log share ratio\|;
  substantive tolerance \|log ratio\| 0.05 PROPOSED. Same §6/§9
  prerequisites as E4 (E5 is the second member of the
  minimal-viable-audit scope).

### E6 — worker-flow rates by firm-size × sector (QWI)

- **Cells**: QWI firm-size groups (mapped through
  `firms/banding.py`; `firmsize1` "0–19" is an inexact
  `LT10+B10_49` span, recorded) × sector, hire and separation rates
  (#223 `e6_e7`; 36 YoY pairs, full and ex-pandemic committed).
- **Recommendation**: **ex-pandemic** floors, same rationale and
  same referee alternative as E2. Gate the size margin; sector cells
  report-only at first lock (the sector × size axis is close to the
  calibration margin — §10 keeps gated cells strictly disjoint).
- **Substantive tolerance**: 10% relative rate error PROPOSED.

### E7 — mean earnings by firm-size (QWI `EarnS`)

- **Unit rule** (ADR 0003, restated §11): QWI publishes **mean**
  monthly earnings of full-quarter employees, never medians; E7 is
  stated on means.
- **Finding** (#223 `method_findings.e7_nominal_trend`): raw EarnS
  YoY variation embeds aggregate nominal wage growth — a trend, not
  noise. Both the raw and the **aggregate-relative** (cell relative
  to the all-size aggregate) floors are committed.
- **Recommendation**: gate on the **aggregate-relative** EarnS
  moment (the size *gradient* of earnings, which is what firm-size
  policy needs) with its ex-pandemic floor; the raw-level cell is
  report-only. Referee alternative: gate raw levels with a
  deflation rule (requires choosing a deflator — a new registered
  input, hence not recommended for first lock).
- **Substantive tolerance**: 5% relative gradient error PROPOSED.

### E8 — nonemployment incidence and duration by age (SIPP holdout)

- **Cells**: 6 age bands × {any-nonemployment share, long
  (multi-month) nonemployment share}
  (#212 `sipp_e8_e9_floors_draft_v0.json`; floors 0.05–0.20
  \|log ratio\|, all non-thin; censoring-free 12-month-observed
  restriction recorded in the artifact and carried into the cell
  definition).
- **Recommendation**: uniform policy on \|log share ratio\|;
  substantive tolerance \|log ratio\| 0.10 PROPOSED. E8 needs no
  worker-to-firm link (ADR 0004 §4), so it has no linkage
  prerequisite.

### E9 — earnings-change distribution by transition type (SIPP holdout)

- **Cells**: transition classes {stay, j2j} × {median log change,
  IQR log change} (#212 `e9_transitions`; entry/exit earnings-change
  cells are not defined — one side has no earnings).
- **Degeneracy finding** (`stay_median_heaping_caveat`): within-job
  SIPP monthly earnings are wave-constant under dependent
  interviewing, so the stay median log-change heaps at exactly 0 and
  its floor is degenerate (0.0/0.0) — same failure class as the E3
  integer heaping.
- **Recommendation**: gate **stay on the IQR only**
  (0.0656, floor 0.0 — substantive tolerance operative, PROPOSED
  0.02 absolute IQR gap in log points); stay-median becomes a
  report-only identity check (a model reporting a nonzero stay
  median is informative, not gated). Gate **j2j on both** median
  (0.2264, floor 0.0766 ± 0.0345) and IQR (1.069, floor
  0.0842 ± 0.0522). Referee alternative for stay: a full
  distributional distance (e.g. the harness energy-distance block)
  — richer, but no floor is committed for it, so it cannot gate at
  first lock (floors-before-thresholds).
- Note the j2j cell counts 545 unweighted pairs — above the thin
  flag but the thinnest gated cell in the block; recorded.
- **Prerequisite**: §9 (E9 consumes transition classification;
  phase-1 imputed-band adjudicability is an open referee item for
  its firm-size-conditional variants, which are **not** in the first
  lock).

### E10 — regression gate (locked PSID gates)

- **Definition**: after the spell layer attaches, every already
  locked PSID earnings gate (`gates.gate_1` and successors) must
  still pass, under its existing thresholds and floors, unchanged.
- **No new floor, no new threshold, nothing PROPOSED**: E10
  re-scores existing locked cells.
- **Hard rule** (ADR 0004 §4, adopted): **linkage failure never
  weakens E10**. An invalid or failed employer-side cell cannot be
  traded against, or used to reinterpret, any PSID threshold; E10 is
  gated from the first lock and in every subsequent phase.

### E11 — J2J flows by origin × destination firm size — see §7.

### E12 — within/between-firm variance and coworker correlation — see §8.

## 4. Degenerate-floor register

For the referee's convenience, the committed degeneracies the block
must not gate on directly:

1. **E3 tenure quantile gaps**: exactly 0.0 in 36/63 cells (integer
   heaping) → ECDF max-gap recommended (#212).
2. **E9 stay median**: floor exactly 0.0/0.0 (wave-constant
   reporting) → IQR recommended (#212).
3. **E1 sector axis**: no floor derivable from committed extracts
   (single-vintage SUSB) → report-only (#223).
4. **E11 detail cells**: no floor derivable (five-quarter published
   window, no YoY pair structure) → report-only (#228, §7).
5. **E12**: no reference extract at all → deferred (#223, §8).

## 5. Views and cell registration (schema sketch)

Following the `views:` convention of `gates.yaml`, the amendment PR
registers (names PROPOSED):

- `sipp_job_spells` — loader `populace_dynamics.data.sipp_jobs`,
  pu2023 (ref. year 2022), person-disjoint holdout, WPFINWGT;
  feeds E4/E5/E8/E9; floor runs: the three #212 artifacts
  (promoted from `draft_v0` to `v1` at lock, sha256-pinned,
  reproduction-tested per the `runs/` convention).
- `cps_tenure` — CPS Jan supplements 2020/2022/2024; feeds E3;
  floor run: `tenure_floors` v1.
- `employer_firm_targets` — the committed
  `data/external/{susb,bds,qwi,j2j,j2jod,j2j_sexage}_us_*.csv`
  extracts (external references, never scored model output);
  feeds E1/E2/E6/E7/E11; floor run: `employer_firm_floors` v1 plus
  the pre-lock sex×age floor build (§3-E2).

## 6. The seam ruling (formal proposal, from #214)

`runs/seam_reconciliation_draft_v0.json` measured, on linked SIPP
job-months: within-wave monthly separation ≈ 1.7–1.8% (means 0.0171
file-year 2022, 0.0182 file-year 2023) against a Dec→Jan across-wave
seam rate of **9.45%**, while the J2J national benchmark's
monthly-equivalent main-job separation rate runs ≈ 3.5–4.6%. The
artifact's concept deltas (jobs-vs-persons, main-job-vs-all-jobs,
UI-covered universe, the 1−(1−q)^{1/3} approximation) are carried
into the record, including the naming note that SIPP's ~0.35%
person-level direct J2J rate and Census J2J's separation benchmark
are different universes, not a contradiction.

**Proposed ruling (to be ratified by this referee round — the
artifact itself is marked NOT RATIFIED):**

1. **J2J is truth for rate LEVELS** (separation/hire/J2J levels in
   E2/E6 gate against the LEHD references, not against SIPP's
   seam-distorted levels).
2. **SIPP is truth for persistence STRUCTURE** (E4/E5/E8/E9
   conditional and distributional shape — J2J publishes no
   within-person structure).
3. **Phase-1 hazards are estimated seam-aware**: wave-frequency
   estimation with within-wave interpolation, so the 5.5× within-wave
   vs seam contrast never enters a hazard as if it were a real
   monthly time-pattern.

**Blocking check before ratification** (concept-delta 5, UNVERIFIED
ASSUMPTION): the 9.45% seam rate assumes SIPP `EJB` job IDs are
longitudinally consistent across the pu2022→pu2023 boundary. If IDs
are reassigned at wave boundaries, part of the seam contrast is a
linkage artifact. The cross-wave job-ID consistency check (ADR 0004
referee item 7) is a **required pre-lock artifact**; the ruling above
is conditional on its result.

## 7. E11 — the window decision (explicit referee choice)

Material finding from #228, test-pinned (`test_j2jod_detail_window`):
the national origin × destination firm-size cross in J2JOD is
published **only 2015Q1–2016Q1** — five quarters. From 2016Q2 every
detail cell carries status flag 11 (a state coverage gap propagated
to the national aggregate; J2J Explorer suppresses identically).
The one-sided margins run through 2025Q1.

The referee round must choose between:

- **(a) Margins-gated + detail report-only — RECOMMENDED.** Gate the
  origin-size and destination-size margins (the `ee_hire_rate` /
  `ee_separation_rate` cells already floored in #223 `e2`, which the
  artifact's `e11.margin_proxy` records as exactly these margins),
  ex-pandemic basis; score the full 6×6 detail on the 2015Q1–2016Q1
  window **report-only**. Rationale: a five-quarter window supports
  no temporal-stability floor (floors-before-thresholds forbids
  gating it), and a gate pinned to a nine-year-old suppressed
  vintage would be an evidentially weak lock; but the detail window
  is still the only published look at the size ladder, so it stays
  on the record.
- **(b) Detail-gated on the five-quarter window.** Requires the
  referee to accept a non-temporal floor basis (e.g. noise-infusion
  margin slack) and the vintage staleness; on the record as the
  alternative.

Under (a), ADR 0004's E11 ordered-pair linkage rules attach to the
report-only detail cells and to the margins-gate's origin/destination
assignments respectively; the joint-pair audit becomes operative only
if (b) is chosen or the detail cell is later promoted.

## 8. E12 — deferred; no-go rule

Per #223 (`method_findings.e12_deferred`) there is no committed E12
reference extract: AKM within/between decompositions require linked
employer-employee microdata, and published decompositions (Song et
al.; KSS-corrected) are research outputs, not a recurring
provenance-pinned release. No floor is buildable. Per ADR 0004 §5
and referee item 2, E12 additionally needs an **adjudicable truth
source** for the worker-to-firm-type assignment at the exact
co-assignment unit the estimand uses.

**Proposal**: E12 is registered in the C3 block as **deferred** —
definition and estimand-candidates recorded, no threshold, no floor,
`locked: false` permanently until both (i) a committed
provenance-pinned reference extract and (ii) an admissible
adjudication frame exist. **Phase-2 consequence** (#192 phase gate +
ADR 0004): if no adjudicable truth source can be identified, that is
a **phase-2 no-go** — claims scope down to firm-size-keyed policy
only; calibration fit to aggregates is not a substitute; the
firm-type register may not relabel type agreement as firm-identity
or coworker validation.

## 9. Linkage-QC integration (ADR 0004 as amended)

ADR 0004 is adopted **with the three changes from the Workstream B
review** (PR #224 review comment), which this block operationalizes:

1. **Minimal viable audit scope for the first lock.** The full
   two-arm, five-band × NAICS-major × transition-class × primary_job
   powered audit would move the lock well past the #192 schedule.
   First lock therefore audits **E4/E5 SIPP-internal employer
   attachment only** — the assignment where hand-adjudicable truth
   demonstrably exists (within-panel `EJB` job-ID attachment, with
   the §6 cross-wave ID check as its natural companion artifact).
   The full grid (E9 transition classes, E11 ordered pairs) phases
   in behind it; a gate promoted later (e.g. E11 detail) must clear
   its ADR 0004 audit at promotion time, never inherit the E4/E5
   pass.
2. **Phase-1 imputed-band adjudicability is an open referee item**
   (blocking, §13 item 6). QRF-imputed firm-size bands on CPS hosts
   may have no defined "adjudicated true class" (SIPP measures
   establishment size; NOEMP is a noisy self-report; no
   representative source observes the joint — ADR 0003). C3 must
   record what E9/E11 firm-size-conditional cells degrade to if no
   admissible truth frame exists. First lock avoids the question by
   scoping those conditional cells out (§3-E9, §7); it must not be
   left implicit.
3. **Named owners.** Workstream B (@vahid-ahmadi) owns the
   **firm-side sidecar specification** (schema, versioning; sidecars
   join C1 on `person_id`/`spell_id`, never amend C1) and the
   firm-side audit artifacts. Workstream A (@daphnehanse11) owns the
   E4/E5 SIPP-internal adjudication frame and coder-panel operation.
   Numeric `P_floor`, `P_design`, α, power, and the recall-gating
   decision are referee items (ADR 0004 §6.1), not owner decisions.

Unchanged ADR 0004 rules restated as binding on this block: audit
precedes the one-shot run; passing uses the one-sided confidence
bound, not the observed proportion; a failed floor invalidates the
cell (no threshold shopping); linkage failure never weakens E10.

## 10. Calibration/gate cell partition (explicit lists)

Per ADR 0003's partition rule, registered here as the explicit lists
that lock with C3:

**Calibration cells (consumed by `microcalibrate`; never gated):**

- SUSB employment margins: canonical size band × NAICS sector
  (`susb_us_sector_size_2022.csv`), SUSB universe (private,
  excluding NAICS 92, crop/animal production, non-employers —
  `class_of_worker` scoping per C1).
- QWI flow margins: firm-size × sector hire and separation rates
  (`qwi_us_firmsize_sector_2015on.csv`, ownership `op`).

**Gate cells (held-out axes; calibration never touches):**

- E2: **sex × age** hire/sep/J2J rates (#228
  `j2j_us_sexage_2015on.csv`).
- E11 margins: origin-/destination-size EE flow margins (J2JOD).
- **Firm-age axis** (QWI/BDS firm-age): registered as a held-out
  gate axis; report-only at first lock pending a committed firm-age
  extract and floor.
- **State axis** (QWI state-level): registered held-out; report-only
  at first lock, same condition.
- E1's gated national size margin is scored against BDS/SUSB
  *stability*, with the calibration consuming the SUSB level margins
  — the referee round must confirm this E1 treatment (gating a
  margin whose level is calibrated is only meaningful on held-out
  structure; the recommended E1 gate is therefore the coarsened
  BDS-partition margin, which calibration does not target, plus the
  report-only sector cross). This is §13 item 3.
- The SIPP-side gates (E3/E4/E5/E8/E9) and E10 are disjoint from
  calibration by construction (different sources).

## 11. The three unit rules (ADR 0003) — C3 dispositions

1. **QWI/J2J cells count jobs, not persons.** Phase 0 is
   primary-job-only, so person-spells vs job-count cells carry a
   wedge on the order of the multiple-jobholding rate (~5%,
   time-varying). **Disposition (PROPOSED)**: a pre-registered
   jobs→persons adjustment factor, published per cell alongside its
   source (CPS multiple-jobholding rate series), applied to E2/E6/E11
   comparisons before scoring; the factor's series and vintage are
   registered in the amendment PR — an explicit pre-registered item,
   not a footnote.
2. **QWI publishes mean earnings (`EarnS`), never medians.**
   **Disposition**: E7 stated on means (§3-E7); no median-based
   employer earnings gate exists in the block.
3. **J2J ownership `oslp` vs QWI `op` vs SUSB (no government).**
   NAICS 92 is dropped from the J2J extracts, but state/local
   employment embedded in sectors 61/62 remains. **Disposition
   (RECOMMENDED)**: carry the scope difference as a pre-registered
   per-cell caveat on E2/E11 (the #228 extracts quantify the small
   LED-vs-flat-file margin delta from excluded public "N" flows),
   rather than restating J2J on a private-comparable basis — a
   restatement would require a new extract and re-floor.
   The restatement option stays on the record as the referee
   alternative (§13 item 4).

## 12. Registration and lock ceremony

Per repo convention (gate-1 precedent PR #33/#39; `gate_m6` ceremony
in `docs/design/m6_projection_engine.md`):

1. This document circulates for the **referee round** (adversarial
   review; every §13 item answered on the record).
2. Required pre-lock artifacts land: the E2 sex×age floor build
   (§3-E2), the cross-wave job-ID check (§6), the E4/E5 minimal
   audit manifest (§9), and promotion of the four draft floor
   artifacts to sha256-pinned `v1` with reproduction tests.
3. A single **amendment PR to `gates.yaml`** adds the employer block
   with referee-resolved numbers, machine-checkable derivations
   (`tests/test_gates_derivations.py` pattern), and `locked: true`;
   authorship by one workstream owner plus approval by the other
   (the ADR 0003 cross-workstream approval norm). Merge of that PR
   is the ratification event; nothing in this draft binds before it.
4. **No one-shot candidate runs before the lock.** Employer-block
   candidates register on issue #42 only after the amendment PR
   merges. Subsequent changes to locked thresholds require a public
   amendment plus a fresh referee round.

## 13. Open questions for the referee round

Every decision this draft leaves to the referee, enumerated:

1. **Threshold policy**: accept `floor mean + 4 × floor sd` with
   per-gate substantive tolerances, or set k per-gate? Every
   PROPOSED tolerance in §3 needs a ratified number.
2. **Temporal-floor basis**: ex-pandemic (recommended) vs
   full-sample floors for E2/E6/E7/E11-margins (#223 commits both).
3. **E1 treatment**: national coarsened-margin gate + sector
   report-only (recommended) vs CV-bound-based size × sector gate;
   and confirmation of the calibrated-margin vs gated-stability
   partition logic (§10).
4. **E3 formulation**: ECDF max-gap (recommended) vs quantile gaps
   with a substantive-tolerance patch over the heaping zeros.
5. **E9 stay formulation**: IQR-only gate (recommended) vs a
   distributional distance (would require a new floor build before
   it could gate).
6. **Phase-1 imputed-band adjudicability** (blocking, from the #224
   review): does an admissible truth frame exist for QRF-imputed
   firm-size bands, and what do E9/E11 firm-size-conditional cells
   degrade to if not?
7. **Linkage-QC numerics** (ADR 0004 §6.1): `P_floor`, `P_design`,
   α, power, multiplicity, and whether recall gates; plus
   ratification of the E4/E5-first minimal audit scope (§9).
8. **The seam ruling** (§6): ratify J2J-levels / SIPP-persistence /
   seam-aware estimation, conditional on the cross-wave job-ID
   consistency check.
9. **E11 window** (§7): margins-gated + detail report-only
   (recommended) vs five-quarter detail gate.
10. **E12**: confirm the deferred/no-go registration and the phase-2
    consequence wording (§8).
11. **Jobs→persons adjustment** (§11.1): adopt the proposed
    published-factor mechanism and its source series.
12. **`oslp` vs `op` scope** (§11.3): per-cell caveat (recommended)
    vs private-comparable restatement of the J2J references.
13. **Held-out firm-age and state axes** (§10): confirm report-only
    registration at first lock, with promotion conditions.
14. **Thin-cell rules**: confirm `THIN_CELL_PERSONS = 200`
    (SIPP-side) and the 10,000-job minimum denominator (B-side,
    a draft choice in #223) as the locked thin flags.
15. **ASEC reference-period mismatch** (ADR 0003 conditioning DAG):
    confirm its carriage as a known label-misalignment note on the
    affected gate cells.
