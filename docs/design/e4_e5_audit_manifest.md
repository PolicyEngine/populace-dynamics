# E4/E5 minimal-audit manifest (ADR 0004 instantiation, pre-lock)

**Status: REGISTERED DESIGN, sample not yet drawn.** This is
Workstream A's §12.2 pre-lock artifact for the C3 block (#230 §9.1):
the complete ADR 0004 adjudication design for the first-lock scope —
**SIPP-internal employer attachment**, the assignment E4 (retention
pairs) and E5 (attachment runs) consume. The numeric slots marked
`REFEREE` are ADR 0004 §6.1 items; the sample is drawn only after the
referee round fills them, by the committed draw script, under the
seed registered here. Owner: @daphnehanse11 (frame + coder-panel
operation, per #230 §9.3).

## 1. The assignment under audit

E4/E5 treat two job records in adjacent reference months as **the
same employer** iff they share a within-panel `EJB` job ID
(`populace_dynamics.data.sipp_jobs`, pu2023, reference year 2022).
The "matcher" is therefore Census's dependent-interview job-ID
assignment as consumed by the reader — not a model we train. What
hand adjudication can verify from the public-use record: whether the
month-*m* and month-*m+1* job records describe the same employer,
using industry code, occupation code, class of worker, work
arrangement, establishment-size code, monthly earnings, and the
`BMONTH`/`EMONTH` spell edges. This is the "hand-adjudicable truth
demonstrably exists" claim of #230 §9.1, made concrete.

## 2. Frame and the two arms (ADR 0004 §2.1)

- **Eligible universe**: all ordered adjacent-month pairs
  (person, month *m*, month *m+1*), *m* = 1..11 plus the Dec→Jan
  cross-file pair, where the person holds ≥1 job in month *m* and is
  present in the panel in month *m+1* (presence from the
  person-month universe — exits to nonemployment are in-universe;
  sample leavers are not). This is exactly the #235 population;
  the check's counts (384,747 within-wave job-holdings; 10,828 at
  the seam) size the frame.
- **Candidate generation**: within-person only — all ≤7 job slots of
  month *m+1* are visible to the coder. Candidate-generation recall
  is 1 by construction (SIPP cannot attach a person's job to another
  person's employer record), so candidate-recall and selector-recall
  collapse; this is registered as a structural property, not
  measured.
- **Accepted-assignment arm (a)**: pairs where a month-*m* job's ID
  recurs in month *m+1* (the "stay" assignment). Coders judge
  same-employer vs not → **precision** of ID-based attachment.
- **Truth-search arm (b)**: pairs where a month-*m* job's ID does
  NOT recur (separations). Coders search all month-*m+1* job records
  for a true same-employer counterpart → **recall** (the missed
  attachments are exactly #235's re-key class; its signature rate,
  17.5% at the seam vs 2.4% within-wave, is the prevalence prior for
  powering this arm).
- **Dual-frame overlap**: a person-pair can contribute a stay job to
  arm (a) and a separated job to arm (b); the unit of audit is the
  **job-pair**, not the person-pair, so the arms partition job-pairs
  and no combined-inclusion estimator is needed. Registered as such.

## 3. Stratification (ADR 0004 §2.2, scoped per #230 §9.1)

First-lock scope excludes firm-size-conditional cells, so the
C2-band stratification of the full ADR grid does not apply (its
strata are registered for promotion-time audits, not this one).
Operative strata:

- **Arm (a)** (precision): {within-wave, seam} × {age 16–44,
  45+} — 4 strata. Seam pairs are deliberately oversampled (they are
  the risk locus established by #214/#235 and are only ~2.7% of the
  frame).
- **Arm (b)** (recall): {within-wave, seam} × {re-key signature
  present, absent} — 4 strata. The signature (same industry + class
  of worker + earnings within 20%; parameters **frozen** in
  `scripts/build_crosswave_jobid_check.py` before any label exists)
  concentrates the plausible false negatives, so signature-present
  strata are oversampled.
- Any pooling of sparse strata is published in the draw artifact
  before adjudication; every inclusion probability is retained.

## 4. Power and target sizes (ADR 0004 §2.3)

Registered parameters — `REFEREE` slots per ADR 0004 §6.1:

| Parameter | Value |
|---|---|
| `P_floor` (precision) | REFEREE |
| `P_design` | REFEREE |
| `alpha` (one-sided) | REFEREE |
| `1 - beta` | REFEREE |
| Multiplicity rule | REFEREE (recommended: intersection-union across the 4 arm-(a) strata with joint power computed by simulation) |
| Recall: gates or reported-with-bound | REFEREE |

**Worked example** (illustrative only, not a proposal): for a
simple-random operative stratum, `P_floor = 0.95`,
`P_design = 0.99`, `alpha = 0.05`, `1 − beta = 0.80` gives
`n = 124`, critical count `c = 122` (smallest binomial solution);
(0.90, 0.97) gives `n = 76, c = 73`. Per ADR 0004, job-pairs
cluster within worker: the draw script computes design-based
power by simulation with **worker** as the registered clustering
unit and applies the anticipated design effect, and targets are
inflated for indeterminate/unusable rates (assumed 10% until the
calibration round measures them). Four strata at the example
numbers imply an arm-(a) total near 550 adjudications before
inflation — feasible for a two-coder panel.

**Arm (b) target**: based on expected true-counterpart prevalence
per stratum (prior: the #235 signature rates), powered for the
registered recall-bound width or floor, per the same machinery.

## 5. Coding protocol (ADR 0004 §2.4)

- **Blinding**: coders see both months' job-record fields with all
  `EJB` job IDs **masked**, and never see the matcher outcome
  (same/different ID), the #235 signature flag, any downstream gate
  quantity, or the other coder's decision.
- **Labels**: `same_employer`, `different_employer`,
  `insufficient_evidence`, under a frozen coding manual
  (`docs/design/e4_e5_coding_manual_v1.md`, to be committed before
  labels open; version hash registered in the draw artifact).
- **Disagreements**: adjudicated by a third coder without disclosure
  of the split; dispositions logged.
- **Indeterminates** (registered now, before labels):
  `insufficient_evidence` counts **conservatively against** the
  audited assignment — as incorrect in arm (a) precision and as a
  missed true counterpart in arm (b) recall bounds — with the
  partial-identification bounds also reported. Hard cases are never
  dropped after labels are known.
- **Calibration**: coders first train on vetted cases excluded from
  both arms; blinded repeats (10% of assignments) measure continuing
  accuracy and agreement.
- The precision test publishes the sensitivity bound to
  reference-label error per ADR 0004 (the label is hand-adjudicated
  reference truth, not infallible ground truth).

## 6. Provenance (ADR 0004 §2.5)

The draw artifact (`runs/e4_e5_audit_draw_v1.json`, committed when
the sample is drawn) will contain: reader version and pu-file
sha256s, frame query (this document's §2 verbatim), stratum
definitions and inclusion probabilities, the random seed
(**registered now: 20260717**), sha256 of each selected job-pair's
public-use identifiers, coding-manual and evidence-sheet versions,
coder-assignment protocol, and counts. Replacements or exclusions
are logged; the sample is never silently refreshed. All fields are
public-use-derived; no restricted data exists in this design.

## 7. No leakage (ADR 0004 §2.6)

Nothing here trains a matcher (Census assigns the IDs), but two
freezes are registered so audit labels cannot leak backward:

1. The #235 re-key-signature parameters (industry + class of worker
   + earnings tolerance) are frozen at their committed values; they
   may stratify this audit but may never be re-tuned on its labels.
2. Audit labels and dispositions may not inform any reader-side
   attachment heuristic, imputation feature, or phase-1 hazard
   specification. If a leak occurs, the sample retires and a new
   manifest issues.

## 8. Deliverables and sequence

1. This manifest (pre-lock, per #230 §12.2) — the registered design.
2. Referee round fills the §4 slots.
3. `scripts/build_e4_e5_audit_draw.py` draws the sample under seed
   20260717 → `runs/e4_e5_audit_draw_v1.json` + the blinded evidence
   sheets.
4. Coding manual v1 commits; calibration round runs; panel codes.
5. `runs/e4_e5_audit_v1.json` reports precision/recall with
   confidence bounds, agreement, dispositions. **Sequence (pinned
   per the #230 round-1 referee review, S4, matching ADR 0004
   §1.5)**: this manifest is the pre-lock artifact; the C3 block
   may lock with the audit *designed but undrawn*; the audit
   **results must exist before the first one-shot candidate run**
   that scores any E4/E5 cell. Passing uses the one-sided bound,
   never the observed proportion. A failed floor invalidates the
   E4/E5 cells — no candidate can then pass the block — and the
   audit artifact publishes regardless of result: a designed stop
   is a graded, publishable outcome, not a re-scoping event.
