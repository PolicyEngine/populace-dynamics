# M6 candidate-3 program — the earnings-margin frontier after the candidate-2 verdict

Status: PROPOSED (draft PR; adversarial referee round required before any
design, selection, or registration step executes).

This document is the candidate-2 → candidate-3 analog of the ratified
candidate-2 program (`m6_candidate2_program.md`, #227 pattern): it publishes
the artifact-only forensics of the ratified candidate-2 verdict, adjudicates
the design directions the verdict licenses, draws the holdout-leakage boundary
that constrains all of them, and fixes the governance ladder to a fresh
registration. It reads no holdout data: every number in §2 is quoted from the
two committed, referee-verified artifacts (`runs/gate_m6_candidate2_v1.json`,
ratified via PR #269 at `ddcd6ef`; `runs/m6_holdout_floors_v4.json`, locked via
#237 at `9e5bab4`) or from the #269 referee record (comment 5039585201).

## 1. Disposition and design home

Candidate 2 FAILED the gate_m6 acceptance conjunction 3-of-5 seeds
(contract requires ≥4-of-5), with `must_not_regress` PASS 5/5 and
`valid = True` (artifact `candidate2_acceptance`). The artifact records
`fresh_registration_required: true`; registration 11 is closed (issue #42
comment thread, closure comment after PR #269 merged).

Candidate 3 is a sibling program under the same locked gate: no gates.yaml
cell, tolerance, seed law, or scoring reducer changes. Its design home is
the same two-stage §2.7 pattern candidate 2 used: any mechanism change is a
prospective design amendment plus a reviewed lock addendum re-finalizing
`gate_m6.design_commit`, then a fresh registration on issue #42, then one
one-shot run under the hardened direct-runner topology proven by invocation
#6-re (incident record: issue #42 comment 5028176439 — no network-dependent
babysitter in the process ancestry; `nohup` + launchd orphaning; `caffeinate
-sim` held by a child of the runner).

## 2. Frozen evidence — artifact-only forensics of the candidate-2 verdict

### 2.1 The three breaches and their unanimity of direction

The gate's three breaching seed-cells, with every seed's truth-side value
(`rate_a`), candidate value (`rbar`, mean of 20 draws), and score, quoted
from `candidate2_acceptance.gate_contract_result.per_seed[*].gated_cells`:

**F1 — `earn_dlog_mean.prime` (metric `abs_gap_log`, tolerance 0.043):
UNDER in 5/5 seeds.**

| seed | truth | candidate | score | % of tol |
|------|-------|-----------|-------|----------|
| 0 | 0.14414 | 0.10357 | 0.04058 | 94.4% |
| 1 | 0.13743 | 0.10168 | 0.03575 | 83.1% |
| 2 | 0.14914 | 0.09326 | **0.05587** | **129.9%** |
| 3 | 0.13376 | 0.09910 | 0.03467 | 80.6% |
| 4 | 0.12794 | 0.09742 | 0.03052 | 71.0% |

Across seeds: mean score 0.03948, sd 0.00984. The candidate under-generates
prime-age mean Δlog earnings in every seed; even the best single draw of
seed 2's twenty (0.10399) sits 0.045 below that seed's truth.

**F2 — `earn_autocorr_lag2` (metric `abs_gap_corr`, tolerance 0.087):
OVER in 5/5 seeds.**

| seed | truth | candidate | score | % of tol |
|------|-------|-----------|-------|----------|
| 0 | 0.67723 | 0.75603 | 0.07880 | 90.6% |
| 1 | 0.69535 | 0.75426 | 0.05891 | 67.7% |
| 2 | 0.69568 | 0.76542 | 0.06974 | 80.2% |
| 3 | 0.67465 | 0.76944 | **0.09479** | **109.0%** |
| 4 | 0.71323 | 0.75711 | 0.04387 | 50.4% |

Across seeds: mean 0.06922, sd 0.01933. The candidate's projected earnings
process is more lag-2 persistent than truth in every seed.

**F3 — `remarriage.18-64` (metric `log_ratio`, tolerance 0.403):
OVER in 5/5 seeds.**

| seed | truth | candidate | score | % of tol |
|------|-------|-----------|-------|----------|
| 0 | 0.03927 | 0.04981 | 0.23761 | 59.0% |
| 1 | 0.03737 | 0.05019 | 0.29507 | 73.2% |
| 2 | 0.03562 | 0.05389 | **0.41386** | **102.7%** |
| 3 | 0.03858 | 0.05195 | 0.29757 | 73.8% |
| 4 | 0.03810 | 0.05099 | 0.29155 | 72.3% |

Across seeds: mean 0.30713, sd 0.06462. The candidate remarries the 18-64
population at 1.27-1.51× the truth-side hazard in every seed; the wide
tolerance (0.403 in log space ≈ 1.5×) absorbs it in four of five.

### 2.2 Floor-relative sizing

The floors-v4 method derived each tolerance as `round(mean + 3·sd, 3)` of
the candidate-blind split-half anchor-noise distribution (100 seeds;
`floor.method`, `floor.cells`). Placing each cell's candidate mean score in
that distribution:

| cell | floor mean | floor sd | candidate mean score | floor-SDs above floor mean |
|------|-----------|----------|----------------------|-----------------------------|
| earn_dlog_mean.prime | 0.01411 | 0.00980 | 0.03948 | **+2.59** |
| earn_autocorr_lag2 | 0.02674 | 0.01993 | 0.06922 | **+2.13** |
| remarriage.18-64 | 0.12206 | 0.09353 | 0.30713 | **+1.98** |

All three candidate discrepancies sit 2-2.6 anchor-noise SDs above the
anchor-noise mean: real model-truth divergence beyond sampling noise, below
the 3-SD tolerance line on average, breaching per-seed at the dispersion
tail. The across-seed score dispersion is comparable to the anchor's own
split-half sd for the earnings cells (0.0098 vs 0.0098; 0.0193 vs 0.0199)
and below it for remarriage (0.0646 vs 0.0935).

### 2.3 Pivotality

The #269 referee's counterfactual (comment 5039585201, verified against the
artifact): removing the earnings-cell failures alone yields 4/5 PASS;
removing the remarriage failure alone leaves 3/5 FAIL. **The earnings cells
are verdict-pivotal; remarriage is not.**

### 2.4 The forecast inversion, for the record

Registration 11's mandated candid forecast predicted FAIL with
`remarriage.18-64` the failing cell, citing the #251 residual-transport
measurement (111.5/125.4/192.7% of tolerance at train boundaries 2006/2008/
2010, same-direction, worsening toward the scored window). The scored
window measured remarriage at 59-103% of tolerance (median 73.2%). The
transport extrapolation overstated the scored-window residual by roughly a
factor of two. Any future use of boundary-transport measurements as
scored-window forecasts must carry this calibration datum.

### 2.5 What the artifact cannot decide

The scored cells are window-aggregates; the artifact carries no per-year
earnings moments (verified: no per-year moment tables exist under the
earnings families). Therefore the artifact alone cannot separate two
mechanisms for F1:

- **(a) persistence-suppressed growth**: the rank-refresh share leaves rank
  trajectories too sticky, damping the mobility contribution to mean Δlog
  (coherent with F2's excess lag-2 autocorrelation via one mechanism);
- **(b) indexation drift**: the §2.7.6 rank-to-level projected-index law
  under-carries the holdout window's level growth, an error that would
  accumulate over 2015-2022 independently of persistence.

Distinguishing them is the program's first mandatory diagnostic (§5.1) and
is executable train-only.

## 3. Controlling law

Verbatim anchors that bind everything below:

- **§2.7.7 selection discipline (candidate-2 program §5.4):** "No 2015+ row,
  realized post-2014 macro value, candidate-1 seed score, or candidate-2
  score may enter numerical estimation." The one-SE rule text: "select the
  **smallest** `q` with `J(q)≤J(q_min)+SE[J(q_min)]`. Exact ties choose the
  smaller `q`."
- **The W1 surface-license precedent:** a scored failure licenses *which
  surface* the next candidate works, through a public amendment plus referee
  round; it never licenses reading holdout values into estimation or
  selection.
- **One-shot law:** each registered candidate gets one scored run;
  re-running an unchanged candidate against fresh seeds is seed-fishing and
  is prohibited (`fresh_registration_required` binds candidate 3 to a
  registered delta).

## 4. The leakage boundary, drawn explicitly

The candidate-2 verdict is holdout information. The following uses are
**licensed** by precedent: naming the failing cells as candidate 3's target
surface; motivating a train-only diagnostic; adjudicating the #249 surface
question's necessity. The following uses are **prohibited**: re-selecting
q* (or any parameter) because the scored window preferred a different
value; adding scored-window moments to any objective; tuning any tie-break
rule (including the one-SE smallest-q rule) in response to the scored
outcome *and applying it to the same selection evidence*. The boundary
test: if the same numerical choice could not have been justified from
≤2014 evidence under a rule written before the verdict, it is leakage.

A subtlety this program must respect: the c2 selector's objective `J` was
built on the older-worker cells (`earn_dlog_sd.older`, `earn_zero_rate.older`).
A candidate-3 selector whose objective *adds train-window analogs of the
failing moments* (prime-age mean Δlog level; lag-2 autocorrelation) is
train-only in its data and is the standard surface-license move — but the
*decision to include those moments* is verdict-motivated. That is exactly
the W1-precedent shape (scored failure → surface amendment → train-only
estimation), and it must be ratified as a design amendment with this
disclosure, not slipped into a re-run of the old selector.

## 5. Candidate-3 design directions, adjudicated

### 5.1 Mandatory first diagnostic — the F1 mechanism split (train-only)

Before any mechanism amendment is drafted, run and publish a train-only
diagnostic separating §2.5(a) from §2.5(b): within ≤2014, hold out the last
two waves (the 2011/2013-style internal split already used by the c2
selector's pseudo-boundaries), project with the certified engine, and
decompose the mean-Δlog gap into (i) the component explained by the
projected wage-index path versus (ii) the residual conditional on the
index. Publish as a findings artifact (#231 pattern: train-only, no
contract surface). The result routes the design:

- predominantly (b) → the amendment targets the §2.7.6 projected-index law;
- predominantly (a) → the amendment targets the refresh/persistence
  mechanism jointly for F1+F2;
- mixed → both, with the selector extension of §5.2.

### 5.2 Primary direction — selector-objective extension (two-stage §2.7 amendment)

Extend the train-only selector objective `J` to `J'` adding the two
verdict-motivated moments as train-window analogs (prime-age mean Δlog;
lag-2 autocorrelation), re-run the full 21-rung ladder + feasibility +
one-SE machinery unchanged in *rule* but on `J'`, and freeze the selected
`q*'` (which may equal or differ from 0.55) via the same two-stage
amendment + lock-addendum + design-commit re-finalization pattern as
amendment 4. All estimation data ≤2014; the ladder, caps, jackknife SE,
and smallest-q tie-break carry over verbatim. Disclosure: the objective
extension is verdict-motivated (per §4); the numbers are not.

### 5.3 Secondary direction — F2-specific refresh-correlation law

If the §5.1 diagnostic shows the persistence channel dominant and §5.2's
`J'` selection cannot move lag-2 autocorrelation without damaging the
protected cells (feasibility guards), adjudicate a narrow structural
amendment to the refresh law's temporal correlation (e.g., refresh-event
persistence across adjacent periods) — options to be enumerated in the
amendment itself, each with train-only estimation and the q=0
bit-reproduction preflight analog.

### 5.4 Rejected directions

- **Seed-fishing re-run of candidate 2 unchanged** — prohibited (§3).
- **Scored-window re-selection of q\*** — prohibited (§4). Recording for
  the permanent record: the c2 one-SE band was {0.55, 0.60} and the
  smallest-q rule chose 0.55. Whether 0.60 would have scored differently
  is unknowable without leakage and is not a licensed question; the rule
  stands unless a *rule-level* amendment is ratified prospectively and
  applied only to fresh train-only selection evidence under `J'`.
- **Tolerance relief for the earnings cells** — no floors change is
  licensed by this verdict; the floors are candidate-blind by construction
  and the breaches are candidate-side.
- **Remarriage-law surgery inside candidate 3** — see §6.

### 5.5 Sizing the bar candidate 3 must clear

For a ≥4-of-5 verdict, at most one seed may fail any cell. Under the
observed across-seed dispersions (§2.2), the central discrepancy reductions
needed to put the per-seed breach probability at or below the 4-of-5 bar
are approximately: F1 mean score 0.0395 → ≤0.031 (≈ −22%), F2 mean 0.0692
→ ≤0.055 (≈ −20%), assuming unchanged dispersion. These are planning
magnitudes, not acceptance criteria; the gate's own conjunction remains the
only acceptance test.

## 6. Remarriage disposition and the #249 adjudication input

Candidate 3 leaves the remarriage law unchanged (the ratified no-op, #247,
stands). The scored evidence for the #249 candidate-blind surface question:
a real, unanimous +27-51% remarriage over-rate exists, and it is absorbed
by the incumbent 0.403 tolerance in 4/5 seeds; the cell is not
verdict-pivotal (§2.3). Recommendation to the #249 ceremony: proceed
prospective-only as chartered — the surface question retains scientific
value for tolerance semantics — but **certification does not wait on it**,
and no candidate-3 step depends on its outcome. The F3 bias is hereby
documented as the standing measurement the #249 instruments should explain.

## 7. Must-not-regress constraints for candidate 3

Carried forward from the ratified c2 regression block, plus the c2 verdict:

1. The original five candidate-1 cells at their original thresholds
   (`divorce.18-44` 0.379, `earn_dlog_sd.older` 0.269,
   `earn_zero_rate.older` 0.163, `incidence.20-66` 0.404,
   `recovery.20-66` 0.314) — unchanged, per §2.8.4a distinct from the live
   gate values.
2. **New:** the eight cells candidate 2 passed in ≥4 seeds must not
   regress to <4 seeds under candidate 3 (registered as a candid
   constraint in the c3 registration; scored by the unchanged gate
   machinery, no new reducers).

## 8. Candidate-3 protocol

Box ladder to registration (all boxes referee-gated):

1. **This program ratified** (draft PR → adversarial referee → fixes →
   verification → ratify-by-merge).
2. **§5.1 diagnostic findings published** (train-only artifact + PR,
   #231 pattern).
3. **Design amendment(s)** per the diagnostic's routing (§5.2 primary;
   §5.3 only if routed), two-stage with lock addendum + design-commit
   re-finalization.
4. **Selection executed and frozen** (ledger + checksums + reviewed lock,
   amendment-4 pattern), if the routed design selects parameters.
5. **Fresh registration on issue #42**: pins per the registration-11
   template (new design_commit, same floors v4, new spec shas, new
   selection-evidence shas, remarriage no-op sha unchanged); candid
   forecast per house rule — including the §2.4 transport-calibration
   datum; the disclosed re-execution allowance resets per the standing
   contract, with the incident-5028176439 topology mandatory.
6. **One one-shot run** (~23h at candidate-2 cost; plan compute
   accordingly), verdict PR, referee, ratify.

No 2015-2019 row, score, residual, or truth moment enters any numerical
step of boxes 2-4. Box 8-style attestation repeats at registration.
