# M6 amendment 6 candidate-3 ρ* lock addendum

- **Addendum id:** `2026-07-23-m6-amendment-6-rhostar-lock`
- **Parent authority:** §2.7.8 of
  `docs/design/m6_projection_engine.md`, ratified in [PR #272][amendment]
  at `5b5c9c641d0cf38926d42afb839b45434c4b1b60`
- **Ceremony stage:** `DRAFT_NOT_OPERATIVE`. This document publishes the
  train-only finding and proposes the lock for review. It is not ratified law,
  a production implementation, a candidate-3 registration, a score, or run
  authorization.
- **Implementation and referee record:** [PR #273][prototype] at
  `097038bc67c4bc006288f67aeaa0183a947a1034`, referee
  `VERIFIED-WITH-NOTES` in [issue comment 5043789093][referee]
- **Frozen selector commit:**
  `097038bc67c4bc006288f67aeaa0183a947a1034`
- **Proposed selected value:** `ρ*=-0.60`
- **Exact raw-ledger SHA-256:**
  `c444c46a8df5f61609519a7ad66a79cd7cd020715e503082976409a922e5c434`
- **Committed findings ledger:**
  [`m6_rhostar_train_only_selection_results.json`][ledger], file SHA-256
  `db7fe83547cad6a4ea477bac9f71f11279e9f21c8399b60d8f77a5ca14d463ff`
- **Freeze/ledger/reducer precedent:** the amendment-4
  [q-star lock addendum][qstar-lock], ratified in [PR #255][qstar-pr] at
  `aaebfb3f988b95bdcbf73da4164316b48da2c542`

## 1. Proposed lock

Subject to referee review and orchestrator ratification, this addendum freezes
the following as one indivisible amendment-6 candidate-3 law:

1. the within-person refresh correlation is
   **`ρ*=-0.60`** at the already-frozen `q*=0.55`;
2. the complete selection evidence is the raw strict-JSON stdout whose
   SHA-256 is
   `c444c46a8df5f61609519a7ad66a79cd7cd020715e503082976409a922e5c434`,
   reduced to the committed findings ledger whose whole-file SHA-256 is
   `db7fe83547cad6a4ea477bac9f71f11279e9f21c8399b60d8f77a5ca14d463ff`;
3. the code-4 threshold and person-level refresh-state law are exactly §2
   below; and
4. earnings substream codes 4 and 5 and their unconditional
   eligible-continuer consumption law remain exactly §3 below.

There was no post-outcome tuning. The selected value is the direct output of
the 17-rung selector frozen at `097038b`.

## 2. Exact correlated-refresh and refresh-state law

For every eligible positive continuer, the only amended primitive is the
threshold applied to the existing code-4 uniform:

```text
B_refresh = 1{ U₄ < p(B_prev_state) }
p(∅)  = q*                      (no prior eligible indicator)
p(1)  = q* + ρ(1−q*)
p(0)  = q*(1−ρ)
```

Here `q*=0.55`, `ρ=-0.60`, and `B_prev_state ∈ {∅, 0, 1}`. The state law is:

1. initialize to `∅` at the first eligible positive-continuation
   transition after the anchor, so the stationary threshold `q*` applies;
2. after each eligible transition, carry the realized `B_refresh` to the
   person's next transition; and
3. reset to `∅` on a gap: non-participation, a zero-earnings year,
   stream-3 re-entry, or exit from the `[25,64]` support.

The chain therefore correlates only consecutive eligible transitions. At
`ρ=0`, all three thresholds equal `q*` and the generator is bit-identical
to the ratified candidate-2 generator. The stable-coordinate donor law,
conditional donor law, participation feedback, re-entry law, `CellMarginal`,
projected wage index, inverse-rank map, support, binning, biennial timing, and
odd-year carry remain byte-carried under §2.7.8.3.

The [PR #273 referee][referee] adjudicated the support-exit endpoint as note
N1. When a continuer's target age exits `[25,64]`, the implementation masks
the carried refresh state to `∅` before applying that same exiting
transition's threshold. The transition therefore uses `q*` and degrades
exactly to incumbent candidate-2 behavior.
This choice is out of the scored support: truth and projection both restrict
scored rows to ages 25–64, and the selector asserts their support checksums
on every draw. Monotone age prevents old-side re-entry, generation remains
person-local, and the threshold-independent draw discipline preserves common
random numbers. The committed support-exit tests and each rung's
`resets.support_exit` counts pin this adjudication.

## 3. Unchanged RNG registry lock

The earnings registry remains exactly:

```text
1: gate
2: donor-draw
3: re-entry-draw
4: memory-refresh-gate
5: memory-refresh-rank
```

For every eligible positive continuer in canonical person order, code 4 draws
one refresh uniform and code 5 draws one stable-donor uniform regardless of
`ρ` or the realized refresh outcome. Codes 1–5, their addresses, states,
and draw order remain unchanged. All ρ-grid rungs use common random numbers
at each fixed draw seed. Odd years retain deterministic carry and consume no
RNG. Candidate-3 registers no new substream.

## 4. Frozen selector and decision

The selector used exactly:

```text
RHO_GRID = {-0.80, -0.75, ..., -0.05, 0.00}
q = q* = 0.55
PSEUDO_BOUNDARIES = {2006, 2008, 2010}
FIT_SEED = 5200
SELECTION_DRAW_SEEDS = {6200, ..., 6219}
FIXED_HALVES = {6200, ..., 6209} / {6210, ..., 6219}
FLOOR_SEEDS = {0, ..., 99}
```

All 17 rungs were valid and feasible. Every nonzero rung strictly improved on
`ρ=0` in the all-20 objective and both fixed halves, so the retained pool is
the complete grid, including the baseline.

| Quantity | Result |
|:---|---:|
| `J(0)` | 81.53194122143351 |
| retained ρ pool | -0.80, -0.75, ..., -0.05, 0.00 |
| `ρ_min` | -0.80 |
| `J(ρ_min)` | 74.62524158208294 |
| ρ-min delete-one mean | 74.63218711790704 |
| ρ-min jackknife SE | 1.6519902288232524 |
| one-SE cutoff | 76.27723181090619 |
| retained rungs within one SE | -0.80, -0.75, -0.70, -0.65, -0.60 |
| closest ρ to zero within one SE | **-0.60** |
| disposition | `LOCK_ADDENDUM_ELIGIBLE` |

The executed rule must agree with §2.7.8.5 in this exact order:

1. retain `ρ=0` after the global validity gate;
2. retain a nonzero rung only if it is feasible and strictly improves on
   `ρ=0` in the all-draw objective and both fixed-half objectives;
3. minimize the all-draw objective over that retained pool, resolving an
   exact argmin tie toward zero;
4. compute the registered delete-one jackknife standard error at that
   argmin; and
5. select the retained rung closest to zero satisfying
   `J(ρ) ≤ J(ρ_min) + SE[J(ρ_min)]`.

The independent execution trace recovered the complete 17-rung retained pool.
The all-draw argmin tie set was the singleton `{-0.80}`, so the fixed
tie-toward-zero rule resolved `ρ_min=-0.80` without an objective tie. Applying
the registered jackknife cutoff admitted exactly
`{-0.80, -0.75, -0.70, -0.65, -0.60}`. The closest-to-zero member is
`-0.60`, so the fixed rule returns **`ρ*=-0.60`**.

Weakening only the all-draw improvement comparison from `<` to `≤`, while
leaving both half-draw comparisons strict, retained the same 17-rung pool and
also selected `ρ*=-0.60`.

## 5. Evidence and equivalence record

The completed execution supplies the required proofs:

- `ρ=0` reproduced the ratified candidate-2 generator bit for bit across
  all three boundaries and all 20 draw seeds: person-period keys, levels,
  participation states, all six moments, support, and final stream-1–5 states
  matched;
- the reset-law discriminator and endogenous-participation fixture passed
  before any ladder value was computed;
- the object-level proof confirmed the unchanged participation
  formula/fit/coefficients, `CellMarginal`, `I_proj` and its leakage fence,
  inverse-rank map and corner semantics, frame state, odd-year behavior, and
  pre-existing substreams;
- all 51 ρ-by-boundary fits were fresh complete QRF refits with fit seed
  5200, and each per-rung fit signature matched the pinned q-star ledger;
- every selected cell was defined and regenerated across all 20 draws, each
  projection began from fresh state, every projection support checksum
  equaled its boundary truth-support checksum, and all transition-chain
  conservation checks passed;
- the committed reducer independently revalidated the preflights, protocol,
  fences, rung structure, support hashes, and transition conservation;
- a separate numeric recomputation reproduced feasibility, all-draw and
  half-draw improvement, the retained pool, exact argmin tie-break,
  delete-one jackknife, one-SE pool, closest-to-zero choice, and final
  disposition without trusting the selector summary; and
- the ledger publishes the realized transition-pair counts entering every
  rung's chain and the train-F1 analog for every rung as disclosure with no
  additional selection role.

The execution passed 60 of 60 `ρ=0` boundary-by-draw equivalence cells before
starting the ladder. Across the 51 fits, the eight exact-age-bin stable pools
were nonempty; the minimum held 3,201 records. Each boundary carried one
invariant fit signature across all 17 rungs. The run exited zero after 51
refits and 1,020 rung-by-boundary draws. The attempt-4 wrapper preserved the
complete raw stdout but did not publish the reduced findings path. After
confirming that the selector and wrapper had exited, the Stage 2 process ran
the committed reducer and its full validation path once against those raw
bytes. A second reducer pass reproduced the findings file byte for byte.

## 6. Execution incidents and reproduction conditions

The ladder completed only on attempt 4. The coordinator's
[launch-mechanics disclosure][incident] and the local execution receipts
support the following incident chain, which is part of this lock record:

1. Attempt 1 used the default OpenMP blocktime. After 7h39m with 2.3 rungs
   complete, it permanently livelocked in an OpenMP join barrier.
   Stack samples established the stable join-barrier state.
2. The restart guard then caught fitting-stack `HEAD` drift introduced by a
   concurrent lane. The guard worked as designed. The coordinator resolved
   the drift with a lane-local clone of the populace repository pinned at
   `ee8f7fc139271de5d4e448549c35e8c5eb992534`, verified both populace-fit and
   populace-frame tree SHAs against the selector pins, and repointed the
   virtual environment to that clone.
3. Attempt 2 set `KMP_BLOCKTIME=0` and `OMP_WAIT_POLICY=PASSIVE`. It ran
   approximately three times slower than the active/default schedule and was
   killed.
4. During attempt 3, incident response established that the lane's original
   launcher was an undisclosed launchd `KeepAlive` job,
   `org.policyengine.m6.c3proto.ladder`. It respawned killed trees and briefly
   raced a duplicate process on the same output files. The coordinator
   removed it with `launchctl bootout`, killed every ladder tree, and cleaned
   the output files. Before attempt 4, no
   `docs/analysis/m6_rhostar_train_only_selection_results.json` artifact
   existed at any point, so no partial ledger could contaminate the
   selection. The sealed selector regenerates every preflight, fit, and rung
   in process.
5. Attempt 4 was the completed run: one process tree under
   `nohup + caffeinate -sim`, the exact eight-variable eight-thread pins, and
   `OMP_WAIT_POLICY=ACTIVE`. The policy change is scheduling-only: the thread
   count was unchanged, and numerics and the fit signature were preserved.
   Every rung asserted its fit signature against the pinned q* ledger.

The findings ledger records the exact eight-variable thread block and the
pinned editable fitting-stack paths. Attempt 4 additionally used
`OMP_WAIT_POLICY=ACTIVE`; this addendum freezes that policy with the recorded
block as the artifact's reproduction conditions:

```text
python=3.14.4
numpy=2.5.1
pandas=3.0.3
scikit-learn=1.8.0
scipy=1.18.0
quantile-forest=1.4.2
populace-fit=0.1.0
populace-frame=0.1.0
policyengine-us=1.752.2
policyengine-core=3.30.3
LOKY_MAX_CPU_COUNT=8
POPULACE_FIT_N_JOBS=8
POPULACE_FIT_PREDICT_WORKERS=8
OMP_NUM_THREADS=8
OPENBLAS_NUM_THREADS=8
MKL_NUM_THREADS=8
VECLIB_MAXIMUM_THREADS=8
NUMEXPR_NUM_THREADS=8
OMP_WAIT_POLICY=ACTIVE
populace-fit editable source = /Users/maxghenis/m6-sol-lanes/populace-pinned/packages/populace-fit
populace-frame editable source = /Users/maxghenis/m6-sol-lanes/populace-pinned/packages/populace-frame
populace repository HEAD = ee8f7fc139271de5d4e448549c35e8c5eb992534
populace-fit tree = 5c866378fdf5906b7a61da9977b8d028d1d36e9f
populace-frame tree = 7cfb9ee78beb74911963913f202a4471aae2f52b
POPULACE_DYNAMICS_PSID_DIR = /Users/maxghenis/PolicyEngine/psid-data
POPULACE_DYNAMICS_PE_US_DIR = /Users/maxghenis/PolicyEngine/social-security-model-worktrees/sol-c3-proto/.venv/lib/python3.14/site-packages
```

`OMP_WAIT_POLICY=ACTIVE` is a reproduction condition for the completed
artifact even though it is scheduling-only. It does not amend the frozen
model or selector.

## 7. Information bright line and exit fence

Every selection-relevant earnings reference row was dated no later than 2014.
The retrospective product's collection wave 2015 supplied only verified
reference-year-2014 labor fields and incumbent same-row demographic
covariates. No 2015 earnings-reference field or later family file was
requested. Each boundary fit admitted earnings only through the boundary and
read the certified NAWI file only through that boundary, so realized
post-boundary macro values never entered the selection.

No candidate-1 seed score, candidate-2 seed score, candidate-1 or candidate-2
scored artifact, gate tolerance, unpublished holdout result, gate
configuration, or existing run artifact entered the selection. The ledger
carries `no_candidate_1_or_candidate_2_artifact_read=true`,
`no_gate_score=true`, and `no_runs_write=true`.

The [PR #273 referee's][referee] note N2 is resolved at process-exit
adjudication: `no_runs_write` is computed against the process's exit state,
not inferred solely from the literal fence field. At
`2026-07-23T12:08:03-0400`,
`git status --porcelain=v1 --untracked-files=all -- runs/` returned zero
bytes, whose SHA-256 is
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The tracked diff was clean, the untracked-path count was zero, and no file
under `runs/` was newer than the attempt-4 launch record. The exit
adjudication therefore computed `no_runs_write=true`.

## 8. Accepted §2.7.7.7 notes with candidate-3 names

The following governance notes are explicit and indivisible from the proposed
lock:

1. At this lock addendum's ratification, the orchestrator changes **exactly
   one field**, `gate_m6.design_commit`, from the live amendment-5 pin
   `0e067a910fde7e479240c472087ece6a7ce29bcd` to amendment 6's exact
   squash-merge commit
   `5b5c9c641d0cf38926d42afb839b45434c4b1b60` before the candidate-3
   registration. This Stage 2 commit performs no gate edit. `design_pr`,
   `design_commit_note`,
   `design_verified_comment`, every cell and reducer, every tolerance and
   floor binding, all seeds, `K`, the 4-of-5 conjunction, and every other
   gate byte remain unchanged.
2. That one-field flip intentionally leaves the pre-amendment gate narratives
   stale, including `gates.yaml:5324-5346`, the `gates.yaml:5354-5361`
   `design_commit_note`, and the amendment-4/5 history entries. They narrate
   superseded pins as endpoints. No narrative byte is an authorized edit.
3. Writing a placeholder or anticipated merge hash into `gates.yaml` is
   prohibited. Only the orchestrator writes the exact ratified amendment-6
   squash-merge commit during the one-field lock flip.
4. The strict-versus-weak note is restated in §2.7.8.5. If
   `J(0) ≤ J(ρ)` for a nonzero rung and that rung satisfies the one-SE
   cutoff, retained `ρ=0` does too, and the closest-to-zero rule returns
   `ρ=0`, the designed pause.
5. After the candidate-3 registration, `ρ*`, the ledger and its SHA, the
   refresh-state law, and every registered implementation byte are immutable.
   A designed pre-score abort cannot authorize a silent retry. Any changed
   byte requires a reviewed fix and fresh registration before another run.

## 9. Operative boundary and next action

This addendum is **`DRAFT_NOT_OPERATIVE`**. Its publication does not itself
freeze production bytes, register candidate 3, authorize a score, or certify
any projection. The selector returned the nonzero value `ρ*=-0.60`, so the
designed-pause clause is not triggered. That result makes this proposal
lock-addendum-eligible; it does not promise a gate pass or waive any remaining
ceremony.

The proposed `ρ*` pin becomes operative only through referee review and
the orchestrator's ratification action. At that ratification, the orchestrator
performs the one-field design-commit flip in §8; this PR performs none of it.
No production or registered candidate-3 implementation, registration, or
score may precede all four §2.7.8.7 steps. The train-only, non-scoring
prototype authorized by the ratified amendment is the sole exception.

Any candidate-3 registration must restate the registration-8 one-run terms
verbatim, adopt the post-2014 attestation wording, and carry the candidate-3
program's §2.4 transport-calibration datum with its pinned comparators and the
§2.7.8.1 conditioning caveat.

A later accepted candidate-3 PASS would first-certify the
correlated-refresh forward law only on the registered 2016/2018 `gate_m6`
surface. No `gate_1` certificate transfers, and no result certifies 2100
earnings or any report-only path.

[amendment]: https://github.com/PolicyEngine/populace-dynamics/pull/272
[prototype]: https://github.com/PolicyEngine/populace-dynamics/pull/273
[referee]: https://github.com/PolicyEngine/populace-dynamics/pull/273#issuecomment-5043789093
[incident]: https://github.com/PolicyEngine/populace-dynamics/pull/273#issuecomment-5050319960
[qstar-lock]: ../amendments/m6_amendment_4_qstar_lock_addendum.md
[qstar-pr]: https://github.com/PolicyEngine/populace-dynamics/pull/255
[ledger]: ../analysis/m6_rhostar_train_only_selection_results.json
