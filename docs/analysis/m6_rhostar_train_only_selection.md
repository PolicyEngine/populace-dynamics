# M6 candidate-3 train-only rho selection protocol

This prototype implements the ratified correlated-refresh law in
`docs/design/m6_projection_engine.md` §2.7.8. It is a train-only,
non-scoring analysis. It does not register candidate 3, read a candidate-1 or
candidate-2 scored artifact, invoke a gate score, write below `runs/`, or edit
`gates.yaml`.

## Fixed protocol

The harness holds the amendment-4 refresh probability at `q*=0.55` and runs
the 17-rung grid `{-0.80, -0.75, ..., -0.05, 0.00}`. It reuses the three
pseudo-boundaries 2006, 2008, and 2010; fit seed 5200; draw seeds 6200–6219;
fixed halves 6200–6209 and 6210–6219; the six retained earnings cells; the
four-cell standardized objective; both older-worker feasibility guards; and
the 20 delete-one jackknife replicates from §2.7.7.5.

Every rho-by-boundary cell performs a fresh complete candidate-2 QRF refit.
The correlated prototype is an unregistered clone of that fitted generator.
It changes only the threshold applied to the existing code-4 uniform and
adds the nullable generated-lag column `gen_earn_refresh_state`. Candidate 1
and candidate 2 continue through their existing ndarray generation path and
do not receive that column.

## Mandatory preflight pass gate

Before any ladder value is computed, the runner executes and pass-gates:

1. rho-zero byte identity against the production candidate-2 generator at
   every annual generated person-period and participation state for all three
   boundaries and all 20 draws, including all six reduced moments and exact
   final states of streams 1–5;
2. a reset-law fixture whose post-gap code-4 uniform distinguishes `q*` from
   both `p(0)` and `p(1)`;
3. an endogenous-participation fixture in which a later correlated refresh
   changes the carried level read by the unchanged next participation gate;
4. object-level identity proofs for the fitted participation gates,
   marginals, donor pools, projected wage index, rank and inverse maps, old
   frame state, odd-year behavior, and old substreams.

The non-selector-changing preflight command is:

```bash
python scripts/select_m6_rhostar_train_only.py --preflight-only
```

Full mode repeats the same pass gate in the same process and refuses to enter
the rho loop if any leg fails.

## Selection rule

A nonzero rho is retained only when it is feasible and strictly improves on
rho zero in the all-draw objective and both fixed-half objectives. Among rho
zero plus retained rungs, the all-draw argmin resolves an exact tie toward
the value closest to zero. The runner computes the registered delete-one
jackknife standard error at that argmin and selects the retained value closest
to zero within one standard error. If zero is within the cutoff, the published
result is `DESIGNED_PAUSE`; the runner does not force a nonzero result.

The ledger also publishes the train F1 analog for every rung as disclosure
without adding a selection criterion beyond that cell's existing role in
`J`. For every rung, boundary, and draw it publishes stationary entries,
realized 0/1 transition-pair counts, reset-cause counts, and conservation
checks.

## Detached execution outputs

The reviewed code is committed before execution. The detached job writes:

- progress to
  `docs/analysis/m6_rhostar_train_only_selection_progress.txt`;
- the reduced final ledger to
  `docs/analysis/m6_rhostar_train_only_selection_results.json`;
- exact raw stdout outside the repository at
  `~/m6-sol-lanes/c3-proto-rhostar-full.json`.

The progress and final ledger remain uncommitted for the lock-addendum review
ceremony. The expected runtime is approximately 18 hours on the frozen
eight-thread environment.
