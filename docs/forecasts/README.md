# Timeline forecast ledger

This directory holds the campaign's schedule forecasts under the same
discipline the evaluation gates use: registered before the fact, decidable
by a stated criterion, never edited after registration, and graded in
public when they resolve.

The ledger is `timeline_ledger.json`. `tests/test_forecast_ledger.py`
validates its structure in CI.

## Rules

1. **Append-only.** A registered entry is never edited except to fill its
   `resolution` block when it resolves. Changing an estimate means
   registering a NEW entry whose `supersedes` field names the old id and
   whose `revision_reason` states the evidence that moved it. The
   superseded entry's `status` becomes `superseded` — its dates stay
   untouched and it is still graded at resolution time against reality,
   so revision cannot launder a miss.
2. **Decidable criteria.** Every entry states a `resolution_criterion` a
   third party could check from the public record (a merged PR, a
   published verdict artifact, a run in `runs/`). No vibes.
3. **Two dates per entry.** `p50` and `p80` calendar dates (UTC): the
   forecaster's 50% and 80% confidence dates for the criterion being met.
4. **Grading.** At resolution, fill `resolution` with `resolved_at`, the
   evidence URL/sha, `error_days_p50` (signed: positive = late), and
   whether the p50/p80 bounds held. The running calibration table below is
   recomputed from the ledger — never hand-edited out of sync.
5. **Cadence.** Register a revision when material evidence moves a p50 by
   more than ~1 day, and resolve entries promptly when their criteria are
   met. Registering forecasts is the orchestrator's job; nothing here is
   part of the evaluation contract in `gates.yaml`, and nothing here may
   cite an unresolved forecast as evidence in any gate ceremony.

## Calibration record

Recomputed from resolved entries; append rows, never rewrite history.

| id | claim (short) | p50 | p80 | resolved | error vs p50 (days) | within p50? | within p80? |
|----|---------------|-----|-----|----------|--------------------:|-------------|-------------|
| (none resolved yet) | | | | | | | |
