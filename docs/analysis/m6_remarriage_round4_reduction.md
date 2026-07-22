Outcome: (c) crash/abort — UNRATIFIABLE_FINDING; canonical result NOT_DETERMINED; divorced NONE; widowed NONE; no law or NO_OP_DESIGNED_PAUSE is ratified, and the designed pause continues.

# M6 round-4 frozen-stdout reduction execution finding

## Machine outcome

The one authorized execution of the frozen round-4 reducer exited 1 while
validating the SHA-pinned round-3 stdout. It emitted no stdout and therefore
produced no reduced findings JSON. The attempt is consumed. It was not
rerun, retried, patched, stripped, manually completed, or otherwise rescued.

| Origin | Canonical outcome | Determination |
| --- | --- | --- |
| divorced | NONE | NOT_DETERMINED |
| widowed | NONE | NOT_DETERMINED |

`NONE` is the canonical non-determination caused by the failed mandatory
reduction. It does not ratify the raw selector's R0 claim or its
`NO_OP_DESIGNED_PAUSE` disposition. The canonical selected joint law is
null, the canonical no-op flag is false, the designed pause continues, and
registration 8 remains forbidden.

## Frozen authorization and pre-run record

The amendment was ratified by PR #242 at squash and round-4 freeze
`d63f36b64f58b53f9ff39a2c54b8c682fea93ac5`. Referee issue comment
`5009101768` approved that freeze after a 168-assertion static sweep returned
zero failures. The referee explicitly did not run the real reduction.

Before the attempt, the execution log recorded:

- `git diff d63f36b -- scripts/
  docs/design/m6_remarriage_round4_amendment.md`: exit 0, empty stdout;
- frozen reducer SHA-256:
  `ea050de7716e8a4485bb34bcafdb6701ced1c594072e18233acfbc8f672d633d`;
- frozen input: 14,606,212 bytes, SHA-256
  `370bef8d00313e847c9cbb0fe09d6b6a5b9dddcf2658449ab61b02a42534614b`;
  and
- CPython 3.13.12, NumPy 2.4.2, pandas 2.3.3, and SciPy 1.17.0 from the
  provisioned interpreter recorded in the frozen stdout.

Every required byte and runtime value matched. The worktree was clean before
the execution log was created. No selector, staged-source preflight, smoke,
or real-data process was run.

## One execution attempt

At `2026-07-18T01:27:29Z`, attempt 1 used exactly:

```text
/Users/maxghenis/m6-sol-lanes/remarr-r2-runtime/bin/python scripts/reduce_m6_remarriage_round4.py < docs/analysis/m6_remarriage_round3_selection_full.json > docs/analysis/m6_remarriage_round3_selection_results.json 2> docs/analysis/m6_remarriage_round4_reduction_stderr.txt
```

The process exited 1 after 0.279 seconds. Its exact terminal path was:

```text
module main                  reduce_m6_remarriage_round4.py:2954
  main                       reduce_m6_remarriage_round4.py:2948
    reduce                   reduce_m6_remarriage_round4.py:2857
      _validate_cube         reduce_m6_remarriage_round4.py:2074
        _validate_aggregate_record
                             reduce_m6_remarriage_round4.py:1626
          _validate_group_array
                             reduce_m6_remarriage_round4.py:769
            _validate_mean_record
                             reduce_m6_remarriage_round4.py:729
              _finite_number
                             reduce_m6_remarriage_round4.py:177
ValueError:
  boundaries.2006.laws.R0.mean.groups[11].rate must be a JSON number
```

Before the exception, the reducer validated the config and authority locks,
strict raw schema, frozen bytes and commit, runtime, input audit, publication
contract, and all three fit-validation records. It then validated the 2006
truth and fit locks, the R0 construction and direct record, and all 40 R0
per-seed rows. It recomputed the full R0 mean in memory and validated the
pooled mean, both origin means, and publication groups 0 through 10.

It did not complete group 11, validate either fixed block, finish the
boundary-law cube, recompute the objectives, jackknife, seven rules, or
one-SE selection, compare the raw and reduced selections, validate the final
disposition, remove publication arrays, or serialize reduced findings.

## Bounded defect finding

The first rejected record is the 2006 R0 full-mean group
`widowed|age_18_34|ysd_10_120`. Its 40 contributing seed records, its truth
record, and its aggregate all have zero exposure, zero numerator, and a null
rate. The selector explicitly uses a null rate when exposure is zero and
normalizes non-finite aggregate means to JSON null during serialization.

The reducer accepts that nullable shape for truth and per-seed rate records.
Its aggregate `_validate_mean_record`, however, requires `rate` to be a
finite JSON number before it compares the aggregate with its independent
recomputation. It therefore aborts on the first null aggregate rate.

A bounded, read-only `jq` census of the preserved stdout found 90 full-mean
or fixed-block group records with a null rate: 75 have zero aggregate
exposure and 15 have positive aggregate exposure. This diagnostic describes
the exposed contract surface only. It neither executes the reducer nor
adjudicates any computation or selection beyond the observed failure.

The conservative defect class is
`NULLABLE_AGGREGATE_MEAN_RATE_CONTRACT_MISMATCH`. Because the reducer failed
before its independent selection comparison, the failure is evidence of
neither ratification nor disagreement with the raw claim.

## Raw claim preserved but unratified

The unchanged round-3 stdout has status `SELECTION_COMPLETE`. Its raw fields
claim joint law R0, divorced `NONE`, widowed `NONE`, disposition
`NO_OP_DESIGNED_PAUSE`, and reason `no_eligible_nonzero_law`. The final 2014
fit was not run because the raw selected law was R0.

Those values remain public evidence only. Round 4 did not independently
complete the metrics, rules, jackknife, one-SE choice, or disposition. The
raw claim is not promoted to the canonical outcome.

## Zero-staged-source file-open audit

The round-4 reducer consumed only the pinned stdout through stdin. Its
explicit repository reads before failure were its frozen config, the four
round-3 freeze-hash paths, the round-2 plan and validation, and the round-1
ledger. It also used Git metadata to validate frozen objects and imported its
normal runtime libraries.

No staged or PSID source path appears in those explicit reads. The selector
was not executed, the reducer did not import the selector or round-1 chassis,
and no staged-source preflight ran. The round-4 staged/PSID source-open count
is zero and real-data contact is false. This audit is the invocation record
plus a static enumeration of the frozen reducer's reached file-read paths;
no rerun was used to add a dynamic hook.

## Publication record

Reduced results-JSON SHA-256: **NOT PRODUCED**. The reducer emitted no valid
results JSON. No substitute is committed at the configured canonical path
`docs/analysis/m6_remarriage_round3_selection_results.json`.

The exact captured streams and incident evidence are:

- [stdout capture](m6_remarriage_round4_reduction_stdout.txt): 0 bytes,
  SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- [stderr capture](m6_remarriage_round4_reduction_stderr.txt): 2,114 bytes,
  SHA-256
  `ce7460c38e2b6ae3686e14bb06a719a1a7114906f105aec3128e0ff2b9308521`;
- [execution log](m6_remarriage_round4_reduction_execution.txt): 3,909
  bytes, SHA-256
  `648393e64322288bd89449bc6c6ed6b10f524f1f0c8b9d4021b84fdcae64ebce`;
  and
- [incident ledger](m6_remarriage_round4_reduction_execution_defect.json):
  11,922 bytes, SHA-256
  `81ba7dda672c89519f534ec13e397b01906f7e10ad4d5ee61acc44c2ae3ad236`.

The stdout SHA-256 is the digest of the empty captured stream. It is not a
results-JSON digest.

## Scope and checks

The incident ledger parses as strict JSON. Repository-wide Black at line
length 79 reports 430 files unchanged, and Ruff reports all checks passed.
The diff against `d63f36b` remains empty for `scripts/`, the round-4
amendment, `gates.yaml`, `src/`, `tests/`, and `runs/`. This publication adds
only the five round-4 analysis artifacts listed above. No existing incident
ledger is modified.

## Governing disposition

This is outcome (c), a crash/abort and `UNRATIFIABLE_FINDING`. The one
authorized attempt is consumed. No selector or reducer rerun, patch, field
stripping, manual reduction, output repurposing, or self-rescue was
performed.

The designed pause remains active and registration 8 remains forbidden. A
future attempt requires a prospective amendment, an independent referee
decision, and a new ratifying freeze. This execution supplies no authority
for an in-lane retry or patch.
