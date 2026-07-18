Outcome: (a) RATIFIED_NO_OP_DESIGNED_PAUSE — the frozen reducer reproduced
R0 and `NO_OP_DESIGNED_PAUSE`; the remarriage reduction pause RESOLVES as a
ratified no-op.

# M6 round-6 frozen-stdout reduction finding

## Machine outcome

The one authorized execution of the frozen round-6 reducer consumed the
SHA-pinned round-3 selector stdout, completed every strict validation, and
exited 0. It reproduced the complete raw selection object through canonical
strict-JSON equality and emitted the configured reduced findings JSON.

This is outcome (a), `RATIFIED_NO_OP_DESIGNED_PAUSE`. The raw R0 claim is
ratified by the independent frozen reduction. The remarriage reduction pause
**RESOLVES as a ratified no-op**.

| Origin | Ratified law | Published `per_origin_outcome` |
| --- | --- | --- |
| divorced | R0 | `NONE` |
| widowed | R0 | `NONE` |

The two `NONE` values are the frozen per-origin fields for joint law R0. They
do not mean that the selection is undetermined: the published selected and
selected-joint laws are both R0, status is `SELECTION_COMPLETE`, and the
ratified disposition is `NO_OP_DESIGNED_PAUSE`.

## Frozen authorization and pre-run record

PR #246 was ratified at squash and round-6 freeze
`982d2c251245504b5c114625c4822950d51f7bbc`. Referee issue comment
`5010108053` returned `AMENDMENT APPROVED — freeze-ready`. In read-only
scratch space, the referee had reconstructed the selection over the pinned
public input and recorded `canonical_equal=True`, zero key-presence
differences, and zero value differences. That was prospective evidence only;
it did not execute the authorized reduction.

Before the attempt, the execution log recorded:

- `git diff 982d2c2 -- scripts/ docs/design/`: exit 0, empty stdout;
- frozen reducer SHA-256:
  `94bc5d67f80243b874c425284a676549dc2524d42be6c0b1061c11a52fbfa431`;
- frozen amendment SHA-256:
  `8e1a6c7705ec0338959b881c9386d4da678f9726fdc4aaaceac754cecdee5767`;
- frozen input: 14,606,212 bytes, SHA-256
  `370bef8d00313e847c9cbb0fe09d6b6a5b9dddcf2658449ab61b02a42534614b`;
  and
- CPython 3.13.12, NumPy 2.4.2, pandas 2.3.3, and SciPy 1.17.0 from the
  provisioned interpreter.

Every required byte and runtime value matched. The full worktree was clean
before the execution log was created. No selector, smoke, staged-source
preflight, or real-data process was run.

## One execution attempt

At `2026-07-18T05:50:20Z`, attempt 1 used the provisioned interpreter and
exact frozen reducer:

```text
/Users/maxghenis/m6-sol-lanes/remarr-r2-runtime/bin/python \
  scripts/reduce_m6_remarriage_round6.py \
  < docs/analysis/m6_remarriage_round3_selection_full.json \
  > docs/analysis/m6_remarriage_round3_selection_results.json \
  2> docs/analysis/m6_remarriage_round6_reduction_stderr.txt
```

The process exited 0 in the same recorded UTC second. The attempt is
consumed. It was not rerun, retried, patched, instrumented, stripped,
manually completed, or otherwise rescued.

The canonical results path is the exact stdout capture. Stderr is an exact
zero-byte capture. The round-6 selection-diff diagnostic is emitted only if
raw and independently recomputed selection objects differ; it emitted
nothing. The subsequent mandatory `_assert_same` comparison returned, and
the reducer reached findings serialization.

## Ratified finding

The reduced findings publish this headline:

```json
{
  "status": "SELECTION_COMPLETE",
  "selection": {
    "selected_law": "R0",
    "selected_joint_law": "R0",
    "per_origin_outcome": {
      "divorced": "NONE",
      "widowed": "NONE"
    },
    "disposition": "NO_OP_DESIGNED_PAUSE",
    "selection_reason": "no_eligible_nonzero_law",
    "eligible_nonzero_laws": [],
    "Lbest": null,
    "one_SE_cutoff": null
  }
}
```

The final-information-fit record names R0, reports
`NOT_RUN_R0_SELECTED`, and carries `construction_pass: true` and
`designed_pause_continues: true`. That last field is the frozen R0 finding's
no-op disposition: no nonzero law or final 2014 fit is forced. It does not
undo the resolved adjudication. The reduction question is determined as R0,
and the disposition is the ratified designed no-op.

The reducer independently validated the complete boundary-law-seed cube,
projected means, fixed-block objectives, delete-one jackknife, seven
eligibility rules, one-SE selection, final-fit disposition, frozen byte and
commit bindings, and strict raw schema. It then removed only the 600
publication-group arrays, containing 10,800 records, and retained every
other raw finding. The removed arrays have canonical SHA-256
`599b54ee1fe1bb7ff3864f17e1a632393fcdc5d9e9dd55b5d3e0298f7628d864`.

## Zero new staged-source or PSID contact

This lane read the authorized frozen selector stdout through stdin. A static
enumeration of the reducer's explicit repository reads finds its own frozen
script, the round-3 config and four freeze-lock paths, the round-2 plan and
validation, and the round-1 ledger. It also reads Git metadata for frozen
object validation and imports its pinned runtime libraries.

The findings retain the original selector's historical file-open audit,
including path strings for the sanitized sources used when that already
public stdout was produced. Those strings are embedded records, not files
opened by this round-6 process. No staged or PSID source path occurs in the
round-6 reducer's explicit reads. The selector and round-1 chassis were not
executed or imported, staged-source preflight was not run, and this lane made
zero new staged-source or PSID contacts.

## Publication record

The canonical reduced results JSON is 5,714,733 bytes with SHA-256:

```text
28e635fdd12d090e23066ea836b853af7c7f1760fc80fc4b214b9d529f93bfd0
```

It is the reducer's complete, verbatim stdout at
`docs/analysis/m6_remarriage_round3_selection_results.json`, not a copied,
normalized, or manually reconstructed substitute. It is strict JSON and
records the frozen reducer SHA-256 exactly.

Its `publication.report_path` remains the frozen round-3 config value
`docs/analysis/m6_remarriage_round3_selection.md`. That inherited metadata
was validated and retained without rewriting. The applicable execution
report for this round is this round-6 reduction report.

The captured stderr is 0 bytes with SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
That is the empty-stream digest. The complete pre-run, execution, stream, and
post-attempt record is preserved in
`docs/analysis/m6_remarriage_round6_reduction_execution.txt`.

## Scope and governing disposition

This publication adds only the canonical reduced findings JSON, the
round-6 report, the execution record, and the exact stderr capture under
`docs/analysis/`. It changes no selector, frozen script, amendment, prior
incident or disagreement ledger, `gates.yaml`, `src/`, `tests/`, or `runs/`.

Outcome (a) consumes the one authorized attempt and ratifies the raw R0 /
`NO_OP_DESIGNED_PAUSE` claim. The remarriage reduction pause **RESOLVES as a
ratified no-op**. This finding does not authorize a nonzero law or
registration 8. The frozen candidate-blind W1 surface-question ladder and
the plan's downstream governance continue.

Because this findings publication is a draft, the amendment's applicable
independent findings review and publication ratification remain required
before repository governance treats the record as canonical. That review
does not authorize or require a second reducer execution.
