Outcome: (b) DISAGREEMENT_VERBATIM — the frozen independent selection and
one-SE recomputation disagreed with the raw selection object; no reduced
results JSON was produced, no side is adjudicated in lane, and the designed
pause continues.

# M6 round-5 frozen-stdout reduction disagreement finding

## Machine outcome

The one authorized execution of the frozen round-5 reducer consumed the
SHA-pinned round-3 stdout and reached the mandatory comparison between the
raw selection object and its independent recomputation. The canonical strict
JSON bytes were unequal. The reducer exited 1 with the exact terminal
diagnostic:

```text
ValueError: selection and one-SE outcome does not match its independent recomputation
```

This is outcome (b), disagreement verbatim. Section 7 of the frozen round-5
amendment leaves exact equality of the complete recomputed metrics,
eligibility, and one-SE tree as the adjudication itself. Section 8 assigns a
mismatch at that residual to the disagreement branch; its incident branch
covers any other exception. The empty stdout is a consequence of the frozen
comparison occurring before publication-group removal and findings
serialization. It does not reclassify this observed disagreement as an
incident.

| Origin | Canonical outcome | Determination |
| --- | --- | --- |
| divorced | NONE | NOT_DETERMINED |
| widowed | NONE | NOT_DETERMINED |

`NONE` is the canonical non-determination after the mandatory raw-versus-
independent comparison returned unequal. It does not promote the raw R0 claim
or ratify `NO_OP_DESIGNED_PAUSE`. The canonical selected joint law is null, no
`NO_OP_DESIGNED_PAUSE` finding is ratified, the designed pause continues, and
registration 8 remains forbidden.

## Frozen authorization and pre-run record

The amendment was ratified by PR #244 at squash and round-5 freeze
`135b165432bffd9a4a30701f75cdc46ac6427b06`. Referee issue comment
`5009765292` returned `AMENDMENT APPROVED — freeze-ready`. The audit's author
verified all 90 repaired null-rate sites and all 810 aggregate group records
with zero violations through strict serialization. The referee expressly did
not run the real reduction.

Before the attempt, the execution log recorded:

- `git diff 135b165 -- scripts/ docs/design/`: exit 0, empty stdout;
- frozen reducer SHA-256:
  `58a8eb2da008739b7b11dcc7eab752d00ed822f0363a1a42ac6067755cea3e00`;
- frozen input: 14,606,212 bytes, SHA-256
  `370bef8d00313e847c9cbb0fe09d6b6a5b9dddcf2658449ab61b02a42534614b`;
  and
- CPython 3.13.12, NumPy 2.4.2, pandas 2.3.3, and SciPy 1.17.0 from the
  provisioned interpreter.

Every required byte and runtime value matched. The worktree was clean before
the execution log was created. No selector, staged-source preflight, smoke,
or real-data process was run.

## One execution attempt

At `2026-07-18T03:49:49Z`, attempt 1 used exactly:

```text
/Users/maxghenis/m6-sol-lanes/remarr-r2-runtime/bin/python scripts/reduce_m6_remarriage_round5.py < docs/analysis/m6_remarriage_round3_selection_full.json > docs/analysis/m6_remarriage_round3_selection_results.json 2> docs/analysis/m6_remarriage_round5_reduction_stderr.txt
```

The process exited 1 by `2026-07-18T03:49:50Z`. The attempt is consumed. It
was not rerun, retried, patched, instrumented, stripped, manually completed,
or otherwise rescued.

Before the comparison returned unequal, the reducer completed the config and
authority locks, strict raw schema, frozen-byte and commit validation,
runtime validation, input/file-open audit validation, publication-contract
validation, the complete boundary-law-seed cube, aggregate and fixed-block
recomputation, delete-one jackknife, seven-rule recomputation, independent
one-SE selection, and final-fit validation. It did not remove publication
arrays or serialize reduced findings.

The exact terminal path was:

```text
module main                  reduce_m6_remarriage_round5.py:3000
  main                       reduce_m6_remarriage_round5.py:2994
    reduce                   reduce_m6_remarriage_round5.py:2937
      _assert_same           reduce_m6_remarriage_round5.py:212
ValueError:
  selection and one-SE outcome does not match its independent recomputation
```

The complete 1,138-byte traceback is preserved verbatim in the stderr
capture.

## Raw side preserved verbatim

The unchanged frozen input remains the verbatim raw side at
`docs/analysis/m6_remarriage_round3_selection_full.json#/selection`. Its
published SHA-256 is
`370bef8d00313e847c9cbb0fe09d6b6a5b9dddcf2658449ab61b02a42534614b`.
Its headline fields are:

```json
{
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
```

The raw final-information-fit object names R0, reports
`NOT_RUN_R0_SELECTED`, and carries `construction_pass: true` and
`designed_pause_continues: true`. These remain public raw claims only. The
round-5 comparison did not establish equality for the complete raw selection
object, so none of its claims is promoted to the canonical result.

## Independent side and disclosure limit

The frozen control flow establishes exactly that an independent selection
object was computed in memory, final-fit validation returned, and its
canonical strict JSON bytes did not equal the raw selection object's bytes.
The reducer did not print that object, name a differing field, or serialize a
partial finding before exit. The only verbatim evidence about the independent
side available from the captured artifacts is:

```text
ValueError: selection and one-SE outcome does not match its independent recomputation
```

The granular recomputed metrics, eligibility tree, one-SE values, and
selected-law fields are therefore not available for verbatim publication.
They are recorded as `NOT_SERIALIZED / NOT_AVAILABLE`, not inferred. Obtaining
them would require a prohibited rerun, patch, instrumentation, or manual
reduction. This lane performs no adjudication of which field differed and
does not force either side toward R0 or any nonzero law.

## Zero-staged-source reachability and file-open audit

The reducer consumed the authorized pinned stdout through stdin. A static
enumeration of the explicit repository reads reached before the comparison
found its own frozen script, the round-3 config and four freeze-lock paths,
the round-2 plan and validation, and the round-1 ledger. It also used Git
metadata for frozen-object validation and imported its normal runtime
libraries.

No staged or PSID source path appears in the explicit reached reads. The
selector was not executed, the reducer did not import it or the round-1
chassis, and no staged-source preflight ran. No dynamic file-open hook was
installed, so this is a static reached-path audit rather than a syscall-level
trace. It identifies zero explicit staged/PSID source reads on the reached
frozen path.

After the attempt, the already-public raw selection and final-information-fit
fields were read only for verbatim publication; no selection arithmetic was
performed. The reducer's authorized frozen-stdout contact and this publication
read are disclosed. Neither is staged/PSID source-data contact.

## Publication record

Reduced results-JSON SHA-256: **NOT PRODUCED**. The reducer emitted no stdout
and therefore no valid results JSON. No substitute is committed at the
configured canonical path
`docs/analysis/m6_remarriage_round3_selection_results.json`.

The exact captured streams and disagreement evidence are:

- [stdout capture](m6_remarriage_round5_reduction_stdout.txt): 0 bytes,
  SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- [stderr capture](m6_remarriage_round5_reduction_stderr.txt): 1,138 bytes,
  SHA-256
  `f24632ad45c3797e012af3d8d3402d1301707116ea0a3834c3157b88a968e9fb`;
- [execution log](m6_remarriage_round5_reduction_execution.txt): 4,665
  bytes, SHA-256
  `d1ba40c3e7afa433e906bee63cd447df186b9e6163f01703caf8cd94ff63833c`;
  and
- [disagreement ledger](m6_remarriage_round5_reduction_disagreement.json):
  11,287 bytes, SHA-256
  `7f24958e47ae669e0ded70f5646e8a3bd799d06b5cba5df88e5cafb452e33996`.

The stdout SHA-256 is the digest of the empty captured stream. It is not a
results-JSON digest.

## Scope and checks

This publication adds only the round-5 disagreement report, ledger,
execution log, and captured stdout/stderr. It does not modify the selector,
frozen round-2/3/4/5 scripts, a prior incident ledger, `gates.yaml`, `src/`,
`tests/`, or `runs/`.

The disagreement ledger parses as strict JSON. Black at line length 79
reports 432 files unchanged, Ruff reports all checks passed, and
`git diff --check` passes. The diff against `135b165` remains empty for
`scripts/`, `docs/design/`, `gates.yaml`, `src/`, `tests/`, and `runs/`.

## Governing disposition

This is outcome (b), disagreement verbatim. The one authorized attempt is
consumed. The raw side, exact terminal diagnostic, and every emitted byte are
preserved. The independent object was not emitted and is not reconstructed.
No selector or reducer rerun, patch, field stripping, manual reduction, or
self-rescue was performed.

Neither the raw claim nor an independent result is ratified in this lane. The
canonical determination remains `NOT_DETERMINED`, the designed pause remains
active, and registration 8 remains forbidden. The applicable independent
findings review and ratification must precede any canonical disposition.
