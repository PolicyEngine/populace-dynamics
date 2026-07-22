Outcome: UNRATIFIABLE_FINDING — divorced NONE; widowed NONE; the selector's raw stdout claimed R0 / `NO_OP_DESIGNED_PAUSE`, but the frozen reducer failed before validating that claim, so no law or no-op disposition is ratified and the designed pause continues.

# M6 round-3 train-only remarriage reduction execution finding

## Machine outcome

The one permitted frozen selector attempt completed and emitted a full JSON
document. Its raw claim was R0 with `NO_OP_DESIGNED_PAUSE` because no nonzero
law was eligible. The required frozen deterministic reducer then exited 1
before independently validating the boundary-law-seed cube or recomputing the
selection.

The ratified procedure requires both the full selector JSON and its
deterministic reduced ledger. It also requires stop and disclosure for a new
post-contact protocol defect. The raw R0 claim is therefore preserved as
evidence but is not promoted to the investigation outcome.

| Origin | Canonical outcome | Determination |
| --- | --- | --- |
| divorced | NONE | NOT_DETERMINED |
| widowed | NONE | NOT_DETERMINED |

Here `NONE` means that no reducer-validated outcome was determined. It does
not mean that R0 or `NO_OP_DESIGNED_PAUSE` is ratified. The canonical selected
joint law is null, canonical `NO_OP_DESIGNED_PAUSE` is false, the designed
pause continues, and registration 8 remains forbidden.

## Frozen execution

- Round-3 freeze: `841c48c6f0316a08f7584402012f00b0a2b535d9` on
  `sol/remarr-r3-run`.
- `git diff 841c48c -- scripts/` was empty immediately before launch.
- Pre-run selector SHA-256:
  `fb07ac198105138e5ce9a89daba96295cadf17bf651fab86029794c87c0c33c3`.
- Pre-run reducer SHA-256:
  `41db8a271d08596ad6383494f37695e658b6e4f7531459d95221f684b09c256e`.
- Pre-run config SHA-256:
  `f52c30653001afeed2822fda382dab86078b0fce884199bc5141b1873b7b16ac`.
- Runtime: CPython 3.13.12, NumPy 2.4.2, pandas 2.3.3, SciPy
  1.17.0, scikit-learn 1.8.0, and quantile-forest 1.4.2.
- Standard runtime environment: no thread or worker cap was added.
- Hardened smoke: `SYNTHETIC_SMOKE_PASS` with no staged-data or candidate
  contact.
- Candidate-blind preflight: `PRE_OUTCOME_PREFLIGHT_PASS`; all three frozen
  fit-side boundary validations matched.
- Launch load gate: `PROCEED` at load-5 18.93, below the threshold of 40.
- Selector attempts: one, from 17:53:05 to 17:54:34 EDT, exit 0. Selector
  reruns: zero.
- Reducer attempts: one, from 17:55:03 to 17:55:04 EDT, exit 1. Reducer
  reruns or patches: zero.

The selector and reducer ran with `PYTHONDONTWRITEBYTECODE=1` under the pinned
round-2 Homebrew 3.13.12 runtime. Output redirection was outside the worktree,
so the selector's committed-clean-freeze assertion passed.

## Raw selector claim — unratified

The preserved raw JSON has schema
`m6.remarriage.learning_plan.round3.selection.full.v1`, status
`SELECTION_COMPLETE`, and one machine document. It contains all 15
boundary-law arrays and 40 seeds per array, for 600 CRN projection draws.

Its raw selection fields are:

- selected joint law: `R0`;
- per-origin raw outcome: divorced `NONE`, widowed `NONE`;
- disposition: `NO_OP_DESIGNED_PAUSE`;
- reason: `no_eligible_nonzero_law`;
- eligible nonzero laws: none; and
- final-information-fit status: `NOT_RUN_R0_SELECTED`.

Every nonzero law passed raw rules 1 through 4 and failed raw rules 5, 6, and
7. These are disclosed selector claims, not independently reduced findings.
The reducer failed before recomputing the metrics, seven rules, jackknife,
one-SE choice, or final disposition, so this lane cannot publish the raw R0
claim as a no-op result.

The full selector audit reports maximum selection information year 2014, no
selection-relevant 2015--2019 read, no `gates.yaml` or `runs/` read, no M6
scorer import, and no helper write. The reducer validated the input/file-open
audit before it reached the failure.

## Frozen reducer failure

The exact terminal path was:

```text
module main                  reduce_m6_remarriage_round3.py:2792
  main                       reduce_m6_remarriage_round3.py:2786
    reduce                   reduce_m6_remarriage_round3.py:2695
      _validate_cube         reduce_m6_remarriage_round3.py:1730
        _validate_fit_public reduce_m6_remarriage_round3.py:1170
          _assert_same       reduce_m6_remarriage_round3.py:212
ValueError:
  fit_validation.2006.reference_spells does not match its
  independent recomputation
```

Before this exception, the reducer validated the config and authority locks,
raw top-level schema, freeze bytes and commit, runtime identity, input and
file-open audit, and publication contract. It then entered cube validation
and stopped at the first fit record. It did not finish the cube or perform
the independent selection and final-fit checks.

## Defect

The selector's fit validation intentionally uses expected-subset semantics.
For each boundary and origin, its public `reference_spells` object contains
every field from the frozen validation JSON plus seven audit fields:

- five category checksums: duplicate, included, missing/nonpositive,
  no-potential-path, and same-year-remarriage; and
- two excluded-weight fields: missing/nonpositive and no-potential-path.

The reducer instead compares the entire expanded `reference_spells` mapping
for canonical equality with the narrower frozen validation mapping. It
therefore rejects the additive fields at boundary 2006. The reducer's later
code separately expects the checksum fields when validating
`reference_exclusion_category_hashes`, but that code is unreachable after the
earlier exact comparison.

A read-only comparison of the preserved stdout and validation JSON found the
same seven raw-only keys in all six boundary-origin records, no validation-only
keys, and equality for every shared frozen field. This bounds the observed
failure as an additive validation-schema mismatch; it does not replace the
frozen reducer, independently ratify the selection, or permit stripping fields
and retrying.

## Publication record

The exact selector stdout is 14,606,212 bytes with SHA-256
`370bef8d00313e847c9cbb0fe09d6b6a5b9dddcf2658449ab61b02a42534614b`.
It is published unchanged at the configured full-output path. The selector
stderr/progress is 4,837 bytes with SHA-256
`b3a7462d200ed92321ab4d1fb2d8643384ea0516f8a7fb64afe9b97e533bbe3a`.

The reducer emitted zero stdout bytes, whose SHA-256 is
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
Because it produced no findings JSON, nothing is committed at the configured
`m6_remarriage_round3_selection_results.json` path. No substitute reduction
was fabricated. The verbatim reducer stderr is 1,631 bytes with SHA-256
`c9a74b4dd8961e9e5a983af4e09ed012b87101f3dc79d62a5ab1368544bf4071`.

The published evidence is:

- [Incident ledger](m6_remarriage_round3_selection_execution_defect.json),
  SHA-256
  `8de73e6e30bc973697e91d17b688959041fc5ebf0be6ff291d9887f1cec2a220`;
- [Exact full selector stdout](m6_remarriage_round3_selection_full.json);
- [Exact selector stderr](m6_remarriage_round3_selection_progress.txt);
- [Exact reducer stderr](m6_remarriage_round3_selection_execution_defect.txt);
- [Frozen hardened smoke](m6_remarriage_round3_selection_smoke.json); and
- [Frozen candidate-blind preflight](m6_remarriage_round3_selection_preflight.json).

## Governing disposition

Sections 6 and 7 of the
[ratified plan](../design/m6_remarriage_learning_plan_round2.md) require
publication of the deterministic reduced ledger and require stop/disclosure
for a new defect after candidate contact. The
[round-3 amendment](../design/m6_remarriage_round3_amendment.md) likewise
forbids self-rescue or repurposing partial procedure output.

No selector or reducer rerun was performed. No output field was deleted, no
manual reduction was used, and no script, config, source, test, gate, floor,
registry, or `runs/` artifact is changed. A future investigation requires a
prospective reducer amendment, a new referee decision and freeze, and a new
investigation without reusing this unratified raw result. Until then, the
designed pause continues.
