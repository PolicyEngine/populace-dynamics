Outcome: UNRATIFIABLE_FINDING — divorced NONE; widowed NONE; the selector did not determine a law or `NO_OP_DESIGNED_PAUSE`, and the designed pause continues.

# M6 round-2 train-only remarriage selection execution finding

## Machine outcome

The frozen selector exited with `KeyError: 'weight'` after its first
candidate-outcome contact. It emitted zero stdout bytes, completed no law
record, and made no selection. This is a selector execution defect, not an R0
selection and not a train-only no-op result.

The ratified plan requires a stop and disclosure when a selector defect appears
after first candidate-outcome contact. No rerun was performed, the empty stdout
was not repurposed, and the deterministic reducer was not run. Under the
plan's frozen no-op ladder, this unratifiable finding leaves the designed pause
active.

| Origin | Published outcome | Selector determination |
| --- | --- | --- |
| divorced | NONE | NOT_DETERMINED |
| widowed | NONE | NOT_DETERMINED |

Here `NONE` means that no per-origin law was selected because the joint
selector did not complete. It does not mean that R0 won. The joint selected law
is null, `NO_OP_DESIGNED_PAUSE` is false, and registration 8 remains forbidden.

## Frozen execution

- Freeze commit: `f44246964b4e54080ab4c7b43e7e5b0c4b78ea7c`.
- Runtime: CPython 3.13.12, NumPy 2.4.2, pandas 2.3.3, SciPy 1.17.0,
  scikit-learn 1.8.0, and quantile-forest 1.4.2.
- Synthetic smoke: `SYNTHETIC_SMOKE_PASS`, with no staged-data read, pseudo
  truth construction, or candidate outcome.
- Candidate-blind preflight: `PRE_OUTCOME_PREFLIGHT_PASS`; all 2006, 2008,
  and 2010 fit-side rows matched the frozen validation.
- Load gate immediately before execution: `PROCEED`, load-5 35.00, below the
  threshold of 40.
- Full selector attempts: one. Disclosed re-executions used: zero.

The preflight's field-aware audit caps candidate information at 2014, excludes
calendar-2014 flow, and records no selection-relevant 2015--2019 read. The
effective fit maxima are 2005, 2007, and 2009 for the three pseudo boundaries.
It also publishes the dependency versions, frozen-tree import paths, and source
audit.

The full-run dynamic file-open hook was installed before selector reads, but
the helper serializes its accumulated path set and final forbidden-path flags
only while assembling a completed payload. The exception occurred before that
step. The exact path set is therefore unavailable and cannot be reconstructed
without a forbidden rerun; this missing full-run audit finding is disclosed
rather than inferred.

## How far the selector got

The full run loaded the sanitized staged sources and repeated the fit-side
validation for boundaries 2006, 2008, and 2010. It then crossed the outcome
freeze at boundary 2006 by constructing the locked pseudo truth and entering
R0 direct standardization.

Before the exception, it validated the boundary-2006 expected support, built
40 raw transition-uniform checksums, and completed the pooled and two
origin-level portions of the R0 direct diagnostic. It then entered the 18-cell
publication-group loop. It did not complete the R0 diagnostic or any R0 law
record.

No CRN projection draw ran. No spouse-gap draw, nonzero law, 2008 or 2010
pseudo truth, eligibility rule, objective, jackknife, one-SE selection, or
final-2014 information fit was reached.

## Defect

The exact terminal path was:

```text
_evaluate_boundary                 select_m6_remarriage_round2.py:2729
  _direct_standardization
    publication-group record       select_m6_remarriage_round2.py:2255
      selected_unmatched["weight"] select_m6_remarriage_round2.py:2179
KeyError: 'weight'
```

For an event-free publication group, the list-comprehension mask is empty.
pandas interprets `selected_events[[]]` as a zero-column selection rather than
a zero-row Boolean selection. The following `weight` lookup therefore fails.
The source event frame did contain `weight`; the identity of the first empty
group was not emitted and is not inferred here.

## Publication record

The selector stdout is exactly 0 bytes with SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
Because it is not JSON, no file is committed at the configured
`m6_remarriage_round2_selection_full.json` path. The frozen reducer requires a
completed raw schema, so it was not run and no substitute is committed at the
configured `m6_remarriage_round2_selection_results.json` path.

The published evidence is:

- [Execution-defect ledger](m6_remarriage_round2_selection_execution_defect.json)
- [Exact selector stderr](m6_remarriage_round2_selection_execution_defect.txt)
- [Frozen synthetic smoke](m6_remarriage_round2_selection_smoke.json)
- [Frozen candidate-blind preflight](m6_remarriage_round2_selection_preflight.json)

The exact stderr is 4,365 bytes with SHA-256
`6c02d24548b8687bb5e81a8ca347ca5fbe2b1a235e5f6e1e99e3f2f61df60c3c`.
The incident ledger records all freeze, authority, runtime, information-boundary,
contact-scope, and publication hashes.

## Governing disposition

The controlling stop/re-freeze rule is
[section 6 of the ratified plan](../design/m6_remarriage_learning_plan_round2.md),
lines 770--773. A future investigation requires a prospective selector
amendment, a new referee decision, a new freeze, and a new run. Section 7,
lines 777--781, keeps the designed pause active for this unratifiable finding.
This lane makes no `src/` model or harness, test, gate, floor, registry, or
`runs/` change.
