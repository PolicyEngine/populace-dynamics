Outcome: `SELECTION_COMPLETE` — selected C `0.001`; final fit eligible.

# M6 candidate-2 first-marriage C-selection finding

## Machine finding

The single recorded invocation of the frozen train-only selector completed with
exit code 0. All 27 pseudo-boundary fits were eligible. The selector chose
`C = 0.001` as the unique minimum of the equal-boundary mean deviance, then
completed one eligible final fit on the full train-information frame.

```text
C          boundary 2006   boundary 2008   boundary 2010   mean
0.0001     0.311636832     0.302836314     0.298101383     0.304191510
0.0003     0.308826125     0.299683047     0.291647336     0.300052170
0.0010     0.307405621     0.301286756     0.289764052     0.299485476
0.0030     0.307957520     0.304562908     0.289894862     0.300805097
0.0100     0.309081962     0.308167355     0.289522172     0.302257163
0.0300     0.310651190     0.312164237     0.289180833     0.303998753
0.1000     0.313014604     0.317478148     0.288287414     0.306260055
0.3000     0.315148896     0.322671067     0.287367348     0.308395770
1.0000     0.317538767     0.328626634     0.286924622     0.311030008
```

The tie set contains only `0.001`. The machine ledger reports selected and
registerable C as `0.001`, with no substitution or reselection. This document
publishes box-2 evidence only; it does not post a registration or adjudicate
any other registration precondition.

## Converged-fit record

The complete converged-fit record is the `final_fit` object in the
[frozen findings](m6_first_marriage_c_selection_findings.json). Those findings
are the exact output of the frozen `reduce_ledger()` function. The same object
is retained in the exact full selector stdout.

- Input rows: 391,761, all with positive fit weight.
- Events: 20,392.
- Features: 64.
- Solver termination: successful, status 0, after 64 iterations.
- Iteration ceiling reached: false.
- Warnings and convergence warnings: zero.
- Independent gradient infinity norm: `8.750257124591878e-9`, below the
  frozen `1e-6` eligibility ceiling.
- Coefficients, design, linear predictors, and probabilities: finite.
- Probability range: `0.00006630900002404425` to
  `0.2013026068323596`, strictly inside the unit interval.
- Preflight: passed; all seven support, standardization, weight, row,
  design, coefficient, and selected-C checksums replayed exactly.
- Age-18–29 transport and hazard records retained: 288.

The final coefficient checksum is
`bad9d4b0777e4927858e5b11f8201f8e32ed61de1e96152c80790b950b9a967c`.
The final support checksum is
`4327e2090f957851faa3af3a8006ef98d2f95624eac522683a194fc73931fc3a`.
Every binding checksum is published in `final_fit.fit_audit.checksums` and
is repeated byte-for-byte in `final_fit.preflight.recomputed_checksums`.

## Frozen execution

- Freeze commit:
  `69880c39b60b5c54f8fb8959ef4fadead7d88c5d`.
- Pre-run `git diff 69880c3 -- scripts/ src/`: empty.
- Frozen selector SHA-256:
  `031447023863e6866acb59b58f60d9618a0f84a061a32fec47f3e4bdbe6676be`.
- Runtime: CPython 3.13.12, NumPy 2.4.2, pandas 2.3.3, SciPy 1.17.0,
  and scikit-learn 1.8.0.
- Launch gate: 5-minute load 31.77, below the hold threshold of 60;
  launched immediately with zero holds.
- Selector attempts: exactly one. Exit code 0; no rerun, patch, or rescue.
- Launch record: 2026-07-18T11:39:10-04:00.
- Output completion: 2026-07-18T11:41:50-04:00.

The exact command and pre-run evidence are preserved in the
[execution record](m6_first_marriage_c_selection_execution.txt). The selector
wrote one strict JSON stdout document and progress only to stderr.

## Information boundary

The ledger records `real_train_only_psid` mode. Native retrospective products
were constructed before field-aware truncation, so later report values may be
read only to establish earlier history. No row dated after 2014 entered any fit
or evaluation frame. No holdout truth table, candidate score, M6 scorer,
`gates.yaml`, or `runs/` value entered numerical selection.

## Validation and reduction

The selector called `validate_complete_ledger()` before emitting stdout. After
the process exited, the exact stdout was parsed and passed through the frozen
`validate_complete_ledger()` and `reduce_ledger()` functions under the pinned
runtime. The reducer completed with no exception and retained every rung,
support cell, selection field, final-fit field, and registration disposition.
The full schema is `m6.first_marriage.c_selection.full.v1`; the findings schema
is `m6.first_marriage.c_selection.findings.v1`.

The reducer's `source_ledger_sha256` is the hash of compact canonical JSON:

```text
e5693f1bdfe17c8c339c75a562e64b90c9783d06153c337ac3284f57712aa1e1
```

That canonical-content hash intentionally differs from the requested hash of
the pretty-printed stdout file bytes.

## Published evidence

- [Exact full selector stdout](m6_first_marriage_c_selection_full.json):
  874,278 bytes; SHA-256
  `b2e51b099c20f2f0021a3073d3678b88b97708e758a184caf208642226d64f01`.
- [Frozen findings](m6_first_marriage_c_selection_findings.json):
  450,055 bytes; SHA-256
  `4ff69bd87a5dc1580128ccc33844cf5c573a6d69437d626f622b9f1fe378b14d`.
- [Exact selector stderr](m6_first_marriage_c_selection_progress.txt):
  244 bytes; SHA-256
  `05f63809b7981a0fb980b0f0969561d7bc4db7867cbf3274a08291596eb4b92b`.
- [Execution record](m6_first_marriage_c_selection_execution.txt): pre-run
  byte verification, load decision, command, terminal status, and hashes;
  SHA-256
  `616db13c2773120778aa13b7356aeeaf7305260e0a5656a8e1e5b7e4a3cefe2d`.

The pre-freeze registry specification already exists at
`docs/analysis/m6_first_marriage_registry_spec_prefreeze_v1.json` from the
selector freeze. It remains unchanged, including its deliberately null
pre-selection bindings. A later, separately authorized freeze may bind this
public ledger and its eligible final-fit checksums.

The selector freeze also already committed the
[failing synthetic preflight](m6_first_marriage_preflight_abort_synthetic.json)
required by box 2. That evidence remains unchanged.

No selector, source, test, gate, run artifact, candidate-1 surface, or existing
registry specification changed in this publication.
