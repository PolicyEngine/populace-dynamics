Outcome: `FIRST_MARRIAGE_PRESCORE_ABORT` — no scored verdict; projection and scoring never started.

# M6 candidate-2 registered-run designed abort

## Machine outcome

The sole execution authorized by Registration 8 exited 2 through the runner's
structured designed-abort path. The canonical terminal record reports:

- type: `FirstMarriagePreflightAbort`;
- stage: `transport_disclosure`;
- message: `first-marriage diagnostic support lacks canonical birth year`;
- `converted_to_gate_fail: false`; and
- status: `FIRST_MARRIAGE_PRESCORE_ABORT`.

This is neither a PASS nor a scored FAIL. There is no per-cell result or gate
aggregate to report.

## One-shot execution

- Registration: [issue #42 comment 5015653634](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5015653634).
- Referee record: [issue #42 comment 5015647566](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5015647566).
- Guarded source: `aa6c6cf9214ca5fc4059c055228656083c9cfead`, with a clean worktree at launch and at the runner's final revalidation.
- Started: `2026-07-19T08:29:36-0400`.
- Ended: `2026-07-19T08:32:27-0400` (171 seconds).
- Exit code: 2.
- Attempts: exactly one. No retry, run-surface patch, instrumentation,
  environment change, or self-rescue followed the abort.

The exact runner invocation was:

```text
.venv/bin/python scripts/run_gate_m6_candidate2.py --registration-id 5015653634 --input-factory registered_m6_candidate2_inputs:build_input_plan
```

`POPULACE_DYNAMICS_PSID_DIR` was
`/Users/maxghenis/PolicyEngine/psid-data`. No worker-control environment knob
was set.

## Reached and fenced phases

The registered selected-C first-marriage fit preflight completed and was
eligible at `C = 0.001`. Its audit records zero warnings and zero convergence
warnings, and all seven recomputed fit checksums match. Full inputs were then
loaded and both candidate and incumbent materialization completed.

The transport disclosure stopped on a gated-support person without a
canonical birth-year mapping. The runner's fences record:

```json
{
  "full_inputs_loaded": true,
  "materialization_completed": true,
  "projection_started": false,
  "score_started": false,
  "candidate_artifact_written": false
}
```

Consequently, gate preflights 1 and 2, projection, scoring, the per-cell
summary, and the aggregate verdict were not reached. The combined execution
log contains two scikit-learn maximum-iteration warnings before the terminal
record; they do not alter the structured abort reason or the selected fit's
eligible zero-warning audit.

## Registered environment

The prelaunch record verifies CPython 3.14.4, NumPy 2.5.1, pandas 3.0.3,
SciPy 1.18.0, scikit-learn 1.9.0, quantile-forest 1.4.2, pip 26.1.2, and
PolicyEngine US 1.752.2. `populace-fit` and `populace-frame` came from the
run-7 fitting-stack revision
`b33524189e6a65100acdbc84fe43229ecc14162a`. The runner captured a 164-line
sorted dependency snapshot and revalidated every registered binding before
emitting the abort record.

## Published evidence

- [Exact canonical abort record](m6_candidate2_registered_abort_v1.json):
  350,817 bytes including its terminating LF; SHA-256
  `cc67a10e91cd656ec8acaf3767ad4d9b924cd6fc0b32f5726622e9c6afa2c81c`.
- [Exact combined execution log](m6_candidate2_registered_abort_execution_v1.txt):
  358,088 bytes; SHA-256
  `bc656c5d3d870cb866d23593ee8f729bcc7f894a349521680cc60e7eca08fe3a`.

The canonical candidate output
`runs/gate_m6_candidate2_v1.json` and its environment sidecar
`runs/gate_m6_candidate2_v1.json.env.json` are both absent, as required by the
designed-abort contract. No substitute artifact or sidecar has been created.
The sidecar SHA-256 retained in the machine record binds the registered
in-memory snapshot; it is not the digest of a file written by this execution.

This publication adds only the exact abort record, the exact execution log,
and this report. It does not modify `gates.yaml`, frozen artifacts, the
registration, runner or model source, tests, or any `runs/` path. This lane
stops after publishing the abort evidence. The coordinator alone adjudicates
any future execution or one-shot accounting; neither is decided here.
