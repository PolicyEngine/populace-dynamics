PROJECTION_RUNTIME_INCIDENT — NO_SCORED_VERDICT. Invocation 5 passed both locked gate preflights and entered the first candidate seed's household-side projection setup, then exited 1 during period-module assembly before `ProjectionEngine` construction, any completed projection draw, numeric per-cell scoring, artifact assembly, or pair write.

# M6 candidate-2 registration-10 invocation-5 incident

## Machine outcome

This lane made exactly one direct candidate-2 runner invocation under
[registration 10](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5018346030).
It exited `1` with an uncaught `ValueError` in the first candidate seed's
first-draw projection setup. This is neither a designed abort nor a scored
PASS or FAIL. It produced no gate aggregate, artifact SHA-256, or per-cell
summary.

Both canonical outputs remain absent:

- `runs/gate_m6_candidate2_v1.json`
- `runs/gate_m6_candidate2_v1.json.env.json`

No retry, environment repair, source change, instrumentation, or self-rescue
followed the exception.

## Frozen invocation

- Overall candidate-2 invocation ordinal: 5, after the four registration-9
  invocations disclosed by registration 10. This was the first and only
  runner invocation under registration 10 by this lane.
- Branch and source: `sol/c2-run6` at clean
  `92e23e96291ea387202c8d0be914d2dd4bd3dd87`, the dispatcher-authorized
  source and local `origin/master` at launch.
- Registered design commit:
  `0e067a910fde7e479240c472087ece6a7ce29bcd`.
- Runner SHA-256:
  `c568d8f2744b488e5a9b11d761fae5db63befddf4f6866ac2ccb0b4196bc3ea9`.
- CLI SHA-256:
  `c4c2cb88296649f025c759e33a381301216e4826e02d5ca88404012257fc7e56`.
- Input-factory SHA-256:
  `acb46989f0f412292e7ea4fb1a15acc5b3001ab6d45067d611e4d49f9d55ddbf`.
- Unified execution session: `27970`; exit code: `1`.
- The exclusive combined log was created at `2026-07-20T06:51:13Z` and
  last modified at `2026-07-20T06:57:13Z`.

The documented CLI was invoked directly, with no `--out` override and no
worker knobs:

```sh
.venv/bin/python scripts/run_gate_m6_candidate2.py \
  --registration-id 5018346030 \
  --input-factory registered_m6_candidate2_inputs:build_input_plan
```

The invocation environment contained:

```text
POPULACE_DYNAMICS_PSID_DIR=/Users/maxghenis/PolicyEngine/psid-data
POPULACE_DYNAMICS_PE_US_DIR=/Users/maxghenis/PolicyEngine/social-security-model-worktrees/sol-c2-run6/.venv/lib/python3.14/site-packages
```

The venv had CPython 3.14.4, pip 26.1.2, NumPy 2.5.1, pandas 3.0.3,
SciPy 1.18.0, scikit-learn 1.9.0, quantile-forest 1.4.2,
policyengine-us 1.752.2, policyengine-core 3.30.1, populace-fit 0.1.0,
and populace-frame 0.1.0. The mandatory mirrored check immediately before
launch recorded:

```text
guard_equality: True
pe_us_version: 1.752.2
```

Its resolved parameter path was
`.venv/lib/python3.14/site-packages/policyengine_us/parameters/gov/ssa`.
The runtime version and directory guards passed.

## Compute-care record

Before launch, this lane found unrelated batch workers whose observed RSS
made concurrent execution unsafe; the largest single observation was
102,227,440 KiB. It did not touch those processes and held the launch until
both batch parents exited naturally. At `2026-07-20T06:50:10Z`, the global
check found no process above 5 GiB RSS and `memory_pressure -Q` reported 87%
free. The final prelaunch check reported 88% free. After launch, the lane
polled only the unified execution session and never sampled, attached to,
signaled, or reniced the runner.

## Terminal projection-assembly failure

The direct cause recorded by Python was:

```text
TypeError: cannot pickle 'mappingproxy' object
```

That exception was wrapped as the terminal error:

```text
ValueError: fitted marital core is not byte-comparable
```

The first-marriage support-aware model carries a `FirstMarriageFitAudit`.
Its checksum mapping is deep-frozen as a `MappingProxyType`. During the first
candidate seed, draw 0, household-side projection setup,
`assemble_period_modules` called `_fit_digest(inputs.family)`, whose
`pickle.dumps(..., protocol=5)` reached that checksum mapping and failed.
The authoritative digest therefore did not complete; the embedded digest was
not evaluated and no byte comparison occurred. `ProjectionEngine` was not
constructed, `engine.project` was not called, the person-side projection was
not entered, and zero draws completed.

The combined log contains two preceding scikit-learn `ConvergenceWarning`
messages from the registered logistic fits. They were warnings, not the
terminal exception: execution continued through both preflights and into the
candidate seed phase.

## Reached and not reached

| Stage | Reached |
| --- | --- |
| Runner governance and registration guard | yes — completed |
| Correct policyengine-us version/directory guards | yes — passed |
| Train-only input load and candidate refit | yes — completed |
| Selected-C first-marriage fit preflight | yes — completed |
| Post-repair incumbent refit | yes — completed |
| Sanctioned full-input load and both materializations | yes — completed |
| First-marriage transport disclosure | yes — completed |
| Preflight 1, candidate-9 re-certification | yes — passed |
| Preflight 2, earnings sign path | yes — passed |
| First candidate-seed orchestration and truth-cell construction | yes |
| First draw, household-side projection path | yes — assembly entered |
| `ProjectionEngine` construction / `engine.project` | no |
| Any completed candidate projection draw | no — zero |
| Numeric `score_gate_seed` cell scoring | no |
| Any completed candidate or incumbent seed | no |
| Gate aggregation, domain-floor report, or artifact assembly | no |
| Final source guard or artifact/sidecar pair write | no |

The externally required `PHASE: refit` marker announced process launch.
After exit, source order and the traceback established and justified the
`PHASE: preflight_1`, `PHASE: preflight_2`, and `PHASE: projection` markers.
No `PHASE: scoring` marker was emitted because `score_gate_seed` was never
called. `PHASE: publish` marks this incident publication, not scored-artifact
publication.

## Information boundary

The registered full-input loader completed inside the harness after both
registered fits and the selected-C preflight. The first seed's truth-cell
construction then ran inside `score_m6_seed` before projection setup. This is
the sanctioned harness scoring path. The lane performed no post-2014 read or
numeric inspection outside that path before or after the failure; its
post-failure inspection was limited to the combined runner transcript and
tracked source call order. No truth or per-cell value is reproduced here.

## Published evidence

- `docs/analysis/m6_candidate2_registration10_execution_incident_v1.json` —
  strict machine-readable incident ledger.
- `docs/analysis/m6_candidate2_registration10_execution_incident_v1.md` —
  this human report.
- `docs/analysis/m6_candidate2_registration10_execution_incident_v1.txt` —
  exact combined runner capture: 5,106 bytes / 78 lines / SHA-256
  `14d960eaf17f8c6d3b17928a8c4655db700c97bf4ad08338c71c3e0f81e495db`.

The transcript contains only the runner's two warnings and nested traceback.
Separate stdout/stderr byte attribution is unavailable because the command
used combined redirection.

## Validation

- Parsed the machine ledger as strict JSON with duplicate-key rejection.
- Compared the published transcript byte-for-byte with the frozen ignored
  `c2-run.log`.
- Recomputed the transcript's byte size, line count, terminating LF, and
  SHA-256.
- Recomputed the tracked source hashes and the unchanged venv's 164-line
  canonical `pip freeze` digest.
- Confirmed both canonical outputs absent after exit.
- Confirmed only the three incident files differ from the registered source,
  then ran scoped formatting and diff checks before publication.

## Governing disposition

This lane applies stop-and-disclose under the #238/#262–#264 incident
pattern. It changes no source, test, gate, floor, registry, frozen artifact,
registration, or `runs/` path. It does not adjudicate one-shot consumption or
authorize re-execution. The coordinator alone adjudicates this projection
runtime incident, one-shot accounting, and any future execution or
registration.
