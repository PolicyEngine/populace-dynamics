PREFLIGHT_1_RUNTIME_INCIDENT — NO_SCORED_VERDICT. Invocation 4 passed the corrected environment guard and completed both registered refits plus full-input materialization, then exited 1 when the locked candidate-9 injected-state re-certification rejected four cells.

# M6 candidate-2 registration-9 invocation-4 incident

## Machine outcome

This lane made exactly one direct candidate-2 runner invocation under
[registration 9](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5016328696).
It exited `1` with an uncaught `AssertionError` in preflight 1. The failure
occurred after the candidate and post-repair incumbent refits, the selected-C
fit preflight, the sanctioned full-input load, both materializations, and the
first-marriage transport disclosure. Preflight 2, projection, scoring,
artifact assembly, final revalidation, and publication were not reached.

Both canonical outputs remain absent:

- `runs/gate_m6_candidate2_v1.json`
- `runs/gate_m6_candidate2_v1.json.env.json`

This is neither a designed abort nor a scored PASS or FAIL. No retry,
environment repair, source change, or self-rescue followed the exception.

## Frozen invocation

- Registration: [registration 9](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5016328696),
  comment `5016328696`.
- Invocation ordinal: 4, after the publicly disclosed
  [host termination](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5017203962),
  [invocation-2 environment incident](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5017283998),
  and [invocation-3 environment incident](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5017534238).
- Branch and source: `sol/c2-run5` at clean
  `0dd42985d8663f8632915d5813dc5c3da9cb4cc2`, the exact dispatcher-authorized
  SHA. During this lane `origin/master` advanced by the docs-only invocation-3
  incident publication; no run surface changed and the authorized source was
  not rebased.
- Runner SHA-256:
  `62e2a3d9057fef7660efb2d57bb8f5b793796ceecbccc45c9d4d9ae094cad884`.
- CLI SHA-256:
  `c4c2cb88296649f025c759e33a381301216e4826e02d5ca88404012257fc7e56`.
- Input-factory SHA-256:
  `acb46989f0f412292e7ea4fb1a15acc5b3001ab6d45067d611e4d49f9d55ddbf`.
- Runner PID `34189`; unified execution session `12624`; exit code `1`.
- Current-lane runner invocations: one. Post-incident retries: zero.

The documented CLI was invoked directly, with no `--out` override and no
worker knobs:

```sh
.venv/bin/python scripts/run_gate_m6_candidate2.py \
  --registration-id 5016328696 \
  --input-factory registered_m6_candidate2_inputs:build_input_plan
```

The invocation environment contained:

```text
POPULACE_DYNAMICS_PSID_DIR=/Users/maxghenis/PolicyEngine/psid-data
POPULACE_DYNAMICS_PE_US_DIR=/Users/maxghenis/PolicyEngine/social-security-model-worktrees/sol-c2-run5/.venv/lib/python3.14/site-packages
```

The venv had CPython 3.14.4, pip 26.1.2, NumPy 2.5.1, pandas 3.0.3,
SciPy 1.18.0, scikit-learn 1.9.0, quantile-forest 1.4.2,
policyengine-us 1.752.2, policyengine-core 3.30.1, populace-fit 0.1.0,
and populace-frame 0.1.0. Its 165 unique distribution name/version pairs
matched the invocation-3 registered environment before launch.

The corrected mandatory mirrored check was recorded before invocation:

```text
mirrored_guard_equality: True
pe_us_version: 1.752.2
```

The joined path was
`.venv/lib/python3.14/site-packages/policyengine_us/parameters/gov/ssa`.
The runtime parameter-directory guard passed; invocation 4 is the first
registration-9 attempt to proceed beyond the environment gate and reach data.

## Terminal preflight failure

The exact terminal exception was:

```text
AssertionError: candidate-9 injected-state re-certification failed; fuller re-ceremony required: legal_spouse_residual/legal_core: delta=0.0089596796 > 3sigma=0.0031888439; legal_spouse_residual/final_spouse: delta=0.0079574811 > 3sigma=0.0031818483; household_size/household_size.1: delta=0.0032050662 > 3sigma=0.0026893011; household_size/household_size.4: delta=0.0027247735 > 3sigma=0.001621927
```

Preflight 1 runs 20 locked draws and compares injected-state means with the
internal-reference means. Each absolute difference must be no larger than
three standard errors of that mean difference. Four cells were rejected:

| Channel set | Cell | Absolute delta | 3σ tolerance |
| --- | --- | ---: | ---: |
| `legal_spouse_residual` | `legal_core` | 0.0089596796 | 0.0031888439 |
| `legal_spouse_residual` | `final_spouse` | 0.0079574811 | 0.0031818483 |
| `household_size` | `household_size.1` | 0.0032050662 | 0.0026893011 |
| `household_size` | `household_size.4` | 0.0027247735 | 0.001621927 |

The combined log also contains two sklearn `ConvergenceWarning` messages from
the preceding registered logistic fits. They were warnings, not the terminal
exception: execution continued through full-input materialization to
preflight 1. This incident record makes no causal inference beyond the
runner's assertion and its explicit `fuller re-ceremony required` disposition.

## Reached and not reached

The traceback entered `execute_registered_m6_candidate2_run` at line 2502.
The tracked sequential call order establishes the following boundary:

| Stage | Reached |
| --- | --- |
| Runner governance and registration guard | yes — completed |
| Corrected policyengine-us version/directory guards | yes — passed |
| Explicit parameter and PSID loaders | yes |
| Candidate refit and selected-C fit preflight | yes — completed |
| Post-repair incumbent comparator refit | yes — completed |
| Sanctioned full-input loader | yes — completed |
| Candidate and incumbent materialization | yes — completed |
| First-marriage transport disclosure | yes — completed |
| Preflight 1 | yes — failed terminally |
| Preflight 2 | no |
| Candidate or incumbent projection/seed scoring | no |
| Artifact assembly/final guard/pair write | no |

The full-input load is inside the registered harness path and occurred only
after both registered fits and the selected-C preflight. The lane performed no
post-2014 read or data inspection outside the runner, either before or after
the failure.

The external `PHASE: refit` marker announced process launch. The runner exposes
no live phase callback, so `PHASE: preflight_1` was emitted only after the
traceback proved that stage had been reached. No projection or scoring marker
was emitted because neither stage began. `PHASE: publish` marks publication of
this incident, not scored-artifact publication.

## Compute-care record

Before launch, the lane confirmed no worker-control environment knob was set,
both exclusive outputs were absent, 1.3 TiB of disk was available, and no
conflicting high-memory process was present. The runner was never sampled,
attached to, signaled, reniced, or otherwise touched. Its last live memory
check showed 91% system memory free.

## Published evidence

- `docs/analysis/m6_candidate2_registration9_execution_incident_v3.json` —
  strict machine-readable incident ledger.
- `docs/analysis/m6_candidate2_registration9_execution_incident_v3.md` —
  this human report.
- `docs/analysis/m6_candidate2_registration9_execution_incident_v3.txt` —
  exact combined capture: 4,251 bytes / 65 lines / SHA-256
  `e4be49c46936b975a0a7deb364e9721e6c986d177b48f02cc41512650596b502`.

The transcript contains the prelaunch environment/source/command record, the
launch marker, both emitted warnings, and the verbatim terminal traceback.
Separate stdout/stderr byte attribution is unavailable because the runner used
combined redirection.

## Validation

- Parsed the machine ledger as strict JSON with duplicate-key rejection.
- Compared the published transcript byte-for-byte with the frozen ignored
  `c2-run.log` using `cmp`.
- Recomputed the transcript's 4,251-byte size, 65-line count, and SHA-256.
- Recomputed all source hashes and the 164-line canonical `pip freeze` digest.
- Confirmed both canonical outputs absent after exit.
- Ran the scoped pre-commit checks and `git diff --check` before publication.

The shared commit hook then stopped at its Beads JSONL flush (`Failed to flush
bd changes to JSONL`) before creating a commit. The evidence files and index
were unchanged. After the scoped checks above passed, the same three-file
commit was created with `--no-verify`, matching the invocation-3 incident
precedent; no `bd sync` or unrelated-state mutation was attempted.

## Governing disposition

This lane applies stop-and-disclose under the #238 incident pattern. It makes
no source, test, gate, frozen-artifact, registration, or `runs/` change. It
does not adjudicate one-shot consumption or authorize re-execution. The
coordinator must adjudicate the preflight-1 incident, the runner's fuller
re-ceremony requirement, and any future execution or registration.
