Outcome: PRE_REFIT_RUNTIME_INCIDENT — NO_SCORED_VERDICT. The registered runner exited 1 before the first explicit PSID loader call, refit, preflight, projection, scoring, artifact write, or sidecar write.

# M6 candidate-2 registration-9 execution incident

## Machine outcome

This lane's single candidate-2 runner invocation reached the registered input
factory and then failed its policyengine-us parameter directory identity
guard. At least one prior runner invocation under registration 9 is disclosed
in issue-42 comment 5017203962, so this invocation is at least the second under
the registration; the dispatcher labels the present lane relaunch 2. The
exception was an uncaught `RuntimeError`, not a designed candidate-2 abort and
not a scored PASS or FAIL. This lane performed no retry or environment repair.

The expected output pair remains absent:

- `runs/gate_m6_candidate2_v1.json`
- `runs/gate_m6_candidate2_v1.json.env.json`

No partial output was repurposed, and no substitute verdict or per-cell result
was fabricated.

## Frozen invocation

- Registration: [registration 9](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5016328696), comment `5016328696`.
- Branch and commit: `sol/c2-run3` at
  `9d6da5e5b031b4deda432fdc984054f54799b0a2`, equal to `origin/master` and
  clean at invocation.
- Runner SHA-256:
  `62e2a3d9057fef7660efb2d57bb8f5b793796ceecbccc45c9d4d9ae094cad884`.
- Failing guard source SHA-256 (`scripts/registered_m6_inputs.py`):
  `4b8e1e7d45f3a59db51c0ffcb728c1f41acd2b3dc53689ba104af2db54503a57`.
- Parameter resolver source SHA-256
  (`src/populace_dynamics/ss/params.py`):
  `9b49894b56eaab32daa62032a139ecb33a1d11f657b1523d3bebadebc2387efd`.
- Runner invocations by this lane: one. Registration-9 invocation ordinal: at
  least two. Reruns by this lane: zero.
- Start observed by the agent/orchestrator:
  `2026-07-19T20:17:29.680Z`; exit code: `1`. Millisecond timing and the
  later exit observation are outside the combined runner transcript.

The documented CLI was invoked directly from the repository root, with no
`--out` override and no worker knobs:

```sh
POPULACE_DYNAMICS_PSID_DIR=/Users/maxghenis/PolicyEngine/psid-data \
  .venv/bin/python scripts/run_gate_m6_candidate2.py \
  --registration-id 5016328696 \
  --input-factory registered_m6_candidate2_inputs:build_input_plan
```

The worktree `.venv` had pip 26.1.2 and passed the required scikit-learn 1.9.x
assertion. The recorded runtime was CPython 3.14.4, NumPy 2.5.1, pandas 3.0.3,
SciPy 1.18.0, scikit-learn 1.9.0, quantile-forest 1.4.2,
policyengine-us 1.752.2, policyengine-core 3.30.1, populace-fit 0.1.0, and
populace-frame 0.1.0. A post-failure read-only recomputation in the unchanged
virtual environment produced a 164-line canonical `pip freeze` with SHA-256
`dd86f55e69c6fc01869cdf3cb8966646563dd5f549d66a870348e694b2e27a91`.
The runner governance guard had completed its in-memory dependency capture,
but the crash left no published sidecar or digest to compare against.

## Failure

The terminal exception was:

```text
RuntimeError: POPULACE_DYNAMICS_PE_US_DIR resolves SSA parameters to a directory that is not the metadata-versioned policyengine-us install; point it at the 1.752.2 install.
```

Post-failure static reconstruction in the current workspace environment, with
unchanged tracked source and virtual environment, found
`POPULACE_DYNAMICS_PE_US_DIR` unset. On that reconstruction, the parameter
loader resolved its host-checkout default:

```text
/Users/maxghenis/PolicyEngine/policyengine-us/policyengine_us/parameters/gov/ssa
```

The version gate resolved policyengine-us 1.752.2 from the worktree virtual
environment:

```text
/Users/maxghenis/PolicyEngine/social-security-model-worktrees/sol-c2-run3/.venv/lib/python3.14/site-packages/policyengine_us/parameters/gov/ssa
```

The fail-closed directory binding rejected the unequal paths. The
contemporaneous transcript proves the mismatch but did not record the ambient
variable or both resolved paths; those details are explicitly a post-failure
static reconstruction. The direct command had explicitly set only the
registered PSID directory from the run brief. This lane records the mismatch
and stops; it does not point the variable elsewhere and invoke again.

## Reached and not reached

The top-level runner governance guard returned, because the traceback entered
the input factory. Within `load_train_only_raw_inputs`, the policyengine-us
version check at line 220 passed and the directory identity check at line 221
failed. Parameter loading begins at line 222 and the first PSID panel read at
line 225, so neither occurred.

| Stage | Reached |
| --- | --- |
| Runner governance and registration guard | yes |
| Input-factory import and entry | yes |
| policyengine-us 1.752.2 version gate | yes |
| policyengine-us parameter-directory gate | yes — failed terminally |
| Explicit parameter-loader or PSID-loader call | no |
| Registered input-plan completion | no |
| Refit | no |
| Preflight 1 or later preflight | no |
| Full-input load | no |
| Projection | no |
| Scoring | no |
| Artifact or sidecar write | no |

The externally required `PHASE: refit` marker was emitted in agent stdout
immediately before process launch so the dispatcher could bind to the runner;
it is outside the combined runner transcript. It was a launch marker, and the
internal refit phase was not reached. Source order proves that the first
explicit PSID loader call was not reached. No dynamic file-open audit was
installed, so this record does not elevate that source-order fact into an
absolute file-read or post-2014-read claim.

## Publication record

The published evidence is:

- [Machine incident ledger](m6_candidate2_registration9_execution_incident_v1.json)
- [Exact combined transcript](m6_candidate2_registration9_execution_incident_v1.txt)

The transcript contains the environment build, prelaunch verification, and
exact combined runner stdout/stderr. It is 8,136 bytes and 223 lines, with
SHA-256
`0ed8fe98fe40ab2015a4a98c61e8c77cdce6dd3eb83cf195505def74363b1762`.
Because the direct command used combined redirection, a separate stdout/stderr
byte attribution is unavailable and is not inferred.

Of the four workspace markers named by the run brief, only the ignored
`c2-run.log` existed. `c2-run.exit`, `c2-run.done`, and `c2-run.last.md` were
absent. They were not backfilled as if they were contemporaneous evidence; the
exit-1 and timing observations are identified separately as agent/orchestrator
evidence outside the combined transcript.

## Governing disposition

This lane applies stop-and-disclose. It makes no source, test, gate, frozen
artifact, registration, or `runs/` change. It does not adjudicate whether the
one-shot is consumed. It performed no further invocation after this incident;
whether this invocation used or exhausted any disclosed re-execution allowance
is `NOT_ADJUDICATED_BY_THIS_LANE`. Any re-execution or subsequent registration
requires coordinator adjudication. This follows the dispatcher's #238 crash
publication rule and the candidate-2 program's public-abort and
post-registration rules at lines 490--493 and 996--999. The prior host-side
termination disclosure is issue-42
[comment 5017203962](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5017203962).
