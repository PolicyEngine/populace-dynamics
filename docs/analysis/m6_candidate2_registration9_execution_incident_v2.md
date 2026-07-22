Outcome: PRE_REFIT_RUNTIME_INCIDENT — NO_SCORED_VERDICT. Invocation 3 followed the mandatory environment recipe and passed its required equality/version check, then the runner exited 1 because the runtime guard applies a different path formula.

# M6 candidate-2 registration-9 invocation-3 incident

## Machine outcome

This lane made one direct candidate-2 runner invocation under registration 9.
It exited 1 in the registered input factory's policyengine-us parameter
directory identity guard, before the first explicit parameter loader, PSID
loader, refit, preflight, projection, scoring, or output write. This is an
uncaught pre-refit runtime incident, not a designed abort and not a scored PASS
or FAIL.

Both canonical outputs remain absent:

- `runs/gate_m6_candidate2_v1.json`
- `runs/gate_m6_candidate2_v1.json.env.json`

No retry, environment repair, source change, or self-rescue followed the
exception.

## Frozen invocation

- Registration: [registration 9](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5016328696), comment `5016328696`.
- Invocation ordinal: 3, after the publicly disclosed host termination
  ([comment 5017203962](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5017203962))
  and invocation-2 environment incident
  ([comment 5017283998](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5017283998)).
- Branch and source: `sol/c2-run4` at clean
  `9d6da5e5b031b4deda432fdc984054f54799b0a2`.
- Runner SHA-256:
  `62e2a3d9057fef7660efb2d57bb8f5b793796ceecbccc45c9d4d9ae094cad884`.
- Current-lane runner invocations: one. Post-incident retries: zero.
- Exit code: `1`; the terminal log was last modified at
  `2026-07-19T21:54:03Z` and the terminal state was observed at
  `2026-07-19T21:54:11Z`.

The documented CLI was invoked directly with no `--out` override and no worker
knobs:

```sh
.venv/bin/python scripts/run_gate_m6_candidate2.py \
  --registration-id 5016328696 \
  --input-factory registered_m6_candidate2_inputs:build_input_plan
```

The command environment contained the registered PSID directory and the exact
mandatory invocation-3 value:

```text
POPULACE_DYNAMICS_PSID_DIR=/Users/maxghenis/PolicyEngine/psid-data
POPULACE_DYNAMICS_PE_US_DIR=/Users/maxghenis/PolicyEngine/social-security-model-worktrees/sol-c2-run4/.venv/lib/python3.14/site-packages/policyengine_us
```

The venv had pip 26.1.2 and passed the required scikit-learn 1.9.x assertion.
Recorded versions were CPython 3.14.4, NumPy 2.5.1, pandas 3.0.3, SciPy
1.18.0, scikit-learn 1.9.0, quantile-forest 1.4.2, policyengine-us 1.752.2,
policyengine-core 3.30.1, populace-fit 0.1.0, and populace-frame 0.1.0.

## Mandatory check passed, runtime check failed

The invocation-3 dispatcher made the environment recipe mandatory and required
this check before launch:

```python
a = Path(os.environ["POPULACE_DYNAMICS_PE_US_DIR"]) / "parameters" / "gov" / "ssa"
b = Path(md.distribution("policyengine-us").locate_file("policyengine_us")) / "parameters" / "gov" / "ssa"
```

It was run twice before launch, including immediately before invocation, and
both executions recorded exactly:

```text
guard_equality: True
pe_us_version: 1.752.2
```

The runner nevertheless raised:

```text
RuntimeError: POPULACE_DYNAMICS_PE_US_DIR resolves SSA parameters to a directory that is not the metadata-versioned policyengine-us install; point it at the 1.752.2 install.
```

The reason is a formula mismatch between the mandatory precheck and the tracked
runtime guard. In `src/populace_dynamics/ss/params.py`, `_resolve_pe_us(None)`
returns the environment value as the root and `_SSA` is
`policyengine_us/parameters/gov/ssa`. In
`scripts/registered_m6_inputs.py`, `assert_pe_us_param_dir` therefore evaluates
`_resolve_pe_us(None) / _SSA`.

With the mandatory value, the paths are:

| Path | Value | Exists |
| --- | --- | --- |
| Mandatory precheck `a` | `.../site-packages/policyengine_us/parameters/gov/ssa` | yes |
| Metadata-versioned `b` | `.../site-packages/policyengine_us/parameters/gov/ssa` | yes |
| Runtime guard resolution | `.../site-packages/policyengine_us/policyengine_us/parameters/gov/ssa` | no |

Thus the prescribed precheck's equality was true while the runtime guard's
equality was false. The package-directory component was applied once by the
mandatory value and again by `_SSA`. This is a post-failure static
reconstruction from the prescribed value, distribution metadata, and unchanged
tracked source. No alternative value was set or tried.

## Reached and not reached

The runner governance guard returned because the traceback entered the input
factory. Within `load_train_only_raw_inputs`, the policyengine-us version check
at line 220 passed and the parameter-directory check at line 221 failed.
Parameter loading starts at line 222 and the first explicit PSID panel reader
at line 225, so neither was reached.

| Stage | Reached |
| --- | --- |
| Runner governance and registration guard | yes |
| Input-factory import and entry | yes |
| policyengine-us 1.752.2 version gate | yes |
| policyengine-us parameter-directory gate | yes — failed terminally |
| Explicit parameter or PSID loader | no |
| Registered input-plan completion | no |
| Internal refit | no |
| Preflight 1 or later preflight | no |
| Full-input load | no |
| Projection | no |
| Scoring | no |
| Artifact or sidecar write | no |

The required external `PHASE: refit` marker announced process launch; internal
refit was not reached.

## Compute-care hold

Before launch, the lane detected an unrelated process using 45--74 GB RSS.
It did not touch that process and did not launch the candidate concurrently.
The lane waited until the external work exited on its own, then rechecked
memory pressure, the clean tree, absent outputs, hashes, worker-knob absence,
and the mandatory equality/version recipe. The incident is unrelated to that
hold: it occurred in the deterministic parameter-directory guard.

## Published evidence

- `docs/analysis/m6_candidate2_registration9_execution_incident_v2.json` —
  machine-readable incident ledger.
- `docs/analysis/m6_candidate2_registration9_execution_incident_v2.txt` —
  exact combined transcript: 2,219 bytes / 31 lines / SHA-256
  `1884a50147f41920c6326d060e12d66bb58a651fd84034c81dd116e39504f004`.

The transcript contains the recorded source and runtime identities, both
successful mandatory equality/version checks, the compute-care hold marker,
the launch marker, and the verbatim terminal traceback. Separate stdout/stderr
byte attribution is unavailable because the runner used combined redirection.

## Governing disposition

This lane applies stop-and-disclose under the #238 incident pattern. It makes
no source, test, gate, frozen-artifact, registration, or `runs/` change. It
does not adjudicate one-shot consumption or authorize another execution. The
coordinator must adjudicate the mandatory-recipe inconsistency and any future
execution or registration.
