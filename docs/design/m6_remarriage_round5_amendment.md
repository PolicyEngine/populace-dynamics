# M6 remarriage round-5 amendment: audit-certified null-contract repair

- **Status:** **PROSPECTIVE AMENDMENT — UNRATIFIED.** This lane authors a
  corrected reducer and extends the synthetic reducer smoke. It performs no
  selection, no reduction of the real round-3 stdout, and no staged or PSID
  data contact. A new independent referee decision and ratification must
  precede the one round-5 reduction attempt authorized below.
- **Plan authority:** the ratified [round-2 learning plan][round2-plan], PR
  [#234][pr-234] at squash
  `004c57d306741e1d97275a720effbb41f378d763`.
- **Prior amendment and freeze:** the ratified [round-4 amendment][round4], PR
  [#242][pr-242] at squash and freeze
  `d63f36b64f58b53f9ff39a2c54b8c682fea93ac5`.
- **Incident authority:** the ratified [round-4 reduction incident][incident],
  PR [#243][pr-243] at squash
  `54c5182bcf19e828a93519ce13b7f1b346811e11`.
- **Completeness certificate:** issue comment
  [5009594753][completeness-certificate] performed a class-closing static
  audit over the entire pinned input and the frozen reducer. It found exactly
  one remaining defect class, with the two paired surfaces repaired here at
  the same 90 sites, and verified that class absent everywhere else through
  strict serialization.

## 1. Canonical state after round 4

The only authorized round-4 reduction attempt exited 1 while validating the
first null aggregate group rate. It emitted no findings JSON. The attempt is
consumed and the ratified incident is an `UNRATIFIABLE_FINDING`. The canonical
result remains `NOT_DETERMINED`, both per-origin outcomes remain `NONE`, and
neither a law nor `NO_OP_DESIGNED_PAUSE` is ratified. The designed pause
continues and registration 8 remains forbidden.

The input remains the complete selector stdout already published in
[#241][pr-241]:

- path: [the round-3 full selector stdout][raw-stdout];
- byte count: `14,606,212`;
- SHA-256:
  `370bef8d00313e847c9cbb0fe09d6b6a5b9dddcf2658449ab61b02a42534614b`;
- schema/status: one strict JSON object and `SELECTION_COMPLETE`; and
- coverage: 3 boundaries x 5 laws x 40 common-random-number seeds.

Its raw R0 and `NO_OP_DESIGNED_PAUSE` fields remain public but unratified.
This amendment neither adopts nor rejects that claim.

## 2. Central operative provision: one frozen-stdout reduction

> **If and only if a new referee accepts this amendment and the accepted
> squash is ratified as the round-5 freeze, that ratification expressly
> authorizes the exact 14,606,212-byte stdout with SHA-256
> `370bef8d00313e847c9cbb0fe09d6b6a5b9dddcf2658449ab61b02a42534614b`
> to be consumed as round 5's frozen input for exactly one execution attempt
> of the frozen round-5 reducer under CPython `3.13.12`, NumPy `2.4.2`, and
> pandas `2.3.3`.**

This is a new, narrow prospective authorization after the consumed round-4
attempt. It is not a retry under round-4 authority. It carries forward the
same input exception ratified in the round-4 amendment and changes neither
the input bytes nor the numeric runtime.

The authorization does not:

- retroactively ratify the raw R0 claim or the failed round-4 attempt;
- authorize a selector, preflight, staged-source, or PSID rerun;
- authorize a manual reduction, field stripping, repair after execution, or
  second reducer attempt;
- authorize any other reuse of the stdout or any partial output; or
- weaken the stop, preserve, disclose, amend, referee, and refreeze rule for
  a further defect.

Round 5 is reducer-only. Before execution, the input hash and byte count, all
three runtime versions, the frozen script bytes, and a clean frozen-scope
diff must match. A mismatch consumes the attempt and takes the incident
branch. The ratified squash is the round-5 freeze; no later patch may be
substituted for its reducer.

## 3. Completeness certificate and exact defect surface

The [class-closing audit][completeness-certificate] used a static profiler
that mirrored every type, shape, strict-key, nullness, flag, and hash-format
requirement in frozen reducer execution order over the entire pinned input.
It also evaluated every statically decidable `_assert_same` family and
followed null propagation through recomputation and strict serialization.
It collected all violations instead of stopping at the observed first one.
No reducer, selector, or reduction-value computation was executed for that
audit.

The exhaustive result is one value-type/null-contract defect class with two
paired surfaces at the same 90 aggregate publication-group records:

| Boundary | Group indices | Shape | Records across laws and aggregates |
|---|---|---|---:|
| 2006 | `groups[11]` | all contributing rates null; zero exposure | 15 |
| 2008 | `groups[10]` | mixed null/finite seed rates; positive aggregate exposure | 15 |
| 2008 | `groups[11]` | all contributing rates null; zero exposure | 15 |
| 2010 | `groups[2]`, `[11]`, `[13]` | all contributing rates null; zero exposure | 45 |

Each group is law-invariant and each count spans 5 laws x the full mean and
both fixed-block aggregates. Thus 75 records have zero aggregate exposure
and 15 have positive aggregate exposure. The positive-exposure 2008 case is
load-bearing: aggregate exposure alone cannot determine rate nullness.

Across all 810 aggregate group records (3 aggregate forms x 3 boundaries x
5 laws x 18 groups), the audit certified the exact biconditional

```text
stored aggregate rate is null
    <=> at least one contributing per-seed rate is null.
```

There were zero exceptions. The audit found no other value-type, schema-key,
shape, flag, or format mismatch in any remaining reducer path through strict
serialization. This certificate is specific to the immutable pinned input,
frozen configuration, reducer lineage, and runtime named here; it is not a
claim about arbitrary future inputs.

## 4. Exact two-surface repair

### Surface A: validate the aggregate null-rate contract

The frozen `_validate_mean_record` unconditionally passes `rate` to
`_finite_number`. That rejects the selector-legal null at the first of the 90
records before comparison with the independent recomputation.

Round 5 explicitly evaluates the contributing per-seed rates for each full
mean and fixed-block aggregate. It requires
`(stored rate is None) == any(contributing rate is None)`. A stored finite
rate is therefore required when every contributing rate is finite, and a
stored null is required when one or more contributing rates are null. Every
non-null rate remains finite and nonnegative. The validator does not infer
nullness from aggregate exposure, so it covers both the all-null,
zero-exposure shape and the mixed-null, positive-exposure shape.

This is enforcement of the certified biconditional, not a general relaxation
that merely tolerates null. All other aggregate field, ratio, key, count,
order, and equality checks remain unchanged.

### Surface B: normalize recomputed non-finite means before comparison

The frozen `_mean_record` converts per-seed values with
`np.asarray(..., dtype=np.float64)` and takes the NumPy mean. A null rate
becomes NaN, so any contributing null produces a NaN projected rate. Surface
A alone would only move the failure: `_assert_same` would pass that NaN to
`_canonical_bytes`, whose `json.dumps(..., allow_nan=False)` raises, and
`_ratio_fields` would also propagate a NaN `rate_ratio` when the truth rate is
positive.

Round 5 keeps that mean arithmetic byte-equivalent and then applies the
frozen selector's `_plain` semantics to the projected values: a non-finite
float becomes `None` before `_ratio_fields` runs. A null projected rate thus
produces null `rate_ratio` and `log_rate_ratio`, and both canonical equality
and final strict serialization remain NaN-free. Finite projected values are
unchanged.

The 810-record biconditional makes the two surfaces converge by construction:
the recomputation produces `None` exactly where the stored aggregate rate is
null, including all 90 audited sites, and nowhere else in the pinned group
surface. Both repairs are required and neither widens an unrelated contract.

Except for a previously requested cosmetic correction of the stale round-3
name in the copied module docstring, these are the only reducer changes from
the frozen round-4 implementation. The round-2, round-3, and round-4 scripts
remain frozen and untouched.

## 5. Defect latency

The class-closing audit compared the relevant implementations across the
round-2, round-3, and round-4 reducers. `_validate_mean_record`,
`_mean_record`, `_ratio_fields`, `_canonical_bytes`, `_assert_same`, and their
routing are byte-identical across that lineage. The null-contract defect was
therefore latent since round 2; it was not introduced by either later
amendment.

Two earlier contract classes masked it in execution order. Round 3 first
reached the `fit_validation.2006.reference_spells` schema-key contradiction.
Round-4 preparation then found the companion `divorced_calibration`
schema-key contradiction before execution, and the round-4 amendment repaired
those fit-comparison legs. Only after those earlier surfaces were cleared did
the round-4 attempt reach the aggregate value-type contract. The latency is
disclosed here even though round 5 changes only the now-exhaustively bounded
null class.

## 6. Full-path synthetic smoke

[`smoke_m6_remarriage_round5_reducer.py`][smoke] extends the existing
synthetic cube without reading the 14,606,212-byte stdout. It mirrors all 90
audited aggregate-null sites with two selector-faithful group shapes:

1. an all-null group in which every contributing seed has zero exposure and
   a null rate; and
2. a 2008-`groups[10]`-style mixed group in which at least one seed has zero
   exposure and a null rate while other seeds have positive exposure and
   finite rates, leaving the aggregate exposure positive and its rate null.

The values and identities are synthetic. Frozen pre-contact locks are used
only to satisfy the reducer contract; the fixture does not inspect or derive
values from the real stdout and creates no real candidate finding.

The frozen round-4 reducer must fail on the extended fixture at the aggregate
null-rate validator, demonstrating Surface A. Fixture construction itself
uses round 5's `_aggregate_projection`/`_mean_record` and then strict
`json.dumps(..., allow_nan=False)`, so producing the fixture bytes already
requires Surface B. The round-5 subprocess then accepts both shapes,
recomputes every aggregate and block, compares them with `_assert_same`,
completes selection and publication-group removal, and emits strict JSON.
Without the post-arithmetic NaN-to-`None` normalization, the fixture would
carry NaN `rate` and `rate_ratio` values and strict serialization would raise
before the full reducer path. A separate Surface-A-only artifact is
unnecessary.

Two silent companion probes exercise both directions of the biconditional:
one stores null where every contributing seed rate is finite, and one stores
a finite rate where a contributing block seed rate is null. Round 5 must
reject both with the dedicated biconditional diagnostic before the smoke can
report success.

The pinned-runtime smoke prints:

```text
ROUND3_REDUCER_SYNTHETIC_SMOKE=EXPECTED_FAIL exit=1 error="ValueError: fit_validation.2006.reference_spells does not match its independent recomputation"
ROUND4_A0C9D916_REDUCER_SYNTHETIC_SMOKE=EXPECTED_FAIL exit=1 error="ValueError: fit_validation.2006.divorced_calibration does not match its independent recomputation"
ROUND4_REDUCER_SYNTHETIC_SMOKE=EXPECTED_FAIL exit=1 error="ValueError: boundaries.2006.laws.R0.mean.groups[11].rate must be a JSON number"
ROUND5_REDUCER_SYNTHETIC_SMOKE=PASS exit=0 selected_law=R0 removed_arrays=600 null_rate_aggregates=90 zero_exposure=75 positive_exposure=15 biconditional_rejections=2 strict_serialization=PASS aggregate_assert_same=PASS
```

The synthetic R0 result is fixture behavior only. It has no authority over
the frozen real stdout.

## 7. Honestly bounded residual

The completeness certificate closes the input contract and value-type class;
it does not pre-adjudicate the reduction. Two residuals are intentionally
left for the one frozen execution:

1. `_assert_same` still requires exact float-representation equality for
   finite aggregate entries and ratios and for the complete recomputed
   `selection.metrics`, `eligibility`, and one-SE tree. Aggregate agreement is
   expected because the selector and reducer share byte-identical arithmetic
   lineage under the pinned CPython and NumPy versions. Selection-tree
   agreement is an independent recomputation whose equality is the
   adjudication itself. Statically declaring it equal would defeat the
   reduction.
2. `_validate_final_fit` follows the independently recomputed selected law.
   The input carries exactly the five-key R0 final-fit shape. If the
   recomputation selects a non-R0 law, it will require the corresponding 2014
   fit record that the raw R0 claim did not publish. That branch is the
   intended raw-versus-reduced disagreement path, not another latent
   null-contract defect.

No amendment code may coerce those residuals toward the public raw R0 claim.
Their observed result must be preserved and published under section 8.

## 8. Referee, freeze, one attempt, and three outcomes

This draft stops before the authorized reduction. If and only if a new
referee endorses the exact amendment, the procedure is:

1. Verify the input-specific completeness certificate, the two-surface diff,
   explicit biconditional enforcement, byte identity of every other reducer
   path, full-path synthetic smoke, and zero-real-stdout-contact design.
2. Ratify the accepted amendment. The ratifying squash is the round-5 freeze
   for the amendment, reducer, and smoke bytes.
3. From a clean tree, reverify the frozen reducer, exact input bytes, and
   CPython `3.13.12`, NumPy `2.4.2`, and pandas `2.3.3` runtime.
4. Execute the frozen reducer exactly once over only the SHA-pinned
   `370bef8d...` stdout. There is no selector attempt, staged-source open,
   retry, patch, field stripping, manual completion, or substitute result.
5. Preserve and publish the complete finding, captured streams, execution
   record, and zero-staged-source file-open audit under exactly one of these
   outcomes:
   - **Ratified no-op:** if the reducer reproduces R0 and
     `NO_OP_DESIGNED_PAUSE`, publish that finding and obtain an independent
     findings referee and ratification. Only that ratification resolves the
     #243 `NOT_DETERMINED` reduction pause into the plan's canonical no-op;
     the frozen section-7 designed no-op ladder still governs.
   - **Disagreement verbatim:** if the independent recomputation disagrees
     with the raw R0/one-SE/final-fit claim, preserve and publish the
     disagreement verbatim, including the exact terminal diagnostic and both
     sides available from the frozen artifacts. Do not relabel the audited
     non-R0 final-fit branch as a null-contract incident, reconstruct a result
     manually, or change either side to force agreement.
   - **Incident:** for any other exception, runtime or byte mismatch, new
     contract defect, or serialization failure, stop and publish an incident
     with an unratifiable finding.
6. Obtain the applicable independent findings referee and ratification before
   treating any reduction result or disposition as canonical.

Every branch consumes the one attempt. None authorizes an in-lane rerun or
self-rescue.

## 9. Designed pause and registration 8

Until a round-5 finding completes independent review and ratification, the
canonical result remains `NOT_DETERMINED`, the designed pause remains active,
and registration 8 is forbidden. A ratified R0/
`NO_OP_DESIGNED_PAUSE` resolves the #243 reduction indeterminacy but then
continues down the frozen candidate-blind W1 surface-question ladder in the
round-2 plan. It is not authority to register a nonzero law. A disagreement
or incident likewise leaves registration 8 forbidden and requires the
prospective process appropriate to that published result.

The post-2014 holdout cannot break a tie, force a nonzero law, repair a
disagreement, or rescue an incident. Registration 8 remains forbidden unless
and until the applicable pause-resolving downstream amendment,
implementation, tests, immutable registry specification, candidate-2 lock,
and ratification are complete.

## 10. Scope

This amendment lane changes only:

- `scripts/reduce_m6_remarriage_round5.py`;
- `scripts/smoke_m6_remarriage_round5_reducer.py`; and
- this document.

It does not modify any frozen round-2, round-3, or round-4 script, a selector
or config, an incident ledger, `gates.yaml`, `src/`, `tests/`, `runs/`,
`pyproject.toml`, or a per-file lint exemption. It performs no real reduction
or selection and no staged/PSID source contact or real-stdout open. The
round-5 reducer and smoke must pass Black at line length 79 and Ruff without
an exemption.

[round2-plan]: m6_remarriage_learning_plan_round2.md
[round4]: m6_remarriage_round4_amendment.md
[incident]: ../analysis/m6_remarriage_round4_reduction_execution_defect.json
[raw-stdout]: ../analysis/m6_remarriage_round3_selection_full.json
[smoke]: ../../scripts/smoke_m6_remarriage_round5_reducer.py
[pr-234]: https://github.com/PolicyEngine/populace-dynamics/pull/234
[pr-241]: https://github.com/PolicyEngine/populace-dynamics/pull/241
[pr-242]: https://github.com/PolicyEngine/populace-dynamics/pull/242
[pr-243]: https://github.com/PolicyEngine/populace-dynamics/pull/243
[completeness-certificate]: https://api.github.com/repos/PolicyEngine/populace-dynamics/issues/comments/5009594753
