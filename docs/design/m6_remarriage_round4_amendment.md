# M6 remarriage round-4 amendment: reducer contract repair and frozen-stdout reduction

- **Status:** **PROSPECTIVE AMENDMENT — UNRATIFIED.** This lane authors a
  corrected reducer and runs a synthetic reducer smoke only. It performs no
  selection, no reduction of the real round-3 stdout, and no staged or PSID
  data contact. A new independent referee decision and ratification must
  precede the one round-4 reduction attempt authorized below.
- **Plan authority:** the ratified [round-2 learning plan][round2-plan], PR
  [#234][pr-234] at squash
  `004c57d306741e1d97275a720effbb41f378d763`.
- **Prior amendment and freeze:** the ratified [round-3 amendment][round3], PR
  [#240][pr-240] at squash and freeze
  `841c48c6f0316a08f7584402012f00b0a2b535d9`.
- **Incident authority:** the ratified [round-3 reducer finding][incident], PR
  [#241][pr-241] at squash
  `23eda4247b25cc4bd94f78957b377d201d88217b`.
- **Independent verification:** issue comment [5008315609][verification]
  returned **VERIFICATION PASSED** after reproducing the failure, both sides
  of the unsatisfiable reducer contract, the attempt ledger, the frozen-input
  hash, and the raw selection claim.

## 1. Canonical state after round 3

The frozen round-3 selector completed its only attempt with exit 0. It
published one strict JSON document containing the complete three-boundary,
five-law, forty-seed cube. Its raw claim was R0 with
`NO_OP_DESIGNED_PAUSE`, because no nonzero law passed all seven conjunctive
rules.

The frozen reducer then failed before independently recomputing and ratifying
that claim. The canonical result therefore remains `NOT_DETERMINED`, both
per-origin outcomes remain `NONE`, and the ratified incident disposition is
`UNRATIFIABLE_FINDING`. `NONE` is not R0 and is not a ratified no-op. The
designed pause continues and registration 8 remains forbidden.

The complete selector stdout is already public as part of [#241][pr-241]:

- path: [the round-3 full selector stdout][raw-stdout];
- byte count: `14,606,212`;
- SHA-256:
  `370bef8d00313e847c9cbb0fe09d6b6a5b9dddcf2658449ab61b02a42534614b`;
- schema/status: one strict JSON object and `SELECTION_COMPLETE`; and
- coverage: 3 boundaries x 5 laws x 40 seeds, or 600 common-random-number
  draws.

## 2. Central operative provision: one frozen-stdout reduction

> **If and only if a new referee accepts this amendment and the accepted
> squash is ratified as the round-4 freeze, that ratification expressly
> authorizes the exact 14,606,212-byte stdout with SHA-256
> `370bef8d00313e847c9cbb0fe09d6b6a5b9dddcf2658449ab61b02a42534614b`
> to be consumed as round 4's frozen input for exactly one execution attempt
> of the frozen round-4 reducer.**

This is the amendment's central provision. It is a narrow prospective
exception to the round-2 plan section 6 instruction, "Do not repurpose
output," and the round-3 amendment section 6, step 5 instruction to amend and
refreeze "instead of repurposing partial output." Those provisions did not
contemplate a later deterministic reduction of a captured, complete stdout.
For this input and this attempt only, the provision above controls.

The authorization does not:

- retroactively ratify the raw R0 claim;
- authorize manual field stripping, a hand reduction, or a reducer retry;
- authorize a selector, preflight, or staged-source rerun;
- authorize any other reuse of this stdout or any partial output; or
- weaken the stop, disclose, amend, referee, and refreeze rule for a further
  defect.

Round 4 is reducer-only. It makes no staged-source or PSID file contact. The
round-4 finding must include a zero-staged-source file-open audit. The
ratified squash is the round-4 freeze; no later patch may be substituted for
its reducer.

## 3. Public disclosure creates no adaptive freedom

The raw claim and the full metric surface are already public in ratified
[#241][pr-241]. The publication includes the R0/no-op claim and reason, every
per-law rule flag, full and fixed-block `J`, delete-one jackknife values,
per-boundary exposure and rate errors, pooled and per-origin direct
deviances, all 600 seed draws, and the fit-side audit objects. It also records
`final_information_fit=NOT_RUN_R0_SELECTED`; no final-2014 fit was run.

Re-reduction cannot adapt to that disclosure:

1. The input bytes and byte count are immutable and independently checked.
2. The reducer is deterministic and may execute once.
3. The round-4 diff is confined to the three statically enumerated comparison
   legs in section 4. The divorced-calibration repair admits only the three
   selector-published fields named there, projects them away, and retains
   exact canonical equality on every shared field.
4. Every objective, block, jackknife, eligibility rule, one-SE choice,
   disposition, publication removal, and other non-comparison path remains
   byte-for-byte the round-3 implementation.
5. The referee must mechanically compare the new reducer with the frozen
   round-3 reducer, whose SHA-256 is
   `41db8a271d08596ad6383494f37695e658b6e4f7531459d95221f684b09c256e`,
   and reject any change outside those three legs.

No disclosed metric can set or alter a law, tolerance, seed, order, rule, or
tie-break. A corrected reduction is allowed to disagree with the raw claim;
it must not be coerced toward it.

## 4. Verified defects and exact three-leg repair

The frozen selector's `_reference_spells` creates a 20-key
`references[origin]["audit"]` mapping and `_fit_boundary` appends `R0_area`,
making exactly 21 keys. It publishes that mapping at
`fit_validation.<boundary>.reference_spells.<origin>`. The frozen validation
object has 14 keys per origin. Across all six boundary-origin records, the
verification found the same seven raw-only keys, no validation-only key, and
exact equality for every shared field.

The authorized 21-key surface is exactly:

1. `R0_area`
2. `eligible_spells_before_exclusions`
3. `duplicate_key_groups`
4. `duplicate_spells_excluded`
5. `duplicate_spells_excluded_weight`
6. `duplicate_spells_checksum_sha256`
7. `same_year_remarriage_spells_excluded`
8. `same_year_remarriage_spells_excluded_weight`
9. `same_year_remarriage_spells_checksum_sha256`
10. `missing_required_or_nonpositive_weight_spells_excluded`
11. `missing_required_or_nonpositive_weight_spells_excluded_weight`
12. `missing_required_or_nonpositive_weight_checksum_sha256`
13. `no_potential_path_spells_excluded`
14. `no_potential_path_spells_excluded_weight`
15. `no_potential_path_spells_checksum_sha256`
16. `included_spells`
17. `included_spell_weight`
18. `included_spells_checksum_sha256`
19. `potential_path_years`
20. `working_age_path_year_terms`
21. `path_checksum_sha256`

The seven additive keys are the five category checksums for duplicate,
same-year, missing-required-or-nonpositive-weight, no-potential-path, and
included spells, plus the missing-required-or-nonpositive-weight and
no-potential-path excluded-weight values.

### Leg A: frozen-validation comparison

Round 3 called `_assert_same` on the whole 21-key selector mapping and the
14-key frozen mapping. Canonical equality was impossible, so the first
reachable comparison raised at
`fit_validation.2006.reference_spells`.

Round 4 retains exact canonical comparison for all 14 shared frozen fields.
It additionally requires both origins, requires each observed audit to have
exactly the enumerated 21 keys, validates all six checksum-shaped values and
the two additive excluded weights, and rejects a missing or unknown extra
key loudly. It then projects the accepted mapping onto the frozen 14-key
surface for `_assert_same`. No global comparison helper changes.

### Leg B: exclusion-category hashes

The later round-3 leg derived six checksum-suffixed values from
`reference_spells` and exact-compared them with
`reference_exclusion_category_hashes`. If Leg A instead received the narrow
14-key form it demanded, only `path_checksum_sha256` existed, while the
selector's hash object contained six fields. The later leg would therefore
reject the only shape the earlier leg accepted. No complete output of the
frozen selector could satisfy both legs.

Round 4 reads an explicit six-key tuple from the 21-key audit already
accepted by Leg A: the five category checksums plus `path_checksum_sha256`.
It exact-compares those values with the selector's separately published hash
object. An explicit tuple prevents a future suffix-named field from silently
widening the contract.

### Leg C: divorced calibration

The selector publishes `area_R0`, `candidate_area`, and
`pairwise_term_count` inside every
`fit_validation.<boundary>.divorced_calibration.<alpha>` mapping, while the
frozen validation mapping omits those three additive fields. Round 3 filters
only the top-level alpha names and then passes each inner mapping whole to
`_assert_same`, so the selector-shaped and validation-shaped mappings cannot
be canonically equal.

Round 4 explicitly enumerates the three additions, requires them in every
observed alpha mapping, validates both areas as finite JSON numbers and
`pairwise_term_count` as a nonnegative integer, and rejects a missing or
unknown extra key loudly. It projects only those three fields out and
exact-compares every shared field with frozen validation.

These three legs are the complete reducer amendment. `_assert_same` itself
and every other reducer path remain unchanged.

## 5. Defect latency

The mismatches were inherited from round 2, not introduced by the round-3
amendment. At the round-2 freeze
`f44246964b4e54080ab4c7b43e7e5b0c4b78ea7c` and the round-3 freeze
`841c48c6f0316a08f7584402012f00b0a2b535d9`, the retained round-2 artifacts
are byte-identical:

| Frozen round-2 artifact | SHA-256 at both freezes |
|---|---|
| selector | `deed33105cffabda4477ac3b7d8b6f0edf9b90bb3ddb83ee42d7ef3434578268` |
| reducer | `6b6f4e0459be23c936894308e6e040219d623762a15926037a7e2d9a2ac83918` |

The round-2 selector published the same audit shape and the same three
additive divorced-calibration fields. Its reducer carried both the same exact
reference-audit comparison and the same wholesale inner-dict calibration
comparison, so `divorced_calibration` has the same round-2 latency. Round 2
died earlier in the selector, at the boundary-2006 event-free publication
group, so the reducer was never reached. The round-3 equivalence review
correctly established that the reducer logic was inherited; a
diff-equivalence review could not detect inherited selector-to-reducer
contract contradictions. A selector-faithful full reducer-shaped synthetic
smoke is the missing check supplied here.

## 6. Full-path synthetic smoke

[`smoke_m6_remarriage_round4_reducer.py`][smoke] creates one synthetic fixture
in memory that is selector-faithful on all three repaired comparison legs. It
takes the shared fit fields from the frozen validation mapping, adds exactly
seven deterministic synthetic audit fields to form the 21-key surface, and
adds deterministic synthetic `area_R0`, `candidate_area`, and
`pairwise_term_count` values to every divorced-calibration alpha mapping. It
uses candidate-blind preflight law tables only where the frozen incumbent
table hashes require them. Frozen pre-contact config, validation, preflight,
truth, support, freeze, and runtime locks are inherited only to satisfy the
full reducer contract. Candidate projection, direct, publication-group, RNG,
downstream, and per-seed values are generated synthetically. No value is read
from the 14,606,212-byte stdout, and that file is never opened.

The minimal contract-valid cube still contains the reducer-required
3 x 5 x 40 rows and 18 publication groups per row. The harness sends the same
generated bytes to the actual round-3 reducer, the exact hash-checked
`a0c9d916` round-4 reducer blob, and the extended round-4 reducer. Round 3
fails first at `reference_spells`; `a0c9d916` passes the first two repaired
legs and fails at `divorced_calibration`; and only the extension proceeds
through config, freeze, runtime, input audit, fit, all 600 rows, aggregates,
blocks, jackknife, seven rules, selection, final disposition, and removal. A
silent companion probe adds one unknown calibration key and requires the
extension to reject it with the strict-key diagnostic. The smoke reads no
staged source and constructs no real pseudo-holdout or candidate outcome.

The pinned-runtime smoke prints:

```text
ROUND3_REDUCER_SYNTHETIC_SMOKE=EXPECTED_FAIL exit=1 error="ValueError: fit_validation.2006.reference_spells does not match its independent recomputation"
ROUND4_A0C9D916_REDUCER_SYNTHETIC_SMOKE=EXPECTED_FAIL exit=1 error="ValueError: fit_validation.2006.divorced_calibration does not match its independent recomputation"
ROUND4_REDUCER_SYNTHETIC_SMOKE=PASS exit=0 selected_law=R0 removed_arrays=600
```

The synthetic R0 result is fixture behavior only. It is not evidence about
the frozen real stdout and has no selection authority.

## 7. Round-4 referee, freeze, and one-attempt procedure

This draft stops before the authorized reduction. If and only if the new
referee endorses the exact amendment, the procedure is:

1. The referee verifies the central exception in section 2, the input hash
   and byte count, the exact three-leg diff, byte identity of every other
   reducer path, the full-path smoke, and the zero-data-contact design.
2. Ratify the accepted amendment. The ratifying squash is the round-4 freeze
   for the amendment, reducer, and smoke bytes.
3. From a clean tree, reverify the frozen reducer, exact input bytes, and
   runtime: CPython `3.13.12`, NumPy `2.4.2`, and pandas `2.3.3`.
4. Execute the frozen reducer exactly once over only the SHA-pinned
   `370bef8d...` stdout. There is no selector attempt, staged-source open,
   retry, patch, field stripping, self-rescue, or substitute result.
5. Publish the complete finding and zero-staged-source file-open audit
   regardless of outcome:
   - a reducer-validated R0 publishes a `NO_OP_DESIGNED_PAUSE` finding that
     may become ratified only after step 6;
   - if deterministic reduction disagrees with the raw claim, publish the
     selected nonzero law and every recomputed check; or
   - if any further defect occurs, stop and publish another incident with an
     unratifiable finding.
6. Obtain an independent findings referee and ratification before treating
   the reduction or any disposition as ratified.

Any runtime, byte, open-audit, or contract mismatch consumes the one attempt
and follows the incident branch. It does not authorize a retry.

## 8. Outcomes, the designed pause, and registration 8

| Round-4 publication | Effect on the designed pause |
|---|---|
| Ratified R0 / `NO_OP_DESIGNED_PAUSE` | Resolves the #241 indeterminacy but does not lift the pause; it proceeds down the frozen candidate-blind W1 surface-question ladder in plan section 7. |
| Reducer-validated nonzero law | Is learning evidence only; the pause can lift only after a separate prospective amendment, implementation and tests, immutable registry specification, candidate-2 lock, and their required ratification. |
| Another incident / unratifiable finding | Leaves the pause active and requires another prospective amendment and referee decision. |

Thus none of the three bare reduction publications independently resolves
the designed pause or authorizes registration. A pause-resolving downstream
remedy must itself complete and be ratified under the applicable ladder.
Registration 8 remains forbidden until that occurs. The post-2014 holdout
cannot break a tie, force a nonzero law, or rescue an incident.

## 9. Scope

This amendment lane changes only:

- `scripts/reduce_m6_remarriage_round4.py`;
- `scripts/smoke_m6_remarriage_round4_reducer.py`; and
- this document.

It does not modify the frozen round-2 or round-3 scripts, a selector or
config, an incident ledger, `gates.yaml`, `src/`, `tests/`, `runs/`, or a
per-file lint exemption. It performs no real reduction or selection and no
staged/PSID source contact or real-stdout open. The round-4 reducer and smoke
must pass Black at line length 79 and Ruff without an exemption.

[round2-plan]: m6_remarriage_learning_plan_round2.md
[round3]: m6_remarriage_round3_amendment.md
[incident]: ../analysis/m6_remarriage_round3_selection_execution_defect.json
[raw-stdout]: ../analysis/m6_remarriage_round3_selection_full.json
[smoke]: ../../scripts/smoke_m6_remarriage_round4_reducer.py
[pr-234]: https://github.com/PolicyEngine/populace-dynamics/pull/234
[pr-240]: https://github.com/PolicyEngine/populace-dynamics/pull/240
[pr-241]: https://github.com/PolicyEngine/populace-dynamics/pull/241
[verification]: https://github.com/PolicyEngine/populace-dynamics/pull/241#issuecomment-5008315609
