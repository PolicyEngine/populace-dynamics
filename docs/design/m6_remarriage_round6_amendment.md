# M6 remarriage round-6 amendment: reconstruction completeness

- **Status:** **PROSPECTIVE AMENDMENT — UNRATIFIED.** This lane writes a
  corrected reducer, an extended synthetic smoke, and this amendment. It
  performs no real reduction, real selector run, staged-source contact, or
  PSID contact. The pinned public stdout was read only through a key/type/shape
  projection for the certificate in section 4. A new independent referee
  decision and ratification must precede the one round-6 attempt authorized
  below.
- **Plan authority:** the ratified [round-2 learning plan][round2-plan], PR
  [#234][pr-234] at squash
  `004c57d306741e1d97275a720effbb41f378d763`.
- **Round-5 freeze:** the ratified [round-5 amendment][round5-amendment], PR
  [#244][pr-244] at squash
  `135b165432bffd9a4a30701f75cdc46ac6427b06`.
- **Ratified finding:** the [round-5 disagreement record][round5-finding], PR
  [#245][pr-245] at squash
  `674183e146410701cb142b2d2dd1a9833471de8b`.
- **Forensic prescription:** issue comment
  [5009948384][forensic-referee] verified #245 and classified its divergence
  as **(iii), a comparison-shape artifact with zero adjudicative content**.

## 1. Canonical state after round 5

The one authorized round-5 reduction reached the complete-object comparison
between the published selection tree and the reducer's independent
reconstruction, then exited 1. It emitted no findings JSON. Outcome (b),
`DISAGREEMENT_VERBATIM`, is ratified; the attempt is consumed.

The forensic referee reconstructed both operands out of lane in read-only
scratch space and established all of the following:

- every value defined on both sides was bit-identical, including zero ulp
  deltas for every shared float;
- every eligibility verdict, rule result, metric, one-SE field, and headline
  agreed;
- the common headline was R0, `no_eligible_nonzero_law`, null `Lbest`, null
  `one_SE_cutoff`, and `NO_OP_DESIGNED_PAUSE`;
- the only divergence was 12 key-presence differences, each a selector-only
  `truth_branch_law_independent: true` annotation; and
- deleting only those 12 keys from the raw object made the canonical JSON
  byte strings equal.

That is zero adjudicative content: no quantity used to choose a law or
disposition differed. The frozen round-5 taxonomy nevertheless classified
the observed raise correctly as outcome (b), because it occurred at the
section-7 residual-1 comparison. The later forensic classification is
**(iii)** rather than a relabeling of that execution record.

The referee's recomputation is evidence that the repaired round-6 run will
take the ratified-no-op path. It is not the result and has no canonical
authority. The run remains the authority, followed by its independent
findings review and ratification.

Until then, the canonical determination remains `NOT_DETERMINED`, both
per-origin outcomes remain `NONE`, the designed pause remains active, and
registration 8 remains forbidden. The public raw R0 claim is still not a
ratified reduction result.

## 2. Plan status of the annotation

The plan contains two relevant and non-identical statements:

1. Section 5.1, lines 547–549, declares that the truth-defined branch
   `m_wb > 0` versus `m_wb = 0` is computed once per boundary and is
   law-independent.
2. Section 5.3's rule 7, lines 649–656, enumerates the positive-event and
   zero-event widowed checks. It specifies neither a published
   `truth_branch_law_independent` field nor a law-independence conjunct in the
   published `branch_pass` expression.

The selector implements the section-5.1 declaration as a runtime conformance
assertion: it compares a candidate's
`widowed_matchable_positive_weight_event_rows` with same-seed R0 at the same
boundary, includes that Boolean in either branch's `branch_pass`, and
publishes it. The round-2 through round-5 reducers implement the checks
enumerated in rule 7 but omit that conjunct and field.

Neither side violates the plan's selection semantics. The plan declares the
property but does not specify the selection record's exact key set; wherever
the declaration holds, the added conjunct is vacuous. The strict residual-1
comparison made the selector's documentary annotation load-bearing as record
shape. Round 6 therefore repairs reconstruction completeness and closes the
future semantic gap without changing a plan rule, tolerance, law, order, or
disposition.

## 3. Exact round-6 reducer amendment

[`reduce_m6_remarriage_round6.py`][round6-reducer] is copied from the frozen
round-5 reducer. Only the following three repairs are authorized.

### 3.1 Independently reconstruct the published annotation

For each of the four nonzero laws and each of the three boundaries,
`_independent_selection` independently computes

```text
truth_branch_law_independent =
    candidate widowed_matchable_positive_weight_event_rows
    == same-boundary R0 widowed_matchable_positive_weight_event_rows
```

It emits that Boolean in the corresponding
`widowed_truth_defined_branches.<boundary>` record. It does not copy the raw
selection annotation and does not hardcode `true`.

### 3.2 Adopt the selector's rule-7 conjunct exactly

Both selector branches include the recomputed law-independence Boolean in
`branch_pass`:

- the positive-event branch requires law independence and widowed deviance no
  worse than R0; and
- the zero-event branch requires law independence, positive risk exposure,
  and the unchanged `g`, `omega`, and `B_W` guard.

The published detail fields retain the selector's exact behavior. In
particular, `widowed_deviance_no_worse_R0` and `g_guard` continue to report the
whole branch result. No eligibility condition is relaxed.

### 3.3 Make any future residual-1 disagreement self-documenting

Before raising at the final selection/one-SE comparison, round 6 writes one
canonical, single-line JSON diagnostic to stderr. Its schema is
`m6.remarriage.round6.selection_diff.v1`; its top-level keys are `schema`,
`where`, `key_presence_diffs`, and `value_diffs`. Each key-presence record
names a JSON path and gives `raw.present` and `recomputed.present`, including
the available side's value. Each shared-path value record gives `path`,
`raw_value`, and `recomputed_value`. A list-length mismatch is one value diff
at the list path; equal-length lists are walked by index. The reducer then
raises the same strict comparison error. It does not strip, ignore,
normalize, or accept an unknown field.

This repair changes disagreement visibility, not comparison strictness. A
future outcome (b) can publish both sides at every differing path directly
from frozen captures instead of reconstructing an un-emitted object after the
attempt.

Except for those three line-accounted repairs, the round-5 reducer bytes and
behavior remain unchanged. The frozen round-2, round-3, round-4, and round-5
reducers remain untouched.

## 4. Upgraded reconstruction-completeness certificate

### 4.1 Method, evidence boundary, and reproducible fingerprint

This certificate adds the half of round-5 residual 1 that was statically
auditable: the complete key set and container shape of the reducer-emitted
selection tree. It does not certify scalar equality or select a law.

The public input remains exactly:

- path: [round-3 full selector stdout][raw-stdout];
- byte count: `14,606,212`;
- SHA-256:
  `370bef8d00313e847c9cbb0fe09d6b6a5b9dddcf2658449ab61b02a42534614b`;
- schema/status: one strict JSON object and `SELECTION_COMPLETE`; and
- coverage: 3 boundaries × 5 laws × 40 common-random-number seeds.

For this certificate only, a `jq` projection recursively retained object
keys and array positions while replacing every scalar with its JSON type
name. It did not print or compare a scalar value. Canonical compact,
key-sorted serialization of `.selection` after that erasure has SHA-256:

```text
4e14980fb4ac17da8e8014d4708db0787c5fdd95fdaf12b5438589f2e920093d
```

The erased public tree contains 362 objects, 47 arrays, 413 Boolean leaves,
894 number leaves, 23 string leaves, and 2 null leaves. All five
`metrics.<law>` subtrees have the same erased shape, whose per-law digest is
`fc22b522b3958071c99ba519d7abef5691ad3a3f5d9b406157664e55cdc0cbdc`.
All four nonzero-law `eligibility.<law>` subtrees likewise have one identical
erased shape, whose per-law digest is
`052da8aed16dcc8c015368ed88ccda0ba93d44cc5154e37a6d61a5b5860c604e`.

The second half of the certificate is static source reconciliation. It maps
every selector container constructor to the corresponding round-6
constructor and checks exact child-key sets, array-generating loops, and
branch-specific record constructors. Neither the selector nor any reducer
was executed. No real reduction occurred.

### 4.2 Exhaustive path notation

The manifest below is exhaustive for `.selection`. Abbreviations denote
literal finite expansions, not omitted path classes:

- `L` is each of `R0`, `R_D50_W00`, `R_D75_W00`, `R_D50_W05`, and
  `R_D75_W05`;
- `C` is each of the four nonzero laws;
- `b` is each of `2006`, `2008`, and `2010`;
- `o` is each of `divorced` and `widowed`;
- `i` is each jackknife array position 0 through 39; and
- `cmp` is each of
  `rule_4_boundary_rate_transport`,
  `rule_5_endogenous_exposure_protection`,
  `rule_6_direct_fit.boundary_comparison`,
  `rule_7_origin_protection.divorced_direct`, and
  `rule_7_origin_protection.working_age_widow_exposure` below
  `selection.eligibility.C.rules`.

### 4.3 Top-level and metrics key/shape reconciliation

Every object path and array path in the top-level and metrics half is listed
here. Child sets are exact on both the pinned public tree and the round-6
constructor.

| Path | Exact children or array shape |
|---|---|
| `selection` | `selected_law`, `selected_joint_law`, `per_origin_outcome`, `disposition`, `selection_reason`, `eligible_nonzero_laws`, `Lbest`, `one_SE_cutoff`, `comparison_tolerance`, `simplicity_order`, `metrics`, `eligibility` |
| `selection.per_origin_outcome` | `divorced:string`, `widowed:string` |
| `selection.eligible_nonzero_laws` | array of law-name strings; pinned length 0; cardinality is a selection value |
| `selection.simplicity_order` | array of 5 strings |
| `selection.metrics` | the five literal `L` keys |
| `selection.metrics.L` | `J`, `block_J`, `jackknife`, `pooled_direct`, `origin_pooled_direct`, `boundary` |
| `selection.metrics.L.block_J` | `block_1:number`, `block_2:number` |
| `selection.metrics.L.jackknife` | `replicates`, `Jjack_bar:number`, `SE_J:number`, `same_seed_deleted_across_all_boundaries:boolean`, `reselection_inside_replicates:boolean` |
| `selection.metrics.L.jackknife.replicates` | array of 40 objects |
| `selection.metrics.L.jackknife.replicates[i]` | `deleted_seed:number`, `retained_seed_count:number`, `J_delete_one:number` |
| `selection.metrics.L.pooled_direct` | `weighted_deviance_numerator:number`, `deviance_exposure:number`, `weighted_bernoulli_deviance:number` |
| `selection.metrics.L.origin_pooled_direct` | `divorced`, `widowed` |
| `selection.metrics.L.origin_pooled_direct.o` | same exact three-number direct-deviance record as `pooled_direct` |
| `selection.metrics.L.boundary` | `2006`, `2008`, `2010` |
| `selection.metrics.L.boundary.b` | `rate_error:number`, `exposure_error:number`, `working_age_widow_exposure_error:number`, `pooled_direct_deviance:number`, `divorced_direct_deviance:number`, `widowed_direct_deviance:number`, `widowed_matchable_positive_weight_event_rows:number`, `widowed_direct_risk_exposure:number`, `g_widowed_log_qdir_ratio:number` |

The remaining top-level leaves are present unconditionally:
`selected_law:string`, `selected_joint_law:string`, `disposition:string`,
`selection_reason:string`, `comparison_tolerance:number`, and the
selection-dependent `Lbest:string|null` and `one_SE_cutoff:number|null`. The
pinned public types at the last two paths are null. Their realized types,
the length and contents of `eligible_nonzero_laws`, and every scalar value
remain part of genuine selection equality for the run; their keys and type
domains do not.

### 4.4 Eligibility key/shape reconciliation

Every object and array path in the eligibility half is listed below. The
selector and round 6 use the same finite law, boundary, and comparison loops.

| Path | Exact children or array shape |
|---|---|
| `selection.eligibility` | the five literal `L` keys |
| `selection.eligibility.R0` | `eligible_as_baseline:boolean`, `rule_1`, `R0_bit_equivalence:boolean` |
| `selection.eligibility.R0.rule_1` and `selection.eligibility.C.rules.rule_1_defined_and_conformant` | `pass:boolean`, `checks` |
| either rule-1 `.checks` | `all_required_selector_values_finite:boolean`, `all_required_log_arguments_positive:boolean`, `support_carrier_event_rng_conformance:boolean`, `fit_side_validation_match:boolean` |
| `selection.eligibility.C` | `eligible:boolean`, `rules` |
| `selection.eligibility.C.rules` | the seven literal keys `rule_1_defined_and_conformant` through `rule_7_origin_protection` |
| `...rule_2_per_origin_construction_and_budget` | `pass:boolean`, `boundaries` |
| `...rule_2_per_origin_construction_and_budget.boundaries` | `2006`, `2008`, `2010` |
| `...rule_2_per_origin_construction_and_budget.boundaries.b` | `pass:boolean`, `checks`, `g:number`, `omega:number`, `B_W:number` |
| that boundary record's `.checks` | 12 Booleans: `divorced_area_within_tolerance`, `widowed_area_within_tolerance`, `divorced_delta_exact_target`, `widowed_delta_exact_target`, `positive_divorced_beta`, `positive_exposure_divorced_cell_rises`, `positive_exposure_divorced_cell_falls`, `widowed_cells_exact_uniform_shift`, `g_nonnegative`, `g_no_greater_than_omega`, `omega_nonnegative`, `omega_within_budget` |
| `...rule_3_full_and_block_loss` | `pass:boolean`, `checks` |
| `...rule_3_full_and_block_loss.checks` | `full_J_strictly_better_R0:boolean`, `block_1_J_strictly_better_R0:boolean`, `block_2_J_strictly_better_R0:boolean` |
| each `...cmp` object | `pass:boolean`, `strict_improvements`, `no_worse`, `strict_improvement_count:number`, `required_strict_improvements:number` |
| each `...cmp.strict_improvements` and `...cmp.no_worse` | array of 3 Booleans, one per boundary |
| `...rule_6_direct_fit` | `pass:boolean`, `pooled_direct_strictly_better_R0:boolean`, `boundary_comparison` |
| `...rule_7_origin_protection` | `pass:boolean`, `divorced_direct`, `widowed_truth_defined_branches`, `working_age_widow_exposure` |
| `...rule_7_origin_protection.widowed_truth_defined_branches` | `2006`, `2008`, `2010` |
| each positive-event branch record, pinned at `b=2006,2008` | `pass:boolean`, `branch:string`, `matchable_positive_weight_event_rows:number`, `truth_branch_law_independent:boolean`, `widowed_deviance_no_worse_R0:boolean` |
| each zero-event branch record, pinned at `b=2010` | `pass:boolean`, `branch:string`, `matchable_positive_weight_event_rows:number`, `truth_branch_law_independent:boolean`, `positive_risk_exposure:boolean`, `g_guard:boolean`, `widowed_deviance_compared:boolean` |

The two branch records are intentionally different shapes. The published key
sets show the positive-event constructor at 2006 and 2008 and the zero-event
constructor at 2010 for all four candidates. The selector's constructor
provenance ties those key sets to its event-row branch. Both the selector and
the reducer derive `event_rows` from the same copied boundary direct-event
field and apply the same `> 0` condition. Round 6 therefore emits the same
branch-specific shape without inspecting or copying the raw branch label.

### 4.5 The 12 repaired paths and proof of exclusivity

The exact previously missing paths are:

```text
selection.eligibility.R_D50_W00.rules.rule_7_origin_protection.widowed_truth_defined_branches.2006.truth_branch_law_independent
selection.eligibility.R_D50_W00.rules.rule_7_origin_protection.widowed_truth_defined_branches.2008.truth_branch_law_independent
selection.eligibility.R_D50_W00.rules.rule_7_origin_protection.widowed_truth_defined_branches.2010.truth_branch_law_independent
selection.eligibility.R_D75_W00.rules.rule_7_origin_protection.widowed_truth_defined_branches.2006.truth_branch_law_independent
selection.eligibility.R_D75_W00.rules.rule_7_origin_protection.widowed_truth_defined_branches.2008.truth_branch_law_independent
selection.eligibility.R_D75_W00.rules.rule_7_origin_protection.widowed_truth_defined_branches.2010.truth_branch_law_independent
selection.eligibility.R_D50_W05.rules.rule_7_origin_protection.widowed_truth_defined_branches.2006.truth_branch_law_independent
selection.eligibility.R_D50_W05.rules.rule_7_origin_protection.widowed_truth_defined_branches.2008.truth_branch_law_independent
selection.eligibility.R_D50_W05.rules.rule_7_origin_protection.widowed_truth_defined_branches.2010.truth_branch_law_independent
selection.eligibility.R_D75_W05.rules.rule_7_origin_protection.widowed_truth_defined_branches.2006.truth_branch_law_independent
selection.eligibility.R_D75_W05.rules.rule_7_origin_protection.widowed_truth_defined_branches.2008.truth_branch_law_independent
selection.eligibility.R_D75_W05.rules.rule_7_origin_protection.widowed_truth_defined_branches.2010.truth_branch_law_independent
```

The forensic referee's full structural walk proved these 12 additions are
exclusive: every other key and every common value already matched, and
removing only these leaves equalized the canonical bytes. Sections 4.3 and
4.4 enumerate every container constructor, including every variable branch
shape. Round 6 adds the missing leaf to both branch constructors, so all 12
finite expansions are present, and changes no other selection-tree
constructor. The diagnostic addition is outside the reconstructed object.

Therefore the reducer's emitted key set and nested container shape now
reconcile exactly with the pinned published selection tree. No field is
ignored, projected away, or accepted as an open-ended extra. Only genuine
scalar/list-content equality and the selection-dependent realized union
types remain for the authorized run. That is the deliberately unadjudicated
half of residual 1.

This certificate is specific to the immutable input bytes, round-3 config,
round-6 reducer lineage, and pinned runtime named here. It is not a schema
relaxation for arbitrary future inputs.

## 5. Defect latency and certificate-gap disclosure

The selector has computed, used, and published the law-independence
annotation since round 2. No reducer generation in rounds 2, 3, 4, or 5 ever
emitted it. The defect is inherited, not introduced by the round-5 null
repair.

Earlier defect classes masked it in execution order:

1. round 3 stopped first at the `reference_spells` schema-key contradiction;
2. round-4 preparation found the companion `divorced_calibration`
   contradiction; and
3. the round-4 attempt stopped at the aggregate null-rate contract before
   the complete selection comparison.

Round 5 cleared those classes and exposed the latent selector/reconstructor
shape difference.

The pinned key set and the reducer's constructors were public at every
relevant freeze. The mismatch required no selection-value computation and
was statically auditable. Round 5's completeness certificate exhaustively
audited validator requirements and null propagation but did not reconcile
the reconstructor's emitted selection shape against the pinned tree. Its
statement that no further schema-key mismatch remained was bounded by the
residual-1 carve-out; the value-free part of that carve-out should also have
been certified. This amendment discloses and closes that certificate gap.

## 6. Extended full-path synthetic smoke

[`smoke_m6_remarriage_round6_reducer.py`][smoke] extends the synthetic
full-cube fixture. It does not open the pinned 14,606,212-byte stdout and does
not construct a real candidate finding.

The smoke preserves the four frozen reducer generations and the historical
round-4 preparation checkpoint:

1. round 3 fails at `reference_spells`;
2. the hash-pinned round-4 preparation blob fails at
   `divorced_calibration`;
3. frozen round 4 fails at the aggregate null-rate validator;
4. the original round-5-shaped synthetic fixture still passes frozen round
   5, preserving the prior smoke result; and
5. round 6 separately accepts the selector-faithful fixture and completes
   strict serialization.

Two directional probes prevent the repair from becoming laxity:

- **missing reconstruction annotation:** the selector-faithful fixture is
  sent to round 5, whose reconstructed branch records omit the field; the
  strict comparison must fail, reproducing the observed artifact; and
- **unexpected raw extra key:** an otherwise faithful fixture adds an unknown
  key to the published selection tree; round 6 must emit a key-presence diff
  and fail loudly rather than strip or ignore it.

The faithful round-6 path includes the annotation in both branch record
shapes and must pass. A companion false-annotation probe changes the
underlying synthetic candidate event-row count so the independently
recomputed annotation and `branch_pass` are both false; exact equality with
the selector-shaped expectation proves the newly adopted conjunct is active.

The pinned-runtime smoke prints exactly:

```text
ROUND6_FOUR_GENERATION_LADDER=PASS round3=EXPECTED_FAIL_REFERENCE_SPELLS round4_a0c9d916=EXPECTED_FAIL_DIVORCED_CALIBRATION round4=EXPECTED_FAIL_NULL_RATE round5=PASS_R0
ROUND6_MISSING_RECONSTRUCTION_KEY_PROBE=EXPECTED_FAIL reducer=round5 exit=1 key_presence_diffs=12 value_diffs=0
ROUND6_FAITHFUL_SELECTION_FIXTURE=PASS exit=0 selected_law=R0 annotations=12 removed_arrays=600
ROUND6_UNEXPECTED_EXTRA_KEY_PROBE=EXPECTED_FAIL reducer=round6 exit=1 key_presence_diffs=1 value_diffs=0 machine_diff=PASS
ROUND6_FALSE_ANNOTATION_CONJUNCT_PROBE=PASS law=R_D50_W00 boundary=2006 annotation=false branch_pass=false exact_selector_match=PASS
```

Any synthetic R0 headline is fixture behavior only. It is not evidence about
the frozen real stdout and has no selection authority.

## 7. Central operative provision: one frozen-stdout reduction

> **If and only if a new referee accepts this exact amendment and the
> accepted squash is ratified as the round-6 freeze, that ratification
> expressly authorizes the exact 14,606,212-byte stdout with SHA-256
> `370bef8d00313e847c9cbb0fe09d6b6a5b9dddcf2658449ab61b02a42534614b`
> to be consumed as round 6's frozen input for exactly one execution attempt
> of the frozen round-6 reducer under CPython `3.13.12`, NumPy `2.4.2`,
> pandas `2.3.3`, with SciPy `1.17.0` recorded and matched.**

This is a new, narrow prospective authorization after the consumed round-5
attempt. It is not a retry under round-5 authority. It carries forward the
same pinned-input exception, changes neither the input nor the numeric
runtime, and authorizes no selector or real-data rerun.

Before execution, the input hash and byte count, all recorded runtime
versions, frozen script bytes, amendment bytes, and clean frozen-scope diff
must match. A mismatch consumes the attempt and takes the incident branch.
The ratified squash is the freeze; no later patch may be substituted.

The authorization does not permit field stripping, manual reduction,
instrumentation, repair after execution, a second attempt, staged-source or
PSID contact, post-2014 adjudication, or use of the forensic recomputation as
a substitute result.

## 8. Referee, freeze, one attempt, and three outcomes

This draft stops before the reduction. If and only if a new referee endorses
the exact amendment, the procedure is:

1. Verify the three-line-accounted reducer repairs, strict diagnostic,
   upgraded key-set/shape certificate, both directional probes,
   generation ladder, and zero-real-reduction design.
2. Ratify the accepted amendment. The ratifying squash freezes the round-6
   amendment, reducer, and smoke bytes.
3. From a clean tree, reverify the frozen reducer and amendment, exact input
   bytes, and CPython `3.13.12`, NumPy `2.4.2`, pandas `2.3.3`, and SciPy
   `1.17.0` runtime record.
4. Execute the frozen round-6 reducer exactly once over only the SHA-pinned
   stdout. There is no selector attempt, staged-source open, retry, patch,
   field stripping, manual completion, or substitute result.
5. Preserve and publish the complete finding, captured streams, execution
   record, machine-readable diff if any, and zero-staged-source file-open
   audit under exactly one outcome:
   - **Ratified-no-op path:** if the reducer reproduces R0 and
     `NO_OP_DESIGNED_PAUSE`, publish the finding and obtain an independent
     findings referee and ratification. Only that ratification resolves the
     reduction indeterminacy; the frozen candidate-blind W1 surface-question
     ladder still continues.
   - **Disagreement verbatim:** if the strict raw-versus-independent
     comparison disagrees, preserve the exact terminal diagnostic and the
     machine-readable per-path raw/recomputed diff from frozen stderr. Do not
     relabel it, infer a missing side, strip a field, or force agreement.
   - **Incident:** for any other exception, runtime or byte mismatch, new
     contract defect, or serialization failure, stop and publish an incident
     with an unratifiable finding.
6. Obtain the applicable independent findings referee and ratification before
   treating any result or disposition as canonical.

Every branch consumes the one attempt. None authorizes an in-lane rerun or
self-rescue. The forensic referee's no-op implication sets an evidence-based
expectation only; the frozen execution and subsequent review govern.

## 9. Designed pause and registration 8

Registration 8 is forbidden until the round-6 outcome is resolved through
its applicable independent review and the plan's downstream governance.
Before then the canonical result remains `NOT_DETERMINED` and the designed
pause stays active.

A ratified R0/`NO_OP_DESIGNED_PAUSE` finding does not authorize a nonzero law
or registration. It continues the plan's frozen candidate-blind W1
surface-question ladder. A ratified nonzero selection would still require a
separate prospective amendment, implementation, tests, immutable registry
specification, candidate-2 lock, and ratification. A disagreement or incident
likewise leaves registration 8 forbidden and requires the prospective process
appropriate to that published result.

The post-2014 holdout cannot break a tie, force a nonzero law, repair a
disagreement, or rescue an incident. The candidate-2 program's decision 9
also fixes a train-only no-op or unratifiable law to the designed-pause
disposition, never post-2014 tuning.

## 10. Scope

This amendment lane changes only:

- `scripts/reduce_m6_remarriage_round6.py`;
- `scripts/smoke_m6_remarriage_round6_reducer.py`; and
- this document.

It does not modify a frozen round-2, round-3, round-4, or round-5 script, a
selector or config, an incident or disagreement ledger, `gates.yaml`, `src/`,
`tests/`, `runs/`, `pyproject.toml`, or a per-file lint exemption. It performs
no real reduction or selection and no staged/PSID source contact. The only
read of the pinned public stdout in this lane is the scalar-erased key/type/
shape audit disclosed in section 4. The round-6 reducer and smoke must pass
Black at line length 79 and Ruff without an exemption.

[round2-plan]: m6_remarriage_learning_plan_round2.md
[round5-amendment]: m6_remarriage_round5_amendment.md
[round5-finding]: ../analysis/m6_remarriage_round5_reduction.md
[raw-stdout]: ../analysis/m6_remarriage_round3_selection_full.json
[round6-reducer]: ../../scripts/reduce_m6_remarriage_round6.py
[smoke]: ../../scripts/smoke_m6_remarriage_round6_reducer.py
[pr-234]: https://github.com/PolicyEngine/populace-dynamics/pull/234
[pr-244]: https://github.com/PolicyEngine/populace-dynamics/pull/244
[pr-245]: https://github.com/PolicyEngine/populace-dynamics/pull/245
[forensic-referee]: https://github.com/PolicyEngine/populace-dynamics/pull/245#issuecomment-5009948384
