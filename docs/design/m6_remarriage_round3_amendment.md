# M6 remarriage round-3 amendment: empty-group selector repair

- **Status:** **PROSPECTIVE AMENDMENT — UNRATIFIED.** This amendment and its
  synthetic smoke do not authorize or perform selection. A new referee
  decision and a new freeze must precede the round-3 investigation.
- **Unchanged authority:** the ratified [round-2 learning plan][round2-plan],
  PR [#234][pr-234] at squash `004c57d306741e1d97275a720effbb41f378d763`.
- **Incident authority:** the ratified [round-2 execution finding][incident]
  in PR [#238][pr-238] at squash
  `711a1914a433f3fb986607b1ef02a20077864f71`.
- **Independent verification:** issue comment
  [5007096806][verification] returned **VERIFICATION PASSED** after a
  read-only reproduction of the mechanism, contact bound, leak bound, freeze,
  and ledger hashes.
- **Scope:** new `_round3` selector, reducer, and config files plus this
  document. No real-data contact, selection, `src/`, test, gate, registry,
  floor, or `runs/` change belongs to this lane.

## 1. Why a prospective round 3 is required

The round-2 selector was frozen at
`f44246964b4e54080ab4c7b43e7e5b0c4b78ea7c` before candidate-outcome
contact. Its one permitted execution began on July 17, 2026 at 15:36:40 EDT,
exited 1 at 15:37:16, and was not rerun. The ratified incident disposition is
`UNRATIFIABLE_FINDING`: selection is `NOT_DETERMINED`, both per-origin
outcomes are `NONE`, and `NONE` does not mean R0 or
`NO_OP_DESIGNED_PAUSE`. The designed pause continues and registration 8
remains forbidden.

The controlling post-contact clause in the [round-2 plan][round2-plan]
requires stop, disclosure, a prospective amendment, a new referee decision,
a new freeze, and a new investigation. Stop and disclosure were completed in
[#238][pr-238]. This document supplies only the prospective amendment and the
fixed implementation for referee review and later refreeze. It does not
reuse the failed attempt or convert it into a selector result.

## 2. Round-2 contact and information-leak bound

The exact execution frontier was boundary-2006 R0 direct standardization.
The [ratified ledger][incident] and its [independent verification][verification]
establish the following reached/not-reached boundary.

| Reached before the exception | Not reached |
|---|---|
| Sanitized-source load and fit-side validation for boundaries 2006, 2008, and 2010 | Any nonzero law |
| Construction and frozen-lock validation of boundary-2006 pseudo-truth | Any CRN panel-projection draw or spouse-gap draw |
| Forty law-independent, seed-addressed transition-uniform checksums | Boundary-2008 or boundary-2010 pseudo-truth |
| Boundary-2006 R0 pooled and both-origin direct records | A completed R0 law record or any selection arithmetic |
| Entry into the 18-cell publication-group loop | Eligibility rules, objective `J`, jackknife SE, one-SE choice, or final-2014 fit |

The 40 transition-uniform checksums hash the pre-loop uniform stream. They are
not CRN projection draws. The exception occurred before
`ProjectionRNGRegistry` began panel projection, so the number of CRN
projection draws was exactly zero. R0 was first in the frozen law order, so
no nonzero law could have run. Boundary 2006 was first in the boundary order,
so no 2008 or 2010 truth could have been constructed.

The information-leak bound is airtight:

1. Selector stdout contained zero bytes, with SHA-256
   `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
2. The stderr transcript contains no candidate-outcome number.
3. Every truth statistic disclosed by the incident ledger is a subset of
   `expected_pseudo_holdouts` already frozen in the round-2 config at
   `f4424696`. This includes the boundary-2006 support checksum
   `725494b1b4bcf3faebf06fcb22176041e6e66e23de70cc8d10f6d4c473320c6e`,
   pooled risk/events `1056/46`, divorced `931/40`, widowed `125/6`, and the
   same-year YSD-0 event count/weight `1/3250.0`.
4. The only new post-contact computations, the pooled and origin R0 direct
   records, remained in memory and were never serialized.

Thus the incident disclosed zero truth information beyond the pre-contact
freeze, zero nonzero-law result, zero selection arithmetic, and no 2008/2010
truth. The dynamic file-open audit result remains `NOT_DETERMINED`, not PASS:
the exception preceded payload assembly, where the observed path set would
have been serialized. This amendment does not broaden that claim.

## 3. Verified defect and exact implementation amendment

As confirmed by the [independent verification][verification], frozen selector
lines 2169–2179 used a plain Python list comprehension as a DataFrame indexer
inside the `record` closure:

```python
selected_events[[predicate(row) for row in selected_events]]
```

For a nonempty frame, pandas receives a length-matched Boolean list and
performs row selection. For an event-free publication group, the
comprehension returns `[]`; pandas interprets `frame[[]]` as an empty column
selection. The resulting frame has no `weight` column, so the next
`selected_unmatched["weight"]` lookup raises `KeyError`. The source event
frame did carry `weight`. With six widowed events distributed across nine
widowed age-by-YSD publication cells, at least one event-free cell was
certain.

A whole-file Boolean-intent indexing audit found three instances of the same
empty-list hazard in the frozen selector:

1. `_reference_spells`: the list-comprehension `same_year_mask` later used as
   `spells[same_year_mask]`;
2. `_direct_standardization`: the `matchable_events` event-frame filter; and
3. `_direct_standardization.record`: the incident-causing
   `selected_unmatched` filter.

Round 3 changes all three to an explicitly Boolean NumPy array:

```python
mask = np.asarray([predicate(row) for row in frame], dtype=bool)
selected = frame[mask]
```

For every nonempty frame, this is the same elementwise Boolean row mask. For
an empty frame, it is a zero-length Boolean row mask, so pandas returns zero
rows while preserving every column. No selection formula, ordering,
comparison, tolerance, random address, or publication quantity changes.

The only other source corrections are the existing lint findings: the
selector import block is ordered for Ruff `I001`, and the reducer imports
`Callable` from `collections.abc` for `UP035`. Round-3 files require no new
lint exemption.

The frozen round-2 artifacts remain byte-identical:

| Artifact | Frozen SHA-256 |
|---|---|
| `scripts/select_m6_remarriage_round2.py` | `deed33105cffabda4477ac3b7d8b6f0edf9b90bb3ddb83ee42d7ef3434578268` |
| `scripts/reduce_m6_remarriage_round2.py` | `6b6f4e0459be23c936894308e6e040219d623762a15926037a7e2d9a2ac83918` |
| `scripts/m6_remarriage_round2_selector_config.json` | `e83b331b5a1d7250c11d916dc1028d69e277d4c9ef5d46d8ae987507e2a5bdb2` |
| `docs/analysis/m6_remarriage_round2_selection_execution_defect.json` | `f8d7009516f3a7948baec008a681d69eeb7f7f04c9b0dce60c074caab2ebf72f` |

## 4. Synthetic smoke-gap closure

The round-2 `_synthetic_smoke` checked the fixed-adjacent pairwise sum,
divorced-root solver, omega-zero widowed identity, and raw-age probability
guard. It never called `_direct_standardization` and therefore never entered
its `record` closure. Empty publication groups were structurally unreachable
from the smoke even though they were inevitable in the real diagnostic.

Round 3 adds a synthetic-only three-risk-row/two-event fixture. Pooled and
both-origin records each receive an event and complete. The first publication
cell, `divorced|age_18_34|ysd_0_4`, has one positive-weight risk row and zero
events. The smoke calls `_direct_standardization` end to end and requires:

- all 18 publication groups to be present exactly once;
- the target cell to publish `risk_rows=1`, `event_rows=0`, `exposure=1.0`,
  zero actual/matchable/unmatched numerator quantities, and `actual_rate=0`;
- its expected numerator to equal its one-row table probability; and
- the empty event and unmatched row slices to retain their weight-bearing
  columns, proven by successful `record` completion and the published zero
  weight quantities.

The identical hardened smoke was injected with only the direct-standardizer
implementation switched between the frozen round-2 and fixed round-3
modules. It produced:

```text
round-2 hardened synthetic smoke: FAIL as expected (KeyError: 'weight')
round-3 hardened synthetic smoke: PASS (event_free_publication_group.weight_bearing_columns_intact=True)
```

This comparison directly calls `_synthetic_smoke`; it cannot dispatch
`--preflight` or `--select`, opens no staged source, constructs no
pseudo-holdout truth, and computes no candidate outcome.

## 5. Selection semantics remain byte-for-byte in force

The [#234 plan][round2-plan] is unchanged. Round 3 carries forward all
selection-bearing config values and code semantics. Only round/version/path
identifiers, the three typed empty-safe masks, synthetic smoke coverage, and
the two lint corrections differ.

The following locks remain exact:

| Contract surface | Unchanged round-3 value |
|---|---|
| Law order | `R0`, `R_D50_W00`, `R_D75_W00`, `R_D50_W05`, `R_D75_W05` |
| Family | divorced `k in {0.50, 0.75}`; widowed `omega in {0, 0.05}`; `B_W=0.08956860182931886` |
| Raw-age law | delta only on ages 18–64; exact R0 probability outside that domain |
| Construction | the same contrast, origin targets, fixed-adjacent float64 sums, `[0,16]` first-accepted-midpoint roots, `1e-10` area tolerance, and 200-iteration cap |
| Boundaries | 2006/2008/2010 with the same evaluation years and `n_periods=4/4/3` |
| CRN bank | seeds 7240–7279, blocks 7240–7259 and 7260–7279, and the same `ProjectionRNGRegistry` addresses |
| Numeric runtime | CPython 3.13.12 and NumPy 2.4.2 exactly; mismatch aborts as `ROOT_VALIDATION_MISMATCH` before selection |
| Information boundary | no 2015–2019 row, truth moment, candidate output, gate score, tolerance comparison, or realized post-2014 macro value enters selection |
| Selector | the same seven conjunctive rules, objective, fixed blocks, 40-seed delete-one jackknife, simplicity order, and one-SE/no-op decision |
| Publication | publish all diagnostics and explicit zero-event groups regardless of disposition; never suppress a failed comparison |
| Disposition | no eligible nonzero law or R0 within the one-SE cutoff selects R0 and returns `NO_OP_DESIGNED_PAUSE`; final-fit failure permits no substitution or rerun |

Rule 7 remains publish-not-compare for a truth-defined widowed zero-event
branch: widowed deviance is published but is not compared; positive risk and
`0 <= g <= omega <= B_W` remain mandatory; and working-age widowed exposure
must be no worse than same-seed R0 at every boundary. The `g` guard remains a
construction-conformance and a-priori budget cap, not a post-outcome
empirical selection filter.

The round-3 JSON config is identical to round 2 after normalizing only
`round3` schema and file/publication path identifiers back to `round2`.
The round-3 reducer is likewise identical after normalizing those identifiers
and the `collections.abc.Callable` lint correction.

## 6. Referee, freeze, and one-attempt procedure

This draft PR stops before selection. If and only if a referee ratifies this
exact amendment, the investigation proceeds in this order:

1. Record the referee decision and ratify the prospective amendment.
2. Create a new committed, clean round-3 freeze descending from the ratified
   amendment. Publish the freeze commit and SHA-256 commitments for the
   round-3 selector, config, reducer, and inherited authority artifacts.
3. Re-run the pinned-runtime audit, frozen fit-side validation, and hardened
   synthetic smoke before candidate-outcome contact. Any mismatch aborts.
4. Make one round-3 selector attempt using the unchanged staged-source,
   boundary, law, seed, and CRN procedure. There is no self-rescue or rerun
   under the same freeze.
5. Publish the full selector JSON and deterministic reduction regardless of
   the selected law, R0/no-op disposition, or final-fit failure. If a new
   post-contact defect occurs, stop, disclose, and require another
   prospective amendment and freeze instead of repurposing partial output.

An R0 result, empty eligible set, undefined required input, final-fit failure,
or unratifiable finding continues the `DESIGNED_PAUSE`. No outcome here
authorizes registration 8, changes a source/model/gate byte, or permits the
2015–2019 holdout to break a tie.

## 7. Scope and lint-exemption retirement

The amendment lane changes only:

- `scripts/select_m6_remarriage_round3.py`;
- `scripts/reduce_m6_remarriage_round3.py`;
- `scripts/m6_remarriage_round3_selector_config.json`; and
- this document.

It runs synthetic smoke only. It does not run preflight or selection, contact
PSID data, create an analysis result, or modify `gates.yaml`, `src/`,
`tests/`, `runs/`, or the frozen incident ledger.

The existing `pyproject.toml` per-file ignores for the round-2 selector and
reducer remain necessary because those historical files stay in-tree with
their incident-frozen bytes. Round 3 adds no ignore and lints clean. The
obligation to remove the round-2 exemptions transfers to the retirement of
the round-2 scripts from the tree, not to this refreeze-preparation PR.

[round2-plan]: m6_remarriage_learning_plan_round2.md
[incident]: ../analysis/m6_remarriage_round2_selection_execution_defect.json
[verification]: https://github.com/PolicyEngine/populace-dynamics/pull/238#issuecomment-5007096806
[pr-234]: https://github.com/PolicyEngine/populace-dynamics/pull/234
[pr-238]: https://github.com/PolicyEngine/populace-dynamics/pull/238
