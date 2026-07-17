# M6 amendment-4 q-star selection: frozen train-only protocol

- **Lane:** amendment 4, candidate-2 conditional-rank refresh
- **Authority:** §2.7.7 of `docs/design/m6_projection_engine.md`, merged as
  `64ec6c04bf8f3e6a6f4fcaf71c086a128056a86f` (#229), with the one-field
  design-commit flip at
  `43fd65eedc225555ac368e24b9510446ca85e1b3` (#233)
- **Status at this commit:** `FROZEN_PRE_OUTCOME`. The runner, reducer,
  implementation choices, and synthetic proofs are frozen before any
  outcome-producing real-data execution. This section contains no result.
- **Permitted evidence:** family-earnings rows whose verified income-reference
  period is no later than 2014 under the incumbent panel's field-dating
  convention; anchor-roster demographics at the exact pseudo-boundary
  interviews; certified NAWI values no later than each fit boundary; synthetic
  computation
- **Prohibited evidence:** every post-2014 earnings reference row, realized
  post-2014 macro value, candidate-1 seed score, candidate-2 score, gate
  tolerance comparison, and existing run artifact

## 1. Frozen question and selector

The registered question is whether the generic stable-coordinate refresh in
§2.7.7 has a train-only supported refresh probability. The selector is exactly:

```text
Q_GRID = {0.00, 0.05, ..., 0.95, 1.00}
PSEUDO_BOUNDARIES = {2006, 2008, 2010}
FIT_SEED = 5200
SELECTION_DRAW_SEEDS = {6200, ..., 6219}
FIXED_HALVES = {6200, ..., 6209} / {6210, ..., 6219}
FLOOR_SEEDS = {0, ..., 99}
```

At every `q × boundary` cell, the runner performs a fresh complete QRF refit
on earnings reference periods `<=b`, with the same fit seed and a NAWI mapping
containing no key later than `b`. It does not cache the q-invariant QRF fit.
The 63 independently fitted row, marginal, state, donor, exact QRF-gate-state,
and canonical gate-probability surfaces must have the same checksum within a
boundary or the run aborts.

For each `q`, the projected moment is averaged over the selected draw block
before applying the cell's exact log-ratio or absolute-gap score. The primary
objective sums the squared `score / realized_sigma` values over all three
boundaries and these four cells:

```text
earn_p10.prime
earn_dlog_mean.prime
earn_mob_h1_diag
earn_autocorr_lag2
```

The two feasibility guards are `earn_dlog_sd.older` and
`earn_zero_rate.older`, boundary by boundary, each limited to its q=0
standardized score plus one. A nonzero rung must be valid, feasible, and
strictly improve the all-20 objective and both fixed-half objectives. The
runner retains q=0 explicitly, finds the all-draw minimum with ties toward
smaller q, computes all 20 delete-one-draw objective replicates, applies the
registered jackknife standard error, and selects the smallest retained q
within one standard error.

For the accepted strict-versus-weak note in §2.7.7.7, the runner also derives
a counterfactual that weakens only the all-20 comparison from `<` to `<=`;
both fixed-half comparisons remain strict. It reruns the retained-set minimum,
jackknife, one-SE cutoff, and smallest-q choice and aborts unless the selected
value is unchanged. The proof suite includes an exact all-20 tie whose two
halves improve, so this is an exercised calculation rather than a constant.

## 2. Bright line and source dating

The runner is `scripts/select_m6_qstar_train_only.py`. It does not import the
M6 gate runner or load or instantiate a gate contract; it imports only the
pure support-restriction helper from the scoring module. It does not read
`gates.yaml`, gate/run content, or a candidate artifact, and has no output path
under `runs/`. Strict JSON is the only stdout; progress is flushed to stderr.

The staged PSID product is retrospective. Broad convenience readers are
prohibited because they materialize fields through 2023 before filtering.
Instead, the runner asks `panels.ind_person_period` for only `age`, `sequence`,
`relationship`, `weight`, and `interview` at the explicit collection-wave list
1969 through 2015, and reads only interview ID plus head/spouse labor levels
from each corresponding family file. No accuracy field or later wave is read.
The resulting earnings frame is asserted to have `period <= 2014`.

The field-dating interpretation is explicit rather than hidden. Family labor
income in collection wave 2015 is verified by the source label as measuring
income-reference year 2014. The incumbent family-panel convention also uses
that collection wave's age, sequence, relationship, weight, and interview:
relationship/interview attach the two role-specific labor fields, sequence
defines in-family presence, and age/weight enter the row's cohort, support,
and reduction. Thus the final admitted selection row is dated 2014 by its
income reference while carrying collection-wave covariates, exactly as the
incumbent `family_earnings_panel` does. This is the sole collection wave above
2014; no 2015 earnings-reference field, 2016-2019 field, or family file after
2015 is requested. The raw audit publishes these exact concepts, waves, and
uses.

Only the 2007, 2009, and 2011 positive-weight demographic slices enter the
full-anchor split or person ordinal assignment. The certified SSA YAML is not
loaded generically or through a broader mapping. At each `q × boundary` cell,
an unbuffered prefix reader stops at the newline ending that exact boundary's
NAWI entry, before any byte of the next key. The exact prefix-byte and admitted
mapping hashes are frozen separately for 2006, 2008, and 2010. Thus the shifted
trailing decade `[b-9,b]` never materializes realized `NAWI_{b+2}` or
`NAWI_{b+4}` in its boundary mapping.

Runtime maxima and source, fit-row, QRF-gate, donor-pool, anchor, domain,
support, NAWI, and RNG checksums are emitted in the raw result. Any boundary
escape aborts. The runner also asserts the exact Python/numerical package
versions and the editable `populace-fit` / `populace-frame` repository commit
and source-tree hashes before loading data.

## 3. Frozen implementation choices

The registered prose determines the stochastic law and selector. These
remaining operational choices are frozen here, before outcomes.

### 3.1 Full boundary anchor

The full `b` anchor is every positive-weight demographic person at the exact
collection wave `b+1`, carrying that wave's family interview number and
cross-sectional weight. It is neither the already-restricted earnings domain
nor a future-opener roster. The 100 floor splits and projection RNG ordinals
are assigned on this full roster first; each half/projection is intersected
with the exact-b fitted earnings domain afterward. The exact `b+1` weight is
held fixed across the `b → b+2 → b+4` truth and projection window.

This makes the registered full-anchor-before-domain order operative while
excluding any later pseudo-window presence signal. The earnings domain itself
is the intersection of the generator's exact-b realized earnings and `u_w`
maps with that full anchor. Equality with the fitted exact-b anchor IDs is a
runtime assertion.

### 3.2 Projection addresses and schema-only sex

Each draw seed `s` uses the existing M6 address tree with
`draw_index = s - 5200`, `n_periods = 4`, annual period indices `1...4`, the
earnings module slot, and canonical ordinals over the full boundary anchor.
The existing `apply_earnings` adapter invokes the generator once per in-domain
person. Fresh boundary state is materialized for every q/draw projection.

`sex` is carried as the fixed nonmissing value `selection_schema_only`. The
incumbent generator validates only that this frame column is nonmissing; sex
does not enter either QRF feature list, a marginal, a donor distance, a domain
predicate, or an earnings cell. A fixed sentinel is therefore byte-neutral and
avoids importing the unrelated retrospective death-year source merely to
satisfy a schema check.

### 3.3 Stable-coordinate wrapper and substreams

The prototype is script-local; production `forward_earnings.py` is unchanged.
The wrapper first advances the supplied person-period parent RNG through the
unchanged one-integer bridge. It replays that exact integer into the complete
incumbent `generate` call, so participation, re-entry, triple/pair order,
`u_cond`, and streams 1-3 remain incumbent code. It then constructs isolated
children:

```text
4: memory-refresh-gate
5: memory-refresh-rank
```

For every positive continuer selected by the unchanged participation gate,
the wrapper draws one code-4 uniform and one code-5 uniform in canonical
person order regardless of q or refresh outcome. The code-5 uniform selects
from the target's exact five-year age-bin view of the incumbent
positive-to-positive pair pool. Those views reuse the incumbent
`(person_id, period_tp2)` order, `u_A`, `u_w`, `u_tp2`, `weight_tp2`, `k=25`,
and `_knn_draw` helper. The distance is only the registered asymmetric stable
coordinate versus the target anchor rank; target `u_w`, current rank, and prior
rank are absent. Every one of the eight pool partitions must be nonempty.

At q=0 the stable draws still occur, but the wrapper returns the exact ndarray
from the incumbent call without a copy or reconstruction. The preflight runs
the incumbent and wrapped q=0 separately at all three boundaries and all 20
draws, comparing every annual person-period level byte, participation state,
all six reduced moments, parent state, and final stream-1/2/3 state trace.

## 4. Truth support, floor, and regeneration

The realized scoring support is the positive-weight family-earnings row support
at `b+2` and `b+4`, ages 25-64, intersected with the exact-b domain. The exact-b
rows required by the change reducers are carried on the same domain. Projection
is merged onto those realized keys and passed through the existing symmetric
domain-support restriction. A key mismatch aborts.

The existing parameterized `earnings_cells` reducer is called with
`level_years=(b+2,b+4)` and `change_years=(b,b+2,b+4)`. Exactly the registered
six cells are retained. `run_floor` receives the full boundary anchor; its
`compute` callback intersects the returned half with the domain. Every selected
floor cell must be defined at all 100 seeds and have a finite, positive
`realized_sigma`.

Every q/boundary/cell must also vary over the 20 projected draws, matching the
existing M6 regenerated-surface convention. The raw result publishes each
per-draw moment and annual level/participation checksum; the public reducer
retains the moment range, distinct-surface counts, and record checksum.

## 5. Raw result and public reducer

The frozen reducer is `scripts/reduce_m6_qstar_selection.py`. It reads the
exact stdout bytes, asserts schema `m6_qstar_train_only_selection.v1`, removes
only the repetitive 21 × 3 × 20 per-draw records and 3 × 20 detailed q=0
preflight records, derives their retained ranges/counts/checksums, renames the
schema to `m6_qstar_train_only_selection.findings.v1`, injects the SHA-256 of
the exact raw bytes, and emits indented, sorted, strict JSON.

The committed ledger will retain every q/boundary fit and pool checksum, truth
moment, 100-seed `realized_sigma`, support and RNG checksum, all-20 and fixed-
half simulated moment/score, boundary objective contribution, all 20
delete-one objectives, feasibility decision, retained status, one-SE cutoff,
effective search size, and selected value.

Canonical verification after the run is:

```sh
python scripts/reduce_m6_qstar_selection.py < full.json > reduced.json
diff -u <(jq -S . reduced.json) \
  <(jq -S . docs/analysis/m6_qstar_train_only_selection_results.json)
```

## 6. Pre-outcome validation

The synthetic suite `tests/test_m6_qstar_selection.py` pins:

- the 21 rungs, three boundaries, fit/draw/floor seeds, halves, six cells, and
  substream registry;
- q=0 output bytes, parent state, and stream-1/2/3 final states;
- odd-year zero-RNG behavior, unchanged frame lag shifts, `CellMarginal`,
  projected-wage-index, and inverse-rank maps;
- q=1 replacement of positive continuers only, with re-entry unchanged;
- common code-4/code-5 draws across q thresholds;
- the intended later-participation feedback after a refreshed carried level
  crosses the unchanged gate;
- exact target-age-bin partitioning, asymmetric Q0/non-Q0 stable coordinates,
  and omission of target `u_w` and recent ranks;
- full-anchor-before-domain floor splitting and exact QRF gate state/surface
  hashing;
- all-draw, fixed-half, delete-one jackknife, and smallest-q one-SE selection;
  including fixtures rejected by a half or feasibility guard and a fixture
  whose one-SE selection is smaller than its raw minimum; and
- exact raw-byte hashing, truth/projection support-hash equality, q=0 seed
  order, and repetitive-array removal by the reducer.

No real-data result is permitted until this runner, reducer, document, and
tests are Black-formatted, pass the focused synthetic and incumbent forward-
earnings suites, are committed with the required trailer, and that commit is
pushed. The exact commit becomes the `freeze_commit` in the findings and lock
addendum.

## 7. Result posture (not yet populated)

No selected q, rung objective, feasibility result, jackknife value, or outcome
disposition has been computed at this commit.

There is a governance distinction to preserve after the run. Ratified
§2.7.7.6 says a q=0 selection pauses registration 8, while the dispatcher for
this final computational gate explicitly treats q=0 as a valid no-refresh
candidate-2 outcome. The numerical selector is identical under either
disposition. If q=0 is selected, the report and `DRAFT_NOT_OPERATIVE` lock
addendum must state both facts plainly and leave their operative reconciliation
to the referee/orchestrator; this lane will not edit `gates.yaml` or force a
nonzero rung.
