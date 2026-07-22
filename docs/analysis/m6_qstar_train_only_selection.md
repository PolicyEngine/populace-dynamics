# M6 amendment-4 q-star selection: train-only findings

- **Lane:** amendment 4, candidate-2 conditional-rank refresh
- **Authority:** §2.7.7 of `docs/design/m6_projection_engine.md`, merged as
  `64ec6c04bf8f3e6a6f4fcaf71c086a128056a86f` (#229), with the one-field
  design-commit flip at
  `43fd65eedc225555ac368e24b9510446ca85e1b3` (#233)
- **Freeze status:** `FROZEN_PRE_OUTCOME` at
  `efabdaf2dcba59a4d5ba37312e895846c1da5f59`. The runner, reducer,
  implementation choices, and synthetic proofs were committed and pushed
  before any outcome-producing real-data execution.
- **Result status:** `COMPUTED_TRAIN_ONLY_FINDING`. The exact frozen selector
  selected `q*=0.55`. The accompanying lock addendum remains
  `DRAFT_NOT_OPERATIVE` pending review and ratification.
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

## 7. Execution record

The exact frozen runner at
`efabdaf2dcba59a4d5ba37312e895846c1da5f59` ran from
`2026-07-17T21:35:17Z` through `2026-07-18T18:29:56Z`. It exited zero after
`75,273.01` seconds (`20:54:33.01`) and reported all 63 fresh refits and 1,260
q-by-boundary draws complete. The runner has no checkpoint surface, so this was
one uninterrupted full rerun.

The two-stack runtime was Python 3.14.4, NumPy 2.5.1, pandas 3.0.3,
scikit-learn 1.8.0, SciPy 1.18.0, quantile-forest 1.4.2, populace-fit 0.1.0,
and populace-frame 0.1.0. The QRF backend uses joblib's threaded sklearn forest
fit plus a separate prediction thread pool. The rerun therefore set, by
environment only, `LOKY_MAX_CPU_COUNT=8`, `POPULACE_FIT_N_JOBS=8`,
`POPULACE_FIT_PREDICT_WORKERS=8`, and every registered OMP/BLAS thread cap to
8. No frozen selector byte changed. `/usr/bin/time -l` recorded a maximum RSS
of 15,911,501,824 bytes (14.82 GiB); the 15-second process-tree logger sampled a
peak of 12,438,144 KiB (11.86 GiB). Boundary progress and memory pressure were
monitored throughout.

The exact raw stdout is bound by SHA-256
`24131632e717293551cb539d4da73c772d090ae91adc256c4a029add88103401`.
The frozen reducer reproduced byte-for-byte and published the aggregate
[findings ledger](m6_qstar_train_only_selection_results.json), whose file
SHA-256 is
`d25b8e159384f8a84ed7f2218d863ca63d96fc9cb244536853b0a1f05c4025bb`.
The raw stdout remains outside the repository; the committed ledger retains its
hash and every selection-relevant aggregate and checksum. Neither `gates.yaml`
nor any path below `runs/` was read or written.

## 8. Information-boundary and support audit

The source audit admitted 348,758 sanitized earnings rows and 68,164 anchor-
demographic rows. The maximum earnings reference year requested or admitted was
2014. The sole collection wave above 2014 was wave 2015, whose verified labor-
income fields measure reference year 2014; its age, sequence, relationship,
weight, and interview fields are the incumbent family-panel covariates for that
2014 row. No 2015 earnings-reference field, later family file, candidate/gate
artifact, gate configuration, or realized post-boundary NAWI value was read.

Each q-by-boundary fit stopped its unbuffered NAWI read at that exact boundary.
The full anchor was split before domain intersection, and truth and projection
used identical person-period support.

| Boundary | Fit rows / max year | NAWI max | Full anchor | Domain | Truth support | Endpoint support | Smallest exact-bin pool |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 2006 | 294,603 / 2006 | 2006 | 22,102 | 12,837 | 29,822 | 19,365 | 3,201 |
| 2008 | 307,958 / 2008 | 2008 | 22,928 | 13,355 | 30,636 | 19,730 | 3,438 |
| 2010 | 321,500 / 2010 | 2010 | 23,134 | 13,542 | 30,541 | 19,475 | 3,594 |

All six truth cells were defined at all 100 floor seeds. The frozen truth
moments, observation counts, and `realized_sigma` standardizers were:

| Boundary | Cell | Truth | n | `realized_sigma` |
|---:|:---|---:|---:|---:|
| 2006 | `earn_p10.prime` | 10000.000000 | 8,471 | 0.04650030 |
| 2006 | `earn_dlog_mean.prime` | 0.057575 | 15,677 | 0.01826126 |
| 2006 | `earn_dlog_sd.older` | 0.670236 | 14,145 | 0.06315927 |
| 2006 | `earn_mob_h1_diag` | 0.587778 | 36 | 0.02054701 |
| 2006 | `earn_autocorr_lag2` | 0.647125 | 11,176 | 0.02650091 |
| 2006 | `earn_zero_rate.older` | 0.225891 | 9,447 | 0.05462305 |
| 2008 | `earn_p10.prime` | 10000.000000 | 8,759 | 0.06095119 |
| 2008 | `earn_dlog_mean.prime` | 0.063905 | 16,261 | 0.01849930 |
| 2008 | `earn_dlog_sd.older` | 0.687934 | 14,375 | 0.06624194 |
| 2008 | `earn_mob_h1_diag` | 0.585089 | 36 | 0.02041138 |
| 2008 | `earn_autocorr_lag2` | 0.663794 | 11,650 | 0.02869587 |
| 2008 | `earn_zero_rate.older` | 0.248683 | 9,441 | 0.05344842 |
| 2010 | `earn_p10.prime` | 10000.000000 | 9,007 | 0.07017631 |
| 2010 | `earn_dlog_mean.prime` | 0.133561 | 16,512 | 0.02013713 |
| 2010 | `earn_dlog_sd.older` | 0.672680 | 14,029 | 0.07208434 |
| 2010 | `earn_mob_h1_diag` | 0.609804 | 36 | 0.01769431 |
| 2010 | `earn_autocorr_lag2` | 0.655512 | 11,776 | 0.03487897 |
| 2010 | `earn_zero_rate.older` | 0.245863 | 9,055 | 0.06745151 |

The q-invariant fit signatures were
`f4a1b6400899e89e00ac354629c102920ccf0d5ac53eaa808f26fb60a14eb74a`
(2006),
`73dffc731bd8a75b62b6b60f96f68f9936c97eb86540adbf8d21b4c804313edd`
(2008), and
`287a4a29e8ed506f0ba1a7546495b9bfe1971a8e889ec265f935541030bf2dd3`
(2010). Each signature was identical across all 21 independently refitted
rungs.

## 9. Full registered rung table

`J06`, `J08`, and `J10` are the four-cell all-20 boundary objectives. `J1` and
`J2` are the two fixed ten-seed totals. The delete-one range and per-rung
jackknife SE are derived from the 20 exact replicates retained in the ledger by
the registered formula; only the SE at the all-draw minimum enters the decision
rule. `V/F/I` means valid, feasible, and strict improvement versus q=0 in the
all-20 and both fixed-half objectives. The q=0 row is the explicitly retained
baseline, so its improvement field is `base`.

| q | J06 | J08 | J10 | J(all) | J1 | J2 | Delete-one J range | JK SE | V/F/I | Retained | 1-SE | Selected |
|---:|---:|---:|---:|---:|---:|---:|:---|---:|:---:|:---:|:---:|:---:|
| 0.00 | 164.642382 | 144.745076 | 103.267224 | 412.654683 | 415.464511 | 410.119185 | 411.330265..414.452279 | 3.505390 | Y/Y/base | Y | N | N |
| 0.05 | 145.107662 | 127.420367 | 85.489833 | 358.017862 | 360.518328 | 355.770890 | 356.913737..359.272042 | 2.800868 | Y/Y/Y | Y | N | N |
| 0.10 | 128.436625 | 109.215474 | 69.555369 | 307.207467 | 308.756241 | 305.874977 | 305.909231..308.356755 | 2.861975 | Y/Y/Y | Y | N | N |
| 0.15 | 115.021806 | 94.213371 | 55.002540 | 264.237717 | 266.620277 | 262.122329 | 262.903188..265.397493 | 2.667094 | Y/Y/Y | Y | N | N |
| 0.20 | 101.254212 | 80.872778 | 42.875720 | 225.002710 | 228.749378 | 221.567817 | 223.498262..225.898701 | 2.797211 | Y/Y/Y | Y | N | N |
| 0.25 | 88.814470 | 67.483258 | 32.481767 | 188.779495 | 191.196224 | 186.591591 | 187.433460..189.658992 | 2.443025 | Y/Y/Y | Y | N | N |
| 0.30 | 76.725458 | 56.192838 | 23.693591 | 156.611886 | 159.611529 | 153.832354 | 155.272478..157.405461 | 2.247367 | Y/Y/Y | Y | N | N |
| 0.35 | 66.321596 | 46.788114 | 18.329742 | 131.439452 | 134.446443 | 128.713234 | 130.287499..132.154874 | 2.145356 | Y/Y/Y | Y | N | N |
| 0.40 | 58.310177 | 38.519334 | 14.569480 | 111.398991 | 113.361005 | 109.741552 | 110.595827..112.097516 | 1.835874 | Y/Y/Y | Y | N | N |
| 0.45 | 51.406656 | 32.088327 | 13.303497 | 96.798481 | 98.551086 | 95.418333 | 96.276505..97.733743 | 1.801626 | Y/Y/Y | Y | N | N |
| 0.50 | 44.480334 | 27.123275 | 14.397438 | 86.001047 | 86.325083 | 86.011305 | 85.313921..86.906311 | 1.818279 | Y/Y/Y | Y | N | N |
| **0.55** | **39.840731** | **24.492445** | **17.198765** | **81.531941** | **80.889604** | **82.495118** | **80.997919..82.463232** | **1.899712** | **Y/Y/Y** | **Y** | **Y** | **Y** |
| 0.60 | 37.005697 | 23.070620 | 22.229004 | 82.305321 | 82.135768 | 82.685181 | 81.629453..83.198662 | 2.039153 | Y/Y/Y | Y | Y | N |
| 0.65 | 34.299815 | 22.338076 | 30.397801 | 87.035691 | 86.183755 | 88.130698 | 86.323482..88.265097 | 2.431767 | Y/Y/Y | Y | N | N |
| 0.70 | 33.443260 | 23.207958 | 40.575919 | 97.227137 | 95.839202 | 98.821788 | 96.297169..98.208810 | 2.574111 | Y/Y/Y | Y | N | N |
| 0.75 | 34.296276 | 26.319299 | 52.625182 | 113.240757 | 111.560808 | 115.102622 | 112.224467..114.373057 | 2.696433 | Y/Y/Y | Y | N | N |
| 0.80 | 35.644509 | 31.079490 | 67.159116 | 133.883116 | 133.292323 | 134.647733 | 132.708010..135.020852 | 2.871053 | Y/Y/Y | Y | N | N |
| 0.85 | 37.294253 | 37.134110 | 83.999095 | 158.427458 | 157.897409 | 159.165196 | 156.868891..159.627776 | 3.476501 | Y/Y/Y | Y | N | N |
| 0.90 | 40.651878 | 43.830095 | 101.398024 | 185.879997 | 183.910464 | 188.036459 | 183.859081..186.970750 | 4.068756 | Y/Y/Y | Y | N | N |
| 0.95 | 44.907876 | 52.021925 | 122.584001 | 219.513802 | 216.425049 | 222.821896 | 217.056409..221.193918 | 4.133465 | Y/Y/Y | Y | N | N |
| 1.00 | 50.674581 | 61.085109 | 147.092892 | 258.852582 | 256.182938 | 261.664394 | 255.726998..261.063638 | 4.718208 | Y/Y/Y | Y | N | N |

All 21 rungs were valid and feasible. All 20 nonzero rungs strictly improved
the all-20 objective and both fixed-half objectives relative to q=0, so the
effective retained search set also contained all 21 rungs. The all-draw minimum
was `q_min=0.55` with `J=81.53194122143351`. Its delete-one mean was
`81.53876566170906`, its registered jackknife SE was
`1.899712017216863`, and the one-SE cutoff was `83.43165323865037`.
Exactly q=0.55 and q=0.60 were within that cutoff. The registered smallest-q
rule therefore selected **`q*=0.55`**. Weakening only the all-draw strict
comparison to `<=` retained the same selected value.

The selected rung passed both feasibility guards at every boundary:

| Boundary | `earn_dlog_sd.older` candidate / q0 / limit | `earn_zero_rate.older` candidate / q0 / limit |
|---:|:---|:---|
| 2006 | 0.780278 / 4.502531 / 5.502531 | 1.628272 / 1.693476 / 2.693476 |
| 2008 | 0.155498 / 4.931338 / 5.931338 | 0.737819 / 0.679914 / 1.679914 |
| 2010 | 0.644933 / 4.285401 / 5.285401 | 1.146844 / 1.176955 / 2.176955 |

## 10. Equivalence and publication proofs

- The q=0 wrapper reproduced the incumbent generator bit for bit for all 60
  boundary-by-draw projections: person-period keys, annual level bytes,
  participation states, all six moments, and final stream-1/2/3 states matched.
  The proof covered 3,178,720 incumbent person-period calls and 1,589,360
  refresh-period records.
- Every selected cell was defined and regenerated over all 20 draws at every
  q-by-boundary cell. Each q-by-boundary cell had 20 distinct annual level and
  participation surfaces, fresh initial state, and exact truth/projection
  support-hash equality.
- Every one of the eight exact target-age-bin stable pools was nonempty at every
  fit. All fit-row, donor-pool, marginal, anchor, QRF-gate-state, canonical gate-
  probability, support, NAWI, and RNG checksums are retained in the ledger.
- The raw and reduced-ledger hashes were independently recomputed. Re-running
  the frozen reducer produced the committed findings ledger byte for byte. An
  independent recomputation of all objectives, 420 delete-one totals, per-rung
  jackknife SEs, feasibility and retention decisions, q_min, cutoff, and
  smallest-q rule reproduced `q*=0.55` exactly.

## 11. Interpretation and disposition

The objective curve falls from `J(0)=412.654683` to its minimum at q=0.55 and
then rises to `J(1)=258.852582`. The boundary-specific minima occur at q=0.70,
q=0.65, and q=0.45 for 2006, 2008, and 2010 respectively; the equal-boundary
total balances at q=0.55. At q=0 the total is dominated by mobility and lag-2
autocorrelation contributions (`239.034873` and `164.517363`). At q=0.55 those
fall to `11.187794` and `52.635431`; full refresh further lowers autocorrelation
to `20.982877` but drives mobility back to `205.039822`, producing the aggregate
U shape. q=0.60 is statistically within the registered one-SE band, but the
minimal-departure rule selects the smaller q=0.55. No rung, seed, guard, or
implementation choice was changed after observing the result.

This is train-only, non-scoring evidence. It reads no 2015-2019 earnings signal,
candidate score, gate tolerance, or run artifact and makes no gate-pass claim.
Because the selected value is nonzero, §2.7.7.6's q=0 designed pause is not
triggered. The result supports the accompanying proposed lock addendum, but that
document remains `DRAFT_NOT_OPERATIVE` until reviewed and ratified. This lane
does not implement or register candidate 2, score a holdout, edit `gates.yaml`,
or write a run artifact. The design-commit flip in #233 is an already-landed
antecedent; this result PR neither performs nor reverses it.

## 12. References

- Amendment-4 authority: `docs/design/m6_projection_engine.md` §2.7.7,
  [PR #229](https://github.com/PolicyEngine/populace-dynamics/pull/229), squash
  merge `64ec6c04bf8f3e6a6f4fcaf71c086a128056a86f`.
- Already-landed amendment design pin: [PR #233](https://github.com/PolicyEngine/populace-dynamics/pull/233),
  squash merge `43fd65eedc225555ac368e24b9510446ca85e1b3`.
- Freeze/ledger/reducer precedent: [PR #231](https://github.com/PolicyEngine/populace-dynamics/pull/231),
  squash merge `203017767b4b45e8375d66697195748acfa5c92b`.
- Frozen selector commit:
  `efabdaf2dcba59a4d5ba37312e895846c1da5f59`.
- Aggregate result ledger:
  [`m6_qstar_train_only_selection_results.json`](m6_qstar_train_only_selection_results.json).
