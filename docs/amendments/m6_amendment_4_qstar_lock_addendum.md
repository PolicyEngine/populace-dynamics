# M6 amendment 4 q-star lock addendum

- **Addendum id:** `2026-07-18-m6-amendment-4-qstar-lock`
- **Parent authority:** §2.7.7 of `docs/design/m6_projection_engine.md`,
  ratified in [PR #229][amendment] at
  `64ec6c04bf8f3e6a6f4fcaf71c086a128056a86f`
- **Ceremony stage:** `DRAFT_NOT_OPERATIVE`. This document publishes the
  train-only finding and proposes the lock for review. It is not ratified law,
  a production implementation, registration 8, a score, or run authorization.
- **Frozen selector commit:**
  `efabdaf2dcba59a4d5ba37312e895846c1da5f59`
- **Proposed selected value:** `q*=0.55`
- **Exact raw-ledger SHA-256:**
  `24131632e717293551cb539d4da73c772d090ae91adc256c4a029add88103401`
- **Committed findings ledger:**
  [`m6_qstar_train_only_selection_results.json`][ledger], file SHA-256
  `d25b8e159384f8a84ed7f2218d863ca63d96fc9cb244536853b0a1f05c4025bb`
- **Design-pin antecedent:** [PR #233][design-pin] at
  `43fd65eedc225555ac368e24b9510446ca85e1b3` already landed the amendment-4
  `design_commit` pin. This addendum neither performs nor reverses a gate edit.
- **Freeze/ledger/reducer precedent:** [PR #231][precedent], squash merge
  `203017767b4b45e8375d66697195748acfa5c92b`

## 1. Proposed lock

Subject to referee review and orchestrator ratification, this addendum freezes
the following as one indivisible amendment-4 law:

1. the global refresh probability is **`q*=0.55`**;
2. the complete selection evidence is the raw strict-JSON stdout whose SHA-256
   is
   `24131632e717293551cb539d4da73c772d090ae91adc256c4a029add88103401`;
3. the stable-coordinate pool and distance law is exactly §2 below; and
4. earnings substream codes 4 and 5 and their unconditional eligible-continuer
   consumption law are exactly §3 below.

There was no post-outcome tuning. The selected value is the direct output of
the 21-rung registered selector frozen at `efabdaf`.

## 2. Exact stable-coordinate pool law

For each boundary fit, `u_stable` is a weight-`weight_tp2` single-record draw
from the incumbent positive-to-positive forward-pair donors whose target age is
in the target person's exact five-year age bin. Donor rows retain the stable
`(person_id, period_tp2)` order. No adjacent-bin, pooled-age, re-entry-pool, or
other fallback is permitted; an empty exact-bin pool is non-registerable.

Within that pool, take the `min(25, n_pool)` nearest records under exactly

```text
|u_A(d) - u_A(i)|
```

for Q0 targets and

```text
|0.1 u_w(d) + 0.9 u_A(d) - u_A(i)|
```

otherwise. Target `u_w(i)`, current rank, and prior rank are omitted. The
incumbent weighted no-jitter one-record helper selects from those neighbors.
The unchanged `CellMarginal`, `I_proj`, inverse-rank map, participation law,
zero-to-positive re-entry law, incumbent conditional donor `u_cond`, branch
order, distances, weights, `lambda=0.1`, and `k=25` remain byte-carried.

At the selected value, every eligible positive continuer uses

```text
B_refresh ~ Bernoulli(0.55)
u_out = u_stable  if B_refresh = 1
        u_cond    otherwise.
```

## 3. RNG registry lock

The incumbent earnings registry remains `{1: gate, 2: donor-draw,
3: re-entry-draw}` and appends only

```text
4: memory-refresh-gate
5: memory-refresh-rank
```

For every eligible positive continuer in canonical person order, code 4 draws
one refresh uniform and code 5 draws one stable-donor uniform regardless of q
or realized refresh outcome. Codes 1-3, their addresses, states, and draw order
remain unchanged. All Q-grid rungs use common random numbers at each fixed draw
seed. Odd years retain deterministic carry and consume no RNG.

## 4. Frozen selector and decision

The selector used exactly:

```text
Q_GRID = {0.00, 0.05, ..., 0.95, 1.00}
PSEUDO_BOUNDARIES = {2006, 2008, 2010}
FIT_SEED = 5200
SELECTION_DRAW_SEEDS = {6200, ..., 6219}
FIXED_HALVES = {6200, ..., 6209} / {6210, ..., 6219}
FLOOR_SEEDS = {0, ..., 99}
```

All 21 rungs were valid and feasible. Every nonzero rung strictly improved on
q=0 in the all-20 objective and both fixed halves, so all 21 rungs were
retained. The registered decision quantities were:

| Quantity | Result |
|:---|---:|
| `J(0)` | 412.6546829223998 |
| `q_min` | 0.55 |
| `J(q_min)` | 81.53194122143351 |
| q_min delete-one mean | 81.53876566170906 |
| q_min jackknife SE | 1.899712017216863 |
| one-SE cutoff | 83.43165323865037 |
| retained rungs within one SE | 0.55, 0.60 |
| smallest q within one SE | **0.55** |

The strict-versus-weak all-draw improvement counterfactual also selected 0.55.
The complete boundary, half, and delete-one rung table is in the
[findings report][report], and every exact replicate remains in the ledger.

## 5. Evidence and equivalence record

The completed execution supplies the required proofs:

- all 63 q-by-boundary fits were fresh complete QRF refits with fit seed 5200;
  within each boundary, all 21 fit, donor, gate-state, and canonical gate-
  probability surfaces had the same q-invariant signature;
- all eight exact target-age-bin pools were nonempty in every fit; the smallest
  pool held 3,201, 3,438, and 3,594 donors at boundaries 2006, 2008, and 2010;
- q=0 reproduced the incumbent generator bit for bit across all three
  boundaries and all 20 draw seeds: levels, participation, person-period keys,
  all six moments, and final stream-1/2/3 states matched;
- every selected cell was defined and regenerated across all 20 draws, every
  projection began from fresh state, and every projection support checksum
  equaled its boundary truth-support checksum;
- the frozen reducer reproduced byte for byte, and an independent recomputation
  reproduced all objectives, delete-one jackknives, guards, retained sets,
  one-SE cutoff, and selected value; and
- the two-stack runtime matched every frozen package pin. The run exited zero
  after 63 refits and 1,260 q-by-boundary draws.

## 6. Information bright line

Every selection-relevant earnings reference row was dated no later than 2014.
The retrospective product's collection wave 2015 was used only for its verified
reference-year-2014 labor fields and the incumbent same-row demographic
covariates; no 2015 earnings-reference field or later family file was requested.
Each boundary fit admitted earnings only through b and read the certified NAWI
file one unbuffered byte prefix at a time through exactly b, so realized
`NAWI_(b+2)` and `NAWI_(b+4)` never materialized.

No candidate-1 score, candidate-2 score, gate tolerance, holdout result,
candidate artifact, gate configuration, or existing run artifact entered the
selection. No path under `runs/` was written. The exact source maxima, read
concepts, support checksums, and NAWI prefix hashes are published in the ledger.

## 7. Accepted §2.7.7 notes

The following governance notes remain explicit:

1. The locked gate narrative described in §2.7.7.7 remains intentionally stale;
   no narrative byte is authorized for edit by this addendum. [PR #233][design-pin]
   is an already-landed antecedent, not an authorization to make another gate
   change here.
2. Strict versus weak "improve on q=0" cannot change the selected outcome: if
   `J(0) <= J(q)` for a nonzero rung and that rung lies within the one-SE cutoff,
   retained q=0 does too, and the smallest-q rule returns q=0. The published
   counterfactual exercises the registered comparison and selects 0.55 under
   both readings.

## 8. Operative boundary and next action

This addendum is **`DRAFT_NOT_OPERATIVE`**. Its publication does not itself
freeze production bytes, implement the refresh law, register candidate 2,
authorize a score, or certify any projection. The q*=0 designed-pause clause is
not triggered because the registered selector returned the nonzero value 0.55;
that does not waive the remaining ceremony.

The proposed q* pin becomes operative only through referee review and the
orchestrator's ratification action. Although [PR #233][design-pin] already
landed amendment 4's design-commit antecedent, it did not select or ratify q*.
This PR makes no `gates.yaml` edit and neither repeats nor reverses that flip.
No production implementation, registration 8, or scored run may precede the
addendum's ratification and the remaining §2.7.7.7 proofs.

[amendment]: https://github.com/PolicyEngine/populace-dynamics/pull/229
[design-pin]: https://github.com/PolicyEngine/populace-dynamics/pull/233
[precedent]: https://github.com/PolicyEngine/populace-dynamics/pull/231
[ledger]: ../analysis/m6_qstar_train_only_selection_results.json
[report]: ../analysis/m6_qstar_train_only_selection.md
