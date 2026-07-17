# M6 amendment 1: closed-domain floors v4

**Status: `DRAFT_NOT_OPERATIVE`.** This is a docs-only draft of the narrow,
prospective §2.8.4 amendment required by Resolution B. It does not amend the
live gate contract, and this lane makes no `gates.yaml` edit. Candidate 1 remains
a historical 0/5-seed FAIL under the immutable v3 contract. The separate §2.7
stable-coordinate design pin and its `gate_m6.design_commit` re-finalization are
out of scope here.

## Authority and reproducibility lock

This draft implements the closed-domain floors resolution in the candidate-2
program merged at
[`051b449`](https://github.com/PolicyEngine/populace-dynamics/commit/051b4494ecce9345da14d68488bb2833ed476d22)
and accepted as verified-ratifiable in
[issue comment 5001901052](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-5001901052).
That comment verifies and authorizes the **program** that defined this disclosure
duty; it is not a verification of the later v4 artifact. The independent v4
byte reproduction and adversarial ceremony review are recorded separately in
[referee comment 5003284911](https://github.com/PolicyEngine/populace-dynamics/pull/232#issuecomment-5003284911).
The proposed lock binds all of the following:

| Item | Locked value |
|---|---|
| Builder source | `scripts/build_m6_holdout_floors_v4.py` at commit `2d1704ede69aea4cb1caf174d6dc40653e56d63a` |
| Primary artifact | `runs/m6_holdout_floors_v4.json` |
| Primary SHA-256 | `4cd2d01a9fd76064e701ae77a9226208cbae94d743f76f502d3d0a5f657d9523` |
| Environment sidecar | `runs/m6_holdout_floors_v4.json.env.json` |
| Sidecar SHA-256 | `75ae57ceacad7abb0958d322ba52f72a7404e45001cdde463539b96a215b37e8` |
| Derivation-core SHA-256 | `b34ab1943f7793e41641819ff3482187e8f0acf492ea2cce7f0e694f9fd01cc0` |

An independent clean rerun of the pinned builder produced byte-identical primary
and sidecar files and therefore the same two SHA-256 values. The v3 input remains
read-only at `runs/m6_holdout_floors_v3.json`, SHA-256
`e931c88622fad84e8f8b2cf18940cbe27da1c93e0d009dfbaa3d6c6cae050c77`.

## Registered derivation

The numerical derivation is truth-side only. Its inputs are the frozen v3 flow
floor records and protocol, the registered staged PSID truth tables, and the
registered forward-earnings fit using data through 2014. It reads neither a
candidate artifact nor projection output.

Resolution B's domain is the truth anchor person IDs intersected with the fitted
realized-2014 earnings keys and fitted `u_w` keys. Scored support is realized,
positive-weight, valid-earnings support in 2016/2018, intersected symmetrically
with that domain. The artifact records:

- 29,792 persons in the full truth anchor;
- 13,561 domain persons and 45,606 domain earnings person-period rows;
- 10,441 domain gated-earnings persons, versus 13,163 on full gated support; and
- 2,722 unique excluded later entrants. The period-specific cohort tallies are
  2,199 prime plus 590 older, or 2,789 appearances; they exceed 2,722 because a
  person can occupy both cohorts across periods, so they are overlapping counts
  rather than a decomposition of the unique-person total.

The F7 order is binding: for each seed `0..99`, split the **full** anchor 50/50 by
person first, then intersect each half with the domain inside the unchanged
`earnings_cells` reducer. The artifact publishes all 100 split ledgers and their
support counts. All six gated earnings cells are defined for all 100 seeds.

All non-earnings-v4 cell records are byte-carried from v3, including the five
gated flow floors. Only the six gated earnings floors are rederived, using the
unchanged cell definitions, metrics, reducers, seeds, and
`round(mean + 3 * sample_sd, 3)` rule with the registered metric caps:

| Earnings cell | Mean | Sample SD | Realized sigma | Minimum weaker-half support | v3 → v4 tolerance |
|---|---:|---:|---:|---:|---:|
| `earn_autocorr_lag2` | 0.026737 | 0.019927 | 0.033346 | 5,636 | 0.087 → 0.087 |
| `earn_dlog_mean.prime` | 0.014107 | 0.009797 | 0.017175 | 8,314 | 0.043 → 0.043 |
| `earn_dlog_sd.older` | 0.083015 | 0.065268 | 0.105600 | 6,202 | 0.269 → 0.279 |
| `earn_mob_h1_diag` | 0.017712 | 0.011948 | 0.021365 | 36 | 0.052 → 0.054 |
| `earn_p10.prime` | 0.105097 | 0.059535 | 0.120788 | 4,549 | 0.221 → 0.284 |
| `earn_zero_rate.older` | 0.055589 | 0.037601 | 0.067112 | 3,964 | 0.163 → 0.168 |

The autocorrelation cap is `0.15`; the other five metric caps are `ln(1.5)`, or
`0.4054651081081644`. No domain tolerance reaches a cap. No gated cell falls below
the registered minimum weaker-half support of 20, including the mobility cell at
36. The fresh two-directional checks find no near-unpassable, near-unfailable, or
near-tautological cell; no vacuity flag fires; the unchanged flow surface clears;
and the ceremony may proceed.

## Operating characteristics

The registered independence approximation computes each cell's pass probability
from its registered tolerance and unrounded realized sigma, multiplies the exact
cell probabilities for `p_seed`, and applies the unchanged at-least-4-of-5 seed
rule for `p_gate`.

| Surface | Cells | `p_seed` | `p_gate` |
|---|---:|---:|---:|
| Ratified v3 tolerances on v3 full support | 11 | 0.8934 | 0.9087 |
| v3 tolerances on the closed-domain earnings cells | 6 | 0.8760 | 0.8809 |
| v3 tolerances on the closed-domain combined surface | 11 | 0.8373 | 0.8114 |
| v4 tolerances on the closed-domain earnings cells | 6 | 0.9300 | 0.9575 |
| v4 tolerances on the closed-domain combined surface | 11 | 0.8889 | 0.9018 |

The program's provisional combined values, approximately `0.8115` and `0.9019`,
multiplied rounded subfamily `p_seed` values. The artifact's `oc_comparison`
correctly records that rounded-subfamily calculation as `operative: false`. The
fresh ceremony instead combines the exact individual-cell probabilities before
rounding, yielding `0.8114` and `0.9018`; this independently reproduced
one-basis-point reconciliation does not change the registered OC method.

### Required lock disclosure: thin clearance

At full precision the combined v4 values are
`p_seed=0.8889341210928279` and `p_gate=0.9018301422467547` (unrounded
`p_gate=0.9018301…`). The latter clears `0.90` by
`0.0018301422467547`, with no rounding assist. The earnings-subfamily value is
`0.9574884…`, displayed as `0.9575` under the same convention.

A delete-one-seed jackknife over the 100 floor seeds gives SE(`p_gate`) about
`0.0186` when tolerances are held at their locked values and about `0.0129` when
tolerances are co-derived in each replicate: approximately `0.013–0.019` across
the two conventions. Four of the 100 co-derived delete-one replicates fall below
`0.90`, and the full-sample clearance is only about `0.1 SE`. Co-derivation
partly self-stabilizes the result because an over-estimated sigma also inflates
its tolerance, which explains the smaller co-derived SE.

Accordingly, the weak-power floor is met under the registered point-estimate
arithmetic convention—the same convention used for the ratified v3 value
`0.9087`—but the clearance is **not statistically resolved at the 100-seed
design**. Ratification knowingly adopts that registered arithmetic convention;
it does not claim that this design statistically resolves the margin.

## S2 candidate-blindness disclosure

The derivation core is independent of candidate-1 scores: the exact-domain trigger
and truth-only procedure were pre-registered in §2.8.3a; the builder records
`candidate_artifact_or_projection_read=false` and
`candidate_score_used_in_floor_tolerance_or_oc=false`. The S2 table below is a
separate governance disclosure and is excluded from the derivation core and its
SHA. Its historical evidence is bound to `runs/gate_m6_candidate1_v1.json`,
SHA-256 `546a9739f8d1c7d21a91a07eb902c8af9bda92cdaa8f7917f312894f6a861b24`,
but that artifact was not read by the numerical derivation.

| Failed earnings cell | Candidate-1 seed scores | v3 → v4 tolerance | Non-operative movement |
|---|---:|---:|---|
| `earn_p10.prime` | seeds 0..4: `[0.251827, 0.308309, 0.271703, 0.327174, 0.356735]` | 0.221 → 0.284 | Seeds 0 and 2 would flip to pass; seeds 1, 3, and 4 remain FAIL. |
| `earn_mob_h1_diag` | range 0.155–0.180 | 0.052 → 0.054 | All five remain FAIL. |
| `earn_autocorr_lag2` | range 0.151–0.185 | 0.087 → 0.087 | All five remain FAIL. |
| `earn_dlog_mean.prime` | range 0.053–0.076 | 0.043 → 0.043 | All five remain FAIL. |

This movement rescues nothing. Every seed still fails the other
persistence/growth cells, the mobility and autocorrelation margins remain wide,
and candidate 1 remains a 0/5-seed FAIL under v3. Applying v4 retrospectively to
candidate 1 is prohibited. The same facts show that the truth-only floor movement
cannot manufacture a persistence pass for candidate 2.

Candidate-2's must-not-regress block also remains unchanged and distinct from the
live v4 gate. In particular, it retains the original `earn_dlog_sd.older=0.269`
and `earn_zero_rate.older=0.163` thresholds even though the proposed live v4
thresholds are `0.279` and `0.168`. Candidate-2 acceptance still requires both
the live 11-cell gate and the original-threshold regression block.

## Proposed narrow §2.8.4 amendment text

> **2.8.4a Resolution-B amendment — prospective closed-domain floor lock.** For M6
> registrations made only after this amendment and its gate-contract lock are
> ratified, replace the v3 floor-artifact reference with
> `runs/m6_holdout_floors_v4.json`, SHA-256
> `4cd2d01a9fd76064e701ae77a9226208cbae94d743f76f502d3d0a5f657d9523`.
> The six gated earnings tolerances are `earn_autocorr_lag2=0.087`,
> `earn_dlog_mean.prime=0.043`, `earn_dlog_sd.older=0.279`,
> `earn_mob_h1_diag=0.054`, `earn_p10.prime=0.284`, and
> `earn_zero_rate.older=0.168`. They are derived on realized scored support
> intersected with the 2014 earnings domain by splitting the full anchor at seeds
> `0..99` before domain intersection. The five gated flow tolerances and all cell
> definitions, metrics, reducers, support rules, floor seeds, metric caps,
> `K=20`, and the at-least-4-of-5 conjunction are unchanged. The v3 artifact and
> every candidate-1 conclusion remain immutable and historical; v4 has no
> retrospective effect. Candidate-2's must-not-regress thresholds remain 0.269
> for `earn_dlog_sd.older` and 0.163 for `earn_zero_rate.older`. The ratifying
> lock may change only the gate path/hash and six earnings-tolerance positions,
> every nested copy of that gate path, the two runtime path/hash constant pairs,
> and the guard-test dispositions enumerated below. No other gate-contract,
> runtime-source, or test byte is authorized by this amendment.

### Complete authorized lock-flip edit surface

The orchestrator executes this edit set only at ratification. The positions and
their lock-time dispositions are exhaustive:

1. **`gates.yaml` — 16 scalar positions, 14 changed values.**
   `gates.gate_m6.floor_run` and `floor_run_sha256` at `gates.yaml:5459-5460`
   re-point to the v4 path and primary SHA above. All eight nested path strings
   also re-point to v4; none remains on v3: each `floor_run` and
   `derivations.floor_run` under `marital_flows` (`gates.yaml:5622,5628`),
   `disability_flows` (`gates.yaml:5638,5643`), `earnings_log_ratio`
   (`gates.yaml:5652,5659`), and `earnings_abs_gap`
   (`gates.yaml:5670,5675`). This includes the flow views even though their
   floor records are byte-carried, because `M6GateContract.from_block` requires
   both nested strings in every view to equal the top-level path
   (`src/populace_dynamics/harness/m6_scoring.py:136-140`). The six tolerance
   positions are `gates.yaml:5654-5657` and `gates.yaml:5672-5673`, with the
   values listed in §2.8.4a. Four values change; `earn_autocorr_lag2=0.087` and
   `earn_dlog_mean.prime=0.043` are verified in place and remain byte-identical.
2. **The two runtime constant pairs.** Set `FROZEN_FLOOR_RUN` and
   `FROZEN_FLOOR_SHA256` to the same v4 path and SHA in
   `src/populace_dynamics/harness/m6_scoring.py:39-42` and
   `src/populace_dynamics/harness/m6_runner.py:108-111`. The scoring contract
   checks its pair at `src/populace_dynamics/harness/m6_scoring.py:113-117`;
   the runner enforces its pair at
   `src/populace_dynamics/harness/m6_runner.py:292-293` and its resolved path at
   `src/populace_dynamics/harness/m6_runner.py:318-319`.
3. **Pre-lock and live-binding guard tests.** Invert
   `tests/test_gate_m6_floors_v4.py::test_live_gate_remains_bound_to_frozen_v3_before_ratification`
   (`tests/test_gate_m6_floors_v4.py:172-183`) to bind the ratified v4 path,
   SHA, and all nested path copies. In the v3 historical suite, retain the
   draft-only v3 byte lock at `tests/test_gate_m6_floors.py:134`, but invert the
   live-gate pin at `tests/test_gate_m6_floors.py:562` and admit exactly the
   authorized v4 live-versus-draft deltas in the equality guard at
   `tests/test_gate_m6_floors.py:575-582`. Update the hard-coded resolved path in
   `tests/test_m6_runner.py::test_frozen_floor_is_byte_verified`
   (`tests/test_m6_runner.py:242-245`). Rebind the live path, SHA, six
   tolerances, and corresponding mutation guards in
   `tests/test_gate_m6_derivations.py::test_gate_m6_locked_added_as_temporal_holdout_top_level_gate`,
   `::test_gate_m6_floor_run_sha_binds`,
   `::test_gate_m6_tolerances_recompute_capped`,
   `::test_gate_m6_v1_v2_v3_lineage_shas`, and
   `::test_gate_m6_zero_threshold_movement_vs_frozen_floor`
   (`tests/test_gate_m6_derivations.py:418-445,465-466,505-513`) and their path,
   SHA, tolerance, and lineage helpers
   (`tests/test_gate_m6_derivations.py:158-202,317-336`) to v4. Carry those
   dispositions through `::test_mutations_are_caught`
   (`tests/test_gate_m6_derivations.py:523-530,571-581`), while preserving the
   frozen v1/v2/v3 lineage assertions and unchanged flow records.

A `gates.yaml`-only flip with stale source constants cannot false-PASS: it fails
loudly during `resolve_m6_contract`. The current call order rejects the stale
binding at `src/populace_dynamics/harness/m6_scoring.py:113-117`; the runner's
independent resolved-path guard at
`src/populace_dynamics/harness/m6_runner.py:318-319` rejects a stale pairing as
`gate_m6 floor path is not the frozen v3 artifact`. Nothing outside the exact
surface above is authorized.

## Ratification and lock checklist

- [x] New v4 artifact created without overwriting frozen v3.
- [x] Builder source commit, primary SHA, sidecar SHA, and derivation-core SHA
  recorded.
- [x] Independent rerun reproduced the primary and sidecar byte-for-byte.
- [x] Exact Resolution-B domain, F7 split order, all 100 seeds, and support counts
  published.
- [x] Six earnings tolerances, combined 11-cell OC, caps, minimum support,
  two-directional power, and vacuity checks published.
- [x] S2 tolerance-movement table and candidate-blind separation published.
- [ ] Referee ratifies this prospective amendment and the fresh thin-margin OC.
- [ ] At ratification, the orchestrator applies exactly the 16-position
  `gates.yaml`, two-source-pair, and named guard-test edit surface enumerated
  above; until that lock lands, v3 remains operative.
- [ ] Registration 8 occurs only after the prospective floor lock is final.
- [ ] The separate §2.7 amendment lane re-finalizes its design pin independently;
  it is not bundled into this floor amendment.
