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
- 2,722 excluded later entrants: 2,199 prime and 590 older persons (cohort counts
  can overlap outside the gated-person total's decomposition).

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
multiplied rounded subfamily `p_seed` values. The fresh ceremony instead combines
the exact individual-cell probabilities before rounding, yielding `0.8114` and
`0.9018`; this reconciles the one-basis-point differences without changing the
registered OC method. The v4 combined `p_gate=0.9018` clears the `0.90` weak-power
floor by only 0.0018, so the margin is explicitly thin.

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
> for `earn_dlog_sd.older` and 0.163 for `earn_zero_rate.older`. No other gate
> contract byte is authorized by this amendment.

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
- [ ] A later ratifying floor-lock PR updates only the authorized v4 artifact
  reference/hash and six earnings tolerance fields in `gates.yaml`; until that
  lock lands, v3 remains operative.
- [ ] Registration 8 occurs only after the prospective floor lock is final.
- [ ] The separate §2.7 amendment lane re-finalizes its design pin independently;
  it is not bundled into this floor amendment.
