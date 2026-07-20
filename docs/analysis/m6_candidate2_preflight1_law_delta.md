# M6 candidate-2 pre-flight-1 train-law delta

## Outcome and authority

This is the signed transport record that invocation 4's raw `AssertionError`
failed to carry. It records the 15-cell difference between the injected
candidate-2 family law and candidate-9's historical internal candidate-16 law
on the training surface. It is not a score, an expected-delta band, a tolerance
change, a new selection, or a registration-9 re-execution.

The #264 classification referee proved classification **(ii), STALE PREFLIGHT
SURFACE**, in [comment
5017886441](https://github.com/PolicyEngine/populace-dynamics/issues/264#issuecomment-5017886441),
ratified at squash `cff216a45c58798d87dfeb561a430f7e4b3b2902` (tree
`8563fe60bd9339634d0444ffb63257a7b2aa4b10`). Candidate 2's injected arm
correctly carried its registered first-marriage law while the internal arm
deliberately remained candidate 16. The four failed cells below are that law
delta's real training footprint, not an injection defect.

The machine-readable companion is
[`m6_candidate2_preflight1_law_delta.json`](m6_candidate2_preflight1_law_delta.json).

## Frozen bindings

| Role | Candidate | SHA-256 |
|---|---|---|
| Injected family | `m6_candidate2_registry_v1` | `734a5b04f347c5d4904bbc6d5ab9a1c2876272d35284eedd2f450518acf1cec5` |
| Historical internal family | `candidate16_registry_v1` | `6d4d2b2beadc87d17404a3deb64a272c2456d7471b3ad6f1cef779d807765aa1` |
| Household composition | `gate2b_candidate9_registry_v1` | `6137d921032c49ccd71c2302418759868689dac731706f1093a21716080800ab` |
| Engine operations, record only | `m6_candidate2_engine_v1` | `8fbfcf4130fd9051aa063061bf7b2d8514773fc6a900c900caab18717ad8e14c` |

The replay used Python 3.14.4, NumPy 2.5.1, pandas 3.0.3, SciPy
1.18.0, and scikit-learn 1.9.0. The complete source-blob map is in the JSON
companion. The SSA parameter-directory binding is named symbolically as
`populace_dynamics.ss.params._PE_US_ENV`.

## Information boundary and protocol

The diagnostic used only the registered train-only input-plan path and data
dated no later than 2014. It built the candidate-2 family fit, the historical
no-spec household fit embedding candidate 16, and the candidate-2 modifier.
Earnings, disability, and mortality fitting were skipped. The full-input
callback was never called. It did not invoke the registered runner, load
post-2014 holdout truth, perform selection, start projection or scoring, read
`gates.yaml` or `runs/`, or read/write a candidate artifact.

The fitted universe contains 431,643 person-waves and 38,252 person IDs. For
`k = 0, …, 19`, the injected arm used the registered candidate-2 family law and
the internal arm used candidate 16 at seed `5200 + k`. The signed delta is
`injected mean − internal mean`. The unchanged runtime rule is

`abs(signed delta) <= 3 * sqrt(var(injected)/20 + var(internal)/20)`,

with sample variances (`ddof=1`) and no tolerance floor.

## Signed 15-cell table

| Channel | Cell | Injected mean | Internal mean | Signed delta | Absolute delta | σ | 3σ | Result |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| cohabitation | `cohabitation_state` | 0.0737912231 | 0.0737912231 | 0 | 0 | 0.0004037634 | 0.0012112903 | PASS |
| cohabitation | `cohabitation_increment` | 0.0345614037 | 0.0350606958 | −0.0004992921 | 0.0004992921 | 0.0002465699 | 0.0007397098 | PASS |
| legal-spouse residual | `legal_core` | 0.5503746723 | 0.5414149927 | +0.0089596796 | 0.0089596796 | 0.0010629480 | 0.0031888439 | **FAIL** |
| legal-spouse residual | `legal_residual_state` | 0.0363711518 | 0.0363711518 | 0 | 0 | 0.0002962168 | 0.0008886505 | PASS |
| legal-spouse residual | `legal_residual_increment` | 0.0142206444 | 0.0147437591 | −0.0005231147 | 0.0005231147 | 0.0001799499 | 0.0005398497 | PASS |
| legal-spouse residual | `final_spouse` | 0.6007324774 | 0.5927749963 | +0.0079574811 | 0.0079574811 | 0.0010606161 | 0.0031818483 | **FAIL** |
| occupancy | `coresident_parent` | 0.1337638829 | 0.1340495264 | −0.0002856435 | 0.0002856435 | 0.0002975750 | 0.0008927250 | PASS |
| occupancy | `multigen` | 0.0419994253 | 0.0422191781 | −0.0002197528 | 0.0002197528 | 0.0003102922 | 0.0009308767 | PASS |
| occupancy | `coresident_child` | 0.3497691716 | 0.3478286876 | +0.0019404840 | 0.0019404840 | 0.0007450030 | 0.0022350090 | PASS |
| occupancy | `coresident_grandchild` | 0.0298373444 | 0.0297725563 | +0.0000647881 | 0.0000647881 | 0.0003275823 | 0.0009827468 | PASS |
| household size | `household_size.1` | 0.1841691503 | 0.1873742165 | −0.0032050662 | 0.0032050662 | 0.0008964337 | 0.0026893011 | **FAIL** |
| household size | `household_size.2` | 0.3576636306 | 0.3579851949 | −0.0003215643 | 0.0003215643 | 0.0008693568 | 0.0026080703 | PASS |
| household size | `household_size.3` | 0.1783878685 | 0.1777478043 | +0.0006400642 | 0.0006400642 | 0.0005365139 | 0.0016095417 | PASS |
| household size | `household_size.4` | 0.1670336601 | 0.1643088866 | +0.0027247735 | 0.0027247735 | 0.0005406423 | 0.0016219270 | **FAIL** |
| household size | `household_size.5+` | 0.1127456904 | 0.1125838977 | +0.0001617928 | 0.0001617928 | 0.0004387976 | 0.0013163929 | PASS |

The four failed absolute deltas and tolerances reproduce invocation 4's
assertion digit for digit.

## Interpretation

The registered law raises `legal_core` by 0.895968 percentage points and
`final_spouse` by 0.795748 points relative to candidate 16. Household-size mass
moves coherently from sizes 1 and 2 into sizes 3, 4, and 5+; the five signed
size deltas sum to zero within binary rounding. The law-insensitive
`cohabitation_state` and `legal_residual_state` controls are exactly zero, and
all four occupancy cells clear the unchanged runtime margin.

This pattern is the cross-law footprint anticipated by the referee, including
candidate 2's boundary-clipped first-marriage behavior where candidate 16
extrapolates. It does not license the repaired check to ignore a genuine wiring
error. Under amendment 5 both arms instead carry the same registered family
law; the separate synthetic fertility-omission fixture still fails, preserving
the check's power.

## Byte contract and disposition

The no-spec candidate-1 route continues to pass the exact embedded candidate-16
fitted-family object and retains the original plain-`AssertionError` failure
surface. Its normal success payload schema is unchanged. The frozen candidate-1
artifact remains unedited at SHA-256
`546a9739f8d1c7d21a91a07eb902c8af9bda92cdaa8f7917f312894f6a861b24`.

This publication is prospective only. Invocation 4 is not regraded, the
one-shot remains unconsumed, and no execution occurs under registration 9.
After referee review and ratification, the coordinator alone re-finalizes
`gate_m6.design_commit`; registration 10 and the next invocation follow.
