# gate_w1 candidate 4 — mapping annotation of record

`runs/gate_w1_candidate4_v1.json` (git blob `d944ed22…`) is the **candidate-4**
one-shot scored run for gate_w1, and the first gate_w1 PASS in the record:

- registration [#42 comment 4964126356][reg] (registry 55/55),
- run [PR #184][pr] (merged `c130517`),
- independent bit-exact verification [#42 comment 4964799327][verify],
- record entry [#42 comment 4964811349][record].

This note is the `runs/` copy of the mapping annotation of record (verifier
finding 1). It does **not** modify the frozen artifact.

## What the run is

- **Model**: `transport_deployment_v3` — byte-identical model and runner to the
  committed candidate 3, with zero code modifications, re-registered on the live
  amendment-3 surface (flip `14658d0`, ratified `cb1ef63`).
- **Surface**: the amendment-3 pair rule — 44 gated cells (43 family-A joints +
  the pair-scoped C2 `elim<->+2pp` compression swap). The runner binds the
  partition from `gates.yaml` (contract blob `1efbf095…`) at run time, so the
  unchanged script scores the amended surface.
- **Verdict**: gate **PASS** — family A in-band on 43/43 cells across 5/5 seeds,
  family C pair-swap realised. The family-A cube is bit-identical to candidate 3's
  committed cube on all 43 shared cells; the four c3-only cells are exactly the
  amendment-3-demoted `hh_size_share.{1,3,4,5plus}`.

## Why the frozen artifact appears to mislabel itself

The byte-copy constraint (candidate 4 *is* the c3 model and runner) leaves three
frozen fields describing the c3 lineage rather than the c4 registration:

- `verdict.candidate` = `"w1_candidate3"` (the model's name),
- `verdict.gate_rule` prose = the pre-amendment "every GATED fingerprint reverses"
  wording, and
- `registration.comment_id` / `verdict.registration_pointer` = `null`.

These are **byte-copy artifacts of the registered c3-model byte-copy**, disclosed
in the registration, PR, and commit. The two operative family-C statistics both
live in `verdict.fingerprints`: the gated C2 fingerprint publishes
`required_swap_realised = true` (the amendment-3 pair rule — the operative gate)
alongside `reversed_to_anchor = false` (the demoted full-ordering rule, published
report-only). Resolve any apparent self-contradiction against this note and
`gates.yaml`; do **not** edit the frozen artifact.

## Record

| Item | Pointer |
|------|---------|
| Registration | #42 comment 4964126356 |
| Run PR | #184 (merged `c130517`) |
| Bit-exact verification | #42 comment 4964799327 |
| Record entry | #42 comment 4964811349 |
| Test pin | `tests/test_gate_w1_candidate4_pin.py` |

[reg]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4964126356
[pr]: https://github.com/PolicyEngine/populace-dynamics/pull/184
[verify]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4964799327
[record]: https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4964811349
