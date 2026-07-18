# M6 candidate-2 entry-dissolved conformance audit

## Outcome

Candidate-2 program §7 conditions 4 and 5 both **PASS**.

- The merged #226 fixtures cover entry-divorced, entry-widowed, in-window
  remarriage, sustained no-remarriage exposure, and the immediate entry-year
  boundary. Two audit-only synthetic edges make the same-year law explicit.
- A fresh train-only paired reconstruction publishes the complete
  first-marriage-to-remarriage relabel ledger by pseudo-boundary year, sex,
  age band, entry origin, and draw. All 120 paired draws pass the exact
  forensic invariants.
- The selected incumbent L0/R0 law still overshoots on all three train-only
  pseudo-holdouts. That residual is published regardless; it is not hidden by
  the conformance PASS.
- No gate, floor, candidate-1 artifact, scorer, tolerance, or `runs/` byte was
  read by the paired data process or changed by this publication.

The machine-readable artifact is
[`m6_candidate2_entry_dissolved_conformance.json`](m6_candidate2_entry_dissolved_conformance.json),
schema `m6.candidate2.entry_dissolved_conformance.v1`. Its SHA-256 is
`7c7779408e5c762528ae0334286cfdfa1f2f69cc1785b05f24b20217ded56a54`.
The complete 5,280-row atomic ledger has canonical SHA-256
`0dee6fca6520020783f70ced0ff50c36dec62b9a7da5026ccd049e4a54f5012d`.

## Authority and firewall

This audit closes the two remaining conformance requirements in
[`m6_candidate2_program.md`](../design/m6_candidate2_program.md): the fixture
audit at lines 820–824 and the exact-class reconciliation and gated-cell
ledger at lines 825–833. The repair is #226, squash
`c16cb9d563bd573ce2b537b19e403fbddec3cba6`; its independent referee record is
issue comment `4998474459`. Decision 9 was re-adjudicated by #250 at
`b4d3da1a8dc8cb86b616f91e033cfb65fd5dce56`, disposition comment
`5011257300`.

The paired computation used only the already registered train-only chassis:

- pseudo-boundaries 2006, 2008, and 2010;
- evaluation years 2007–2010, 2009–2012, and 2011–2013;
- 40 common-random-number draws, seeds 7200–7239;
- fixed start-wave F6 weights; and
- the incumbent remarriage law, later ratified as R0/no-op.

The staged source is a retrospective product, so the chassis severed every
post-2014 value before fitting or evaluation. The sanitized maxima are 2013
for demographic, marriage, and event fields and 2014 for death and marital
person-year fields. Calendar-2014 flow is excluded because its establishing
interview is in 2015. No post-2014 value entered this computation, and no
candidate output was contacted.

The process did not import the M6 scorer, read `gates.yaml`, read a gate
tolerance or floor artifact, or read or write under `runs/`. The residual
table below was copied mechanically from the already committed train-only
finding, not recomputed from a scored window.

## Condition 4: fixture-by-fixture audit

| Fixture | Exact coverage | Discriminating result |
| --- | --- | --- |
| `test_entry_dissolved_person_year_history_survives_assembly` | Six entry-divorced persons at YSD 4 remain dissolved without remarriage through 2015–2022; a married control divorces in 2015. | Repaired output has 55/55 dissolved rows and PID 101 carries YSD 4…11. Pre-repair output has 7/55 rows, so the fixture fails with the real-frame 12.73% signature. |
| `test_entry_dissolved_remarriages_keep_event_history_through_assembly` | Entry-divorced female at YSD 4 and entry-widowed male at YSD 7; `start_exposure_year = censor_year = 2015`; forced remarriage. This covers both origins, in-window remarriage, and the immediate entry-boundary year. | Repaired output has two 2015 `remarriage` events with exact origins/YSD and inert `n_marriages = [1, 1]`. Pre-repair output labels both `first_marriage`, so the fixture fails. |
| `test_marital_core_appends_certified_change_points_and_durations` | General `allow_exact_matches=False` boundary: an event in year `t` changes entering-year state only in `t+1`. | The fixture passes before and after #226, proving the repair did not move the existing event-year convention. |

The relevant merged tests are at
[`tests/test_m6_panel_builders.py`](../../tests/test_m6_panel_builders.py),
lines 289–421.

Two audit-only synthetic cases remove any ambiguity in “same-year”:

1. The minimum lawful entry-dissolved boundary sets both origins to YSD 1,
   starts and censors in 2015, and forces remarriage. The 2015 person-years
   remain divorced/widowed at YSD 1; the two 2015 events are remarriages with
   exact origins/YSD; lifetime counts stay `[1, 1]`. YSD 0 is not a lawful
   entry-dissolved seed under `allow_exact_matches=False`, because a
   dissolution in year `t` first produces dissolved entering state in `t+1`.
2. The orthogonal true-YSD-0 case gives each person a prior marriage ending in
   2015 by divorce/widowhood and a second marriage starting in 2015. Assembly
   emits the dissolution and remarriage in 2015, labels the remarriage origin
   exactly, records YSD 0, keeps the 2015 person-year in the prior marriage,
   and gives the new marriage duration 1 in 2016. This confirms
   dissolve-before-marry tie order without pretending YSD 0 is an entry seed.

Targeted validation returned `3 passed, 10 deselected`:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q -p no:cacheprovider \
  tests/test_m6_panel_builders.py \
  -k 'marital_core_appends_certified_change_points_and_durations or entry_dissolved'
```

The fixture matrix therefore closes condition 4 without a source or test edit.

## Condition 5: paired pre-score reconciliation

The audit executed the historical pre-repair
`_simulate_candidate16_with_generators` from parent
`4f8e1d9839f5bed6f2291b9e78b79ff30653efe9` and the landed implementation on
the same seed panel with fresh, identical RNG registries. The historical and
current marital-source SHA-256 values are respectively
`3209361f5e0a841e42690962ab64e9942070e4538e13d98aedc720bfd3667670`
and
`82247d03907231f384e11edb939653d5e45b7785987c81c54538fb0b92cdcbf7`.

The ledger publishes a complete row, including zeros, for every combination
of:

- pseudo-boundary and evaluation year;
- draw seed 7200–7239;
- carried entry origin, divorced or widowed;
- sex; and
- working-age first-marriage band 18–29, 30–44, or 45–64.

Each atomic row records relabeled event count and F6 weight, signed transfer
into and out of both transition labels, and pre/post carrier exposure,
numerator, and rate fields. The artifact also aggregates each
boundary/year/origin/sex/age cell over draws with sum, mean, minimum, and
maximum.

### Candidate-2 first-marriage gated shape

The table below isolates female ages 18–29 over each pseudo-window, the exact
sex/age shape of `first_marriage.18-29|female`. Values are means over 40 draws
and include both carried origins.

| Boundary | Evaluation | Relabeled events | Relabeled F6 weight | False never-married carrier rows before → after | False never-married F6 exposure before → after | False first-marriage F6 numerator before → after |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 2006 | 2007–2010 | 3.575 | 26,583.750 | 24.925 → 0 | 185,092.350 → 0 | 26,583.750 → 0 |
| 2008 | 2009–2012 | 3.150 | 29,249.125 | 20.350 → 0 | 190,837.350 → 0 | 29,249.125 → 0 |
| 2010 | 2011–2013 | 1.375 | 14,975.825 | 9.525 → 0 | 80,370.275 → 0 | 14,975.825 → 0 |

Within this exact carrier class, every false first-marriage event and its F6
weight transfers to remarriage. The post-repair carrier-only first-marriage
rate is intentionally undefined because its corrected never-married
denominator is zero; the artifact does not disguise that as rate zero or
misstate it as the full-population gated rate.

Across all working-age sex/age cells, the relabeled count and F6 weight per
draw are:

| Boundary | Relabeled events/draw | Relabeled F6 weight/draw |
| ---: | ---: | ---: |
| 2006 | 56.900 | 669,363.250 |
| 2008 | 43.500 | 487,293.550 |
| 2010 | 28.150 | 305,685.925 |

The gated female-18–29 rows are origin-resolved in the artifact. Divorced
origin supplies 3.350/2.875/1.375 events per draw at the three boundaries;
widowed origin supplies 0.225/0.275/0.000. No zero origin/cell is omitted.

### Exact forensic invariants

Every one of the 120 paired draws establishes all of the following:

- both pre- and post-repair projected support keys exactly equal the same
  train-only truth keys;
- every landed carrier retains its entry state and exact YSD;
- formation-event identity is exact except for the intended label and restored
  prior-history metadata;
- pooled event count and pooled F6 event weight are conserved;
- divorce/widowhood event rows are exact;
- all married person-years, and therefore every divorce denominator row, are
  exact;
- first-marriage and remarriage count transfers sum exactly to zero;
- first-marriage and remarriage F6-weight transfers sum exactly to zero; and
- no event is created or lost.

The three boundaries contain 690, 581, and 419 entry-dissolved carriers. Their
truth-support row counts are 18,358, 17,997, and 13,458, with exact support
checksums published in the artifact.

This conformance transfer is independent of the future candidate-2
first-marriage fit. Entry-dissolved people enter internal states 2/3 and use
the unchanged remarriage law; the first-marriage estimator is invoked only
for state 0. Candidate 2’s earnings refresh is disjoint. The ledger therefore
isolates the landed assembly repair and does not attribute it to the new
first-marriage hazard. It also makes no claim about the future estimator
delta, which remains unobserved before the registered score.

## Post-repair residual transport — published regardless

The following incumbent-L0 measurement is copied from the committed,
train-only
[`m6_remarriage_train_only_delta_results.json`](m6_remarriage_train_only_delta_results.json),
SHA-256
`aabec7cc1254d1be9bcf57518632a04f4bbf42d842d30b1991b58a832206c05f`.
Carrier conformance passed at every boundary.

| Boundary | Evaluation | Direct expected/actual numerator | Projected/truth exposure | Projected/truth numerator | Projected/truth rate | `ln(rate ratio)` |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 2006 | 2007–2010 | 1.526015 | 1.030674 | 1.613264 | 1.567394 | 0.449414 |
| 2008 | 2009–2012 | 1.618915 | 1.030281 | 1.705660 | 1.657659 | 0.505407 |
| 2010 | 2011–2013 | 2.038506 | 1.058255 | 2.295647 | 2.174235 | 0.776677 |

The repair restores the missing risk history; it does not rescue the
incumbent remarriage law. The residual remains a same-direction overshoot on
all train-only boundaries and is published even though the conformance
conditions pass. Under the ratified decision-9 disposition, R0 stays unchanged
and candidate 2 carries the candid modal-failure forecast for
`remarriage.18-64`.

## Frozen-byte reconciliation

Part A changes only this report and its JSON artifact. A diff against
`b91573e` confirms no byte movement in `gates.yaml`, any M6 floor or sidecar,
or the candidate-1 artifact or sidecar.

| Frozen surface | SHA-256 |
| --- | --- |
| `gates.yaml` | `50a5a9a5dd4f7d87346cd5dcdcd6ddd46e9c57c5029874933d18a9684b2cc6f5` |
| `runs/m6_holdout_floors_v3.json` | `e931c88622fad84e8f8b2cf18940cbe27da1c93e0d009dfbaa3d6c6cae050c77` |
| `runs/m6_holdout_floors_v4.json` | `4cd2d01a9fd76064e701ae77a9226208cbae94d743f76f502d3d0a5f657d9523` |
| `runs/gate_m6_candidate1_v1.json` | `546a9739f8d1c7d21a91a07eb902c8af9bda92cdaa8f7917f312894f6a861b24` |

The JSON binds all four M6 floor artifacts and sidecars plus the candidate-1
sidecar. No gate or floor change is authorized by this finding.

## Final disposition

Condition 4 is `PASS`. Condition 5 is `PASS`. The post-repair residual is
`PUBLISHED_REGARDLESS`. No changed event semantic, support definition, truth
reducer, cell, gate, or floor was detected, so the narrow prospective §2.8
amendment escape path is not triggered.
