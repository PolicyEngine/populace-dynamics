# M6 projection engine — design and temporal-holdout gate proposal

- **Design id**: `2026-07-12-m6-projection-engine`
- **Roadmap**: #113 M6 (the projection engine), the last build before #113 M7
  (trust-fund accounting) and #113 M8 (integrated scoring). Workstreams #100
  (W1/W2/W3 seams), ADR-0001 (`docs/adr/0001-populace-axiom-seam-ownership.md`).
- **Status**: DESIGN (draft, revision 7). This document seeded the `gate_m6` lock
  ceremony (design review → floor build → adversarial referee → verification →
  ratify-by-merge → lock), now **locked** (`gates.gate_m6`, v3 floor
  `runs/m6_holdout_floors_v3.json` sha256 `e931c886…`). **This document edits no
  `gates.yaml` cell, moves no threshold, builds no floor, and writes no test.**
- **Revision 7 (design amendment 3c)** corrects §2.8.6 and the §10 `preflight_2`
  field, which **inverted** the certified earnings sign-path. They pinned
  pre-flight 2 to verify the `draw_sign` branch and to reject the `_target_models`
  reconstruction as a "test seam," but today's certified `RegimeGatedQRF` exposes
  `_target_models` and **no** `draw_sign`, so the faithful-to-spec pre-flight
  aborts every real run (harness-referee finding F1, PR #185 comment 4966859161).
  The corrected pin verifies the `_target_models` reconstruction deploys and
  rejects the `draw_sign` seam — restoring the direction of engine-referee
  observation 6 (PR #173 comment 4962620806), with the designed-abort semantics
  unchanged. Docs-only; edits no `gates.yaml` cell and writes no test.
- **Revision 6 (design amendment 3b)** adds §2.8.3a, the **year-0 earnings-domain
  law**, closing the harness build lane's round-4 blocker (Sol,
  `~/PolicyEngine/sol-worktrees/m6-harness-REPORT.md`): the closed-panel universe
  includes people the certified forward-earnings initializer has no year-0 state for
  (non-head/spouse members; 2017/2019 openers), and `materialize_initial_frame`
  raises for any such person. §2.8.3a pins the earnings-module domain = the
  2014-anchored realized-earnings **closed panel**, an explicit non-scored marker +
  wrapper-level domain filter for everyone else (later earnings-entrants are
  report-only open additions, §4.8), the 2014-fixed anchor rule, and the
  domain-restricted symmetric gated-earnings scoring — with the empirical finding
  that the frozen v3 floor over-includes ~21 % open-additions in its earnings
  support (a disclosed conservative delta + self-check, not a threshold move).
- **Revision 4 (design amendment 3)** adds §2.8, the **scored-run harness** — the
  deliverable whose absence stopped the registered `gate_m6` candidate-1 run before
  scoring (#42 comment 4962640241 registered it; comment 4962773701 is the designed
  pre-scoring stop, graded a registration error, naming this amendment as the
  unblock path). §2.8 pins the projected-slice→native-panel builders per field
  against the certified `MaritalPanel`/`HouseholdCompositionPanel` schemas, the
  year-0 realized-2015-interview slice mirroring the floor, the drift-scoring layer
  bound to the frozen v3 floor's cell machinery, and the two real-data pre-flights
  — so the harness build lane implements with zero design choices, candidate-blind
  (builders and scorer exercised on synthetic frames only until the registered run).
- **Revision 2** adds the forward earnings generator spec (§2.7), a design
  amendment unblocking the engine build's honest blocker (Sol, PR #173): the
  certified gate-1 candidate-11 generator is a **backward** biennial imputation
  chain and cannot be composed into the M6 forward loop without a new stochastic
  law. §2.7 specifies that law — a forward mirror of the certified conditioning
  structure, fit `≤2014` by construction and **first-certified by `gate_m6`** (not
  `gate_1`) — with the annualization rule, the certification framing, and the
  implementation contract for the engine lane; §4.2 and decision 10 are updated.
- **Revision 3 (design amendment 2)** adds §2.7.6, closing the engine build's
  round-2 blocker (Sol): §2.7 specified the forward rank draw but not the
  leakage-safe rank↔level map for the 2016/2018 draws (the certified de-index uses
  target-period marginals that do not exist post-`T*`). §2.7.6 pins the rank-to-level
  law (a NAWI-normalized calendar-invariant `≤2014` marginal, re-indexed by a
  `≤2014`-projected wage index — never realized post-`T*` NAWI, which would leak into
  the gated level cells), the re-ranking law (the fit-population CDF, the exact
  inverse), and the four secondary byte-determinism gaps, so the engine lane
  implements with zero design choices. The `gate_m6` lock flip's `design_commit`
  must finalize to this amendment's merge.
- **Revision 1** incorporates the adversarial design referee round (PR #170
  comment 4953818376, verdict MAJOR REVISION) and the ten-decision adjudication
  (comment 4953722912, whose conditions bind). The referee cleared the two hard
  gates — the P1 leakage BLOCKER is **not** triggered (§4.2 refits ≤`T*`) and the
  P2 decision-8 premise **passes** (the embedded marital core is literally the
  certified `CANDIDATE_16` object) — and returned fourteen findings, mapped to
  sections in the revision log below.
- **Evidence base cited by path+field**: the roadmap (#113); the locked contract
  `gates.yaml` (`holdout_protocol`, `noise_floor`, `views`, `moment_battery`,
  `gates.gate_m4`, `gates.gate_w1`); the module registries
  (`family_transitions/registry.py`, `household_composition/registry.py`,
  `household_composition/base_simulator.py`,
  `household_composition/components/marital_core_adapter.py`); the M4 candidate
  (`disability_hazard_sim.py`); the gate-2c candidate
  (`couple_formation_sim_v2.py`, `data/couple_earnings.py`); the gate-1 earnings
  candidate (`scripts/run_gate1_candidate11.py`); the claiming module
  (`claiming.py`); the PSID readers (`data/family.py`, `data/transitions.py`,
  `data/deaths.py`); the W1 transport candidates (`transport_deployment_v1.py`,
  `transport_deployment_v2.py`); the W2 seam (`scripts/w2_seam_caregiver.py`,
  `runs/w2_seam_caregiver_v1.json`); the M2 pseudo-projection
  (`runs/m2_pseudo_projection_v1.json`); the contract loader (`evaluation.py`,
  `contract.py`, `artifacts.py`); the W1 amendment
  (`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md`); the standing
  process addendum (#42 comment 4948637741).

## Revision log (finding → section)

- 1 (spec-selection-on-full-sample disclosure) → §4.2, §4.9, decision 10.
- 2 (earnings QRF in the refit enumeration) → §4.2.
- 3 (spec-vs-fit distinction; §7↔§4.2) → §4.2, §7.
- 4 (gate-2c modifier composed at a named step) → §2.2 step 3, §2.6.
- 5 + 6 (single-source-of-truth dataflow; injection line) → §2.6.
- 7 + 9 (support basis; symmetric presence-conditioning; attrition) → §4.4,
  friction F7.
- 8 (weight semantics over the projection window) → friction F6, §4.7.
- 10 (dual-axis reference-year/event-year shock-window pin) → §4.1.
- 11 (claiming-table vintage) → §2.2 step 7, §7; **frozen §2.8.10** (2014-edition 6.B5.1).
- 12 (biennial boundary peek at `T*`) → §4.2.
- 13 (household-ID persistence + weight carriage) → §2.2 step 8, §9.
- 14 (compute the OC before lock; weak-power pause) → §9.
- amendment 3d (≤`T*` external-reference input bindings: claiming / SSA-params / mortality
  vintages + the input factory) → §2.8.10, §10 revision 10 (resolves finding 11's freeze
  branch; referee SPEC-SOUND #42 comment 4968160638, 8 findings landed).
- amendment 3e (mortality-rates shape bridge + parameter-dir binding) → §2.8.10.5, §10
  revision 11 (closes factory-referee F1/F2, PR #191 comment 4969829131: the pinned
  seven-band external-rates shape vs `AgeSexMortalityModel`'s 0-start requirement, and the
  version gate's unbound parameter directory; N1 `_era_mapping` docstring soften).
- amendment 3f (demographic-seed sex source correction) → §2.8.3f, §10 revision 12 (corrects the
  §2.8.3 per-field list, which attributed `sex` to `panels.demographic_panel`; the
  demo frame has no `sex`, so `build_realized_population` now sources it from
  `data.deaths` — closes the third registration's crash-2, graded #42 comment
  4972045579; adds the `m6_schema_audit` full-phase column contract).
- amendment 3g (marital projection domain law for sub-`START_AGE`-at-anchor
  persons) → §2.8.2g, §10 revision 13 (corrects the §2.8.2 marital-builder universe
  pin — it claimed `_valid_persons` resolves the missing-entry-row case, but that
  certified-span test runs *before* the anchor override, so person-years presence at
  the anchor requires `anchor_wave ≥ birth_year + START_AGE`; adopts option B
  seed-at-marital-entry over exclude-and-mark to keep the frozen v3 floor
  byte-identical; closes the fifth registration's execution failure, graded #42
  comment 4979269487, forensics comment 4979437110).
- amendment 3h (fertility / open-additions roster-materialization domain law) →
  §2.8.2h, §10 revision 14 (separates the frame-independent §2.8.2 at-risk *schedule*
  that governs scoring from the simulated-mortality *roster* that governs
  materialization — §2.2 step 1 already mandates the decedent exit for "all
  subsequent year-`t` steps" but no §2.8 law stated it for step-4 materialization;
  scheduled maternal births materialize only for roster-present mothers, and a
  wave-loop-killed mother's scheduled birth drops into the report-only reconciliation
  via a filter *after* the draw, leaving the fertility RNG stream byte-unchanged; the
  guard `steps.py:412` stays as the invariant backstop; the frozen floors are
  byte-identical because open additions are report-only (§2.1); closes the sixth
  registration's execution failure, graded #42 comment 4984699959, forensics comment
  4984997277).

## 1. Summary

M6 composes the M1–M5 certified generators into a year-by-year forward
simulation on the unified person-period panel (#113: populace persons × years,
future births and immigrant entrants as synthetic persons entering the panel).
It fits nothing new except a versioned alignment layer; its certification move is
a **temporal holdout** — fit the dynamics through year `T*`, project the held-out
window `T*+1 … T_end`, and score projected observables against their realized
PSID values under the same locked machinery every prior gate uses
(`gates.yaml` `noise_floor` half-split floors; K=20-draw mean estimator on
`numpy.random.default_rng(5200 + k)`; tolerance `round(mean + k·sd, 3)` capped at
`ln(1.5)`; seeds `0–4` with a 4-of-5 conjunction; the one-shot rule enforced by
`artifacts.write_new`).

The design record below covers the engine architecture (§2), the composition
resolution under the plumbing/surgery rules (§2.6), the RNG stream registry (§3),
the `gate_m6` temporal-holdout proposal with cell-level observable definitions and
the support-basis/attrition treatment (§4), the per-year PolicyEngine seam (§5,
the W2 pattern), the M7 interface (§6, interface only), the explicit non-goals
(§7), the numbered open decisions with their adjudications (§8), and the
floors-ceremony plan (§9).

Two scope statements this revision makes explicit and does **not** overstate:

- The holdout is out-of-sample for fitted **parameters** but **in-sample for
  spec/structure selection**: `CANDIDATE_16`/`CANDIDATE_9`, M4, and the gate-1
  chained-QRF were structurally chosen by audited ladder searches that observed
  1998–2022 including the 2015–2022 window. A `gate_m6` pass reproduces held-out
  moments *after* an audited iterative search whose structure was chosen on the
  full window — not a structural out-of-sample claim (§4.2, §4.9, decision 10,
  per #42 comment 4948637741).
- The temporal holdout is **PSID-projected vs PSID-realized**, so — unlike
  `gate_w1` family B — it needs no concept bridge (§4.3). But the realized side is
  PSID-attrition-selected, and the gated cells condition on realized
  wave-presence; this is disclosed as conditioning-not-leakage and priced per
  cell, with mortality the worst case (§4.4).

## 2. Engine architecture

### 2.1 The unified person-period panel

The canonical object is the #113 person-period panel: one row per
`(person_id, year)`, with the full variable surface as column groups (earnings,
marital/roster, disability/claiming state, AIME/PIA, per-year calibrated weight).
Every product is a slice — the current populace cross-section is one year-slice,
the DYNASIM-class projection is the whole panel. M6 is the engine that writes the
future year-slices. It does not own the panel schema (a populace-side decision
that lands with M5, per #113); it consumes the pinned M5 frame as year 0 and
appends `T` future year-slices.

Two population strata coexist in the panel and must be kept distinct because they
have different ground truth:

- the **closed panel** — persons present at year 0, projected forward from their
  realized year-0 state. Those who remain in PSID over the holdout window have
  realized futures; **attriters do not** (friction F7, §4.4). The gated surface
  (§4) is defined on realized-wave-presence with that selection disclosed.
- the **open additions** — synthetic births (from the fertility/roster module)
  and immigrant entry cohorts, materialized as new person-rows entering in their
  entry year. These have no PSID ground truth over the holdout window and are
  therefore report-only in `gate_m6` (family B, §4.8, decision 4).

The panel carries a **per-year calibrated weight**, but the certified generators
and their floors use **start-wave weights** (`gate_m4`; `disability_hazard_sim`
"start-wave weights"). Which weight the gated rates use, and how a start-wave
weight extends across the projection window, is friction F6 (§2.5) and is pinned
in §4.7.

### 2.2 The wave loop and the per-period order of operations

The engine is a loop over projection years `t = 1 … T`. Within each period the
modules compose in a fixed order of operations, chosen so each module reads only
predetermined state (the year `t-1` panel plus already-resolved year-`t`
sub-states):

1. **mortality** — draw deaths for the surviving `t-1` population from the
   differential-mortality component (`data/deaths.py` basis; NCHS × PSID-band
   hazards). Decedents leave the risk set for all subsequent year-`t` steps.
2. **aging** — advance age by one year and roll calendar state (indexing series,
   wage base) for survivors.
3. **couple formation / dissolution** — the single marital core: the certified
   `CANDIDATE_16` hazards (first marriage, divorce, remarriage, widowhood,
   spousal-age-gap), **with the certified gate-2c first-marriage-by-earnings
   modifier composed onto the first-marriage events** — `couple_formation_sim_v2.
   apply_first_marriage_modifier`, a no-RNG marginal-preserving reweight reading
   each person's **person-constant career-earnings-capacity tercile** (§2.6, R2).
   This step emits the period's **authoritative** marital state (`sim_years`);
   §2.6 lists every downstream reader. Widowhood interacts with step 1 (friction
   F1).
4. **fertility / roster** — births from the fertility component; new child
   person-rows attach to mothers; the roster (household membership) updates.
   Fertility reads authoritative marital state (`marital_core_adapter.
   paternal_births` selects `marital_state == "married"`), so it must run after
   step 3 — the constraint that forces the single core to sit at step 3 (§2.6).
5. **disability** — work-limitation incidence / recovery and the near-FRA
   disability-exit split, from the M4 hazard machinery (`disability_hazard_sim.py`;
   `disability_conversion.py` for the FRA auto-conversion, factor `1.0`, 42 USC
   402(a)/416(i)(2)(D)). Conditions on (age-band, sex) only; marital-blind (§2.6).
6. **earnings** — draw the year-`t` earnings increment from the gate-1-certified
   chained-QRF earnings generator, conditioned on age and the person's
   latent-permanent/anchor component (`scripts/run_gate1_candidate11.py`).
   Marital-blind (§2.6, R2).
7. **claiming** — draw claim age from the calibrated Table 6.B5.1 distribution by
   (sex, entitlement year) (`claiming.claim_age_pmf(exclude_conversions=True)`),
   with DI→retirement conversions supplied by step 5. The 6.B5.1 reference is the
   *2023* Supplement (`claiming.py:7`), a post-`T*` external table; it touches no
   gated cell but feeds report-only benefit levels, so it is frozen to a ≤`T*`
   vintage or its vintage leak is named (finding 11, §7; **the ≤`T*` freeze is pinned
   in §2.8.10 — the *2014*-edition 6.B5.1**).
8. **household composition reconciliation** — resolve cohabitation, legal-spouse
   residual, parental-home exit, multigenerational and skip-generation occupancy,
   non-family bridge, and household size from the `household_composition` registry
   (`CANDIDATE_9`), **consuming the step-3 authoritative marital state through the
   adapter's read-only mappers rather than re-simulating it** (§2.6, findings
   5/6). Household identity and per-period household weight carry across
   composition changes per the rule pinned in §9 (finding 13).

The order encodes a directed acyclic graph. Deaths first so no dead person is
married, made disabled, or paid; the single marital core at step 3 so fertility,
and the household reconciliation, read one authoritative marital state; earnings
after the demographic state resolves (though it reads none of it — §2.6);
household composition last as a reconciliation over the resolved roster.

### 2.3 The module registries as the interface

The two flattened registries are the engine's module interface. Each is an
immutable `ComponentRegistry` of `ComponentDefinition`s resolved by
`(kind, implementation_id)` with frozen `params`, fitted in declared dependency
order:

- `family_transitions/registry.py` — `CANDIDATE_16`, `contract_revision
  gate_2_amendment_1`; seven kinds in fit order `initial_states → first_marriage
  → divorce → widowhood → remarriage → fertility → spousal_age_gap`.
- `household_composition/registry.py` — `CANDIDATE_9`, `contract_revision
  gate_2b_locked`; ten kinds.

The certified gate-2c object — the first-marriage-by-earnings modifier — lives in
**neither** registry (`couple_formation_sim_v2.py`); the engine composes it
explicitly at step 3 (§2.2, §2.6, finding 4). A `gate_m6` run pins the resolved
`CandidateSpec.sha256` of each registry plus the 2c modifier's committed spec.
`CandidateSpec.sha256` hashes params/impl, **not** estimates or data
(`family_transitions/registry.py:118-124`), which is the hinge for the spec-vs-fit
distinction (§4.2): the same spec sha with a truncated-panel fit is a different
fitted artifact, not a changed certified object.

### 2.4 Where the W1 transport layer sits

W1 transport (`transport_deployment_v1.py` / `v2.py`, `gate_w1`) is **period 0**
of the projection, not a separate stage. Deployment places the certified
generators onto the pinned populace frame (`gate_w1` `deployment_frame`: bundle
`us-4.18.8`, dataset `populace_us_2024`, artifact sha
`c2065b642ab00da74746afdfd9f06890e5f32f9b10bd6610ff236452d40f39c5`, `n_persons
166302`, `n_households 57240`) and seeds each frame adult's entry state
(`transport_deployment_v2` Q1). M6 takes W1's deployed year-0 panel as its initial
condition and runs the §2.2 loop forward. Two distinct initial conditions exist
and the gate must not conflate them (decision 5): the **deployment initial
condition** (W1, re-drawn entry state on the CPS frame, for the production
projection) and the **holdout initial condition** (`gate_m6` family A, each PSID
person's *realized* year-`T*` state, so the temporal holdout tests forward
dynamics in isolation from transport error).

### 2.5 Registry frictions against the composition order

The registries were sealed for single-period gates; the period axis and the
mortality-linked composition order expose seven concrete resistances the engine
must resolve. These are reported, not hidden. F1–F5 verify against code (referee
"what is NOT wrong"); F6–F7 are added in revision 1 (findings 8, 9).

- **F1 — endogenous vs exogenous widowhood.** `family_transitions/registry.py`
  fits `widowhood` as a standalone PSID hazard (`_WIDOWHOOD_PARAMS
  period_trend_applied=False`, reading `initial_states.support.by_person`), not a
  simulated spouse death. With mortality first, a mortality-killed spouse and the
  fitted widowhood hazard double-count unless reconciled. The gated closed-panel
  test uses the certified exogenous hazard (what `gate_2a` certified); the open
  production projection needs an endogenous-widowhood reconciliation named out of
  M6-certified scope (decision 5, successor gate).

- **F2 — household composition embeds a second marital core.**
  `household_composition/registry.py` `marital_core_adapter._fit_marital`
  (`registry.py:251-261`) calls `fit_family_transitions`, whose body
  (`marital_core_adapter.py:36-44`) returns `ft.REGISTRY.fit(ft.CANDIDATE_16,
  context)` — the identical estimated object as the standalone certified core (P2,
  referee-verified). Candidate-9 then **re-simulates** the core internally
  (`base_simulator.py:72-77`, `ft.simulate(mpanel, holdout_ids,
  fitted.family_transitions, draw_seed)`). Running both step 3 and this internal
  simulation is a double-run with inconsistent marital state. Resolution in §2.6
  (injection, forced by R1+R2).

- **F3 — frozen `core_seed` has no period axis.** `household_composition`
  components seed from frozen params (`_LEGAL_PARAMS core_seed=5200`; child streams
  `SeedSequence([5200 + k, 0xC2/0xC5/0xC7])`, `_CHILD_PARAMS
  persistence_fit_seed=0xC7`). A component that constructs `default_rng(core_seed)`
  internally draws the same numbers every projection period, but the params are
  frozen (`MappingProxyType`) and the registry rejects any mismatch
  (`registry.py:148-153`), so the period cannot be folded into `core_seed`. The
  period is threaded at the engine level (§3), spawning the per-`(module, period)`
  generator and injecting it — engine-side plumbing, not a change to certified
  params (R1, §2.6).

- **F4 — the earnings ⇄ marriage dependency is person-constant, not a
  within-period cycle.** `couple_formation_sim_v2` conditions first-marriage
  timing on an earnings tercile, but the tercile is the person's **career-mean
  indexed positive-year earnings** — a person-constant capacity measure
  (`data/couple_earnings.py:17-45`; `couple_formation_sim_v2.py:261`), **not** a
  contemporaneous or lagged annual level. So there is no within-period annual
  cycle: the modifier reads a predetermined permanent-earnings tercile (§2.6, R2).
  This supersedes decision 6's lagged-edge framing (§8, revised decision 6).

- **F5 — the M4 simulator preserves observed support.**
  `disability_hazard_sim.simulate_draw` (lines 221-259) preserves realized support
  (persons, waves, ages, sexes, start-wave weights) and re-draws only state. It is
  a holdout-*reproduction* simulator, not a forward-*projection* one. The gated
  disability path reuses reproduction mode (§4.4); the open production projection
  needs a forward variant (decision 5's successor gate).

- **F6 — weight semantics over the projection window (finding 8).** `gate_m4` and
  `disability_hazard_sim` use **start-wave weights**; the panel (§2.1) carries a
  per-year calibrated weight. A projected-vs-realized rate must use one consistent
  weight on both sides or every gated numerator/denominator is biased. The
  projection-window convention is pinned in §4.7 (extend the `gate_m4` start-wave
  definition: the gated rates use the person's realized-`T*` start-wave weight held
  fixed across the window on both the projected and the realized side, matching the
  floor).

- **F7 — PSID attrition on the truth side (finding 9).** The realized holdout
  rates and the half-split floor are computed on survival-and-retention-selected
  persons; a non-death for an attriter is unobservable, so the realized
  death-hazard exposure is retention-selected. Treated by symmetric
  presence-conditioning with per-cell dispositions, mortality worst (§4.4).

### 2.6 Composition under the plumbing/surgery rules (R1–R3)

The adjudication (4953722912, findings 5/6) sets three rules and directs deriving
the concrete design from code, not picking a side a priori. **R1** — engine-supplied
RNG generators are plumbing; changing a certified object's fitted params or
internal code paths is surgery; feeding a certified object externally-simulated
**input state through an interface it was certified with** is plumbing; **bypassing
an internal simulation it was certified with** is surgery unless accompanied by an
explicit re-certification note. **R2** — every fitted object must be fed covariates
with the timing/conditioning it was **estimated** with. **R3** — if R1+R2 jointly
force injection into candidate-9, adopt it with a targeted re-certification note.

**R2, pinned from code — the marital/earnings timing each downstream object was
estimated with:**

- gate-1 earnings QRF: predictors are age plus a fixed latent-permanent/anchor
  component and the zero-anchor participation regime
  (`scripts/run_gate1_candidate11.py`; gate-1 view `covariate_columns: [age]`). It
  reads **no marital state** (grep for marital/married/marriage over the candidate:
  zero hits).
- M4 disability: `incidence`/`recovery`/`prevalence` by (age-band, sex),
  `exit_retirement_share` by sex (`disability_hazard_sim.py`). **No marital
  state.**
- claiming: Table 6.B5.1 shares by (sex, entitlement year) (`claiming.py:7-11`,
  `claim_age_pmf` line 262). **No marital state.**
- gate-2c first-marriage modifier: `m(tercile | age band, sex)` where the tercile
  is the person's **career-mean indexed positive-year earnings** — a
  person-CONSTANT capacity measure (`data/couple_earnings.py indexed_earnings_
  supply`; `couple_formation_sim_v2.py:261 tmap = {pid: ce._tercile_of(v,
  axis.cuts) for pid, v in axis.earn.items()}`), not an annual level.

Two consequences follow:

- **Only fertility (step 4) and household composition (step 8) read authoritative
  marital state** among the composed objects; earnings, disability, and claiming
  are marital-blind. So the single marital core can run once at step 3 and be
  consumed read-only downstream with no R2 reorder conflict.
- **The F4 "cycle" is not contemporaneous.** The 2c modifier conditions on a
  person-constant career-capacity tercile, whose projection-time realization is
  the person's permanent/anchor earnings component (predetermined at panel entry;
  the realized ≤`T*` career-capacity in the gated closed-panel test). Feeding it a
  `t-1` annual level (decision 6's lagged edge) is as much an R2 violation as
  feeding it `t`; the R2-faithful composition reads the permanent tercile, which
  has no within-period timing cycle. Decision 6 is revised accordingly (§8).

**R1+R2 force injection into candidate-9 — adopted with the R3 note.** Fertility
(step 4) reads married state (`marital_core_adapter.paternal_births` selects
`marital_state == "married"`), so the marital core must be simulated by step 3 — it
cannot wait for candidate-9 at step 8. But candidate-9 re-simulates the core
internally (`base_simulator.py:72-77`) and consumes the result through its
read-only mappers (`marital_core_adapter.spouse_from_marital` lines 66-87,
`simulated_marital_binary` lines 99-113). Running both is the F2 double-run. The
only consistent resolution is to **inject step-3's authoritative `sim_years` into
candidate-9, bypassing its internal `ft.simulate`, and let the read-only mappers
consume it.** Under R1 that is surgery (bypassing a certified internal
simulation), so per R3 it carries a targeted re-certification note. R1 is the
binding line and is strictly more conservative than finding-6's proposed
"params/impl unchanged ⇒ plumbing" reading; the conservative reading is adopted so
the floors-ceremony referee re-verifies rather than assumes.

**What the re-certification note must establish** (targeted margin check, not a
gate-2b re-ceremony):

1. the estimated core is byte-identical — `ft.REGISTRY.fit(ft.CANDIDATE_16,
   context)` in both the step-3 path and candidate-9's `_fit_marital` (P2);
2. the composition path changed: candidate-9 consumes injected step-3 `sim_years`
   (carrying the 2c marginal-preserving modifier) instead of its internal
   `ft.simulate(draw_seed)` draw;
3. the two differences leave candidate-9's actual conditioning input intact — (a)
   the draw seed differs but the distribution is identical (same estimated object,
   P2); (b) the 2c modifier **preserves the (band, sex) first-marriage marginal by
   construction** — `apply_first_marriage_modifier` enforces `sum_t m·phi_cert = 1`
   and raises on violation (`couple_formation_sim_v2.py:520-525`), and candidate-9
   reads only the married/not-married marginal via `spouse_from_marital`, not the
   within-band earnings-tercile composition the modifier shifts;
4. a pre-named distributional margin check (gate_m4 ≥3σ style) on candidate-9's
   household-composition **output** moments under injected-vs-internal marital
   state, over the gate seeds. Pass ⇒ certification transfers; fail ⇒ the
   composition depends on the bypassed internal draw and a fuller re-ceremony is
   required.

**Dataflow, once per period** (the single-source-of-truth trace finding 5 asks
for): step 3 emits authoritative `sim_years` (CANDIDATE_16 core + 2c modifier).
Readers: step 4 fertility (`paternal_births`, reads `married`); step 8 household
composition (via `spouse_from_marital`, `simulated_marital_binary`,
`father_marital_by_year` — all read-only, no `ft.simulate`). Non-readers (verified
marital-blind, R2): step 5 disability, step 6 earnings, step 7 claiming. No step
re-simulates the core.

### 2.7 The forward earnings generator (design amendment, unblocks #173)

**The blocker (Sol, PR #173 "M6 engine: wave-loop composition", BLOCKED
honestly).** The engine build composed steps 1–8 and passed the §2.6
re-certification margin check, but reported a BLOCKER: the certified gate-1
candidate-11 earnings generator (`scripts/run_gate1_candidate11.py`, run
`gate1_rank_knn_v5`; byte-identical to candidate-10, `run_gate1_candidate10.py`) is
a **backward** biennial imputation chain — it anchors each person's chronologically
**latest** real earnings, predicts wave `t−2` participation from later earnings,
and draws earlier ranks from pair/triple/re-entry pools conditioned on **later**
ranks. M6 starts from realized 2014 and needs **forward** earnings. Sol correctly
refused to (i) invert the certified backward conditional, (ii) extrapolate
period-specific marginals beyond the last `≤2014` fit cell, or (iii) map the
two-year kernel to annual steps — each a different stochastic law than the one
gate-1 certified — shipping only the admissible `≤2014` refit plumbing and an
**explicit external earnings seam** (`CertifiedEngineInputs` still requires an
externally supplied earnings generator). This subsection specifies that generator.

**2.7.1 The forward law.** A **biennial forward conditional-rank chain**
`2014 → 2016 → 2018` for the scored path — the gate's earnings cells live on
reference years `{2016, 2018}` (§4.1), realized 2014 is the anchor, so two biennial
forward steps produce the two scored draws (the chain continues `2018 → 2020 →
2022` for the report-only/seam path, but those draws are not gated — §4.1 shock
demotion). The scored path stays **biennial**, at the certified `period_step 2`
(`gates.yaml views.psid_family_earnings_pairs`), so no annual kernel is invented
for any gated cell — retiring Sol's concern (iii).

The forward law **mirrors the certified candidate-11 conditioning structure**,
reversed in time, and is **fit from scratch on `≤2014` forward wave tuples** — it is
**not** an inversion of the certified backward conditional (retiring concern (i)).
Because it is a `≤2014`-fit conditional-rank law applied out-of-sample to the
`2014→2016` and `2016→2018` steps — not a period-specific marginal carried past its
last fit cell — it does not extrapolate a marginal (retiring concern (ii)):
applying a `≤2014` transition law out-of-sample is exactly what the temporal
holdout scores.

The mirror, primitive by primitive (certified backward → forward):

| certified backward primitive | forward mirror | machinery |
|---|---|---|
| anchor = chronologically **latest** real earnings | anchor = realized **2014** (the only real forward anchor) | candidate-7 anchor `u_A` |
| predict `t−2` participation from **later** earnings | predict `t+2` participation from **earlier** earnings | RegimeGatedQRF sign gate |
| draw **earlier** ranks conditional on **later** ranks | draw **later** ranks conditional on **earlier** ranks | k-NN conditional rank bootstrap |
| two-step-plus-anchor memory (two later waves + anchor) | two-step-plus-anchor memory (two earlier waves + anchor) | candidate-7/10 |

**What transfers byte-for-byte from candidate-11/10** (`run_gate1_candidate11.py` /
`run_gate1_candidate10.py` docstrings): k-NN `k = 25`; distance weights
`1 / 0.5 / 0.25`; the fixed `λ = 0.1` donor-coordinate blend for non-Q0 targets
(Q0 exempt, its third distance term staying `|u_A(donor) − u_A(target)|`); the
`u_w = Φ(ŵ/σ̂_w)` shrunk-permanent rank — **candidate-8's construct** (PR #58,
`run_gate1_candidate8.py:27-38`) built on candidate-3's Stage-1c correlated-noise
decomposition (c10 itself: "the u_w decomposition is candidate 8's") — as the
latent-permanent coordinate; the RegimeGatedQRF participation sign-gate architecture; the zero-anchor
participation regime with the **full** re-entry pool (no restriction); the weighted
single-record draw with no smoothing/jitter; and age as the only other covariate
(marital-blind, per the verified R2 facts, §2.6).

**What is newly chosen — justified from `≤2014` (fit-window) diagnostics only**,
candidate-blind with respect to the holdout:

- the **forward conditional itself** (`rank_{t+2} | rank_t, rank_{t−2}, u_A, age`)
  and its k-NN donor pools, **re-formed on `≤2014` forward-ordered wave tuples**
  (the certified pools are formed on the reversed order). The pool-formation rule
  is candidate-7's verbatim, applied to forward tuples.
- the **forward participation gate coefficients**, re-fit on `≤2014` forward wave
  pairs (the RegimeGatedQRF architecture and features are transferred; only the fit
  direction is forward).
- the **anchor role flip** (initial rather than terminal) — forced by the projection
  (2014 is the only real forward anchor) and consistent with the certified rule that
  the anchor is the sole real wave available to the chain's direction.
- the **start-of-chain memory ramp**: `2014→2016` has one prior wave (2014, also the
  anchor) so it runs at one-step-plus-anchor memory; `2016→2018` has two priors
  `{2014, 2016}` and runs at full two-step-plus-anchor memory. This mirrors the
  certified chain's own earliest-wave boundary, reversed; the ramp is fixed from the
  `≤2014` wave calendar, not chosen against any 2016/2018 cell.

**Spec-selection disclosure (#42 comment 4948637741; §4.2 finding 1).** This forward
spec is written **knowing the holdout exists**, but every choice above is
**derivable from `≤2014` data alone** — hyperparameters transferred from the
certified `≤2014`-inner-validated candidate, new choices fixed from `≤2014` wave
tuples and diagnostics, none tuned against any 2016/2018 observable. The gate then
scores the composed engine out-of-sample; **`gate_m6` IS the forward generator's
first certification** (§2.7.3). It inherits the §4.2 spec-selection limitation:
out-of-sample for its fitted parameters, in-sample for its structural choice
(mirroring a full-window-selected certified spec).

**2.7.2 Annualization (non-scored), under the real per-year seam.** The engine calls
the earnings generator **every projection year** (§2.7.4: `generate(frame, year,
rng)` invoked by `apply_earnings`, `origin/m6-engine:src/populace_dynamics/engine/
steps.py:205-238`, docstring "Draw annual earnings"), but the forward chain draws
only at even reference years `{2016, 2018}`. The odd years `{2015, 2017, 2019, …}`
are filled by a **non-scored** rule that never enters a gated cell — the scored
earnings-moment cells (§4.5) read **only** the biennial forward draws at
`{2016, 2018}`:

- **carry-forward is the default** (participation and level both): the odd-year value
  is the last available biennial value — realized for `2015` (`= 2014`), the drawn
  `2016` for `2017`, the drawn `2018` for `2019`. This is the only rule computable in
  the **forward per-year** seam: at the `2015` call the `2016` draw does not yet
  exist (the loop runs forward one year at a time, no lookahead — §2.7.4), so no
  interpolation toward the *next* biennial value is possible without drawing `2016`
  early on the wrong period's RNG stream.
- **the interpolation alternative** (log-linear on the NAWI-indexed level between
  bracketing biennial waves, participation piecewise-constant) is available only if
  the engine **batch-computes** the biennial draws first and backfills the odd years
  — a departure from the pure per-year seam and the F3 per-period stream assignment
  (the `2016` draw must consume the `2016` period stream, not be triggered at the
  `2015` call). It is named for the engine lane but is **not** the seam-faithful
  default.

Either way the odd-year values carry a `non_scored_annualization` tag, feed only the
pe-us seam / AIME / claiming, and are excluded from every `gate_m6` cell and floor.
The interp-vs-carry choice is **immaterial to certification** — it moves only
report-only AIME/benefit levels (referee question 3) — and the real forward seam
resolves the revision-1 flag to **carry-forward**.

**2.7.3 Certification framing.** The forward generator is a **different stochastic
law** than gate-1 certified (forward conditional ≠ backward conditional), so it is
**NOT** covered by `gate_1`; **no `gate_1` certificate transfers to it.** It enters
`gate_m6`'s scored run as one composed module of the engine and is **first-certified
there**, on the temporal holdout, exactly like the composition itself (§2.6) — its
`≤2014` fit makes it leakage-safe by construction, and the gate scores its
2016/2018 output against realized PSID earnings (§4.5). §4.2 adds it to the refit
enumeration; decision 10 records that a `gate_m6` pass is the forward generator's
sole certification.

**2.7.4 Implementation contract (for the engine lane) — the real `EarningsGenerator`
seam.** The forward generator implements PR #173's `EarningsGenerator` protocol
(`origin/m6-engine:src/populace_dynamics/engine/steps.py:164-169`):

```
class EarningsGenerator(Protocol):
    def generate(self, frame: pd.DataFrame, year: int, rng) -> np.ndarray: ...
```

`apply_earnings` (`steps.py:205-238`) calls `generate` **every projection year**,
once per eligible person (`age ≥ 15`) with that person's engine-owned
`ProjectionModule.EARNINGS` stream (`context.person_generator(...)`), and writes the
returned per-row earnings level into `frame["earnings"]`; `CertifiedEngineInputs.
earnings` (`assembly.py:86`) is the standalone slot the build left external
(`earnings_step` → `apply_earnings(..., model=inputs.earnings)`, `assembly.py:292-293`;
it is not in `from_refit_bundle`'s required set). The contract is therefore
**per-year, frame-in / `ndarray`-out**, and **stateful across years through the
frame** — not the pure function the revision-1 draft (incorrectly) described.

- **frame columns read** (the conditioning surface; `earnings` is written
  in-place by `apply_earnings` and carried across years by the loop's frame
  reassignment `current = self.modules.earnings(current, ...)`, while
  `assembly._merge_period_columns` threads the marital/disability/household
  update-frames): `earnings` — the **most recent drawn/realized
  level**, `0` encoding non-participation (this is how the **drawn prior biennial
  rank re-enters**: `generate` re-ranks it to `rank_t`); the realized `≤2014` wave
  columns (the earlier-lag `rank_{t−2}` and the `2014` anchor `u_A`); the `≤T*`
  latent-permanent `u_w`; `age`/birth-year; `sex`. No column dated after `T*` is
  read; the fit (donor pools, participation-gate coefficients) is `≤2014`.
- **the parity branch** (the load-bearing per-year fact `generate` must implement):
  branch on `year`. On an **even reference year** `∈ {2016, 2018, …}` compute the
  forward biennial draw — participation gate → later-rank draw conditional on the
  frame's `rank_t` / `rank_{t−2}` / anchor → de-index to a level — consuming the
  §2.7.5 substream slots, and return that level (`0` if non-participating). On an
  **odd year** return the deterministic §2.7.2 carry-forward value (no RNG consumed).
  Either return is written to `frame["earnings"]` and carried forward by the
  loop's frame reassignment, so the drawn `2016` level is the `2018` step's
  `rank_t`.
- **drawn-prior-rank threading (named input via the frame).** The chain conditional
  `rank_{t+2} | rank_t, rank_{t−2}, u_A, age` needs, at `2018`, the **drawn** `2016`
  rank; it is **not** a `generate` argument but is recovered inside `generate` by
  re-ranking the frame's carried `earnings` column (the `2016` drawn level), with
  `rank_{t−2}` and `u_A` from the realized-`2014` columns. The generator holds **no**
  cross-year state of its own; all chain state lives in the frame the engine threads.

**2.7.5 RNG stream slots (F3 registry, §3.2), under the per-year seam.** `generate`
receives the person's `ProjectionModule.EARNINGS` generator for the **current year**
(`context.person_generator(ProjectionModule.EARNINGS, person_id)`,
`steps.py:212-224`) — the engine's realization of the §3.2 `stream(k, t, earnings)`
slot at period `t = year`. RNG is consumed **only at even reference years** (odd-year
carry-forward is deterministic, §2.7.2). At an even year the generator spawns
candidate-7's three canonical sub-streams from that per-person period stream —
**participation gate**, **donor draw**, **re-entry draw** (`run_gate1_candidate10.py:
239` `SUBSTREAM_CODES {gate:1, donor-draw:2, re-entry-draw:3}`; the two-element
`SeedSequence([·, code])` of `:242-252`) — preserving the certified draw order.
Because each biennial draw uses its own period-`t` per-person stream, no two
`(k, t)` earnings draws share entropy, any biennial step is independently
reproducible, and the split stream stays disjoint from `5200 + k` (§3.3).

### 2.7.6 The rank-to-level and re-ranking laws (design amendment 2, closes Sol's round-2 blocker)

Round 2 pinned the forward conditional-RANK draw and the real per-year seam, but
the engine build blocked again (Sol, `~/PolicyEngine/sol-worktrees/m6-forward-
REPORT.md`): §2.7 never specified the **leakage-safe map from a drawn rank to a
positive earnings LEVEL** for 2016/2018, nor its inverse (recovering the drawn-2016
rank at the 2018 step). This subsection pins both, plus the four secondary
byte-determinism gaps Sol enumerated, so the engine lane implements with **zero**
design choices. It closes the blocker; the lock flip's `design_commit` pin
(§2.7.3) must finalize to **this amendment's** merge commit — the `gate_m6` lock is
holding for it (§2.7.6.6).

**2.7.6.1 The certified de-index, and why it does not transfer forward.** The
certified chain draws a RANK (`_knn_draw` returns the donor's `u_prev`, not a level
— `run_gate1_candidate7.py:383-400`) and de-indexes it through a **marginal
quantile map**: `earnings = marginals[(age_bin, period)].quantile(u_prev)`
(`candidate7:648-654`), where each `CellMarginal` (`run_gate1_candidate5b.py:283-313`)
is the weighted `Qhat_pos` quantile of that `(age_bin, period)` cell's **nominal**
positive earnings, with `p0` the zero share and `rhat` the inverse rank map (no
NAWI normalization anywhere in `c7`/`c5b`). A `≤2014` fit has no `(age_bin, 2016)`
or `(age_bin, 2018)` cell, and §2.7.1 forbids carrying a period-specific marginal
past its last fit cell — so the inherited de-index cannot supply the forward
level/rank maps, and a `≤2014`-derivable substitute must be pinned.

**2.7.6.2 Rank-to-level law — options and the pinned choice.** Three admissible
constructions, adjudicated:

- **(a) donor-level bootstrap** — carry the k-NN-selected donor's own realized
  `t+2` level from the `≤2014` forward tuples, wage-index-bridged to the target
  wave. *Rejected.* The certified de-index is a **marginal quantile map**, not a
  donor-level carry: `_knn_draw` returns the donor's rank and the level comes from
  `cell.quantile` (`candidate7:648-654`). Swapping in a donor-level bootstrap
  changes the estimated de-index **mechanism** (an R2 violation — §2.6), and it has
  no clean inverse for the re-rank (§2.7.6.4), which would need the marginal CDF
  anyway.
- **(b1) carry the 2014 period marginal** — apply `marginals[(age_bin, 2014)]` to
  the 2016/2018 draws. *Rejected.* This is exactly "carrying a period-specific
  marginal past its last fit cell," which §2.7.1 forbids; it also omits 2014→2016
  wage growth, biasing the gated `earn_p10/p50` levels low.
- **(b2) NAWI-normalized calendar-invariant marginal, re-indexed — PINNED.** Fit a
  single `CellMarginal` per `age_bin` (the certified `fit_cell_marginals` /
  `_plotting_positions` machinery `candidate5b:283-313` verbatim) on the **pooled
  `≤2014`** positive earnings (one `CellMarginal` per `age_bin` on the `[25,64]`
  grid, §2.7.6.5), each divided by its wave's wage index (NAWI-normalized to a
  common scale). De-index a drawn rank `u`: normalized level `ĝ =
  Qhat_pos_agebin.quantile(u)`, then the nominal target-year level is
  `ĝ × I_proj(target_year)` (§2.7.6.3). This **keeps the certified de-index
  MECHANISM** — it re-fits a **new** `age_bin`, NAWI-normalized object through the
  **same `CellMarginal` machinery** (`fit_cell_marginals` / `_plotting_positions`,
  `candidate5b:283-313`), not the certified `(age_bin, period)` object itself (a
  `CellMarginal` quantile — R2-faithful). It is fully `≤2014`-derivable, is **not** a
  period-specific marginal (calendar-invariant, so §2.7.1-consistent), and has an
  inverse for the re-rank (interior-exact, §2.7.6.4). `p0` (the zero share) and the participation gate are unchanged; a
  drawn zero stays a zero (non-participation).

**2.7.6.3 The wage index `I_proj` — `≤2014`-derivable, pinned; the leakage
prohibition.** `I_proj(2016)` and `I_proj(2018)` are the wage-index (SSA average-
wage index / NAWI — the `ss/params.py` series the M2 `taxable_payroll_convention`
and AIME indexing use) **projected from `≤2014`**, never the realized 2016/2018
values. Pinned form: an **OLS log-linear fit of `ln(NAWI_y)` on `y` over the
trailing decade `y ∈ [2005, 2014]`**, extrapolated to 2016/2018 (equivalently:
apply the `≤2014` trailing-decade geometric-mean annual wage-growth rate forward
from `NAWI_2014`). Justified from `≤2014` diagnostics only: a decade window
averages out biennial sampling noise while reflecting the recent (post-2005)
wage-growth regime and excluding pre-2005 structural breaks; the two scored steps
consume only `I_proj(2016)` and `I_proj(2018)`. **NAWI publication lag, disclosed:**
the SSA average-wage index for a year is published ~2 years later, so `NAWI_2013`
and `NAWI_2014` are *published* after 2014; but the AWI is a **final, unrevised**
series (SSA does not restate prior AWI values once set), so a `NAWI_y` with
`y ≤ 2014` is a fixed function of `≤ 2014` wages — leakage-safe. Only its
publication *date* is post-`T*`, not its *information*; no realized 2016/2018 wage
outcome enters `I_proj`.

> **Leakage prohibition (pinned).** Using any **realized** post-`T*` NAWI / wage-
> index value on the **scored** path is PROHIBITED — it would leak realized wage
> growth into the gated level cells (`earn_p10`, `earn_p50`, the Δlog-earnings SD,
> the mobility diagonal). Concretely: the projection frame **carries a realized
> `nawi` column** (the aging step rolls `nawi_by_year → nawi`, `steps.py:145-156`);
> the scored forward earnings de-index MUST NOT read it for 2016/2018 — it uses the
> generator's own `I_proj`. The realized `nawi` is admissible only for the
> **non-scored** seam / `taxable_max` plumbing and for the **report-only alignment
> path**. The gated path is the **un-aligned** projection (decision 9, §4.8: "gate
> the un-aligned drift"; the alignment layer is report-only, the M2
> `calibration_disclosure` lesson); realized-index alignment — snapping the
> projected level onto realized NAWI — may appear only on the report-only path,
> disclosed per run within the **maximum alignment displacement** (adjudication 9,
> §4.8).

**2.7.6.4 Re-ranking law — options and the pinned choice.** Recovering `rank_t`
from the carried nominal `earnings` level at the next biennial step:

- **within-frame cross-sectional percentile** — rank the level within the current
  projected frame's cross-section. *Rejected.* It references the **synthetic
  projected cohort**, a population the chain was never fit on (an R2 violation), and
  it is not the inverse of §2.7.6.2, so the rank→level→rank round-trip would not
  close.
- **fit-population marginal CDF — PINNED.** Recover `rank_t = rhat_agebin( ℓ /
  I_proj(wave_of_ℓ) )` — normalize the carried nominal level by its wave's
  `I_proj`, then evaluate the **same `age_bin` `CellMarginal`'s inverse rank map
  `rhat`** used to de-index in §2.7.6.2 (`candidate5b` `CellMarginal.rank`). It is
  the **inverse** of §2.7.6.2 — both are `np.interp` on the one `(wtil, yval)` grid —
  **exact on the strictly-monotone interior**. At the corners it is
  **deterministic but inexact**: `quantile` clamps to `[ymin, ymax]` and `rank` to
  `[0.001, 0.999]` (`RANK_CLAMP_LO/HI`), and a flat-`yval` tie maps back to the first
  `wtil` (`candidate5b:250-259`; the certified code tracks this corner mass as
  `corner_bottom` / `corner_top`). **Consequence for the 2018 re-rank:** `rank(ℓ)`
  is a pure deterministic function of the carried level, so **chain integrity holds**
  (the 2018 draw conditions on a well-defined `rank_t`); the inexactness is only a
  **bounded rank perturbation at the corners** (a tail level clamps into
  `[0.001, 0.999]`; a tie collapses to one grid rank), never a break in
  reproducibility. A carried zero maps to the `p0` / zero-anchor (Q0) regime.
  **Conditioning-consistency (R2):** the
  certified chain's ranks **are** `CellMarginal` CDF positions (it draws `u_prev`
  and de-indexes via `cell.quantile`), so composing as-estimated requires the
  deployed re-rank to reference the **fit-population marginal**, exactly what the
  pinned rule does — not the projected frame's own cross-section.

**2.7.6.5 The four secondary gaps, pinned (byte-determinism).**

- **Memory after 2018 (generated-lag columns).** The frame carries the **two most
  recent generated biennial levels** — `gen_earn_w2` (the `t−2` wave) and
  `gen_earn_w4` (the `t−4` wave) — updated at each biennial step, so every draw has
  its two-step-plus-anchor memory (`rank_t` from `gen_earn_w2`, `rank_{t−2}` from
  `gen_earn_w4`, both via §2.7.6.4; anchor `u_A` from realized-2014). The
  start-of-chain ramp (§2.7.1) fills the early lags from the realized `≤2014`
  columns: `2016` uses realized-2014 (`w−2`) + realized-2012 (`w−4`), `2018` uses
  drawn-2016 + realized-2014, `2020` uses drawn-2018 + drawn-2016.
- **Generator → substream bridge.** The seam passes an `np.random.Generator`; the
  certified helper wants an integer seed. Pinned: reduce the per-person `EARNINGS`
  generator to a seed with `engine.rng.seed_from_generator(rng)` (the existing
  bridge — `int(generator.integers(0, 2**64-1, dtype=uint64))`,
  `origin/m6-engine:engine/rng.py`) and construct the three certified sub-streams as
  `default_rng(SeedSequence([seed, code]))` for `code ∈ {1: gate, 2: donor-draw,
  3: re-entry-draw}` (`SUBSTREAM_CODES`, `candidate10:239`), the `_substream`
  construction verbatim.
- **Frame schema (concrete columns).** The generator reads exactly:
  `person_id`, `age`, `sex`, `u_w` (the §2.7.1 latent-permanent rank),
  `realized_earn_2014` and `realized_earn_2012` (the anchor + `t−4` start-lag),
  `earnings` (current level), and the rolling `gen_earn_w2` / `gen_earn_w4`. The
  **engine `≤2014` refit bundle** (`engine/refit.py`) materializes `u_w` and the
  `realized_earn_*` columns at period 0 (deployment); `_merge_period_columns`
  carries them and the rolling generated-lag columns forward. No post-`T*` column
  (including the realized `nawi`) enters the scored de-index.
- **Age support — the forward law fits `[25,64]`, extending the transferred gate-1
  `[25,59]`.** The forward law is a **new** law first-certified by `gate_m6`
  (§2.7.3), so its fit range is a hyperparameter transferred **where transferable**.
  Here the gate-1 transfer `[25,59]` (`refit.py:802-806`, `age_range [25,59]`)
  **conflicts with the pre-registered truth-side surface**: the gated earnings
  cohorts are `prime = 25–44` and `older = 45–64` (`EARN_COHORTS`,
  `scripts/build_m6_holdout_floors.py:120`; the panel has **no** age cap), so
  `earn_*.older` scores ages up to **64**. Clipping `60–64` through the `55–59`
  marginal would understate the retirement-transition zero-inflation and **kill the
  `earn_zero_rate.older` signal**. The `[25,59]` fit range is therefore a
  **non-transferable** hyperparameter: the forward **participation gate**, **donor
  pools**, **memory ramp**, and the calendar-invariant **NAWI-normalized marginal**
  (§2.7.6.2) all fit on **`[25,64]` pooled `≤2014`** — the `≤2014` PSID panel
  contains ages 60–64 and exactly the retirement-transition signal `earn_zero_rate
  .older` scores, so the extension is fully `≤2014`-derivable. The `age_bin` grid
  extends from the certified 7 five-year bins to **8**: `{25–29, 30–34, 35–39,
  40–44, 45–49, 50–54, 55–59, 60–64}`. The lookup **clips to the nearest fitted bin
  at the new boundary** (`<25 → 25–29`, `>64 → 60–64`); the `<25` and `65+` draws
  are **non-scored** plumbing (feeding only the seam / AIME). **The principle:** the
  law must **cover the surface it is scored on**; narrowing the surface — re-scoping
  or demoting a gated cell — to fit the model's support would be **candidate-informed
  surface design, prohibited** in that direction (the surface is pre-registered
  truth-side; §4.9, decision 10). *Verified:* every gated earnings cell's cohort
  span — `prime 25–44`, `older 45–64` (`EARN_COHORTS`,
  `build_m6_holdout_floors.py:120`) — lies inside `[25,64]`.

**2.7.6.6 Closes the blocker; the lock dependency.** Every under-determined
primitive Sol flagged is now pinned — the rank→level map (§2.7.6.2), its exact
inverse (§2.7.6.4), the `≤2014` index basis and its leakage fence (§2.7.6.3), and
the four byte-determinism gaps (§2.7.6.5) — so the engine lane implements the
forward generator with **zero** design choices. This **closes Sol's round-2
blocker.** Because this amendment changes the design the `gate_m6` block draft
pins, the **lock flip's `design_commit` must finalize to this amendment's merge
commit** (not the round-2 `d6abb16`); the block draft's `design_commit_note`
already records the finalize-at-lock-flip rule, and the `gate_m6` lock is holding
for this amendment.

### 2.8 The scored-run harness (design amendment 3, unblocks the `gate_m6` run)

**The blocker (the run lane's designed pre-scoring stop).** `gate_m6` candidate-1
was registered as a one-shot temporal-holdout run (#42 comment 4962640241) and
then **stopped before any scored phase** (#42 comment 4962773701, graded a
registration error): the engine (PR #173, merged 95a3e05) is **build-only** —
the per-slice native-panel builders (`marital_panel_builder`,
`household_panel_builder`, `assembly.py:59-72`) exist only as test stubs
(`tests/test_m6_engine_assembly.py:75-76,269-270`,
`lambda frame, context: (object(), {1})`), the M6 drift-scoring composition is
unbuilt, and pre-flight 1 has only ever run on synthetic ensembles. The stop was
correct: reconstructing the certified simulators' observed-panel schemas from
projected state inside a scoring lane, unspecified, would be unregistered
modeling. This subsection is the **reviewed design amendment** the stop names as
the unblock path — it pins every modeling decision in the harness so the build
lane implements with **zero** design choices (the §2.7.6 bar), **candidate-blind**:
the builders and scorer are exercised on **synthetic frames only** until the
registered run, and no holdout truth is read anywhere in the harness build.

What is already built and is *not* redesigned here (verified against 95a3e05):
the RNG spawn tree (`engine/rng.py`, §3.2), the injected marital core
(`marital.simulate_marital_step`, §2.6), candidate-9 injection
(`composition.simulate_candidate9_injected`, §2.6), the M4 reproduction adapter
(`disability.simulate_reproduction`, §4.4), the composition re-certification
check (`composition.check_candidate9_recertification`, §2.6), the `≤T*` refit
(`refit.refit_m6_components`, §4.2), and — decisively for §2.8.1 — the
**support/weight/presence contract** (`engine/support.py`:
`prepare_evaluation_support`, `StartWaveWeightSnapshot`, `EvaluationMode`,
`PresenceBasis`, `WidowhoodMode`), which already implements symmetric
presence-conditioning, the F6 start-wave weight on all three sides, and the
forward-mode rejection of realized inputs. The unbuilt deliverables §2.8 designs
are exactly the four the stop enumerated: the projected-slice→native-panel
builders (§2.8.2), the year-0 realized slice (§2.8.3), the drift-scoring layer
(§2.8.4), and the runner with its two pre-flights (§2.8.5–§2.8.7).

**2.8.1 The adjudicated core — the year-0 closed panel carries realized
histories.** *Adjudicated (orchestrator).* For the closed-panel **reproduction**
test, a projected slice's history provenance is **realized**: the per-slice
builders read each person's **realized ≤ their anchor interview** (2015 for the
bulk; the 2017/2019 presence-conditioned openers seed at their later anchor, §2.8.3)
marriage / household history as the seed, and the projected years' contributions to
the native panels are constructed from the engine's **simulated** state under the
per-field rules of §2.8.2. The per-person anchor boundary is the true leakage
fence: seeding a 2019-opener from realized ≤2019 state is the ratified
presence-conditioning (block `f6_weight.start_wave`; §4.4), not a leak — the seed is
the projection's realized *initial condition* (decision 5, generalized to the
anchor wave), no fitter sees it (the refit stays `≤2014`, §4.2), and the scored
quantity is the out-of-sample transition *from* that seed. The argument is entirely
from the ratified record, not a new choice:

- Decision 5 (§8) ratified **realized-`T*` seeding + M4 reproduction mode +
  exogenous widowhood** for the gated holdout, with the forward-projection variant
  a *registered successor gate* (§4.4, §7). §4.4 split the support basis:
  disability is **reproduction mode** (`disability_hazard_sim.simulate_draw`
  preserves realized support and re-draws only state, F5); mortality/marital are
  **presence-conditioned forward-style** on presence-at-start-of-interval
  (conditioning-not-leakage). Both sit on **realized** support.
- The engine already instantiates this for disability: `assembly.disability_step`
  (`assembly.py:257-304`) builds the reproduction support from the **realized**
  `inputs.disability_panel.person_years` filtered to the projection window
  `[start_year+1, end_year]` — **not** from the projected frame — and re-draws
  state via `simulate_reproduction`, seeded from realized wave-one state, on the
  per-period `DISABILITY` streams. The marital and household builders **extend
  this same template**: read realized histories for the year-0 population (they
  are PSID-observed people), simulate state forward.
- `engine/support.py` already encodes the two modes as distinct objects:
  `EvaluationMode.GATED_REALIZED` (requires the realized truth/floor/presence and
  the boundary weight snapshot; `WidowhoodMode.EXOGENOUS_CERTIFIED_HAZARD`) vs
  `EvaluationMode.FORWARD` (rejects every realized holdout input;
  `WidowhoodMode.ENDOGENOUS_RECONCILIATION_REQUIRED`; `structural_delta` names the
  gap). The harness runs the gated path in `GATED_REALIZED`.

**Forward/production path — out of scope, pointer restated.** The forward
production projection **cannot** carry realized histories (no PSID panel over the
projected future), so it seeds from re-drawn transport state (§2.4 deployment
initial condition), needs the endogenous-widowhood reconciliation (F1), and
synthesizes disability support (F5). That object is **decision 5's registered
successor gate**; its delta from the gated presence-conditioned path is already
recorded (§4.4 third bullet; `support.py` `FORWARD`-mode `structural_delta`; §7
"Deferred to the decision-5 successor gate"), and `gate_m6` publishes the
**report-only re-drawn-seed comparison** with a pre-named margin seeding that
gate's flip criteria (decision 5(b), §4.8). §2.8 builds none of it.

**2.8.2 The projected-slice → native-panel builders, per field.** Each builder is
a pure function `(frame, context) -> (native_panel, holdout_ids)` matching the
`assembly.py:59-72` protocols. It constructs a native panel that is
**schema-byte-compatible** with the certified reader's output
(`transitions.build_marital_panel` / `household_composition.build_household_panel`)
so the certified simulators consume it unchanged — R1 plumbing (feeding input
state through the certified interface), not surgery. The panel is **whole-window**
and the certified core is run **once per draw** and cached, exactly mirroring the
built `disability_step` (which caches `state.disability_projection` and slices it
per period): re-running the certified core per period on per-period streams would
stitch a person's trajectory from independent draws and break the longitudinal
marital-event cells, so the reproduction core is one certified realization per
draw `k`, sliced by the loop.

**Cached-core stream address (pinned).** A single whole-window run per draw must
consume **one** deterministic stream address. Pinned: the **period-0 deployment
address** (`rng.py:41-44`; "period zero is addressable for deployment and
compatibility checks") — **not** the period-1 first-invocation streams the loop
passes. Period 0 reproduces the certified single-period topology (`tagged_generator`
preserves the exact `SeedSequence([5200+k, tag])` streams candidate-9 was certified
on, §3.2) and is the **same** address the §2.8.5 internal reference already uses
(`composition.simulate_candidate9_internal_reference`, `composition.py:643-653`,
`n_periods=0`, `composition_rngs_from_registry(…, 0)`), so pre-flight 1's
injected-vs-internal margin check compares like with like. Concretely, per draw `k`
(`draw_index = k`): the cached marital core takes `main_rng =
registry.generator(0, MARITAL_CORE)` and `gap_rng =
registry.child_generator(0, MARITAL_CORE, 1)`; the cached household core takes the
**single** `composition_rngs_from_registry(registry, 0)` `CompositionRngs`
(`composition.py:532-539`); the household-conditioning fertility it consumes is
drawn on `registry.generator(0, FERTILITY)`. Both addresses are bit-exact
reproducible (the §4.9 pass-verification standard).

*Marital builder — the certified `MaritalPanel` schema (episodes → change-points →
per-year state via backward-asof) as the contract.* The certified
`_simulate_candidate16_with_generators` (`marital.py:61-122`) reads from
`panel.attrs` — `person_id`, `birth_year`, `sex`, `start_exposure_year`
(→`start_year`), `censor_year` (→`end_year`) — and from the **entry** (first)
`person_years` row — `marital_state`, `marriage_duration`,
`years_since_dissolution` — seeding `current_start = start_year − duration`,
`dissolution_year = start_year − years_since`. The builder sets, per field:

| field (certified schema) | reproduction-test construction |
|---|---|
| `attrs` **universe** (`person_id / birth_year / sex`, and the whole attrs frame) | `attrs = transitions.build_marital_panel(marriage.marriage_history(), death_records, anchor_weight).attrs ∩ anchor` — the **identical** universe the truth side scores (`marital_tables`, `build_m6_holdout_floors.py:259-261`). This drops anchor persons **absent from the marriage-history file** (children; MH-uncovered adults) on both sides symmetrically — they have no truth-side marital exposure (`_valid_persons`, `transitions.py:239-277`), and a builder that seeded them would project them as `never_married` at-risk (the `pd.isna` entry path, `marital.py:100-101`), biasing the `first_marriage` denominator. `_valid_persons` also enforces `start_exposure_year ≤ censor_year` on the *certified* seed (`birth_year + START_AGE`) — but it does **not** resolve the missing-entry-row case: it runs **before** the builder overrides `start_exposure_year := anchor_wave`, and person-years presence *at the anchor* requires `anchor_wave ≥ birth_year + START_AGE`, which the sub-`START_AGE`-at-anchor children fail (the PSID marriage-history file carries children, so they survive `attrs ∩ anchor`). This is the reg-5 crash class, corrected by amendment 3g (§2.8.2g). |
| `attrs.start_exposure_year` | the person's realized **start-of-holdout wave** (`anchor_wave`; 2015 = the `SEED_WAVE` for the bulk) — the reproduction seed boundary |
| `attrs.censor_year` | the projection horizon 2022, **clipped to the person's realized presence/death censor** (`transitions.person_attributes` from `death_records`), so projected exposure = the realized support the truth side scores (§4.4 symmetric presence) |
| `attrs.n_marriages` | the **lifetime MH18** count `person_attributes` writes (`transitions.py:239-243`; `marriage.py:216`) — the same value the truth-side `build_marital_panel` carries, so the seed is byte-identical. It is **carried-inert**: the C16 core never reads it (the hazard order re-seeds to `1` on entry, `marital.py:107,119`) and `_assemble_panel` **overwrites** it from the simulated episodes + the marriage residual (`marital.py:262-272`). No value-of-record choice — the builder writes MH18 and it is inert. |
| entry `person_years` row (`marital_state`, `marriage_duration`, `years_since_dissolution`) | the **realized** seed-wave marital state, from `transitions.build_marital_panel(marriage.marriage_history(), death_records, weight)` sliced at the seed wave — the identical machinery the truth side runs (`build_m6_holdout_floors.marital_tables:259-261`) |

The projected years then extend this seed under four pinned rules, each stated in
the certified schema so the projected `person_years`, reassembled through
`transitions._assign_state`, reproduce the same state/duration series the truth
side scores:

- **episode extension** — a Y0-intact marriage extends as an **open** episode
  carrying its *realized* `current_start` (`= seed_wave − realized
  marriage_duration`); the C16 pass either keeps it intact to `censor_year`
  (episode emitted `how="intact"`, `end=NA`, `marital.py:243-249`) or closes it at
  a **simulated** dissolution year. Realized pre-Y0 episodes are **not**
  re-simulated — only the most-recent/open state seeds the machine, and the
  projected window appends **new** episodes for simulated (re)marriages.
- **change-point appends** — each simulated transition appends a change-point
  exactly as `transitions._changepoints` derives one from a realized episode
  (`marry` at the simulated year carrying `current_start`; `dissolve` at the
  simulated year carrying `dissolution_year`), preserving the `dissolve`-before-
  `marry` tie order (`kind_order`) and the `allow_exact_matches=False` discrete-time
  convention (`transitions.py:307-352,377-423`). The certified `_assemble_panel`
  already emits episodes in this form; the builder introduces no new change-point
  kind.
- **duration accounting** — `marriage_duration` at projected year `t` = `t −
  current_start` (realized start if the marriage predates Y0, simulated start
  otherwise); `years_since_dissolution` = `t − dissolution_year` (realized if the
  person entered the holdout dissolved, simulated otherwise). This is *verbatim*
  what C16 computes internally (`duration = year − current_start`,
  `years_since = year − dissolution_year`, `marital.py:188,217`); the seed maps the
  realized entry `marriage_duration` / `years_since_dissolution` into
  `current_start` / `dissolution_year` per the certified seed logic
  (`marital.py:104-118`).
- **censor handling at the projection boundary** — the C16 loop simulates
  `[start_exposure_year, censor_year]` (`marital.py:150-152`); at `censor_year` a
  still-married person's open episode is emitted `intact`
  (`marital.py:243-249`). Setting `censor_year` to the realized presence/death
  censor makes projected and realized exposure the same person-year support, which
  is what symmetric presence-conditioning requires (§4.4).

*Household builder — the certified `HouseholdCompositionPanel` schema
(`person_waves` observed states → evolve) as the contract.* Candidate-9's injected
path (`composition._compose_base_injected`) reads the **observed** seed states from
`hh.person_waves` via `padded_person_matrices` — `obs_parent`, `obs_multigen`,
`obs_cohab`, `obs_skipgen` (`composition.py:240-241,302,475-479`) — and evolves
them forward with the fitted entry/exit rates + the **injected** marital binary
(`spouse_from_marital`, `simulated_marital_binary`), re-simulating nothing (§2.6).
The builder therefore constructs `person_waves` whose `{person_id, year, age, sex,
weight, band}` are the realized holdout support and whose
`{coresident_parent, multigen, coresident_grandchild, cohabiting}` are the
**realized ≤ anchor-interview** observed states at the seed wave, read from
`household_composition.build_household_panel` on the realized roster (sliced ≤ the
person's anchor interview) and to the holdout support; the transition helpers
(`has_next` / `next_*`) are rebuilt by `household_composition._add_transitions`. The
projected years are candidate-9's forward evolution over that seeded support,
consuming the §2.8.1 injected marital result and the **single** period-0
`CompositionRngs` (`composition.composition_rngs_from_registry(registry, 0)`;
`simulate_candidate9_injected` takes exactly one `CompositionRngs` per whole-window
call, `composition.py:532-539`) — identical to the certified injected simulator,
only the seed provenance (realized anchor state) and the whole-window caching are
pinned here.

**2.8.2g Amendment 3g — the marital projection domain law for
sub-`START_AGE`-at-anchor persons (closes the reg-5 crash).** The closed-panel
anchor universe (§2.8.3, `build_anchor_frame`, `m6_cells.py:122-140`) admits
**every** person present in a gated start wave with positive weight — **no age
filter** — so it carries minors. The marital builder then overrides
`start_exposure_year := anchor_wave` (`panel_builders.py:187`) and demands the
certified `person_years` **entry row at that anchor wave** (`:200-214`). But the
certified marital `person_years` begin at the risk-set entry
`birth_year + START_AGE` (`START_AGE = 15`, `transitions.py:111,260`), so a person
whose `anchor_wave < birth_year + START_AGE` has **no** certified row at the anchor
and the builder **raises** (`panel_builders.py:209-214`) — the fifth-registration
execution failure (`ValueError: certified marital panel has no entry row at
anchor`; graded #42 comment 4979269487, forensics round-2 comment 4979437110). The
raise reports `missing[:10]` — the 10 ids reg-5 listed are the lowest-id members of
the crash's **seed-0 household-split side**, not the full class's lowest ids — but on the realized
anchor the identical mechanism trips a **single uniform class of 2,850 persons**
(verified read-only through the harness loaders: 2,568 anchored 2015 / 246 anchored
2017 / 36 anchored 2019; ages 7–14 at anchor, mean 10.8; born 2001–2008; every one
`py_year_min == birth_year + START_AGE` exactly, **none** with a certified row at
its anchor wave): sub-`START_AGE`-at-anchor children who age into the marital risk
set mid-window. This is the marital analogue of §2.8.3a's earnings-domain gap — the
closed-panel universe contains persons for whom the certified generator has no
year-0 (here, no anchor-wave) state — but the marital module had **no** domain law.
This subsection is that law.

*Step 1 (uncontested) — correct the universe pin.* The §2.8.2 marital-builder row
(corrected in place above) claimed `_valid_persons` "resolves the missing-entry-row
case by construction." It does not. `_valid_persons` (`transitions.py:265-277`)
enforces `start_exposure_year ≤ censor_year` on the **certified**
`start_exposure_year = birth_year + START_AGE` — a *lifetime at-risk existence* test
(does the person ever reach marriageable age within observation) — and runs
**before** the builder overrides `start_exposure_year := anchor_wave`. The entry-row
invariant instead requires *person-years presence at the anchor*:
`anchor_wave ≥ birth_year + START_AGE`. The two coincide for the bulk (anchored at
or after age 15) and diverge for anchor-present minors; the pin's "drops … children
absent from the marriage-history file" is real but incomplete, because the PSID
Marriage-History file **carries** children (birth year, `n_marriages = 0`), so a
child born early enough to reach 15 by censor **survives** `attrs ∩ anchor`. The
corrected universe is

> marital domain = `attrs ∩ anchor ∩ {anchor_wave ≥ birth_year + START_AGE}`

equivalently: intersect the anchor with the certified marital `person_years` **at
the anchor wave**, not merely with `attrs`.

*Step 2 (the decision) — treat the gap population.* Correcting the test is not
enough: the 2,850 must be assigned a treatment, and the choice is a **modeling
decision**, not a mechanical realignment, because the gap population is **not**
gated-neutral. Of the 2,850, **281** (all born 2001; age 14 at the 2015 anchor,
reaching 18 by 2019) carry truth-side 18-29 at-risk `person_years` at **2019** — 281
of the 3,224 in that **sex-pooled 2019 slice** (8.7 % of persons, 9.0 % of F6
weight). The actual gated floor cell is **female-only and pools the gated flow years
2015–2019** — `first_marriage.18-29|female`, 7,832 at-risk person-years — which the
class touches only at 2019, through its female half: **157 person-years = 2.0 % by
rows, 2.25 % by weight** (all verified read-only). **Zero** of the
2,850 appear in any gated *event*; the remaining 2,569 have no gated footprint at
all (they reach a marital band only at the report-only shock years 2020+, or not
within the window). Two candidate resolutions, both design amendments:

| resolution | (i) frozen v3 floor file (`e931c886…`) | (ii) truth-side gated denominators | (iii) projected-side universe | (iv) report-only |
|---|---|---|---|---|
| **A — exclude-and-mark** (the §2.8.3a analogue): drop the gap class from the marital universe, mark non-scored | **moves → dies.** A *symmetric* exclusion must also drop the class's 157 person-years from the truth-side `first_marriage.18-29|female` denominator (2.0 %) → its floor σ + tolerance recompute (`0.356 → 0.355`) → **not byte-identical** | moves (symmetric) — disqualifying | 2,850 excluded | 2,850 |
| **B — seed-at-marital-entry** (**adopted**): for `anchor_wave < birth+START_AGE`, set `start_exposure_year = birth_year + START_AGE`, read the certified entry row there | **zero.** The builder is **projected-side only**; the truth side scores through `build_marital_panel` directly and is untouched → byte-identical | **zero** — the 281 stay in the frozen denominator exactly | +2,638 seeded (the 212 born-2008, `birth+15 = 2023 > 2022`, are dropped by the existing `start_exposure ≤ censor` filter); the class's 157 female person-years now **also** enter the projected `first_marriage.18-29|female` denominator, **symmetric** with truth | 2,357 |

**B is adopted; A dies on the frozen-floor constraint.** The v3 floor
(`runs/m6_holdout_floors_v3.json`, sha256 `e931c886…`) is ratified-frozen and its
`first_marriage.18-29|female` tolerance was derived on the full female,
2015–2019-pooled at-risk denominator. A's symmetric branch removes the class's 157
person-years (2.0 %) from it and re-derives the tolerance — the frozen tolerance
moves `0.356 → 0.355`, so the floor is **not** byte-identical, which the ratified
lock forbids. The magnitude is immaterial: **any** nonzero move breaks byte-identity.
A's only floor-preserving branch is *asymmetric* exclusion (drop the class from the
projected side only), but that scores a projected denominator missing the class
person-years the truth side still carries — a like-vs-unlike comparison the §2.8.4
identity guard rejects (`support.py:311-315`, which fires on real frames) and which
biases the `first_marriage` rate **upward** on the projection by construction
(dropping zero-event at-risk person-years raises the rate). The §2.8.2 pin's
own rationale — "a builder that seeded them would project them as `never_married`
at-risk … biasing the `first_marriage` denominator" — is the *mirror image* here:
these class person-years **do** carry truth-side `never_married`-at-risk exposure at 2019, so
*dropping* them from the projection alone biases the denominator the other way.
Symmetry, which the gate requires, is exactly what B delivers at zero floor cost.

*B's named leak, adjudicated empirically (not merely "structurally near-certain").*
B seeds the gap class `never_married` at `birth_year + START_AGE`, a wave inside the
holdout window (2016–2022 for the seeded 2,638). Does that seed carry holdout
information? It does not — it is the risk-set **entry** state, a structural
constant. Verified read-only over all 2,850: the certified entry `person_years` row
at `birth_year + START_AGE` is `never_married` for **every** one (0 exceptions),
with `marriage_duration` and `years_since_dissolution` null/zero. Of the 11 with
`n_marriages > 0`, **9** have a dated first-marriage episode; the earliest
first-marriage **event** in the class is age **18** (born-2004 `person_id 2852062`,
first marriage 2022), and the earliest `married` *person-year* is age **19** (the
year after, under `_assign_state`'s discrete-time `allow_exact_matches=False`
convention). Every such event is **strictly after** the age-15 seed — no one is at
marital risk, let alone married, before `START_AGE` — so each realized marriage is
scored out-of-sample from the seed exactly as the bulk's is. Because the entry state is identical (`never_married`) for every person
regardless of any downstream outcome, the seed encodes nothing about the holdout —
**B leaks nothing.** This is not an *assumed* `never_married`: B reads the certified
entry row, which the data confirm is uniformly `never_married` (the same
`pd.isna → never_married` entry the certified core already applies,
`marital.py:100-101`). That B reads the *certified* row, not an assumed state, is
load-bearing: the MH file *does* carry pre-`START_AGE` marriage episodes for other
persons — 23 land in `attrs ∩ anchor` and enter `married` at their `birth+15` — but
**none** is in this class (they are born 1928–1988); that MH data-quality point is
**out of scope for 3g**.

*Pins (B).*

- **The seed-wave clamp.** In `marital_panel_builder`, replace the unconditional
  override `start_exposure_year := anchor_wave` (`panel_builders.py:187`) with
  `start_exposure_year := max(anchor_wave, birth_year + START_AGE)`. For the bulk
  (`anchor_wave ≥ birth+START_AGE`) this is `anchor_wave`, **byte-unchanged**; for
  the gap class it is `birth_year + START_AGE`, the certified risk-set entry. The
  subsequent entry-row read (`:200-205`) then finds the row that exists **by
  construction** (the certified `person_years` start exactly there, and
  `_valid_persons` already guarantees `birth+START_AGE ≤ censor`), and the existing
  `start_exposure_year ≤ censor_year` filter (`:193`, censor clipped to 2022)
  naturally drops the 212 who would first reach 15 after the horizon.
- **Symmetric, floor-inert.** The clamp lives **only** in the projected-side
  builder; the truth side (`marital_tables → build_marital_panel`,
  `m6_cells.py:221-231`) is untouched, so the frozen v3 floor and every gated
  denominator are byte-identical. The 281 born-2001 enter the projected 2019 18-29
  at-risk person-years (the sex-pooled support; the 157 female of them feed the gated
  `first_marriage.18-29|female` cell), matching truth, so the §2.8.4 identity guard
  holds and they are scored like-for-like.
- **Carried inert like the bulk.** The clamped seed feeds the certified C16 core
  exactly as the bulk seed does — `never_married` entry, `n_marriages` carried-inert
  (§2.8.2), episode / change-point / duration rules unchanged (§2.8.2). The gap
  class introduces no new state and no code path beyond the `max(…)` clamp.

*What amendment 3g does NOT change.* The frozen floors are byte-identical
(`runs/m6_holdout_floors_v{1,2,3}.json` untouched; the v3 sha `e931c886…` remains
the gated contract, applied unchanged). No gated cell definition moves
(`MARITAL_BANDS`, `MARITAL_AT_RISK`, `GATED_FLOW_YEARS`; the 5 marital + 6 earnings
gated cells). The certified engine invariant is **preserved**: the raise
(`panel_builders.py:209-214`) stays — a marital projection genuinely cannot be
seeded for a person with no marital state at the seed wave; amendment 3g removes the
*cause* (the sub-`START_AGE` anchor override) so the invariant is satisfied by
construction, exactly as §2.8.3a satisfies `materialize_initial_frame` by
construction rather than weakening it. The fix is harness/universe-side: no
`gates.yaml` cell, no threshold, no floor, no certified-core edit.

*Implementing patch (pinned; lands after ratification, not in this PR).*

- **`engine/panel_builders.py` `marital_panel_builder`:** the one-line clamp above
  (`start_exposure_year = max(anchor_wave, birth_year + START_AGE)` in the certified
  dtype), replacing `:187`. `birth_year` is already on `attrs`
  (`transitions.py:239-243`); `START_AGE` imports from `data.transitions`. No other
  builder, and no certified-core file, changes.
- **Callers/tests touched.** `assembly.marital_step` (`:279`) and the runner path
  (`m6_runner.py` `_project_side → _projected_cells → _project_side`) are unchanged
  (they call the builder, whose signature and return are identical).
  `tests/test_m6_panel_builders.py` is extended, not rewritten.
- **The discriminating test (the class the current fixtures lack).**
  `test_m6_panel_builders.py`'s `_marital_source` fixture carries only born-1980–1982
  persons (`start_exposure_year` 1995–1997), all anchored well after age 15, so **no
  current fixture exercises the crash class**. Add a sub-`START_AGE`-at-anchor person
  — e.g. `birth_year = 2001`, `anchor_wave = 2015` (`birth+15 = 2016 > 2015`),
  certified `person_years` beginning 2016 `never_married` — and assert: (a) under the
  pre-patch override the builder raises `certified marital panel has no entry row at
  anchor`; (b) under the clamp the person is seeded at 2016 `never_married`, appears
  in `holdout_ids`, and the returned panel's entry row sits at 2016; (c) a born-2008
  anchor person (`birth+15 = 2023 > 2022`) is dropped by the censor filter, not
  seeded. This is the fixture class the current builder tests lack.
- **Schema-audit / manifest deltas: none.** The clamp reads only columns already on
  `attrs` (`birth_year`, `anchor_wave`, `censor_year`); no new panel column, no
  reader-schema change, so `m6_schema_audit` (§2.8.3f) needs no new contract row, and
  no `runs/` artifact or `gates.yaml` block is written by this amendment.

Docs-only design amendment (revision 13); the implementing patch above lands after
the referee round, as amendments 3d/3e did. Edits no `gates.yaml` cell, moves no
threshold, builds no floor, and writes no test in this PR.

**2.8.2h Amendment 3h — the fertility / open-additions roster-materialization
domain law (closes the reg-6 crash).** Step-4 fertility draws its maternal-birth
schedule from the **frame-independent** §2.8.2 marital at-risk universe
(`holdout_ids = state.marital_ids`, built **once** per draw with `del frame` and
realized censor, `assembly.py:278-293`) but **materializes** the scheduled births
onto the **frame-dependent** post-mortality roster (`materialize_maternal_births`,
`steps.py:381`; called from `apply_fertility`, `steps.py:472`, itself step 4 of the
§2.2 wave loop via `fertility_step`, `assembly.py:315`). The two universes are keyed
to two different notions of when a person exits — the schedule to the **realized**
marriage-history censor (clipped to `projection_end_year`), the roster to the
**simulated** wave-loop mortality (`apply_mortality`, `steps.py:113`;
`assembly.py:254`) — and nothing reconciles them. A woman whose simulated death
precedes her realized censor stays in the fertility schedule after the roster has
removed her; when the wave-t fertility RNG draws her a wave-t birth, the parent is
absent from the roster and the guard **raises** (`steps.py:412`,
`ValueError: birth parents are absent from the roster`). This is the sixth
registration's execution failure (forensics #42 comment 4984997277, deterministic,
one-shot unconsumed; graded #42 comment 4984699959).

The forensics pinned it to a **single** realized trip on the pinned seed-0/draw-0
person-side stream — `{782173}` at wave **2020** — reproduced byte-identically in
three independent reconstructions. Person 782173 is a realized **survivor** (female,
born 1988, anchored 2015, PSID censor 2022) whom the differential-mortality draw
kills in **simulated 2020**; her realized censor keeps her fertile in the schedule
through 2022; the wave-2020 fertility RNG draws her a 2020 birth two years after the
roster dropped her. Per-wave, waves 2015–2019 materialize 50 / 55 / 63 / 54 / 34
child rows with **zero** absent parents, and only wave 2020 (38 births, 1 absent
parent) raises. The forensics' hypothesis classification is decisive:
**(a) death-then-drawn is the sole mechanism, confirmed 100 %** — mortality is the
only roster-removing step (aging / marital / fertility / disability / earnings /
claiming / household never delete rows; entrants only add), and `valid_ids` derives
from an inner join *with* `anchor`, so the at-risk set is a strict **subset** of the
roster universe, which refutes **(c)** any id/join seam: an at-risk mother absent
from the roster is absent **because** simulated mortality removed her, never because
of an orphan id. **(b)** a non-mortality removal and **(d)** a 3g-seeding interaction
are likewise refuted (782173 is bulk, her 3g clamp `max(2015, 2003)` a no-op). She is
the first-manifesting member of a **structural susceptibility class** — at-risk
fertile women whose simulated death precedes their realized censor and who retain
≥1 fertile-age schedulable year while absent — which numbers **10** on the seed-0
person side (earliest-absent-fertile-schedulable year distributing
`{2015:1, 2016:1, 2017:1, 2019:2, 2020:1, 2021:2, 2022:2}`; the 2015 member, id
5459180 born 1966, would trip first on any draw that drew her a 2015 post-death
birth). Context: **456** persons die in the wave loop over the window on that side,
against **11,552** at-risk, and the schedule observes none of them; the guard is a
live tripwire over the full `5 seeds × 2 sides × 20 draws` ensemble, not a
782173-specific data artifact.

The pins already say where each universe belongs but no law separates them. §2.2
step 1 makes **"decedents leave the risk set for all subsequent year-`t` steps"**
and its DAG rationale is **"Deaths first so no dead person is married, made
disabled, or paid"** — of which *born of* is the unstated fourth. §2.8.9 pins the
at-risk universe **frame-independent for the scoring path** (correct — gated marital
/ earnings / disability score on the §2.8.2 builder's realized support, never the
live roster). The code reuses that one frame-independent universe for the **roster
materialization** step, which §2.2 requires to be frame-dependent. §2.8.2g corrected
the marital builder's *entry* universe; **3h separates the fertility builder's
*scoring* universe from its *materialization* universe**, which §2.2 already implies
and no §2.8 law states.

*The structural tell — this is the one materializing step among reconciling
peers.* Every other wave-loop module that consumes a frame-independent projection
reconciles it against the roster by a **left merge keyed on the roster**
(`_merge_period_columns`, `assembly.py:174-192`, `how="left"`): marital state
(step 3, merged at `fertility_step`, `assembly.py:309-314`), disability (step 5,
`assembly.py:372`), and household composition (step 8, `assembly.py:412`) all **drop**
a decedent's frame-independent row silently, because a left join keyed on the
surviving roster cannot attach a row for an absent person. Step-4 fertility is the
**sole** module that reconciles by **materialize-with-raise** (`pd.concat` of new
child rows behind a hard guard) rather than by drop-on-merge. 3h is not a new
posture — it brings the one materializing step into line with the merge steps'
already-correct **"reconcile, do not raise"** behavior. That parity is what makes 3h
a **law** over the wave loop, not a one-hole patch (the sibling sweep below
discharges the rest of the arm).

*The domain law.*

> Scheduled maternal births materialize **only for mothers present in the live
> post-mortality roster at the birth wave**. The **simulated-mortality universe**
> (the roster, §2.2 step 1) governs **materialization**; the **frame-independent
> at-risk schedule** (the §2.8.2 marital builder, realized presence/censor) governs
> **scoring**. A scheduled birth whose mother the wave loop has already removed does
> **not** materialize; it is recorded in the run's report-only reconciliation.

*The resolution options, adjudicated.* Three treatments for a scheduled birth whose
mother is roster-absent; all three leave the frozen floors byte-identical (open
additions are report-only, §2.1), so the decision turns on determinism and the
report-only realization:

| resolution | determinism (RNG stream) | report-only materialized births | scored / gated surface | verdict |
|---|---|---|---|---|
| **(i) drop-with-reconciliation, filter *after* the draw** (**adopted**) | **pure function of roster state**; the drop is post-draw so the fertility RNG stream is **byte-unchanged** — only the dropped rows differ | **down** by the post-death births (exactly **1** = `{782173}` on seed-0/draw-0; order ~1 birth/draw on this stream), **disclosed, report-only** | **none** — the roster is report-only (§2.1); scored moments read the frame-independent schedule (below) | **adopted** |
| (ii) re-draw a replacement mother/birth | **fails** — a re-draw consumes RNG and shifts every downstream per-person stream; it also **fabricates** a birth for a different mother | up (invented births), non-reproducible | none | rejected on determinism |
| (iii) keep-raising (status quo) | deterministic, but **aborts** the entire scored pre-artifact run | n/a (no artifact written) | none | rejected: a **report-only** mechanism must not abort a scored run |

**(i) is adopted.** The three proof obligations, discharged:

- **Determinism — the drop is a pure function of roster state, with no re-draw that
  shifts a stream.** (i) filters the drawn maternal frame against
  `set(frame["person_id"])` (the live post-mortality roster) **after** the
  `simulate_fertility` draw (`steps.py:469`) and **before** the materialize call
  (`steps.py:472`), dropping the roster-absent-parent rows. Because the draw is
  untouched, the fertility RNG consumption is **byte-identical** to the pre-patch
  run; the only difference is the set of rows concatenated into the roster. This is
  the forensics' explicit RNG-address decision: *filter after the draw* (RNG
  byte-unchanged) over *intersect the at-risk set before the draw* (which would
  change `simulate_maternal_births`'s RNG consumption — fewer at-risk rows — and
  shift the report-only realizations). Both are floor-inert; the post-draw filter is
  the byte-cheaper one and is adopted.
- **Report-only birth-count bias — down, quantified, disclosed.** Dropping the births
  of simulation-decedents biases the **materialized** birth count **down** relative
  to the certified frame-independent fertility moments (which continue to count every
  scheduled at-risk birth). The magnitude is the count of post-death fertile-year
  draws that land: **exactly 1** on seed-0/draw-0 (`{782173}`'s 2020 birth), order
  **~1 birth per draw** on this stream; across the ensemble it is bounded by the
  susceptibility class (10 women, seed-0 person side) against 456 wave-loop deaths.
  This is a **report-only** distortion of the open-additions roster, disclosed and
  netted in the reconciliation — it moves no gated cell (next bullet).
- **Certified-scoring invariance — the scored moments read the schedule, not the
  roster; pinned by code pointer.** The gated and certified surfaces score fertility
  as **distributional moments over self-contained frame-independent panels**, never
  over the materialized roster: `simulate_maternal_births` (`marital.py:288-352`)
  reads **only** the static `panel.attrs` (`start_exposure_year`, `censor_year`,
  `birth_year`, `sex`) across `[start_exposure_year, censor_year] ∩ fertile age` and
  **never** touches the live roster; candidate-16's internal `births` and
  candidate-9's household-composition conditioning fertility (`assembly.py:394`, a
  **separate** `generator(0, FERTILITY)` draw that materializes **no** roster) are
  the certified fertility lineage, whose `censor_year` **is** the person's survival.
  The certified deployment therefore has **no** open-population child roster and **no**
  independent mortality step, so the "parent absent from roster" state cannot arise —
  the invariant holds **vacuously** in certified use, and the roster is report-only in
  `gate_m6` (§2.1; §4.8 decision 4). **Pin: the drop touches only the report-only
  materialized roster and provably cannot reach any gated or certified surface** —
  the frozen floors (`runs/m6_holdout_floors_v{1,2,3}.json`, v3 sha `e931c886…`) and
  every gated cell are byte-identical.

*The sibling sweep (the arm-parity lesson).* 3h is a law only if **every** open-
addition / materialization path in the wave loop that could key a new or dependent
record to a roster-removable person is enumerated and dispositioned. The roster is
removed from by exactly one step (step-1 mortality); the sweep is over every step and
side-path that attaches records:

| wave-loop path | code | attaches by | keys to a roster-removable person? | disposition |
|---|---|---|---|---|
| 1 mortality | `apply_mortality` (`steps.py:113`; `assembly.py:254`) | **removes** survivors | — (the removing step) | the guard's premise; unchanged |
| 2 aging | `advance_age` (`steps.py:137`; `aging=advance_age`, `assembly.py:428`) | in-place age/calendar roll | no (no keying) | invariant N/A by construction |
| 3 marital core | `marital_step` (`assembly.py:271`) → `_merge_period_columns` at `fertility_step` (`assembly.py:309-314`) | **left merge** on roster | drops absent by construction | holds by construction |
| **4 fertility / roster** | `apply_fertility` → `materialize_maternal_births` (`steps.py:472, 381`) | **materialize** (concat + guard raise) | **yes — `parent_person_id` (mothers)** | **3h's law applies** |
| 5 disability | `disability_step` → `_merge_period_columns` (`assembly.py:326, 372`) | left merge on roster | drops absent by construction | holds by construction |
| 6 earnings | `earnings_step` → `apply_earnings` (`assembly.py:379`) | in-place earnings draw | no | invariant N/A by construction |
| 7 claiming | `claiming_step` → `apply_claiming` (`assembly.py:382`) | in-place claim-age draw | no | invariant N/A by construction |
| 8 household composition | `household_step` → `_merge_period_columns` (`assembly.py:385, 412`) | left merge on roster | drops absent by construction | holds by construction |
| paternal shadow births | `materialize_maternal_births` docstring (`steps.py:389-391`) | **not materialized** — household-conditioning draws, deliberately not duplicated into child rows | would-be `father_person_id` | holds by construction (no materialization) |
| candidate-9 conditioning fertility | `household_step` (`assembly.py:394`, `generator(0, FERTILITY)`) | feeds candidate-9 as conditioning; materializes **no** roster | — | holds by construction (separate draw, no roster attach) |
| child-ordinal assignment | wave-end loop (`loop.py:329-331`) | **additive** ordinal keyed to the **post-materialization** roster | keyed to roster-**present** persons only; `if person_id not in person_ordinals` | holds by construction — **rules out the secondary KeyError** |
| scheduled realized openers (loop-native entrants) | wave-loop entrant concat (`loop.py:192-254`): frames validated `:198-211`, concatenated at each period top **before** step-1 mortality (`pd.concat`, `:252`) | **materialize** fresh realized entrant person-rows (the 2017/2019 opener cohorts, 246 + 36; frames from `m6_population.projection_metadata`, `m6_population.py:65-73`; side-split `:97-103`; counted `scheduled_realized_openers`, `m6_runner.py:916-919`) | no — fresh **realized** ids keyed to no roster-removable person | holds by construction — the overlap guard raises on any collision (`loop.py:215-219`, `:244-250`); every scheduled id is ordinal-seeded pre-loop (`loop.py:214`, `:234-237`) so no `KeyError` shape arises; entrants carry a realized year, never a birth drawn against a decedent |
| immigrant entry cohorts (§2.1) | would ride the **same** `SCHEDULED_ENTRIES_KEY` entrant mechanism (not `PeriodModules`, `assembly.py:426-436`, which carries no entrant step) | **dormant** — nothing schedules immigrant frames; the entrants report hardcodes `immigrant_cohorts: 0` (`m6_runner.py:914`) | no | out of scope (unwired; if wired it would ride the openers' benign-by-construction path — fresh ids keyed to no removable person) |
| synthetic id allocation | `synthetic_id_allocator.allocate` call (`steps.py:418`; allocator `loop.py:49`) | fresh child ids | no | holds by construction |

The **child-ordinal** row is load-bearing and answers the forensics' explicit
question of why the materialized children do **not** raise a *second* KeyError. RNG
for a person is drawn through `person_generator`, which **raises**
`KeyError: no stable RNG ordinal for person …` for any id absent from
`person_ordinals` (`loop.py:99-104`). Newborns are materialized at step 4 but the
wave-end loop assigns an ordinal to **every** id in the surviving roster
(`loop.py:329-331`) **after** all eight steps, so by the next wave's step-1 mortality
every newborn already has an ordinal; the assignment is additive (`if … not in`) and
keyed to the post-materialization roster, so it can never key to a removed person.
Verified end-to-end by the forensics' full real-engine run (waves 2015–2019
materialize and ordinal children with no defect; only wave 2020 raises the fertility
guard). The **household** rows note one residual **report-only** consistency
question the sweep surfaces but 3h does **not** reach: candidate-9 may compute a
`coresident_spouse` flag for a living person from the **frame-independent** marital
state even when the spouse was roster-removed by mortality — but that is a
report-only **column value** on a roster-present person (merged, never materialized,
never raised), analogous to the birth-count bias, not a guard violation; it is out of
scope for 3h's guard-closure and flagged for the ceremony's report-only ledger.

*Floor-inertia proof obligations.* Open additions are **report-only** by §2.1:

> the **open additions** — synthetic births (from the fertility/roster module) and
> immigrant entry cohorts … have no PSID ground truth over the holdout window and are
> therefore **report-only in `gate_m6`** (family B, §4.8, decision 4). — §2.1

Because no gated cell reads the live materialized roster, **all frozen floors are
byte-identical by construction** — `runs/m6_holdout_floors_v{1,2,3}.json` untouched,
the v3 sha `e931c886…` remains the gated contract applied unchanged — and the
adopted filter-after-draw reconciliation additionally leaves the fertility **RNG
stream** byte-identical, so even the report-only draw realizations of every
**non-dropped** birth are unchanged. **The guard at `steps.py:412` stays.** Under the
adopted placement the reconciliation filter lives in the open-additions **caller**
(`apply_fertility`), so `materialize_maternal_births` receives only roster-present
parents and its guard becomes **unreachable for the lawful path** — but it is
**retained** as the invariant backstop for any caller that passes an un-reconciled
birth frame (a genuine id/join defect, hypothesis (c), which the forensics refuted
for the M6 at-risk universe but which the primitive must still catch). Cause removed,
invariant satisfied by construction, guard kept — exactly the §2.8.2g / §2.8.3a
posture, and the cleanest of the three because the surface is entirely report-only.

*Implementing patch (pinned; lands after ratification, not in this PR).*

- **`engine/steps.py` `apply_fertility`:** between the draw (`simulate_fertility`,
  `:469`) and the materialize call (`:472`), filter `draws.maternal` to rows whose
  `parent_person_id` is in `set(frame["person_id"])` (the live post-mortality
  roster), and record the dropped rows in the report-only reconciliation. The draw is
  untouched, so the RNG stream is byte-unchanged (the adopted post-draw point). No
  change to the frame-independent `simulate_maternal_births` schedule
  (`marital.py:288-352`), no change to `holdout_ids`/`state.marital_ids`
  (`assembly.py:321`). `apply_mortality` and the guard are unchanged.
- **The reconciliation record shape (report-only).** A per-wave report-only field
  keyed by projection year — e.g. `roster_absent_births[context.year] =
  {"dropped_parent_ids": frozenset(absent), "dropped_count": len(absent)}` —
  published on the same report-only open-additions channel §4.8 decision 4 already
  carries (alongside the existing `birth_store[context.year] = draws`,
  `steps.py:470`). No gated cell, no `runs/` artifact, no `gates.yaml` block reads it.
- **The discriminating test (the class the current fixtures lack).**
  `tests/test_m6_engine_steps.py`'s materialize fixtures use `parent_person_id =
  [10, 10]` with parent 10 **present** in the roster frame (`:203, :241`), so **no
  current fixture exercises a roster-absent mother**. Add a dead-mother-scheduled-
  birth fixture — a `births` frame with a `birth_year == context.year` row whose
  `parent_person_id` is **not** in the roster `frame` (the mortality-removed mother) —
  asserting: (a) under the pre-patch caller `materialize_maternal_births` raises
  `birth parents are absent from the roster`; (b) under the patch `apply_fertility`
  **drops** that birth, records it in the reconciliation, and materializes only the
  roster-present-parent children; (c) the `simulate_fertility` RNG draw is
  **byte-identical** to the same draw against a roster where the mother is present
  (filter-after-draw invariance). **Real-frame proof target:** the **10-person
  seed-0 susceptibility class** — post-patch, a full-window projection of the person
  side must **complete** to 2022 with the drops recorded and the counts matching the
  forensics (the singleton `{782173}` dropped at wave 2020, the class's earliest-
  absent-fertile-schedulable distribution above). This real-frame proof runs in the
  post-ratification patch lane (reusing the run-6 venv), **not** in this docs PR.
- **Schema-audit / manifest deltas: none for 3h.** The filter reads only
  `person_id` (already the roster key) and the drawn `parent_person_id`; it adds no
  panel column and no reader-schema change, so `m6_schema_audit` (§2.8.3f) needs no
  new contract row for fertility. **Erratum ridden here (PR #210 referee NOTE-1,
  the 3g-implementation referee's deferred one-liner):** the `m6_schema_audit`
  `marital_panel_builder` read-set (`m6_schema_audit.py:266-268`) lists
  `marital.attrs = {person_id, censor_year, start_exposure_year, weight}` and
  **omits `birth_year`**, which the ratified 3g clamp
  `max(anchor_wave, birth_year + START_AGE)` (`panel_builders.py:187`) reads
  (`birth_year` is on `attrs`, `transitions.py:239-243`). The 3g-patch lane must add
  `birth_year` to that frozenset; recording the erratum here per the referee's
  request. This is a manifest correction to the **3g** patch, carried as a note — 3h
  itself edits no manifest.

*What amendment 3h does NOT change.* The frozen floors are byte-identical
(`runs/m6_holdout_floors_v{1,2,3}.json` untouched; v3 sha `e931c886…` the unchanged
gated contract). No gated cell definition moves; no threshold, no floor, no
`gates.yaml` cell, no certified-core edit. The frame-independent fertility **schedule**
and its **scored** moments are unchanged — 3h touches only which scheduled births
**materialize** onto the report-only roster. The certified fertility lineage
(candidate-16 `births`, candidate-9 conditioning) is untouched; the guard
(`steps.py:412`) stays; the §2.2 order of operations is unchanged (3h makes step 4
honor the step-1 decedent exit §2.2 already mandates). The related report-only
household-diagnostic consistency question (a `coresident_spouse` flag against a
mortality-removed spouse) is **flagged, not fixed** — out of scope for the guard
closure.

Docs-only design amendment (revision 14); the implementing patch above lands after
the referee round, as amendments 3d/3e/3g did. Edits no `gates.yaml` cell, moves no
threshold, builds no floor, and writes no test in this PR.

**2.8.3 The year-0 slice from the realized 2015-interview state.** The seed slice
mirrors the floor's realized panel **exactly**, so projection and truth condition
identically. `T* = 2014` realizes its state at the **2015 interview**
(`SEED_WAVE = 2015`, reference year 2014;
`build_m6_holdout_floors.py:15-16,58-59`). Pins:

- **Universe / `holdout_ids`** = every person present in a **gated start wave**
  (`GATED_START_WAVES = (2015, 2017, 2019)`) with positive weight — the closed
  panel (§4.8), built by `build_anchor_frame` (`build_m6_holdout_floors.py:153-171`).
  The bulk seed at 2015; persons whose earliest gated presence is 2017/2019 are
  **presence-conditioned openers** whose gated intervals open at their anchor wave,
  handled symmetrically on both sides by the §2.8.4 presence sets (not a separate
  seed population).
- **Per-field seed reads (PSID source → column), each the source the floor reads:**
  `panels.demographic_panel` → person-year presence, `age`, `weight`,
  `interview` (the `household_id` = family interview number; person `sex` is
  **not** a demographic-panel column — it is sourced from `data.deaths`, the
  same reader named for the mortality slices below, see amendment 3f §2.8.3f);
  `marriage.marriage_
  history` → `transitions.build_marital_panel` → `marital_state`,
  `marriage_duration`, `years_since_dissolution`, `n_marriages` at the seed wave;
  `household_composition.build_household_panel` → the `coresident_*` / `multigen` /
  `cohabiting` seed; `data.disability` reader → `disabled` / `retired` seed;
  `family.family_earnings_panel` → realized 2014 anchor earnings **and** realized
  2012 (`t−4` start-lag) and the `u_w` latent-permanent rank the forward earnings
  chain needs (§2.7.6.5; materialized at period 0 by `refit.py`); `data.deaths` →
  `sex`, `death_year` for the mortality slices.
- **F6 weight and `household_id`** = the person's cross-sectional PSID weight and
  family interview number at their realized **start-of-holdout (anchor) wave**,
  held **fixed** across the window (`build_anchor_frame`;
  `StartWaveWeightSnapshot`, §4.7). Never the per-year calibrated weight on the
  gated rates.
- **Presence-conditioning set — flows** = `presence_by_wave` (waves 2013/2015/2017/
  2019/2021, sequence 1–20, positive weight; `build_m6_holdout_floors.py:174-181`),
  applied to projection, truth, and floor **symmetrically** by
  `support.prepare_evaluation_support` with `PresenceBasis.START_OF_INTERVAL` (the
  interval's opening biennial interview), which additionally enforces
  **identical** projection/truth person-period support and applies the F6 snapshot
  to all three sides. This basis keys the marital and disability flow intervals.
- **Support basis — earnings** = the realized `family_earnings_panel`'s own
  **person-period row existence** at the reference years `{2016, 2018}` (present,
  positive-weight, valid-earnings; `earnings_frame`,
  `build_m6_holdout_floors.py:354-378`), an **`EXACT_WAVE`-class** basis on the
  even reference years — **not** the odd-wave `presence_by_wave` flow sets (whose
  keys cannot even address the even reference years). The gated earnings cells are
  scored on this realized support **intersected with the earnings domain**
  (§2.8.3a — the 2014-anchored closed-panel population the certified chain can
  project; §2.8.4, finding 1); it is the earnings analogue of the disability
  reproduction support.

**2.8.3a The year-0 earnings-domain law (closes Sol round-4 blocker).** The
closed-panel universe (§2.8.3) includes persons for whom the certified
forward-earnings generator has **no** year-0 state, and its initializer requires
that state for **every** initial person — a build-stopping contradiction the round-4
build lane correctly refused to improvise (`~/PolicyEngine/sol-worktrees/
m6-harness-REPORT.md`). `family_earnings_panel` emits a reference-year row only for
a **family head or spouse** present in that collection wave (`data/family.py:
785-857`), so non-head/spouse anchor members (children, other relatives) have **no**
earnings row at all, and a person whose first gated presence is 2017/2019 has **no**
2014 row (it comes only from the 2015 interview). `fit_forward_earnings` restricts
`u_w_by_person` / `realized_earn_2014_by_person` to persons with a realized 2014
anchor (`engine/forward_earnings.py:725-736`), and `ForwardEarningsGenerator.
materialize_initial_frame` **raises** for any initial person absent from both maps,
**before** any age rule can exclude anyone (`forward_earnings.py:882-896`).

*Adjudicated (from the ratified F1 + §4.8 closed-panel principles): no synthetic
initialization law is needed or permitted — the excluded people are the earnings
open-additions, report-only by §4.8, not a gated population.* Pins:

- **The earnings-module domain = the realized-state (closed-panel) population** =
  persons with the realized 2014 head/spouse earnings anchor + `u_w` that
  `materialize_initial_frame` requires (`realized_earn_2014_by_person ∩
  u_w_by_person`). This **is** the earnings closed panel — the 2014 realized-earnings
  cross-section (§2.1, §4.8) — the only population the 2014-anchored chain can
  project.
- **Non-domain persons carry an explicit non-scored marker.** Pin the frame column
  `earnings_domain: bool` (True iff the person is in the domain). For
  `earnings_domain == False` the wrapper writes `earnings = 0.0` and leaves the
  chain-state columns (`u_w`, `realized_earn_2014/2012`, `gen_earn_w2/w4`) absent,
  and their per-year seam / report-only earnings follow the pinned **non-scored
  zero** rule (the pe-us W2 seam already tolerates zeros, §5; the marker and the
  count publish). No fallback earnings law, no synthetic `u_w`, no backfilled 2014
  anchor is introduced. **Report-only distortion, disclosed (no gated-cell
  impact):** the ~21 % later earnings-entrants (§2.8.3a floor bullet) are **real
  earners** — they have realized 2016/2018 income — carried at `0.0` throughout, so
  their report-only **AIME / PIA / claiming levels (the §5 W2 seam) are understated**
  and the family-B report-only benefit tier must **not** be read at face value for
  the marked population. This touches **no gated `gate_m6` cell** (AIME, PIA, and
  claiming are report-only, never gated); the marker lets the report-only tier net
  them out or annotate them.
- **The domain filter lives in the earnings-module wrapper, not the certified
  generator — a per-person check at both entry points.** The `EarningsGenerator`
  adapter intercepts the two places the certified earnings state is touched: (a)
  `initialize` → `materialize_initial_frame` (`assembly.py:198-202`), where the
  wrapper materializes the state for the **in-domain rows only** and re-assembles
  the full frame with the out-of-domain rows carrying the marker + `earnings = 0.0`
  and their chain-state columns absent; and (b) `apply_earnings`, whose registry
  path already calls `generate` **per person on a single-row frame**
  (`steps.py:213-232`), so the domain filter is a **per-person check** (skip the
  out-of-domain person, write the non-scored zero), **not** a batch-subframe
  rewrite. The build lane reads it as a per-person predicate at (b) and a
  subframe-materialize-plus-reassembly at (a); both are mechanical, within the
  pinned wrapper remit (R1). The certified generator therefore only ever sees
  in-domain persons and its `materialize_initial_frame` requirement is satisfied by
  construction — the certified stochastic law is untouched.
- **The anchor/later-opener rule (F4-symmetric, earnings-specific).** The certified
  chain is **2014-anchored** (`fit_forward_earnings` anchors on `period ==
  boundary_year == 2014`), so — unlike the marital/household seeds, whose anchor
  wave varies 2015/2017/2019 — the earnings-domain wave is **uniformly the 2015
  interview** (the only interview producing a 2014 reference year). There is **no**
  2017/2019 earnings-domain entry: a later opener has realized 2016/2018 but no
  realized 2014, so it is a **later earnings-entrant = an open addition (§4.8, §2.1)
  → report-only**, carrying the non-scored earnings marker throughout. This is
  symmetric with the histories' rule in the precise sense that each module seeds at
  the wave producing *its* required anchor state; for the 2014-anchored chain that
  wave is fixed. **Per-module open-addition, distinguished from family B:** these
  people are **closed-panel** members — scored for the marital / household /
  disability **flows** — reclassified as open-additions **for earnings only**
  (each module scores the realizable domain of its own certified law). A person can
  therefore be closed-panel for flows and an open-addition for earnings; this is a
  per-module reframe of the §4.8 open-addition concept and is **not** the family-B
  synthetic-birth / immigrant-entrant population (§2.1, §4.8), which has no PSID
  ground truth for **any** module.
- **Gated-earnings scoring support = realized support ∩ the earnings domain,
  symmetric on both sides.** The forward chain is evaluated (per F1) for every
  realized-present **in-domain** person-period regardless of simulated death; the
  truth side (`rate_a`) is restricted to the **same** domain, so projection and
  truth carry identical support and the §2.8.4 identity guard holds. Later
  earnings-entrants are excluded from **both** sides (report-only open additions),
  not scored survivor-/entrant-conditioned.
- **Floor consistency — the empirical delta (replaces the "holds by construction"
  expectation).** Verified against the staged PSID through the frozen floor
  machinery: of the frozen v3 floor's gated-earnings support (**13,163** persons at
  `{2016, 2018}`, prime+older), **2,722 (~21 %: 2,199 prime, 590 older)** are later
  earnings-entrants **outside** the 2014-anchored domain — the frozen floor
  **over-included open-additions** in the gated earnings cells (a lineage artifact
  of `build_anchor_frame`'s multi-wave anchor merge; the flows presence-condition
  those openers at their anchor, but the 2014-anchored chain cannot project them).
  So point-(6)'s "no gated earnings cell's realized support includes any excluded
  person" is **empirically false** for the two level cells and the change cells; it
  **does** hold for non-head/spouse members (no earnings row → in no earnings cell).
  The consequence is **conservative**, not a leak: scoring the closed-panel domain
  against the frozen full-support tolerance faces ~`√(N/N_domain)` inflated
  half-split noise (~1.17× prime, ~1.06× older), so the effective earnings
  tolerances are **tighter** than the closed-panel warrants (harder to pass, no
  false-PASS risk). Bind it with a **harness self-check**: recompute the
  domain-restricted (closed-panel) earnings floor half-splits and publish the
  tolerance / earnings-OC delta vs the frozen v3 floor (report-only — the frozen
  tolerances remain the gated contract, applied conservatively). The escalation is
  **two-directional**, honoring §4.9's both-directions weak-power check: route to
  the **floors-ceremony finding** rule (adjudication-7 / §4.9 — a ceremony finding,
  never a silent redesign; a possible earnings-floor re-derivation on the
  closed-panel domain) if **either** (a) the recomputed closed-panel earnings OC
  falls **below the 0.90** weak-power floor (near-**unpassable** — the modal
  direction here, since the frozen tolerances were non-vacuity-certified on the
  full support and the domain half-split noise is almost-certainly larger), **or**
  (b) the domain-restricted surface trips the §4.9 **vacuity** guard in the other
  direction (near-**tautological**: any domain-restricted gated tolerance lands at
  the `ln(1.5)` cap, or the domain earnings OC approaches 1 with a near-unfailable
  cell). Both directions publish; neither is silently absorbed.

**2.8.3f Amendment 3f — the demographic-seed sex source correction (closes the
third-registration crash-2).** §2.8.3's per-field list above erroneously attributed
`sex` to `panels.demographic_panel`. The certified `demographic_panel` schema is
`{person_id, period, age, sequence, relationship, weight, interview}` (`panels.py:252-253`)
— **seven columns, no `sex`** — so `build_realized_population` reading `sex` from that
frame via `_anchor_rows(columns=("age","sex","interview"))` (`m6_population.py:186`) raised
`ValueError: anchor source is missing columns ['sex']` (`m6_population.py:139`) at the
refit phase (`build_realized_population`, `m6_runner.py:355`) on the **first** real-frame
execution — the second pre-scoring crash of the third registration (registered 4971244215,
graded #42 comment 4972045579). Person sex is person-constant `ER32000` from the PSID
cross-year individual file, read by `data.deaths.read_death_records` — the **same** source
§2.8.3 already names for the mortality slices, and the same canonical attach the certified
builders use (`household_composition.join_demographics`, `disability.attach_sex`).
*Resolution, PINNED:* `build_realized_population` takes the `death_records` frame and joins
person sex by `person_id` before the demographic seed, validating one coded value per person
(no conflicts) and full coded-sex coverage over the anchor persons (raise on any anchor
person with no `male`/`female` sex). The demographic panel's seed reads are `{person_id,
period, age, interview}`; `sex` moves to the `death_records` read. A static full-phase column
audit (`m6_schema_audit`, `tests/test_m6_schema_audit*.py`) now pins every phase's real-frame
column reads against the loaders' committed and real schemas, and the fixture over this path
carries the real seven-column demographic schema plus a real-schema death-records sibling, so
a frame that cannot supply a read fails a test rather than the run. Docs-and-patch; edits no
`gates.yaml` cell, moves no threshold, builds no floor.

**2.8.4 The M6 drift-scoring layer.** The scorer computes, per gated cell,
`|ln(rbar_projected / rate_a)|` against the frozen **v3** floor tolerance
(`runs/m6_holdout_floors_v3.json`, sha256 `e931c886…`), reusing the floor cell
machinery **verbatim**:

- **Truth-side cell functions reused as the single source** (named): the frame
  builders `build_anchor_frame`, `presence_by_wave`, `mortality_slices`,
  `marital_tables`, `disability_pairs`, `earnings_frame`; the cell functions
  `mortality_cells`, `marital_cells`, `disability_cells`, `earnings_cells`
  (with `MARITAL_AT_RISK`, `EARN_COHORTS`, `band_of`); the primitives `_rate`,
  `_wquantile`, `_score`; the change/mobility/autocorr moments in
  `harness.moments`; the split `harness.panel.split_panel_by_person`; the tolerance
  `evaluation.derive_tolerance`. These live in `scripts/build_m6_holdout_floors.py`
  today; the ceremony **extracts them to one importable module** that both the
  floor script and the scorer import, so byte-identity is by construction. If any
  cell value is instead re-implemented, a **harness self-test** computes the
  realized `rate_a` on the same realized side-A via **both** the floor function and
  the scorer's reduction and asserts **byte-identity**, failing the run otherwise
  (the design admits no drift between the two reductions). This is a **cell-function
  identity** on a *given* support; the earnings **domain support-restriction**
  (§2.8.3a — realized ∩ the 2014-anchored closed panel) is a documented support
  filter applied on top of the identical function, with its own recompute self-check
  (§2.8.3a), not a change to the function.
- **The projected side** reduces each K-draw to the **same**
  presence-conditioned, F6-weighted, windowed long frames the truth side uses, then
  applies the **same** cell functions. The **flow** families reduce over realized
  support by construction: projected marital events + at-risk person-years come from
  the frame-independent §2.8.2 marital builder (realized presence/censor), and
  disability pairs from the realized-support reproduction (`assembly.py:257-304`).
  Earnings is scored on the realized earnings support per the finding-1 pin below
  — **not** the survivor-attrited live-frame `earnings` column. `rbar_projected` is
  the **K = 20**-draw mean (`numpy.random.default_rng(5200 + k)`, §3.1) of the
  projected rate; the odd-year earnings carry-forward and the biennial
  `{2016, 2018}` scored draws are as pinned in §2.7.2/§2.7.6.
- **The gated earnings cells score on the realized support, not the live projected
  frame (adjudicated — the one gated family §2.8 must resolve against certified
  step-1 mortality).** Step-1 mortality (`apply_mortality`, `steps.py:108-130`,
  "return only the period's survivors") removes simulated deaths from every later
  frame slice, so the projected frame's `{2016, 2018}` earnings person-periods are a
  **strict subset** of the realized `earnings_frame` support on essentially every
  draw. Scoring that survivor-conditioned subset against all-present realized truth
  would either **abort every draw** under the §2.8.3 identity guard
  (`support.py:311-315`) or silently score **survivor-conditioned** earnings —
  admitting the `not_certified` mortality drift (§4.10) into six gated cells through
  survivorship, with a systematic mortality–earnings direction on `earn_p10.prime`,
  `earn_zero_rate.older`, et al. **Adjudicated**, by the same symmetric-conditioning
  principle §2.8.1 applies to histories and §4.4's cancellation logic ("the same
  wave-presence selection sits on both the projected and the realized rate"), and by
  §4.7's own framing ("the `gate_m4` construction with the time axis substituted",
  whose reproduction precedent scores on **realized** support): the gated earnings
  cells are a **reproduction read on the realized earnings support** — the realized
  `family_earnings_panel` row existence at `{2016, 2018}` (the §2.8.3 `EXACT_WAVE`
  earnings basis) **intersected with the §2.8.3a earnings domain** (the 2014-anchored
  closed panel; later earnings-entrants are report-only open additions, excluded
  symmetrically from both sides) defines the scored person-periods, and the forward
  chain is evaluated for **every realized-present in-domain person-period regardless
  of simulated death**. This is **bit-safe**: `apply_earnings` draws on **person-keyed**
  `EARNINGS` streams (`context.person_generator(EARNINGS, person_id)`,
  `steps.py:222-233`), so evaluating a realized-present-but-simulated-dead person's
  chain consumes **that person's own** entropy and perturbs no survivor's bits —
  the gated earnings reduction is independent of projected mortality's RNG
  consumption. Projected step-1 mortality still runs and still feeds the report-only
  diagnostics and the engine's internal dynamics (AIME / claiming / household
  roster); it does **not** enter the gated earnings read. **What this conditions
  on:** realized reference-year presence (positive-weight, valid-earnings) — not a
  modeled engine outcome, so conditioning-not-leakage, on the same footing as the
  flows' presence and age/sex. **What it does not score:** mortality's effect on the
  earnings *composition* (the survivorship channel) — that is the reproduction
  test's designed scope, identical to the disability and marital flows, and
  mortality drift stays `not_certified` (§4.10).
- **Gate-seed construction (the `gate_m4` construction, time axis substituted;
  §4.7).** For each gate seed `s ∈ {0,1,2,3,4}`, `split_panel_by_person` on the
  cell's **correlation-respecting unit** (the block's `split_units`: **household**
  for marital/mortality, **person** for disability/earnings) with `fraction = 0.5`,
  `seed = s` selects **side-A**; `rate_a` is side-A's realized presence-conditioned
  rate, `rbar_projected` is the K = 20-draw mean of the engine projecting **side-A**
  persons from their realized `T*` state; the score is `_score`'s
  `|ln(rbar/rate_a)|` (log-ratio cells) or `|rbar − rate_a|` (`abs_gap_log` /
  `abs_gap_corr` cells). Scoring on a half matches the tolerance's half-split
  sampling-noise scale.
- **Conjunction and pass rule, read from the locked block (invented nowhere).**
  A cell clears at seed `s` iff its score `≤` the block tolerance; a **seed
  passes** iff **every** gated cell clears at that seed; **family A passes** iff
  `≥ 4 of 5` gate seeds pass — the seed-level 4-of-5 conjunction the block pins
  (`gate_seeds [0..4]`, `conjunction: 4 of 5 seeds`) and the standing
  `gate_2a`/`gate_m4` semantics (`gates.yaml` gate_m4 "seed s passes iff every
  gate-eligible cell passes … the gate passes iff ≥4 of 5"). The 11 gated cells,
  their `split_unit`, `metric`, `k` (FLOW `k=3`), and tolerances are read from the
  locked `gate_m6` `views` block (§4.10 v3 registry); the harness computes no
  tolerance and moves no threshold.
- **Undefined-draw and regeneration guards.** An undefined gated rate on **any**
  draw invalidates the run (`_score` returns `None`; `undefined_draw_rule`, §4.8).
  The artifact records the **across-draw dispersion** (`max_across_draw_sd`,
  `max_per_draw_abs_ln`) and asserts a **regenerated surface** — non-zero
  across-draw dispersion proving the projection *regenerates* every scored column
  rather than passing a degenerate identity (the `conformance` guard prior scored
  runs record).

**2.8.5 Pre-flight 1 — the candidate-9 re-certification margin (real-data,
holdout-blind).** Before any scored phase, the harness runs the §2.6 targeted
transfer check on the **`≤2014`-refit** household panel over the gate-seed draws:
for each `k`, `composition.simulate_candidate9_injected` (injected whole-window
step-3 marital) vs `composition.simulate_candidate9_internal_reference` (the frozen
internal `ft.simulate`), reduced to the pre-named channel moments
(`composition.composition_channel_moments`; `RECERTIFICATION_CHANNEL_SETS`:
cohabitation, legal-spouse-residual, occupancy, household-size), checked by
`composition.check_candidate9_recertification` at the **≥3σ** `gate_m4`-style
margin. The injected arm supplies the §2.8.2-pinned household-conditioning
fertility (`steps.simulate_fertility` drawn on `registry.generator(0, FERTILITY)`,
mirroring the certified `assembly.py` household step) into
`simulate_candidate9_injected`, so it carries the same maternal-birth line the
internal reference generates inline through `ft.simulate`; the two arms differ
**only** in marital-state provenance, not in the presence of fertility. This is
**candidate-blind**: it compares two *simulation* paths on the
fitted panel and reads **no holdout cell**. A failure is a **designed abort** —
`check_candidate9_recertification` raises, the run stops pre-scoring, the one-shot
is **not** consumed, and the fuller re-ceremony §2.6 names is triggered (not a
self-rescue). The per-channel margins publish with the run (registration
observation 3).

**2.8.6 Pre-flight 2 — the certified sign-path verification.** The harness
verifies that the forward earnings generator deploys the **certified
externally-driven** sign path — the `_target_models` reconstruction inside
`forward_earnings._gate_sign_draw` (`forward_earnings.py:820-826`), which
reproduces `populace.fit.qrf.FittedRegimeGatedQRF._gate_draw` (`qrf.py:731`)
bit-for-bit on engine-supplied uniforms and is the branch every certified
candidate-10 `RegimeGatedQRF` sign gate actually executes — rather than the
`hasattr(fitted, "draw_sign")` fast-path (`forward_earnings.py:815-819`), which
is a **test seam** reached only by injected doubles
(`tests/test_m6_preflight.py:92`, `tests/test_m6_engine_forward_earnings.py:33`).
Today's certified `FittedRegimeGatedQRF` exposes `_target_models` and defines no
`draw_sign` method (`qrf.py:612`), so on every real fit `_gate_sign_draw` falls
through to the reconstruction and the `draw_sign` branch never fires outside
tests. The harness runs the participation gate on a **synthetic probe** frame,
records **which branch executed** into the run artifact, and — keeping the
designed-abort semantics — aborts the run if a gate deploys the `draw_sign` test
seam instead of the certified `_target_models` reconstruction. No holdout
contact (synthetic probe only). Should populace-fit ever expose a certified
`draw_sign` on `RegimeGatedQRF`, this pin is revisited so verification still
confirms the deployed path is the certified externally-driven draw (registration
observation 6).

**Reconciliation with observation 6 and harness-referee F1.** Engine-referee
observation 6 (PR #173 comment 4962620806) reads: "`_gate_sign_draw` prefers a
`draw_sign` method when the fitted gate exposes one (test seam; the certified
reconstruction is the fallback and is what production `RegimeGatedQRF` objects
will hit today)." That is correct — `draw_sign` is the seam and the
`_target_models` reconstruction is the certified path. The prior wording of this
section and of the §10 `preflight_2` field **inverted** it, pinning pre-flight 2
to verify the `draw_sign` branch and to reject `_target_models` as a "test-seam
fallback"; taken literally that aborts every real run, because the certified gate
exposes no `draw_sign`. Harness-referee finding F1 (PR #185 comment 4966859161)
proved the abort — `verify_external_sign_path` raises against the real
`_target_models`-only shape, so the registered scored run would stop at phase 3
today. This amendment restores observation 6's direction — certified branch
`_target_models`, seam `draw_sign` — and leaves the designed-abort intact, now
firing on the seam rather than on the certified path. Observation 6 was **not**
wrong about which branch is the seam; the earlier §2.8.6/§10 prose transcribed it
backwards.

**2.8.7 The runner phase structure.** One ordered pass, each phase a pure function
of the prior:

1. **Refit** — `refit.refit_m6_components(M6RefitInputs, boundary_year=2014)`
   returns the `M6RefitBundle` (family/household/earnings/modifier/disability/
   claiming/mortality, each a truncated-panel fit of the frozen spec, §4.2);
   `CertifiedEngineInputs.from_refit_bundle(...)` binds it with the real §2.8.2
   builders, the realized `disability_panel`, and the §2.8.3
   `StartWaveWeightSnapshot`. Record the refit provenance — registry
   `spec_sha256`s, `boundary_year`, `EARNINGS_SPEC_SHA256`, and that certified
   full-window artifacts were neither read nor written (`RefitProvenance`).
2. **Pre-flight 1** (§2.8.5) — abort-on-fail, margins published.
3. **Pre-flight 2** (§2.8.6) — sign-path recorded.
4. **Project + score, per gate seed** — for `s ∈ {0..4}`, project side-A through
   `ProjectionEngine.project(initial_slice, end_year=2022, draw_index=k)` for
   `k = 0..19`, reduce to cells, score against the v3 floor (§2.8.4); collect the
   per-seed `{seed, n_side_a_units, seed_pass, n_cells_pass/fail,
   undefined_draw_cells, worst_cells[]}` and the family-A verdict.
5. **Report-only** — the shock-window diagnostics, the `not_certified` surface
   (mortality drift first, §4.10), the re-drawn-`T*`-seed comparison (decision 5b),
   entrants (family B), and the alignment displacement — all published, none gated.
6. **Assemble + write** — one artifact `runs/gate_m6_candidate<N>_v1.json` via
   `artifacts.write_new(..., sidecar=True)` (exclusive-create enforces the
   one-shot; the `.env.json` sidecar carries `environment_block()` + the contract
   ref). The artifact **stamps** the run's #42 **registration-id** and the
   `refit.EARNINGS_SPEC_REGISTRATION` lineage constant (`refit.py:70`), the floor
   `sha256` (`e931c886…`), the resolved spec `sha256`s, and `publishes_regardless`
   — the schema prior scored runs use (`schema_version`, `run`, `candidate`,
   `registration`, `lineage`, `gate`, `protocol`, `verdict`, `family_a`,
   `family_b`, `family_c`).

**2.8.8 What the harness must NOT do (the §2.7.6 fence, extended).** The harness
is holdout-fenced by construction:

- **No `gates.yaml` read beyond the `gate_m6` block's protocol/cells** — the 11
  gated cells, their `split_units`, `metric`/`k`, tolerances, `gate_seeds`, and the
  4-of-5 conjunction, read-only. It computes no tolerance, moves no threshold, and
  touches no other gate.
- **No holdout-informed choice** — every builder rule (§2.8.2), the year-0 reads
  (§2.8.3), and the cell reductions (§2.8.4) are fixed from the ratified design and
  certified code; nothing is tuned to a 2015–2019 / 2016–2018 observable. The
  builders and scorer are exercised on **synthetic frames only** until the
  registered run, and pre-flights 1–2 read no holdout cell.
- **No realized post-boundary macro read on the scored path** — the §2.7.6.3
  leakage prohibition extends to the harness: the scored earnings de-index uses the
  generator's own `≤2014` `I_proj`, **never** the frame's realized `nawi` column
  (`steps.py:145-156`), which is admissible only for the non-scored seam /
  `taxable_max` and the **report-only** alignment path (decision 9, §4.8). No
  realized 2015–2022 aggregate enters any gated cell.
- **Forward-mode inputs stay rejected** — a gated run uses
  `EvaluationMode.GATED_REALIZED`; `support.prepare_evaluation_support` structurally
  rejects supplying realized truth/floor/presence to a `FORWARD` call, so the
  successor-gate object cannot silently borrow the gated support.

**2.8.9 Closes the blocker; residual open decisions.** Every deliverable the
designed stop enumerated is now pinned — the realized-history provenance (§2.8.1),
the per-field projected-panel construction against the certified schemas (§2.8.2),
the year-0 realized slice mirroring the floor (§2.8.3), the **year-0
earnings-domain law** closing the round-4 build blocker (§2.8.3a, amendment 3b),
the drift-scoring layer reusing the floor cell functions verbatim with a
byte-identity self-test (§2.8.4), the two pre-flights (§2.8.5–§2.8.6), and the
runner with its one-shot stamping and fence (§2.8.7–§2.8.8) — so the build lane
implements the harness with **zero** design choices. **Residual open decisions
(amendment 3g/3h corrections, 2026-07-15): two, now closed** — **(1)** the marital
projection carried **no** domain law for the sub-`START_AGE`-at-anchor class that
`build_anchor_frame` admits (all ages, no filter) but the certified marital
`person_years` do not cover at the anchor wave; the fifth registration's execution
failure surfaced it (forensics #42 comment 4979437110) and §2.8.2g (amendment 3g)
closes it via option B (seed-at-marital-entry; the frozen v3 floor stays
byte-identical). **(2)** the step-4 fertility / open-additions **roster
materialization** had **no** domain law separating the frame-independent scoring
schedule from the simulated-mortality roster, so a mother the wave loop killed
retained her scheduled birth and tripped the report-only parent-roster guard; the
sixth registration's execution failure surfaced it (forensics #42 comment 4984997277)
and §2.8.2h (amendment 3h) closes it via drop-with-reconciliation (roster-present
mothers materialize; the dead mother's birth drops into the report-only
reconciliation; the frozen floors and the fertility RNG stream stay byte-identical),
**promoting item (iii) below from a disclosure to a law.** One **flagged
floors-ceremony finding** (not a harness-design gap): §2.8.3a
found empirically that the frozen v3 floor's gated-earnings support over-includes
~21 % later earnings-entrants (open additions the 2014-anchored chain cannot
project), so the frozen earnings tolerances are applied **conservatively** and the
§2.8.3a self-check may escalate an earnings-floor re-derivation to the ceremony
(the §4.9 / adjudication-7 rule — a ceremony finding, never a silent redesign);
the harness itself is fully specified. Three items are **mechanical
alignments / disclosures, not design choices**, recorded for the build lane: (i)
`marital_step` and `household_step` compute the certified core **once per draw and
cache** (the built `disability_step` pattern) at the period-0 address (§2.8.2),
rather than the stub-era per-period recompute; (ii) the truth-side cell functions
are **extracted to one shared importable module** that both the floor script and
the scorer import (byte-identity by construction, self-tested per §2.8.4); (iii)
step-4 **fertility** remains a per-period recompute on the cached marital result
(`apply_fertility` → `simulate_fertility` per period, `steps.py:440-472`), so
frame-materialized births are stitched across years — **inert on every gated cell**
(the **synthetic newborns** fall outside `holdout_ids` and every gated band —
distinct from the **anchor-present** minors `build_anchor_frame` admits *into*
`holdout_ids`, who *do* age into the gated `first_marriage.18-29` band and are
handled by amendment 3g (§2.8.2g) — and births feed only the report-only
entrant/roster paths). The **materialization** of those synthetic newborns onto the
post-mortality roster is now governed by amendment 3h (§2.8.2h) — roster-present
mothers only, drop-with-reconciliation — closing the sixth registration's execution
failure (a dead mother's scheduled birth tripped the report-only guard and aborted
the scored run); the materialization stays inert on every gated cell and the frozen
floors and fertility RNG stream are byte-identical. This closes the run
lane's blocker; a fresh
`gate_m6` registration follows the harness build and its engine-referee-style
review, with a forecast informed by nothing new (no holdout contact occurs in the
harness build).

### 2.8.10 The ≤2014 external-reference input bindings (design amendment 3d, closes the second designed stop)

The harness of §2.8.1–§2.8.9 is fully specified, but it does **not execute itself**:
`run_gate_m6_candidate1.py` is not self-starting (its docstring: "a fresh issue-42
registration and an explicit input factory are required"), requiring an
`--input-factory module:callable` that returns an `M6HarnessInputs` via
`load_m6_inputs` (`harness/m6_inputs.py:542-564`). That entry point takes **three
caller-owned external references** — `ssa_params` + `ssa_params_vintage`, a
`claiming_reference`, and
`mortality_exposure` + `mortality_external_rates` + `mortality_external_vintage` —
and `_validate_external_inputs` (`m6_inputs.py:198-250`) refuses any of them past
`T*` through `validate_external_vintage` (`engine/refit.py:825-838`; `vintage > 2014`
raises). The run lane's **second designed pre-scoring stop** (registered
4967241464, graded
[#42 comment 4967433717](https://github.com/PolicyEngine/populace-dynamics/issues/42#issuecomment-4967433717))
found that this certified binding **does not exist**: the only committed claiming
reference is the *2023* Supplement, and `claiming_pmfs_from_reference` **correctly
raises** `claiming reference vintage 2023 is post-T* (2014)`. The guard architecture
worked — it kept the runner from silently selecting a new external vintage; the
one-shot was preserved, nothing scored. This subsection is the reviewed amendment the
stop names as the unblock: it pins all three ≤`T*` bindings **and** the factory that
supplies them, at the §2.7.6 / §2.8.9 bar (**zero** build-lane choices,
candidate-blind — the bindings are fixed sources and vintages, tuned to no 2015–2019
observable). It **resolves finding 11** (§2.2 step 7, §9 `claiming_vintage_freeze`) by
taking its **"≤`T*` claiming vintage"** branch, not the "named 2023 leak" branch.

**2.8.10.1 Binding 1 — the claiming reference (the hard blocker).** *Source, PINNED:*
the **SSA *Annual Statistical Supplement, 2014* (the 2014 edition), Table 6.B5.1** —
"percentage distribution of retired-worker awardees by age at month of entitlement, by
sex and year of entitlement." It carries entitlement years **1998–2013** (confirm-at-fetch;
each edition *Y* tabulates 1998..*Y*−1, verified contiguous 1998–2022 in the committed 2023
file — admissibility is insensitive to 2013-vs-2012, since `claiming_pmfs_from_reference`
needs only `supplement_year ≤ 2014` plus ≥1 entitlement year ≤2014 and the harness snaps to
nearest-available) with `supplement_year = 2014`. *Why this edition:* it is the **maximal edition with
`supplement_year ≤ 2014`** — the 2015 edition (data through 2014) has
`supplement_year = 2015` and `validate_external_vintage` rejects it. The 2014 edition's
**information** is entirely ≤2013 claiming behaviour; its publication (~early 2015) is
post-`T*` but its information is ≤`T*`, **leakage-safe on the exact §2.7.6.3
publication-lag principle** ("only its publication *date* is post-`T*`, not its
*information*"). This is the finding-11 freeze. *Fields to extract* (mirror
`scripts/build_ssa_claim_ages.py` verbatim): per (sex, entitlement year) the twelve
published raw age columns `{age62, age63, age64, age65_before_fra, age65_at_fra,
age65_after_fra, age66_before_fra, age66_at_fra, age66_after_fra,
disability_conversion, age67_69, age70plus}` (`null` for the era-inapplicable `. . .`
cells), `number_thousands`, `average_age`, and the footnote-a FRA schedule; the
era-dependent applicable-column map is recorded per year exactly as the 2023 build
does. *Target schema, PINNED:* `data/external/ssa_claim_ages_2014supplement.json`,
**structurally identical** — at every field `claiming._load` parses and consumers read
(`claiming.py:147-166`) — to the committed `ssa_claim_ages_2023supplement.json`:
`schema_version = "ssa_claim_ages.v1"`, `table = "6.B5.1"`, `supplement_year = 2014`,
`column_schema.{raw_columns, collapsed_categories}` (the eight-way partition `{age62,
age63, age64, age65, age66, disability_conversion, age67_69, age70plus}`),
`provenance`, `validation`, `fra_schedule`, and `data.{male,female}.{"YYYY":
{number_thousands, average_age, raw, categories, fra_at, …}}` — so
`claiming.load_claim_age_reference(path)` parses it (`claiming.py:147-166`) and
`claiming_pmfs_from_reference` admits it (`supplement_year 2014 ≤ 2014`; entitlement
years ≤2014 present). This is **structural** identity, not literal `provenance`-key
parity: the 2014 `provenance` additionally carries `source_sha256` and `retrieval_date`,
fields absent from the committed 2023 `provenance`. *Acquisition protocol* (the **orchestrator** performs; ssa.gov
returns HTTP 403 to `curl` / WebFetch): from an authenticated browser session, fetch
**`https://www.ssa.gov/policy/docs/statcomps/supplement/2014/6b.html`** (Table 6.B5.1;
PDF fallback `…/2014/supplement14.pdf`), transcribe verbatim into
`scripts/build_ssa_claim_ages_2014supplement.py` (embedded `RAW_TABLE`, era-column
handling, cross-validated to the digit against a live DOM/PDF extraction of the same
page in the same session), emit the JSON, and **commit the raw fetched source
alongside** (e.g. `data/external/sources/ssa_supplement_2014_6b.html`) recording in
`provenance`: `source_url`, `retrieval_date` (UTC), `fetch_method`, and `source_sha256`
(the committed raw source's hash — **build-time provenance, not the load-time tamper
gate**; the factory instead asserts the JSON file's own hash, §2.8.10.4). *Consumption:*
**report-only** — claiming is R2 marital-blind (§2.6) and feeds only report-only benefit
levels (AIME/PIA/claiming are never gated, §2.8.3a); it is a **hard blocker** solely
because `assemble_m6_inputs` materialises the PMFs and raises on any post-2014 vintage
**before any refit, score, or artifact write** (`load_m6_inputs` does read PSID first via
`load_m6_raw_inputs`, then validates — the guard fences the fit/score/write, not the
read). The nearest-year fallback on the harness path is **`ClaimingSchedule.distribution`**
(`steps.py:299-305`) — nearest-available-year over the PMF dict
`claiming_pmfs_from_reference` materialises from `reference.years() ≤ T*` (`refit.py:972-984`)
— so for the 2014 file (1998–2013) a projected entitlement year >2013 snaps to 2013, the
held-out claim mix frozen at the last ≤`T*` observation. **Caution for the factory-build's
validation tests:** the module-level `claiming.claim_age_distribution` / `draw_claim_ages`
/ `_resolve_year` are **not** on the M6 path and would `KeyError` for entitlement years
2014–2022 against the 2014 file — their `MIN_YEAR` / `MAX_YEAR = 1998 / 2022` are 2023-file
constants that do not adapt to the loaded reference (`claiming.py:86-87, 200-213`); drive
the fallback through `ClaimingSchedule.distribution` / `claiming_pmfs_from_reference`,
never the module helpers.

**2.8.10.2 Binding 2 — the SSA parameters vintage.** *Revision, PINNED:* load from
**policyengine-us 1.752.2** — the deployment-frame pin (`data/deployment_frame.py:48`,
`CERTIFIED_PIN["model_version"]`, shared with every `gate_w1` artifact). *Acquisition*
(there is **no git tag `1.752.2`** on PolicyEngine/policyengine-us; PyPI carries it,
released 2026-07-01): `pip` / `uv install policyengine-us==1.752.2` into the run env and
set `POPULACE_DYNAMICS_PE_US_DIR` to the directory containing `policyengine_us/` (a
site-packages install satisfies `_SSA = policyengine_us/parameters/gov/ssa`,
`ss/params.py:54`), or check out the exact upstream commit for that release. *Version
assert:* the factory checks `importlib.metadata.version("policyengine-us") ==
CERTIFIED_PIN["model_version"] == "1.752.2"` in the same env the directory came from.
**(Amendment 3e, F2):** that metadata assert alone does **not** bind the *directory*
the YAML loads from — `POPULACE_DYNAMICS_PE_US_DIR` can point at a different-vintage
checkout and still pass the gate; §2.8.10.5 adds the factory-side assert that ties the
resolved parameter directory to the metadata-versioned install, so "in the same env the
directory came from" is *checked*, not assumed.
*Provenance consequence:* `load_ssa_parameters` records `pe_us_revision` from a `git log`
in that directory and degrades to `"unknown"` for a non-git site-packages install
(`ss/params.py:309-318`), so the artifact's parameter provenance carries
`pe_us_revision = "unknown"` with the vintage identity pinned by the asserted
`importlib.metadata` version + `CERTIFIED_PIN`, not a git hash. `params_full =
load_ssa_parameters()` (`ss/params.py:206`) runs its load-time cross-check (derived
bend points vs pe-us stored thresholds; `NAWI(1977) = 9,779.44`) on the **realized**
series, which passes. Declare `ssa_params_vintage = 2014`. The **field-level vintage
rule** (mechanically load-bearing — the guard checks only the declared int, not the
dict contents, so leakage-safety is a *design* obligation the factory must honour):

- **NAWI — realized ≤2014, `I_proj` beyond (replace, do not truncate).** Keep the
  realized values for years ≤2014 (the `[2005, 2014]` window intact: 2005 = 36,952.94 …
  2014 = 46,481.52) and **replace every entry for year >2014** with the §2.7.6.3
  projection `I_proj(y) = exp(α + β·y)`, where `(α, β)` is the OLS fit of `ln(NAWI)` on
  year over `[2005, 2014]` — **byte-identical to
  `engine.forward_earnings.fit_projected_wage_index`** (`forward_earnings.py:181-206`) —
  over the **same year-key range** as the pinned revision (so `max(nawi)` is unchanged).
  *Why replace, not truncate:* `couple_earnings.indexed_earnings_supply` admits a person
  to the gate-2c permanent-earnings axis only if `min_nawi ≤ birth+60 ≤ max_nawi`
  (`_admitted`, `couple_earnings.py:350-362`) and indexes their ≤2014 earnings to the
  age-60-year NAWI; a hard truncation to ≤2014 collapses `max_nawi` to 2014, admitting
  only `birth ≤ 1954` — **near-empty for a PSID first-marriage panel** — so
  `refit_first_marriage_modifier` raises (`"gate-2c permanent earnings supply is
  empty"`, `refit.py:879-880`) or, the likelier branch, **degenerates to the neutral 1.0
  modifier** (`couple_formation_sim_v2` sets undefined cells neutral) exactly where the
  gated cells live. That modifier multiplies **first-marriage** hazards directly
  (`apply_first_marriage_modifier`, `marital.py:378`) and so **feeds the scored path of**
  all three gated marital flows (`divorce.18-44`, `first_marriage.18-29|female`,
  `remarriage.18-64` — the divorce / remarriage pair via composed marital-state
  occupancy), and §2.7.6.3 names **AIME indexing** as a NAWI consumer under the **leakage
  prohibition on realized post-`T*` NAWI on the scored path**; a current pe-us checkout carries
  **realized** 2015–2024 NAWI (2015 = 48,098.63 vs `I_proj(2015) = 47,252.12`, an $846
  gap) which would leak realized wage growth into those gated flows. Replacing post-2014
  NAWI with the identical `I_proj` the forward law re-derives internally hands the refit
  **one `T*`-admissible wage-index surface** — realized ≤2014, `I_proj` beyond — so every
  NAWI consumer (the forward de-index, AIME indexing, bend-point derivation) reads only
  ≤`T*` values, while the **admitted universe** (a function of the year-key *range*, not
  the values) is unchanged.
- **Ordering constraint (PINNED).** The replacement **must follow `load_ssa_parameters`
  and must not re-trigger its cross-check**: that check compares derived bend points
  (`nawi[year−2]`) to pe-us's realized-based stored thresholds across the whole stored
  range, and re-deriving post-2014 bend points from `I_proj` — intended for the fit —
  would spuriously trip it. So the order is **load (cross-check on realized) →
  `dataclasses.replace(params_full, nawi=…, wage_base=…)`**. `SSAParameters` is a frozen
  dataclass with no `__post_init__` (`ss/params.py:65-136`), so `replace` bypasses the
  check by construction.
- **Wage base — truncate change-years to ≤2014.** Keep the step-function entries through
  2014 (= 117,000); drop 2015+. Inert on every gated cell (the ≤2014-truncated earnings
  panel credits only ≤2014 earnings, so `benefits.creditable_history` consults only
  ≤2014 bases), and it keeps the declared vintage literally true; the report-only AIME's
  use of the persisted 2014 base for post-2014 projected earnings is a report-only
  immateriality (§2.7.6.3 admits the realized base only on the `taxable_max` /
  report-only path anyway).
- **Bend points — inherit the NAWI rule.** Derived, not stored (`bend_points(year) =
  f(nawi[year−2])`, `ss/params.py:138-149`); realized-derived through eligibility year
  2016, `I_proj`-derived beyond, consumed only on the report-only PIA path — no separate
  step.
- **Statutory step schedules — pass through.** The 416(l) FRA-by-birth-year table, the
  PIA formula factors, the 402(q) early-reduction rates, the 402(w)
  delayed-credit-by-birth-year steps, `max_delayed_years`, and the auxiliary
  spouse/survivor constants (`ss/params.py:87-136`) are keyed by **birth year** or are
  statutory constants — no calendar-year vintage content; every value in the model's
  scope was fixed in statute well before 2014. Unchanged from the pinned revision.
- **Prohibited, enumerated.** (a) any **realized** NAWI for year >2014 on any refit path
  (replaced by `I_proj`); (b) any `wage_base` change-year >2014 in the bundle; (c) any
  pe-us parameter whose effective date >2014 that the refit reads. The 2026 bend-point
  determination (1,286 / 7,749) that `load_ssa_parameters` cross-checks is a **load-time
  self-consistency assertion on the realized series, not a fit input** — it enters no
  fitted estimate and no gated cell; recorded here so the referee does not read it as a
  post-`T*` leak. Relatedly, §2.7.6.3's report-only realized-NAWI **alignment** box is
  **vacuous** under this bundle: no realized post-2014 NAWI exists in-process, and the
  runner records the alignment layer as not collected (`build_alignment_displacement(None,
  None)`, `m6_runner.py:1100`) — the two sections do not conflict.

**2.8.10.3 Binding 3 — the mortality reference (ungated).** *Vintage, PINNED:* **NCHS
*United States Life Tables, 2010*** (NVSR Vol. 63 No. 7, released Nov 2014),
`data/external/nchs_life_tables_2010.json` (`vintage_year = 2010`);
`mortality_external_vintage = 2010` (the vintage the canonical fixture
`tests/test_m6_inputs.py:263` already uses). *Why the vintage is outcome-inert, and why
2010 among the admissible:* `fit_mortality_model` computes `aligned_rate = central_rate ×
(psid_rate / central_rate) = psid_rate` **exactly** (`refit.py:1083-1088`; its docstring:
"the fitted-window level equals the PSID central rate"), so the external vintage is
**outcome-inert on every fitted hazard** — untunable to any 2015–2019 observable by
construction; it enters only the `central_rate > 0` guard and the **report-only** anchor
decomposition (`external_anchor`, `build_mortality_floors.py:281+`). Among the admissible
≤`T*` vintages the pin is still **2010** (NVSR 63-7, released Nov 2014 — ≤`T*` on both the
described-year and publication axes): the 2023 table (NVSR 74-6) is guard-rejected
(`validate_external_vintage` raises, 2023 > 2014, admissible only for
`build_mortality_floors.py`'s same-time reproduction floor), and 2010 is the latest
decennial complete life table over the 14-year-older 2000 (NVSR 51-3) — the choice cannot
move the fit, only the report-only decomposition's anchor. *external-rates
shaping, PINNED* (mirror `build_mortality_floors.py:256-275` `nchs_band_rates`):
`mortality_external_rates` with columns `{lower_age, upper_age, age_band, sex,
central_rate}`; for each band × sex ∈ {female, male}, `central_rate = (l_a − l_{b+1}) /
(T_a − T_{b+1})` from the 2010 table's `lx`/`Tx` columns (open top band `[85,120] = l_85
/ T_85`), over the seven `MORTALITY_BANDS` `(25-34, 35-44, 45-54, 55-64, 65-74, 75-84,
85+)` (`m6_cells.py:70-78`); band strings via `band_label` (hyphen closed, `"85+"`
open); `(lower_age, upper_age)` the band endpoints. *exposure shaping, PINNED — a small
adapter, because neither cited builder emits the pinned shape:* `build_exposure_slices`
(`build_mortality_floors.py:154-183`) returns `{person_id, sex, weight, age, band,
exposure, death, start_wave}` and `m6_cells.mortality_slices` windows to the **2015+**
truth years (`m6_cells.py:212-218`) — so "mirror" means an adapter that re-derives the
≤2014 PSID person-interval single-year slices from the same step-4 readers
`panels.demographic_panel()` + `deaths.read_death_records()` (`build_mortality_floors.py:559-561`)
and emits the pinned columns `{event_year, required_interview_year, age_band, sex,
start_weight, exposure, death}`. *Slice values:* `exposure = 0.5` in the death-year slice
else `1.0`, `death = 1.0` in that slice else `0.0`; `age_band` from `MORTALITY_BANDS`,
`sex` from the death/demo sex code, `start_weight` = the F6 realized start-wave anchor
weight (= `fit_mortality_model`'s "fixed boundary weight"). *Row dating (both fields on
EVERY slice — death and survivor):* `event_year` = the slice's calendar year (the death
year for the death slice); `required_interview_year` = the **interval's closing wave**
(`next_wave` from the demo panel's wave grid, per `build_exposure_slices` — it is **not**
in `read_death_records`, which carries only `person_id` / `sex` / `death_year`,
`deaths.py:81-91`). The **symmetric conservative rule** (closing-wave dating for *every*
slice) is pinned: `prepare_mortality_refit_inputs` applies the flow rule to all rows
(`truncate_estimation_frame(flow=True)`, `refit.py:282-295`), keeping only slices with
**both** `event_year ≤ 2014` **and** `required_interview_year ≤ 2014` — this drops the
2013→2015 interval wholesale (the ≤`T*` window ends with the 2011→2013 interval), chosen
over anchor-wave-for-survivors + closing-wave-for-deaths, which would keep 2013–2014
exposure **without** its 2015-dated deaths (a downward-biased terminal hazard). The
biennial-boundary rule (finding 12) thus applies symmetrically. *Disclosure:*
`start_weight` is read from a 2015+ interview yet enters a ≤`T*` fit — admissible only
because mortality is ungated (below). *Consumption —
UNGATED:* the rates + exposure feed `fit_mortality_model` (the projection's step-1
age×sex hazard) and the report-only mortality / AIME / PIA / household diagnostics; **no
mortality cell is in the 11-cell gated set** (`death.85+` is report-only,
attrition-demoted, `m6_cells.py:79-83`, §2.8.4). Per the F1 adjudication (§2.8.4), gated
flows and earnings score on the **realized** support regardless of simulated death, so
this binding touches **no gated tolerance** — it is required only because
`load_m6_inputs` validates the mortality vintage + shape at assembly.
**(Amendment 3e, F1):** the seven-band table pinned here is the *estimation* shape; the
*projection model* `AgeSexMortalityModel` additionally requires bands from age 0
contiguous through 120 (`engine/steps.py:57-68`, so materialized births get a defined
hazard rather than the `np.zeros` default), which `fit_mortality_model` — deriving its
model bands from these external rows (`refit.py:1061-1068,1089`) — cannot satisfy on the
seven-band shape and raises `mortality bands must start at age zero` at the run's first
phase (`m6_runner.py:348`). The factory bridges this with an inert `(0,24)` projection-
coverage pad on **both** fit inputs; the exact rows, the inertness proof, and what the
`<25` hazard consumes are pinned in **§2.8.10.5**.

**2.8.10.4 Binding 4 — the factory contract.** *PINNED:* a new module
**`scripts/registered_m6_inputs.py`** exposing a **zero-argument** `build_inputs() ->
M6HarnessInputs`, resolved by `run_gate_m6_candidate1.py --input-factory
registered_m6_inputs:build_inputs` — `importlib.import_module("registered_m6_inputs")`
(`run_gate_m6_candidate1.py:42`) resolves it because `scripts/` is `sys.path[0]` when the
runner is invoked from the repo root. Both `data/external/…` paths are
**repo-root-anchored** (`DATA = Path(__file__).resolve().parents[1] / "data" / "external"`,
mirroring `claiming._ROOT`, `claiming.py:80-83`), never CWD-relative. Every binding above
is **hardcoded**; there is no argument and no environment-derived vintage selection beyond
`POPULACE_DYNAMICS_PE_US_DIR` → the pinned 1.752.2 install (§2.8.10.2).
Deterministic steps, in order: **(1)** assert `importlib.metadata.version("policyengine-us")
== CERTIFIED_PIN["model_version"] == "1.752.2"` **and** assert the resolved parameter
directory is the metadata-versioned install (§2.8.10.5, F2); `params_full =
load_ssa_parameters()`; **(2)**
`params = dataclasses.replace(params_full, nawi = {realized ≤2014} ∪ {I_proj(y) : y>2014
in params_full.nawi}, wage_base = {change-year ≤2014})` (§2.8.10.2, `I_proj` via
`fit_projected_wage_index`); **(3)** `claiming_reference =
claiming.load_claim_age_reference(DATA / "ssa_claim_ages_2014supplement.json")`
(repo-root-anchored) and **assert the JSON file's sha256 equals a constant hardcoded in
`registered_m6_inputs`** — the artifact actually consumed; the committed raw-source
HTML/PDF's hash lives separately in `provenance.source_sha256` as build-time provenance,
not the tamper gate; **(4)** build `mortality_external_rates` from
`nchs_life_tables_2010.json` (band collapse) and `mortality_exposure` from the ≤2014 PSID
slices via the §2.8.10.3 adapter, **then append the inert `(0,24)` projection-coverage
pad to both** (§2.8.10.5, F1); **(5)** `return load_m6_inputs(ssa_params=params,
ssa_params_vintage=2014, claiming_reference=claiming_reference, mortality_exposure=…,
mortality_external_rates=…, mortality_external_vintage=2010)` (defaults
`boundary_year=2014`, `earnings_seed=5200`). `load_m6_inputs` then **re-validates every
vintage at assembly** (`validate_external_vintage` ×3, `claiming_pmfs_from_reference`,
`prepare_mortality_refit_inputs`), so the harness re-checks the factory's bindings **before
any refit, score, or artifact write** (not before the PSID read — `load_m6_raw_inputs`
reads first); the sha256 gate plus the three ≤`T*` guards make the one binding that
could silently drift — the acquired claiming source — tamper-evident. The factory is
exercised on the committed references + staged PSID **only at the registered run**; the
build/test lane never executes the runner against real data (runner docstring, §2.8.8).

**Residual.** The mortality-exposure row-dating semantics (§2.8.10.3) were the one
under-pinned item the amendment-3d referee flagged; they are now pinned — the symmetric
closing-wave dating, the re-derivation adapter, the `next_wave` derivation, and the
`start_weight` disclosure. With that, all three references are pinned to a single ≤`T*`
source and vintage each, and the factory is a deterministic **build** over them, not a
design choice. Amendment 3d proceeds as 3b / 3c did: this pin → referee → the factory build
(`registered_m6_inputs`) plus the acquired claiming source with provenance → referee →
merge → a third `gate_m6` registration → the run.

**2.8.10.5 Amendment 3e — the mortality-rates shape bridge + parameter-dir binding
(closes factory-referee F1/F2).** The factory (`scripts/registered_m6_inputs.py`, merged
`67e7fad`) was verified **MERGE-READY (build only)** by the adversarial factory referee
(build `36d6cde`,
[#191 comment 4969829131](https://github.com/PolicyEngine/populace-dynamics/pull/191#issuecomment-4969829131))
with two run-gating findings against the merged **spec/engine** pair — filed as findings,
not build defects, because the build faithfully implements the pinned §2.8.10 shape with
zero build-lane choices. This subsection is the reviewed amendment the referee names as
the third-registration unblock; it edits no `gates.yaml` cell, moves no threshold, builds
no floor, and writes no test.

**F1 (blocking) — the pinned seven-band external-rates shape is rejected by the
projection model at the run's first phase.** §2.8.10.3 pins `mortality_external_rates`
over the seven `MORTALITY_BANDS` (25-34 … 85+). `fit_mortality_model` derives its model
`bands` from the external-rates rows (`refit.py:1061-1068`) and constructs
`AgeSexMortalityModel(bands=…, probability=…)` (`refit.py:1089`), whose `__post_init__`
requires bands to **start at age 0** and run **contiguous through 120**
(`engine/steps.py:57-68`) — because `AgeSexMortalityModel.probabilities` fills a
`np.zeros` default for any age outside the bands (`steps.py:98-109`), so a `<25` gap would
silently apply **zero** hazard to every materialized birth and child. The registered run
hits this at its **first phase**, `refit_m6_phase → fit_mortality_model(bundle.mortality)`
(`m6_runner.py:348`), **before** the pre-flights, score, or artifact write; on the
factory's real outputs it raises `ValueError: mortality bands must start at age zero`
(referee-reproduced). The naïve "add a 0-24 external row" **does not work**: the fit loop
iterates every external row and requires PSID exposure in that cell
(`refit.py:1070-1082`), and the ≤`T*` PSID exposure has **no `<25` slices** — the shared
builder drops every out-of-band age (`build_exposure_slices`,
`build_mortality_floors.py:223-224`) — so it would then raise `undefined mortality fit
cell 0-24|…`. The whole mortality data pipeline is 25+ only: floors
(`build_mortality_floors.BANDS`), holdout truth (`MORTALITY_LADDER`,
`build_m6_holdout_floors_v3.py`), the scored cells (`_projected_mortality_cells` restricts
to the seven bands, `m6_runner.py:708-722`), and the external anchor. The 0-start
requirement is **solely** a projection-model total-population invariant, with no
counterpart on any estimation, truth, or scored surface.

*Resolution, PINNED — option (b): an input-side inert projection-coverage pad, engine
untouched.* The factory appends **one `(0,24)` band per sex** to **both** fit inputs,
after the seven-band construction, leaving `fit_mortality_model` and
`AgeSexMortalityModel` unchanged. Exact rows:

- **External pad (2 rows):** `{lower_age: 0, upper_age: 24, age_band: "0-24", sex,
  central_rate: (l_0 − l_25)/(T_0 − T_25)}` from the same NCHS-2010 `lx`/`Tx` formula
  (`nchs_band_rates`) — **male ≈ 0.000766, female ≈ 0.000451**. **The value is
  outcome-inert:** `aligned_rate = central_rate × (psid_rate / central_rate) = psid_rate`
  exactly (`refit.py:1083-1088`, the identity §2.8.10.3 already invokes), so the external
  rate cancels on every cell; it is pinned to the real NCHS 0-24 rate for provenance
  honesty, but **any positive value yields byte-identical fitted probabilities**.
- **Exposure pad (2 rows):** `{event_year: 2014 (= T*), required_interview_year: 2014
  (= T*), age_band: "0-24", sex, start_weight: 1.0, exposure: 1.0, death: 0.0}`. Dated at
  `T*` so it survives `prepare_mortality_refit_inputs`'s ≤`T*` flow truncation
  (`refit.py:1001-1007`). With `death = 0`, `psid_rate = 0`, so the fitted **`<25` hazard
  = `−expm1(0) = 0` exactly** — the pinned convention: **no modeled mortality below the
  age-25 PSID exposure floor.**

*The pad satisfies the validator without moving the fit — verified by execution against
the merged engine (`fit_mortality_model` on the real `refit.py`).* Unpadded → the model
raises `mortality bands must start at age zero`. Padded → a valid 8-band model
`((0,24),(25,34),…,(85,120))`; the fourteen 25+ cells are **byte-identical** to the
unpadded fit and invariant under any pad value **or** weight (varying the external value
`0.00077 ↔ 0.5` and the exposure weight `1.0 ↔ 17.3` changes no 25+ probability), and the
`<25` hazard is exactly `0`. The pad matches only the `"0-24"` `age_band`, so the seven
real cells' `cell` selections (`refit.py:1071-1074`) are untouched; it clears
`AgeSexMortalityModel.__post_init__`'s band/coverage/cell checks and its `[0,1]` check
(`0 ∈ [0,1]`) without perturbing any fitted value — the inertness thus **covers the
validator path**, not only the arithmetic. The shared seven-band set (`MORTALITY_BANDS`,
`build_mortality_floors.BANDS`) is **not** modified — the pad is appended *after* the
seven-band build, so the `MORTALITY_BANDS == mf.BANDS` guard
(`registered_m6_inputs.py:238`) still holds and the committed floors artifacts are
undisturbed.

*What the fitted hazard consumes, by band group.* The seven 25+ bands carry the
PSID-level hazards (`aligned_rate = psid_rate`) and feed the projection's `apply_mortality`
(`steps.py:113-134`) plus the **report-only** `mortality_drift` / `shock` disclosure cells
(`m6_runner.py:855-874`), which are ungated (`certifies_nothing_about_mortality_drift:
True`, `m6_runner.py:1244`; and the scored `_projected_mortality_cells` drops `<25`,
`m6_runner.py:708-722`). The `(0,24)` pad band is consumed **only** by `apply_mortality`
as a zero hazard for age-`<25` individuals (materialized births + any `<25` holdout
members); it appears in **no** scored or disclosed mortality cell. Because mortality is
ungated (§2.8.10.3, §2.8.4), the zero-hazard convention touches no gated tolerance; its
sole channel to gated (family-A) moments is negligible second-order population composition,
made deterministic by the zero convention. *Rejected — option (a):* extending the external
table with a real NCHS `<25` hazard that the fit **consumes** would require an engine
change (a no-exposure fallback inside `fit_mortality_model`) **and** would import an NCHS
**level** as the hazard, violating the mortality-anchor contract that
`aligned_rate = psid_rate` encodes in code ("the anchor certifies the gradient, not the
level"; PSID levels, NCHS inert — `build_mortality_floors.py` `proposed_thresholds_note`).
Option (b) touches zero verified-engine surface and honours that contract.

*Patch pointer (factory, zero-discovery).* In `scripts/registered_m6_inputs.py`, add one
helper `_pad_below_25_projection_coverage(external_rates, exposure, *, boundary_year)`
that appends the four rows above (2 external + 2 exposure, per sex), and call it in
`build_inputs` step (4) on the outputs of `nchs_2010_external_rates()` and
`mortality_exposure_adapter()`, before `load_m6_inputs` (equivalently, append inside each
builder after its seven-band construction). The patch lane must also add the **bridge
test** the referee flagged missing — no committed test routes factory-shaped rates through
`fit_mortality_model` — asserting the padded factory output fits to an 8-band 0→120 model
with `<25` hazard `0` and the seven 25+ cells unchanged.

**F2 (should-fix) — the version gate does not bind the parameter directory.**
`assert_pe_us_version` reads `importlib.metadata.version("policyengine-us")` in the
running env (`registered_m6_inputs.py:125-137`), but `load_ssa_parameters` reads YAML from
`_resolve_pe_us(None)` — `POPULACE_DYNAMICS_PE_US_DIR` or the default checkout
(`ss/params.py:187-227`) — and records `pe_us_revision` from a `git log` in that dir,
**decoupled** from the metadata. A mismatched dir passes the gate (referee-demonstrated:
1.752.2 pip-installed **and** the env var pointed at a 1.690.7 checkout →
`boundary_ssa_parameters` silently loads the wrong vintage). §2.8.10.2's "in the same env
the directory came from" is an unchecked assumption.

*Resolution, PINNED — a factory-side provenance assert (no data-selection choice).* Add
`assert_pe_us_param_dir()` (mirroring `assert_pe_us_version`, unit-testable via an injected
root) and call it in `build_inputs` step (1), right after `assert_pe_us_version()`. It
binds the directory `load_ssa_parameters` **will read** to the metadata-versioned
distribution's on-disk location, deriving the location from the **same** distribution the
version comes from:

```python
resolved  = (_resolve_pe_us(None) / _SSA).resolve()                    # ss/params.py:52-54,187-193
versioned = (Path(importlib.metadata.distribution("policyengine-us")
                  .locate_file("policyengine_us")).resolve()
             / "parameters" / "gov" / "ssa")
if resolved != versioned:
    raise RuntimeError(
        "POPULACE_DYNAMICS_PE_US_DIR resolves SSA parameters to a directory that is not "
        "the metadata-versioned policyengine-us install; point it at the 1.752.2 install."
    )
```

Verified against the real 1.752.2 install:
`distribution("policyengine-us").locate_file("policyengine_us")` returns exactly
`Path(policyengine_us.__file__).parent`, whose `parameters/gov/ssa/nawi.yaml` is the file
`load_ssa_parameters` reads (`ss/params.py:226`); a site-packages install is not a git
repo, so `pe_us_revision = "unknown"` (matching §2.8.10.2's provenance note). After the
assert, the gate's 1.752.2 metadata genuinely governs the loaded YAML. (`_resolve_pe_us`
and `_SSA` are module-private; the patch may import them, or the patch lane may promote a
thin public `resolved_pe_us_root()` in `ss/params.py` — a factory-side call either way,
per the referee's framing.)

**N1 (nit) — the `_era_mapping` docstring's phantom assert.** `_era_mapping`'s docstring
(`scripts/extract_ssa_claim_ages_2014.py:327-328`) states men/women era parity is
"verified in `build()`," but no such assert exists; the property holds empirically in the
sha-pinned source and `era_map` is provenance-only (read by nothing). *Resolution, PINNED
— soften the docstring, do not add the assert:* an assert would guard a non-load-bearing
property and could spuriously fail on a valid future edition where a sex-specific era
boundary legitimately differs. Replace the clause with: "The men and women panels share
the same era structure empirically in this sha-pinned 2014 source (an observed property,
not asserted; `era_map` is provenance-only and read by nothing); men's panel drives the
derivation."

**Sequence.** Amendment 3e proceeds as 3d did: this pin → factory-referee round → the
F1/F2/N1 patch (`registered_m6_inputs` inert pad + parameter-dir assert + the missing
bridge test; the `_era_mapping` docstring soften) → referee → merge → the third `gate_m6`
registration → the run.

## 3. RNG discipline — the projection stream registry

### 3.1 The inherited convention

The stack scores K=20 draws on `numpy.random.default_rng(5200 + k)`, `k = 0…19`
(`family_transitions/evaluation.py` / `household_composition/evaluation.py`
`DRAW_SEED_BASE = 5200`; `gate_m4.protocol draw_stream_base: 5200`). The estimator
is the mean over K draws, scored **once** as `|ln(rbar / rate_a)|`
(`evaluation.py _load_scoring`; `gate_m4 estimator`). Components spawn disjoint
child streams by tag: `SeedSequence([5200 + k, 0xC2])`, `0xC5`, `0xC7`, occupancy
`0xB2B` (`base_simulator.py:72-73`). The simulation stream is kept **distinct**
from the split stream (`gate_m4` finding 2; `gate_w1` streams `9100`/`9200`).

### 3.2 The (draw × module × period × person) spawn tree

M6 adds a period axis; every module × period × person draw must be addressable and
bit-reproducible. The proposed registry is a hierarchical `SeedSequence` spawn
tree rooted at the inherited base:

```
root(k)              = SeedSequence([5200 + k, GATE_M6_TAG])
period(k, t)         = root(k).spawn(T+1)[t]          # ordered child per year t = 0..T
stream(k, t, module) = period(k,t).spawn per module in the fixed §2.2 order
generator(k,t,module)= default_rng(stream(k, t, module))
```

`GATE_M6_TAG` is a fixed engine constant distinct from any component tag;
`MODULE_TAG`s reuse the existing child-tags (`0xC2`/`0xC5`/`0xC7`/`0xB2B`) at
`t = 0` so single-period draws stay byte-identical. Per-person draws are consumed
in canonical `person_id`-sorted order, so person `i`'s draw at `(module, period)`
is a pure function of `(k, t, module, i)`. The engine passes
`generator(k,t,module)` **into** each component (the F3 fix); it never lets a
component construct `default_rng(core_seed)` from a frozen param — engine-side
plumbing, no change to certified params (R1, §2.6).

### 3.3 Disjointness guarantees and the separate split stream

The spawn tree gives disjointness across draws (distinct `k` root the tree at
distinct entropy → independent K=20 members), across periods (distinct `t` are
distinct ordered children → re-running period `t` is exact, no two periods share
entropy — the F3 fix), and across modules (distinct spawn positions within a
period). The **floor split stream stays separate** from this simulation tree: the
temporal fit/holdout boundary is a time cut, but the `gate_m6` floor (§4.7) draws
person/couple/household-disjoint half-splits of the holdout window, and those
split seeds use their own base disjoint from `5200 + k` (`gate_m4` finding-2
discipline).

### 3.4 K-draw ensembles at the engine level

`gate_m6` reuses K=20 and `5200 + k` verbatim (decision 7), so the draw-noise-free
half-normal OC basis (`gate_2a.faithful_candidate_oc basis_note`) transfers
unchanged. Projection variance **compounds over periods** (a period-`t` draw feeds
`t+1`), so a late-window cell's across-draw sd is larger; this is priced correctly
because the floor and the candidate use the same K and window. If the floors build
reveals degenerate ensemble width under compounding, that is a floors-ceremony
finding to bring back, not a silent redesign (adjudication 7).

## 4. The temporal-holdout gate (`gate_m6`, proposed)

### 4.1 The fit/holdout split and the dual-axis shock-window pin

**Proposed boundary `T* = 2014`: fit on all PSID observations dated `≤ 2014`;
project and score the held-out window `2015–2022`.** Wave facts (verified against
the readers and `gates.yaml views`): earnings are biennial from 1999
(`data/family.py`: waves 1994-2023, income **reference years** 1993-2022), and the
gate-1 earnings view (`views.psid_family_earnings_pairs`) is `period_range [1998,
2022]`, `period_step 2`; marital/fertility/mortality flows are annual to
`MAX_YEAR = 2023` (`data/transitions.py`); PSID death file 1968-2023
(`data/deaths.py`).

Justification: an 8-year horizon is long enough that projection drift — the thing
M6 adds over the M1–M5 single-period generators — accumulates and becomes visible,
short enough that the youngest fit cohorts still carry holdout exposure; it keeps
the fit inside the biennial-from-1999 regime.

**Shock-window demotion, uniform across all modules (adjudication 2, finding 10).**
The 2020–2022 COVID/inflation shock is outside the engine's model class (no
macro/epidemic channel) and hit mortality (2020–21 excess deaths) and
marital-transition rates (the 2020 marriage collapse) at least as hard as
earnings. All shock-window cells across all modules are **report-only** with one
machine reason, `exogenous_shock_outside_model_class`, pinned on a **dual axis**:

- **earnings (reference-year axis):** report-only `{2020, 2022}`; **reference year
  2021 is unobserved** (odd year, no biennial interview — `data/family.py`
  biennial-from-1999); gated earnings reference years `= {2016, 2018}` (2 pre-shock
  biennial waves ⇒ ~2 change-transitions).
- **flows (event-year axis):** report-only `{2020, 2021}` (excess deaths, marriage
  collapse); gated event years `= {2015–2019}` (5 annual years).

This reconciles the earlier "4 biennial earnings holdout waves / gates 2015–2022
flows" wording to the post-demotion surface: gated `= 2015–2019 annual demographic
flows + 2016/2018 earnings waves`. The shock cells are a first-class published
diagnostic (how far a mechanical engine misses a pandemic, quantified against
held-out truth). The whole disposition is **pre-lock** — the amendment-2 lesson
(identify concept-class-unclearable cells before registration) applied before the
gate, exactly as the attrition demotions are (§4.4).

### 4.2 Prohibited re-estimation, spec-selection, and the spec-vs-fit distinction

The discipline that makes the holdout honest for **parameters** is **temporal
disjointness**: persons overlap fit and holdout by design (the projection's input
is today's realized cross-section), but no post-`T*` information may enter any
fitter. The complete refit enumeration — **including the gate-1 earnings generator**
(finding 2), the object §4.1's earnings-moment cells score:

- no hazard, initial-state model, spousal-gap table, fertility kernel, disability
  rate, claiming distribution, **or gate-1 chained-QRF earnings generator** may be
  fit, refit, tuned, knot-selected, or candidate-selected using any person-year
  dated after `T*`;
- no alignment-layer coefficient may be re-anchored to a post-`T*` Trustees vintage
  or realized aggregate;
- concretely, the certified `CANDIDATE_16` / `CANDIDATE_9` / M4 / **gate-1
  earnings-QRF** fits are re-run with their estimation panels truncated at `T*`,
  and the projection re-draws 2015–2022 state seeded from each person's realized
  2014 state.

**Forward earnings — the certified gate-1 generator is backward (design amendment
§2.7).** The enumeration lists the gate-1 earnings generator, but the certified
candidate-11 chain is a **backward** biennial imputation
(`run_gate1_candidate11.py`; §2.7), so its admissible `≤2014` refit is the
participation-gate plumbing (PR #173), **not** a forward generator. The forward
scored earnings at `{2016, 2018}` come from the **forward conditional-rank chain of
§2.7** — a **different stochastic law** fit `≤2014` by construction and
**first-certified by `gate_m6`**, not covered by `gate_1`. It enters this
enumeration as a `≤2014`-fit object (leakage-safe) whose out-of-sample 2016/2018
output the gate scores; no `gate_1` certificate transfers to it (§2.7.3, decision
10).

**Spec-vs-fit (finding 3, reconciling §7's "no new estimation").** Refitting a
frozen spec on ≤2014 produces new fitted numbers — re-estimation of *estimates*,
with the *spec* unchanged. The M6 run pins the same `CandidateSpec.sha256` (which
hashes params/impl, not estimates or data — `family_transitions/registry.py:118-124`;
the registry still rejects any param mismatch, `:148-153`) but a **different**,
truncated-panel **fitted artifact**, and it neither reads nor writes the certified
full-window artifacts. §7's "no new estimation" means no new **spec**; the
truncation is a data input, not a param change.

**Spec-selection-on-full-sample (finding 1, the residual second-order leak).** The
holdout cures parameter leakage but not **structure** leakage: `CANDIDATE_16/9`,
M4 (`disability_hazard_sim`), and the gate-1 chained-QRF (`gates.yaml:122-126`)
were *structurally* chosen by audited ladder searches that observed 1998–2022,
including the 2015–2022 holdout. The temporal holdout tests refit-stability and
drift **conditional on that selection** and cannot cure it. Per the #42 comment
4948637741 search-disclosure addendum, this is a named limitation: the holdout is
out-of-sample for fitted parameters, **in-sample for spec/structure selection**;
any pass is framed as "reproduces held-out moments after an audited iterative
search whose *structure* was chosen on the full window," never a structural
out-of-sample claim (carried into decision 10's certified-claims language and
§4.9).

**Biennial boundary peek (finding 12).** Annual flows reconstructed from biennial
interviews (`data/transitions.py`) can date a reference-year-2014 transition using
the 2015 interview — a one-interview peek across the cut. Conservative rule
adopted: **any flow event whose dating requires a post-`T*` (2015+) interview is
excluded from the ≤`T*` fit** (no post-`T*` interview informs a fit-window flow),
so the fit/holdout cut is unambiguous. The alternative (bounding the peek) is
rejected as harder to audit.

### 4.3 Why PSID-vs-PSID avoids the W1 concept-bridge trap

`gate_w1` amendment 1 demoted all 10 family-B cells because the gate scored a PSID
work-limitation **point-prevalence** against an SSA disabled-worker **beneficiary
stock** across a concept bridge it never defined
(`docs/amendments/gate_w1_amendment_1_family_b_di_bands.md` §2; misses 2.9×–21.9×
tolerance, `concept_delta_dominant_share 0.595`). `gate_m6` family A does not
repeat this: both sides of every family-A cell are the **same PSID instrument** —
projected PSID state vs realized PSID state, same definition, population, severity
threshold, self-report. The seven `gate_m4` `concept_deltas` are **zero** between a
projected PSID rate and a realized PSID rate. The external-aggregate comparison
that *does* carry concept/coverage deltas (CPS, SSA, OCACT) is quarantined in the
report-only / margin-gated family B (§4.8), not in family A. (The remaining
selection concern is not a concept bridge but PSID attrition — §4.4.)

### 4.4 Support basis and symmetric presence-conditioning (findings 7, 9)

A temporal holdout on PSID persons confronts two selection facts: reproduction-mode
conditioning on realized support, and PSID attrition on the truth side.

**Support basis differs by module.** `disability_hazard_sim.simulate_draw`
preserves the realized holdout support and re-draws only state; mortality and
marital cells cannot be scored that way — conditioning a death-hazard test on
realized survival-through-the-interval is circular. So:

- **disability** is scored in reproduction mode (realized support). The same
  wave-presence selection sits on both the projected and the realized rate, so the
  common selection largely cancels in the projected-vs-realized ratio; a disability
  PASS certifies a **temporal extension of `gate_m4`** (the fitted hazards
  reproduce a future PSID window), **not** M6's forward disability path (which
  synthesizes support — F5, decision 5's successor gate). The claim is scoped
  accordingly so certification is not over-transferred (finding 7 (iii)).
- **mortality and marital** cells are scored on **presence-at-start-of-interval**:
  among persons realized-present and at risk at wave `t`, the projected transition
  rate over `[t, t+1]` vs the realized rate. Presence-at-`t` is not a modeled
  outcome of the engine (PSID retention is not simulated), so conditioning on it is
  **conditioning-not-leakage**, on the same footing as age/sex conditioning.

**Symmetric presence-conditioning.** Both the projection cells and the truth cells
(and the §4.7 floor) are computed on the same realized-wave-presence set per
interval, disclosed as a conditioning choice, not a leak.

**Attrition bites worst on mortality (friction F7).** Presence-at-`t` is exogenous
to the interval's transition, but attrition *within* the interval is not fully so:
a person present at `t` who dies in `[t, t+1]` is observed as a death (`data/
deaths.py` `ER32050`), but one who **attrites** (leaves the panel, not dead) is
censored, and if attrition correlates with mortality the realized death-hazard
exposure is retention-selected. Per-cell dispositions:

- mortality/marital cells whose truth is attrition-confounded beyond repair are
  **demoted to report-only pre-lock**, machine reason `attrition_confounded_truth`
  — the amendment-2 lesson applied before the lock, as with the shock window (§4.1);
- for retained cells the residual attrition bias is disclosed; inverse-attrition
  weighting is a floors-ceremony option, not assumed;
- the **forward production engine** cannot condition on realized future presence
  (no PSID panel over the projected future), so its mortality/marital path is a
  different object; its delta from the gated presence-conditioned path is stated,
  and the forward variant is decision 5's registered successor gate.

### 4.5 The gated observables and cell definitions

Four observable classes on the post-`T*`, post-demotion surface (§4.1 shock, §4.4
attrition). Cell naming follows the `gate_w1` family-A convention
(`observable.stratum`).

> **Superseded by §4.10 (v3).** This section describes the *design-time candidate*
> surface at age-band × sex granularity, and reads as if survival curves and all
> four marital transitions were gated. They are **not**. The OC-before-lock pause
> and the candidate-blind coarsening/decompounding ladders (§4.10) demoted this
> surface to the actually-lockable one: **mortality gates nothing** (see
> `not_certified`, mortality drift first); the marital surface gates only
> `first_marriage.18-29|female`, `divorce.18-44`, and `remarriage.18-64` (the
> completed asymmetric-age-2 rung); **widowhood gates nothing**; earnings is
> decompounded to 6 cells. Read §4.10 for the binding registry.

- **Survival curves (mortality flow).** Cell = the realized death hazard
  `q(band, sex)` = deaths ÷ start-of-interval exposure among presence-conditioned
  holdout person-years (§4.4), per (age-band × sex) over bands {…, 55-64, 65-74,
  75-84, 85+}, event years 2015–2019. Scored `|ln(rbar_projected / q_realized)|`.
  Attrition-confounded-beyond-repair cells demoted per §4.4.
- **Marital composition drift.** Gated: the four transition hazards realized in the
  holdout window — `first_marriage`, `divorce`, `widowhood`, `remarriage` — per
  (age-band × sex), event years 2015–2019 (couple-disjoint floor). Report-only /
  margin: end-window marital-state shares (§4.6).
- **Earnings distribution moments.** The `gates.yaml moment_battery` on the gated
  earnings reference years {2016, 2018}: `change_moments` (mean, sd, skew,
  kurtosis of Δlog-earnings, by cohort, log), `age_profile` (quantiles 0.1/0.5/0.9
  by cohort), `mobility_matrix` (5-bin, zero-bin, horizons 1–2 in wave units),
  `zero_spells` (person-disjoint floor). `autocorrelation` lag 5 is **report-only**
  by measurability — a 10-year statistic is not computable inside the holdout,
  machine reason `horizon_exceeds_holdout_span`; its status is re-derived, not
  grandfathered, if `T*` moves (adjudication 1).
- **Disability stocks/flows.** Gated flows (reproduction mode, temporal gate_m4
  extension per §4.4): work-limitation `incidence` and `recovery` hazards per
  (age-band × sex), event years 2015–2019, and the near-FRA disability-exit split.
  Report-only / margin stock: disabled-occupancy `prevalence`. Person-disjoint
  floor.

### 4.6 Flows vs stocks, mixed-k, and the trivially-passable-stock problem

Over the holdout window the end-of-window **stock** is nearly a deterministic
function of the realized `T*` stock plus the gated **flows**, so double-gating a
stock adds little discriminating power and risks a trivially-passable headline.
Disposition (decision 3): gate the **flows** as family-A primary; treat
end-of-window stocks (survival curves, marital shares, disability prevalence) as
report-only with a `gate_m4`-style **≥3σ margin check**, not a `|ln|` level gate.
Where stocks are gated, adopt the `gate_m4` **mixed-k** discipline: FLOW cells
`k = 3`, STOCK cells `k = 4`, tolerance `round(mean + k·sd, 3)` capped at
`ln(1.5)` (`evaluation.derive_tolerance`).

### 4.7 The floor, weights, and presence-conditioning

The floor is the deployment-scale sampling null of the holdout-window observable,
built as the prior gates build theirs (`noise_floor method: half_vs_half`), but
split within the holdout window and **presence-conditioned symmetrically** (§4.4):

- split the presence-conditioned holdout persons into two disjoint halves at the
  correlation-respecting unit (person / couple / household as the observable
  dictates; `gate_w1` family-A `split_note`, `gate_2c` couple-disjoint);
- compute each gated observable on each real half; score `|ln(rate_a / rate_b)|`;
  repeat over floor seeds `0–99`; the (mean, sd) per cell derive the tolerance
  `round(mean + k·sd, 3)` capped `ln(1.5)`.

**Weight convention (F6, finding 8).** Both the projected and the realized side —
and the floor — use the person's **realized-`T*` start-wave weight held fixed
across the projection window**, extending the `gate_m4` start-wave definition to
the projection axis. The per-year calibrated panel weight (§2.1) is **not** used
for the gated rates (it would put a different weight on the two sides and bias
every ratio); it is a report-only alternative. The floor must be built on the same
start-wave weight it scores against.

The candidate is scored `|ln(rbar_projected / rate_a)|` where `rate_a` is side-A's
realized presence-conditioned holdout rate and `rbar_projected` is the K=20-draw
mean of the engine projecting side-A persons from their realized `T*` state. This
is the `gate_m4` construction with the time axis substituted for the re-drawn-state
axis. The floor artifact (`runs/m6_holdout_floors_v1.json`, **not built by this
document**) is authored in the ceremony, sha-pinned, read-never-rewritten.

### 4.8 Families, the one-shot rule, and report-only tiers

Mirroring `gate_w1`'s three-family structure:

- **family A — dynamics drift (floor-priced, certifiable).** The §4.5 flow cells on
  the post-demotion surface, `|ln(rbar/rate_a)| ≤ round(mean + k·sd, 3)` capped
  `ln(1.5)`, K=20 mean on `5200 + k`, gate seeds `0–4`, **4-of-5 seed
  conjunction**. This is the surface a `gate_m6` PASS certifies.
- **family B — external alignment audit + entrants (report-only / margin-gated).**
  The projection's aggregate agreement with realized external series (CPS, NCHS,
  SSA, OCACT triangulation), the per-run alignment-layer interventions and the
  per-year **maximum alignment displacement** (adjudication 9, a reported not
  gated magnitude), and the **entrant cells** (synthetic births + immigrant
  cohorts) with their named bridge requirements (adjudication 4). All report-only
  or bounded/ordinal margin, never `|ln|`-level-gated (the W1 lesson; the M2
  `calibration_disclosure` lesson for the alignment layer).
- **family C — drift-direction fingerprints (binary, optional).** A small set of
  pre-committed qualitative directions the projection must reproduce, binary like
  `gate_w1` family C.

**One-shot rule.** Each candidate is registered on #42 before its run and writes
its artifact through `artifacts.write_new`; the `undefined_draw_rule` carries over
(an undefined gated-cell rate on any draw invalidates the run —
`gate_2a.fresh_run_artifact_schema.undefined_draw_rule`).

**Report-only re-drawn-seed comparison (adjudication 5b).** `gate_m6` also
publishes, alongside the gated realized-`T*`-seed run, the same holdout cells
scored once from **re-drawn** `T*` initial states, with a pre-named margin that
seeds the successor gate's flip criteria — making initialization error visible in
the M6 record rather than deferred.

### 4.9 Operating characteristic, pass-verification, and search disclosure

The OC is computed on the draw-noise-free half-normal basis from the family-A
floor σ (`gate_2a.faithful_candidate_oc basis_note`), reported at the ceremony from
`runs/m6_holdout_floors_v1.json` once built; this document states no OC number
because the floor does not yet exist (stating one would be fabrication, the posture
`gate_w1_amendment_1` §4b takes). The post-adjudication surface is **thin** (the
shock and attrition demotions shrink N), so §9 requires the OC be checked before
lock in **both** directions (near-vacuous from wide tolerances, or unfailable).

Two standing #42 process rules (comment 4948637741) bind `gate_m6`:
**pass-run verification** (a PASS enters the record only after independent
adversarial bit-exact reproduction) and **ladder-search disclosure**. The latter is
strengthened by finding 1: the disclosure names not only the M6 gate's own ≤`T*`
forensics but the **spec-selection-on-full-sample** limitation (§4.2) — a pass is
"reproduces held-out 2015–2019 / 2016–2018 moments after an audited iterative
search whose *structure* was chosen on the full window," and the 2100 production
projection is report-only extrapolation beyond the validated window (decision 10,
§7).

### 4.10 Post-pause surface redesign — the power-derivation and the v1→v2→v3 registry

The floors ceremony's step-1 OC-before-lock check (§4.9) **fired the §9 pause**.
The frozen v1 floor artifact `runs/m6_holdout_floors_v1.json`
(sha256 `16c28d8cd9095e5233ab224c659c8d5b9eb1621099e2524455a3a8ff8e88d318`, the
**pause evidence**) priced the §4.5 age-band × sex surface and found it does not
support a lock in **either** direction: the certifiable **flow** surface was
near-vacuous (a single gated flow cell, `first_marriage.18-29|female`), because an
8-year biennial holdout at that granularity yields ~0.1 half-split log-ratio noise
per flow cell, so at the design-pinned FLOW `k=3` most flow tolerances land just
over the `ln(1.5)` cap; and the combined faithful `p_gate` was **0.8449 < 0.90**
(the named weak-power floor), because 16 earnings-moment cells compound under the
4-of-5 conjunction. The pause is the correct outcome, not a defect: it is the W1
lesson (identify an unlockable surface **before** locking) applied to the power of
the surviving surface.

The adjudicated redesign is executed by two **candidate-blind** ladders
(`scripts/build_m6_holdout_floors_v2.py` → `runs/m6_holdout_floors_v2.json`). They
are candidate-blind *by construction*: every choice is a function only of
truth-side power arithmetic — floor sigmas, weaker-half event counts, and v1
`tolerance/σ` ratios — and never references an engine, a candidate, or what a model
would find easy or hard.

- **Coarsening ladder (flows).** Per transition type, pool **adjacent** strata
  minimally in the pinned order *sex-pool → age-pool-adjacent*, climbing until a
  cell's tolerance `≤ ln(1.5)` **and** `≥ 20` weaker-half events; adopt the
  **minimal** rung with `≥ 1` clearing cell uniformly for that transition type.
  Mortality `85+` stays report-only (`attrition_confounded_truth`) at every rung.
  Outcome (v2): `first_marriage` clears already at age×sex (`.18-29|female`);
  `divorce` clears sex-pooled at ages `18-44`; disability `incidence`/`recovery`
  clear only fully pooled (`.20-66`); **mortality, `widowhood`, and — in v2 —
  `remarriage` cleared nothing at the rungs v2 enumerated.** The v2 gloss "too thin
  for them even pooled" is corrected at v3 for `remarriage` (see the v2→v3
  correction below): it was true of the *enumerated rungs*, not of the data.
- **Earnings decompounding ladder.** Prune gated earnings cells
  **weakest-power-first** (largest v1 `tolerance/σ` first), recomputing the
  combined `p_gate` after each prune, stopping at the **first** `p_gate ≥ 0.90`,
  never below `≥ 1` gated cell per concept family (log-quantiles, dispersion,
  mobility, zero-rate, autocorrelation, change-mean). This retains the **largest**
  surface meeting the power floor under the ladder — *maximum falsifiability
  subject to power*. Outcome: 9 of 16 earnings cells pruned, 7 retained (one per
  concept family, plus a second `change_mean`).

**As-committed v2 gated registry: 11 cells — 4 flow + 7 earnings** (frozen,
`runs/m6_holdout_floors_v2.json`, sha256 `3f273d47…`):

| Family | Gated cells | Adopted rung |
|---|---|---|
| marital | `first_marriage.18-29\|female`, `divorce.18-44` | age×sex; sex-pooled age `18-44` |
| disability | `incidence.20-66`, `recovery.20-66` | sex-pooled, fully age-pooled |
| earnings | `earn_p10.prime`, `earn_zero_rate.older`, `earn_dlog_sd.older`, `earn_mob_h1_diag`, `earn_autocorr_lag2`, `earn_dlog_mean.prime`, `earn_dlog_mean.older` | v1 person-disjoint floor (unchanged), decompounded |

The v2 faithful-candidate OC on this surface is **`p_seed 0.8921` / `p_gate 0.9067`**
(flows-only `0.9882`, earnings-only `0.9518`), and the certifiable flow surface
carries **4** gated cells. Every number is arithmetically clean and independently
reproduced.

#### The v2→v3 correction — completing the marital ladder's rung enumeration

**v2's marital ladder enumeration was incomplete.** The adversarial referee
(comment `4958425437`, amendment 2) rebuilt both artifacts bit-identically from
PSID and swept **every** contiguous adjacent-band pooling with the ceremony's own
machinery. Mortality and widowhood are pooling-invariant (nothing admissible
clears anywhere — mortality best `~0.462`, widowhood best `~0.474`). **Remarriage
is not.** It clears at exactly one rung the v2 marital ladder never enumerated: the
**asymmetric age-2 partition** `[18-64],[65+]` — `remarriage.18-64` at tolerance
**`0.403`** (`σ 0.1538`, 143 weaker-half events, 100/100 seeds, household-disjoint).
So the v2 gloss "too thin for remarriage even pooled" was true only of the
*enumerated* rungs, not of the data. Critically, this is **not a new design
choice**: the asymmetric "merge the lower bands wide, isolate the elderly tail"
shape is one the **mortality ladder itself already used** (its
`[25-54],[55-74],[75-84]` and `[25-64],[65-84]` rungs). Completing the marital
ladder to enumerate the same rung shapes — uniformly per transition type, as
originally pinned — is fidelity to the adjudicated candidate-blind criteria, not a
second design pass. This is the **orchestrator's v3 adjudication** of the referee's
amendment 2.

The completion is a single machinery change (`scripts/build_m6_holdout_floors_v3.py`,
`MARITAL_LADDER` gains `sex_pooled_age2p = [18-64],[65+]` after the symmetric
age-2 rung; ordered by coarseness, since the symmetric split's widest merged group
spans 2 raw bands while the elderly-isolating split's spans 3). Under the same
minimal-rung rule it adopts, uniformly: `first_marriage → age×sex` (unchanged),
`divorce → sex-pooled age 18-44` (unchanged), **`remarriage → sex-pooled age 18-64`
(new)**, `widowhood → nothing` (now recorded as evaluated at the asymmetric rung
too), `death → nothing`. The extra 5th gated flow cell lowers the combined
`p_gate` at every earnings-prune step, so the **same pinned weakest-first earnings
ladder is forced exactly one further prune** (`earn_dlog_mean.older`, the next
non-last-in-concept cell: `0.8944 → 0.9087`), retaining 6 earnings cells (still one
per concept family).

**v3 gated registry: 11 cells — 5 flow + 6 earnings** (`runs/m6_holdout_floors_v3.json`,
sha256 `e931c88622fad84e8f8b2cf18940cbe27da1c93e0d009dfbaa3d6c6cae050c77`; v1
`16c28d8c…` and v2 `3f273d47…` stay **frozen** as lineage):

| Family | Gated cells | Adopted rung |
|---|---|---|
| marital | `first_marriage.18-29\|female`, `divorce.18-44`, `remarriage.18-64` | age×sex; sex-pooled `18-44`; sex-pooled `18-64` (asymmetric age-2) |
| disability | `incidence.20-66`, `recovery.20-66` | sex-pooled, fully age-pooled |
| earnings | `earn_p10.prime`, `earn_zero_rate.older`, `earn_dlog_sd.older`, `earn_mob_h1_diag`, `earn_autocorr_lag2`, `earn_dlog_mean.prime` | v1 person-disjoint floor (unchanged), decompounded |

The v3 faithful-candidate OC is **`p_seed 0.8934` / `p_gate 0.9087`** (flows-only
`0.9822`, earnings-only `0.9626`); the certifiable flow surface now carries **5**
gated cells across **three** marital transition types plus both disability flows —
a strictly larger, more falsifiable flow surface than v2, on the ceremony's own
maximum-falsifiability-subject-to-power principle. The completion held every
invariant fixed: `T* = 2014` (**no window extension** this round — that is a
separate adjudication re-deriving decisions 1–2), lag-5 report-only, FLOW `k = 3`,
the `ln(1.5)` cap, the 4-of-5 conjunction, and the `0.90` floor. `incidence.20-66`
is the weakest gated flow cell (`0.404`, 99.6% of the `ln(1.5)` cap); no gated
tolerance sits at its cap, so the restored third vacuity guard (not all tolerances
capped) is satisfied.

**What v3 still certifies nothing about** (the `not_certified` declaration, block
draft, at the same prominence as the PASS claim — referee amendment 1): **mortality
drift first** (no admissible pooling clears the `25-84` surface even fully pooled,
best `~0.472`; `85+` is `attrition_confounded_truth`; `85+`-inclusive pools clear on
power but ~27% of their events are the confounded stratum, so pooling dilutes
rather than cures), then **widowhood** (best `~0.474`), the **2020–2022 shock
window**, **entrants / open panel**, **autocorrelation lag-5**, the **2100
forward-projection extrapolation**, all **stock margins**, and the remarriage `65+`
tail. Mortality is therefore seeded into M7 **report-only-with-anchor**, never as
certified drift: the block names the **SSA/NCHS life-table mortality anchor** as a
required report-only family-B deliverable before the lock flip can claim any
M7-seeding support (`|ln|`-gating external mortality *levels* stays rejected — the
W1 concept-bridge lesson; referee ruling in comment `4958425437`). The circularity
that the engine's mortality input is itself NCHS×PSID-band anchored is disclosed
with the anchor (the M2 `calibration_disclosure` lesson): agreement validates the
alignment path, not independent drift.

*Public-record citations at flip time:* the **candidate-blind surface-redesign
adjudication** (the two pinned ladders) lives in this §4.10 and the PR #172 body;
the **v3 adjudication** (complete the pinned ladder's enumeration) is referee
comment `4958425437` amendment 2, adopted by the ceremony orchestrator. Both must
be cited on the public record when the lock flip lands, so the flip carries the
adjudication text, not only its result.

## 5. PolicyEngine integration — the W2 pattern per projection year

The seam is the ADR-0001 W2 pattern applied once per projected calendar year. pe-us
takes `social_security` as an **uprated survey input** — no benefit formula exists
upstream (`w2_seam_caregiver.py` header; ADR-0001 W2). For projection year `y`, the
engine's per-person modelled benefit (from `ss/benefits.py`, plus the M4 DI and
402(b)/(c)/(e)/(f) auxiliary paths) **replaces the `social_security` column** of
that year's panel slice, and a standard managed `policyengine.py` microsimulation
yields full tax-and-benefit incidence through existing pe-us machinery unchanged
(`pe.us.managed_microsimulation()`).

Division of ownership: **stays in populace-dynamics** — the stochastic dynamics
(§2 engine), the statutory benefit oracle (`ss/`), the panel, the alignment layer,
and `gate_m6` scoring; **runs in policyengine.py** — the deterministic per-year
cross-sectional tax-benefit incidence, one calendar year at a time, **one
simulation per process** (`w2_seam_caregiver.py` launches separate subprocesses;
`POPULACE_DYNAMICS_PE_US_DIR` / `POPULACE_DYNAMICS_PE_PYTHON`). The seam is
per-year and stateless: pe-us sees a calendar-year cross-section with a modelled SS
column, exactly the W2 object. The `w2_seam_caregiver.py` sparse-57k caveats carry
forward (zeroed untargeted inputs → means-tested offsets move in the right
direction, magnitudes interim). The full cross-program execution across **all**
year-slices at once is M8, not M6 (§7).

## 6. M7 interface sketch (levels the engine must expose)

Interface only — the trust-fund arithmetic is #113 M7. M6 must expose, keyed by
`(year)` and aggregable by cohort from the panel, the level quantities M2 already
computes on the **closed** 1943–1957 frame (`runs/m2_pseudo_projection_v1.json`)
but now on the **open** projected panel with entrants:

- **covered / taxable earnings aggregate by year** — `Σ w · min(earnings_y,
  taxable_max_y)`, the M2 `taxable_payroll_convention`; expose both **capped** and
  **uncapped** person earnings so cap-reform provisions (`cap_150k`, `elimination`)
  recompute without re-simulation (M2 `revenue_side.cap_150k`, `payroll_increments`).
- **taxable-maximum interaction** — each person's year-`y` earnings and wage base.
- **benefit outlays by type by year** — OASI retired-worker, DI disabled-worker,
  402(b)/(c)/(e)/(f) auxiliary, survival-weighted on the calendar (M2
  `outlay_side.convention`, extended with M4 DI and the auxiliaries; FRA conversions
  per `disability_conversion.py`).
- **immigration entries by year** — the synthetic entrant cohorts M2 structurally
  lacks (its frame is a **closed** cohort, `n_common_frame 1549`, `weight_sum
  33696344.0`).
- **the per-year alignment-layer adjustments** — the versioned interventions
  applied each year (§4.8), so M7 reports scheduled-vs-payable on an audited
  alignment (#113 named-hard-part 2).
- **the OASDI rate constants** — pe-us nodes where they exist (M2
  `revenue_side.oasdi_combined_rate combined 0.124`); the discount / Trustees
  vintage assumptions M2 carries (`balance_analogue.discount_rate 0.029`;
  `tr_vintage_cite`) become M7's versioned alignment inputs.

M6 stops at exposing these levels; M2 demonstrates the accounting they feed
(`balance_analogue`, `exhaustion_analogue baseline_exhaustion_year 2034.0`) on the
closed frame, and M7 lifts it onto the open panel.

## 7. Explicit non-goals for M6

- **No behavioral response.** Claiming is drawn mechanically from Table 6.B5.1; no
  labor-supply, claiming-timing, or savings response (#113 named-hard-part 4; M2
  "no behavioural response, no claiming change"). Behavioral modules plug in later
  behind their own gates.
- **No macro feedback.** Wages, prices, interest, the wage base are exogenous
  Trustees inputs; no general-equilibrium loop. The 2020–2022 shock is not modelled
  and its cells are report-only (§4.1).
- **No new spec estimation.** M6 composes the M1–M5 certified **specs** unchanged
  (same `CandidateSpec.sha256`); it re-runs their **estimates** on ≤`T*`-truncated
  panels (finding 3, §4.2) and fits nothing new except the alignment layer (a
  versioned adjustment, disclosed, not a certified generator). `gate_m6` certifies
  composition/projection, not spec re-estimation.
- **No trust-fund accounting (M7); no whole-panel rules scoring (M8).** M6 exposes
  the §6 levels and runs the per-year W2 seam (§5); it computes no actuarial balance
  and runs no simultaneous cross-year rules execution / W3 cross-validation.
- **No validated projection beyond the holdout window.** Only the post-demotion
  2015–2019 / 2016–2018 surface is gated; the 2100 path is report-only
  extrapolation, and — per finding 1 — even the gated claim is conditional on
  full-window spec selection (§4.9).
- **Deferred to the decision-5 successor gate:** the forward-projection variant
  (synthesized support, endogenous widowhood, re-drawn initial states — F1/F5); no
  claim of a certified projection *engine* until it passes.

## 8. Open design decisions (with adjudications)

The ten decisions were adjudicated in comment 4953722912; this section records the
outcome and the revision-1 refinements. Where the design referee's findings changed
a resolution, the change is noted.

1. **Earnings-moment cells — ACCEPTED.** Gate `change_moments`, `age_profile`,
   `mobility_matrix` (h 1–2), `zero_spells`; `autocorrelation` lag 5 report-only,
   machine reason `horizon_exceeds_holdout_span`, re-derived not grandfathered if
   `T*` moves.
2. **`T* = 2014` — ACCEPTED, shock demotion extended to all modules** with the
   dual-axis reference-year/event-year pin and machine reason
   `exogenous_shock_outside_model_class` (§4.1, finding 10).
3. **Flows family-A, stocks report-only with ≥3σ margin — ACCEPTED** (§4.6).
4. **Closed panel gated; entrants family B with named bridge requirements —
   ACCEPTED** (§4.8).
5. **Realized-`T*` seeding + M4 reproduction mode + exogenous widowhood —
   ACCEPTED, with two conditions:** (a) the forward-projection variant is a
   **registered successor gate**, not optional hardening; the 2100 path stays
   report-only until it passes; (b) `gate_m6` publishes a **report-only re-drawn-seed
   comparison** with a pre-named margin seeding the successor gate's flip criteria
   (§4.8). Finding 7 adds the support-basis split: mortality/marital are
   presence-conditioned forward-style cells, disability is reproduction-mode and
   certifies a temporal `gate_m4` extension (§4.4).
6. **One lagged edge — this revises adjudication 6, SUPERSEDED by the R2
   derivation.** Adjudication 6 accepted a lagged edge (couple formation reads
   `t-1` earnings). The code supersedes it: the 2c modifier's tercile is the
   person's **career-mean indexed positive-year earnings** — a person-constant
   capacity measure (`data/couple_earnings.py indexed_earnings_supply`;
   `couple_formation_sim_v2.py:261`), not an annual level, so there is no
   within-period annual cycle and a `t-1` (or `t`) annual level would violate R2.
   The R2-faithful composition feeds the predetermined **permanent-earnings
   tercile** at step 3, and the modifier is a marginal-preserving reweight
   (`apply_first_marriage_modifier` enforces `sum_t m·phi_cert = 1`, raising on
   violation — `couple_formation_sim_v2.py:520-525`), so composing it at step 3
   moves no (band, sex) first-marriage marginal. Adopted in place of the lagged
   edge, per adjudication 6's own instruction to derive the design from code
   rather than pick a side a priori.
7. **K = 20 on `5200 + k` verbatim — ACCEPTED**; degenerate ensemble width under
   compounding is a floors-ceremony finding, not a silent redesign.
8. **Single marital core — INJECTION, forced by R1+R2, with the R3 re-certification
   note.** P2 passed (the embedded core is the certified `CANDIDATE_16`), but the
   step order forces the core to run at step 3 (fertility reads married state), and
   avoiding candidate-9's internal re-simulation double-run requires injecting
   step-3 state and bypassing its internal `ft.simulate` — R1 surgery, adopted with
   the targeted re-certification note (§2.6). This resolves the finding-6
   contradiction by applying R1's line (bypassing a certified internal simulation =
   surgery + note), the more conservative reading.
9. **Alignment layer report-only; gate the un-aligned drift — ACCEPTED,** plus the
   per-run **maximum alignment displacement** published as a reported (not gated)
   magnitude (§4.8).
10. **Certify only the post-demotion holdout surface; 2100 = report-only
    extrapolation — ACCEPTED,** with the finding-1 certified-claims language: even
    the gated claim is conditional on full-window spec selection (§4.2, §4.9).
    **Revision 2 addition:** the certified/not-certified language also records that
    the **forward earnings generator (§2.7) is not `gate_1`-certified** — it is a
    different (forward) stochastic law than the certified backward chain, `≤2014`-fit
    by construction, and a `gate_m6` pass is its **sole** certification. No `gate_1`
    certificate transfers to it; the composed-engine pass certifies the forward law
    together with the composition.

## 9. Floors-ceremony plan (§9 deliverables)

The ceremony that seeds from this revised §4 must produce, before locking:

- **the floor artifact** `runs/m6_holdout_floors_v1.json` — presence-conditioned
  (§4.4), start-wave-weighted (§4.7, F6), correlation-respecting half-splits, floor
  seeds `0–99`, on the post-demotion surface (§4.1 shock, §4.4 attrition). v1 fired
  the pause; the candidate-blind ladders produced v2, and the completed marital
  ladder produced the binding **v3** artifact `runs/m6_holdout_floors_v3.json`
  (§4.10). v1 and v2 stay frozen as lineage.
- **the SSA/NCHS life-table mortality anchor (referee amendment 3)** — a *named*
  report-only family-B deliverable: external SSA/NCHS published death hazards
  (age × sex) for the holdout and projection years, triangulated against the
  engine's projected hazards and published with the candidate run. Required before
  the lock flip can claim M7-seeding support, because mortality certifies nothing
  on the truth side (§4.10 `not_certified`). `|ln|`-gating external *levels* stays
  rejected (the W1 concept-bridge lesson); the NCHS×PSID-band input circularity is
  disclosed with the anchor (agreement validates the alignment path, not
  independent drift — the M2 `calibration_disclosure` lesson).
- **the re-certification margin check (§2.6)** — candidate-9's household-composition
  output moments under injected-vs-internal marital state agree within a pre-named
  gate_m4-style margin; a floors-ceremony deliverable, not assumed here.
- **the OC-before-lock check (finding 14)** — compute the family-A operating
  characteristic on the **surviving** gated surface before locking, with a named
  weak-power threshold below which the ceremony **pauses for surface redesign**
  rather than locking a near-vacuous gate (the W1 lesson in both directions: no
  cell near-tautological, and no cell unfailable from wide small-N tolerances).
- **the household-ID + weight carriage rule (finding 13)** — the cross-period rule
  for household identity across composition changes (divorce → which IDs; new-entrant
  and child household assignment) and the per-period household weight the
  household-disjoint floor split and the household-level report-only cells depend on,
  consistent with the F6 weight convention (§4.7).
- **the claiming-vintage freeze (finding 11)** — a ≤`T*` claiming vintage for the
  report-only benefit levels, or the named 2023-Supplement vintage leak. **Pinned
  §2.8.10: the *2014* Supplement 6.B5.1 (`supplement_year = 2014`).**

## 10. Design-parameters summary (design proposal — not a test-bound ledger)

This block summarizes the proposed parameters and the artifact/code fields they
derive from. Unlike a ratified amendment's consistency ledger, it is **not** bound
to committed artifacts by a test — this is a docs-only design PR that builds no
floor and writes no test. The numbers become test-bound only when
`runs/m6_holdout_floors_v1.json` and the `gates.gate_m6` block are authored in the
lock ceremony.

```json m6-design-parameters
{
  "design_id": "2026-07-12-m6-projection-engine",
  "revision": 14,
  "referee_round": "PR #170 comment 4953818376 (MAJOR REVISION)",
  "adjudication": "issue #42 comment 4953722912",
  "status": "design_draft",
  "floors_ceremony_outcome": {
    "step1_v1": "PAUSED (near-vacuous flow surface + p_gate 0.8449 < 0.90); evidence runs/m6_holdout_floors_v1.json sha256 16c28d8c...",
    "redesign_v2": "candidate-blind coarsening + decompounding ladders (see 4.10); runs/m6_holdout_floors_v2.json; 11 gated (4 flow + 7 earnings), p_gate 0.9067 >= 0.90, ceremony may proceed",
    "gates_yaml_still_untouched": true
  },
  "gates_yaml_untouched_by_this_document": true,
  "fit_holdout": {
    "boundary_T_star": 2014,
    "prohibited_reestimation_includes_gate1_earnings_qrf": true,
    "spec_vs_fit": "same CandidateSpec.sha256 (params/impl), different truncated-panel fitted artifact; certified full-window artifacts neither read nor written",
    "spec_selection_on_full_sample_disclosed": true,
    "biennial_boundary_rule": "exclude any flow event whose dating needs a post-T* (2015+) interview from the fit",
    "person_overlap_intended": true,
    "disjointness": "temporal, not person"
  },
  "shock_window": {
    "machine_reason": "exogenous_shock_outside_model_class",
    "earnings_reference_year_axis": {"report_only": [2020, 2022], "note_2021_unobserved_odd_year": true, "gated": [2016, 2018]},
    "flows_event_year_axis": {"report_only": [2020, 2021], "gated": "2015-2019"}
  },
  "support_basis": {
    "disability": "reproduction mode (realized support); certifies a temporal gate_m4 extension, not M6 forward disability",
    "mortality_marital": "forward-style, presence-at-start-of-interval, conditioning-not-leakage",
    "attrition_friction": "F7; mortality worst; attrition_confounded_truth cells demoted report-only pre-lock"
  },
  "composition_R1_R2_R3": {
    "R2_marital_readers": ["fertility_step4_paternal_births", "household_composition_step8"],
    "R2_marital_blind": ["earnings_qrf", "disability", "claiming"],
    "gate2c_tercile_is_person_constant_career_capacity": true,
    "decision6_lagged_edge": "SUPERSEDED; feed permanent-earnings tercile at step 3",
    "injection_forced": true,
    "injection_class_under_R1": "surgery (bypasses candidate-9 internal ft.simulate)",
    "recertification": "targeted distributional margin check on candidate-9 composition outputs (P2 identical core; 2c marginal-preserving); not a gate-2b re-ceremony"
  },
  "forward_earnings_generator": {
    "blocker": "Sol PR #173 (BLOCKED): certified candidate-11 is a backward biennial chain; cannot compose forward without a new law",
    "law": "biennial forward conditional-rank chain 2014->2016->2018 (scored on {2016,2018}), mirrors certified conditioning structure reversed, fit <=2014 from scratch (not an inversion)",
    "transfers_from_candidate11": ["knn_k=25", "distance_weights_1/0.5/0.25", "lambda=0.1_donor_blend_Q0_exempt", "u_w_shrunk_permanent_rank_candidate8_PR58_on_candidate3_stage1c", "RegimeGatedQRF_participation_gate", "zero_anchor_full_reentry_pool", "weighted_single_record_draw_no_jitter", "age_only_covariate_marital_blind"],
    "newly_chosen_fit_window_only": ["forward_conditional_donor_pools_on_<=2014_forward_tuples", "forward_participation_gate_coefficients", "anchor_flip_initial_2014", "start_of_chain_memory_ramp_2014->2016_one_step"],
    "annualization": "NON-SCORED; real per-year forward seam has no lookahead so DEFAULT is carry-forward (participation+level) of the last drawn/realized biennial value; log-linear interpolation only via batch-compute+backfill (departs from per-period stream assignment); scored cells read only {2016,2018}; interp-vs-carry gate-immaterial",
    "certification": "NOT gate_1-certified (different/forward law); first-certified by gate_m6; no gate_1 certificate transfers",
    "seam_signature": "EarningsGenerator.generate(frame, year, rng) -> np.ndarray (steps.py:164-169); per-year, called by apply_earnings every projection year; stateful THROUGH the frame (drawn earnings column written in-place and carried by the loop's frame reassignment; _merge_period_columns threads the marital/disability/household update-frames), not a pure fn",
    "rng_slots": "per-year person stream context.person_generator(EARNINGS, person_id) = stream(k,t=year,earnings); RNG consumed only at even reference years; even-year spawns candidate-7 SUBSTREAM_CODES {gate:1,donor:2,reentry:3}; odd-year carry-forward deterministic",
    "drawn_prior_rank_threading": "at 2018, drawn-2016 rank recovered inside generate by re-ranking the frame's carried earnings column (0=non-participation); rank_{t-2} + anchor from realized-2014 columns; no generator-held cross-year state",
    "amendment_2_section": "2.7.6 (closes Sol round-2 blocker: rank<->level map undefined)",
    "rank_to_level_law": "PINNED (b2): NAWI-normalized calendar-invariant age_bin CellMarginal on pooled <=2014 positive earnings (certified fit_cell_marginals verbatim, key (age_bin,period)->age_bin); de-index rank u -> Qhat_pos_agebin.quantile(u) normalized level -> x I_proj(target_year) -> nominal level. REJECT (a) donor-level bootstrap (departs from certified marginal-quantile de-index candidate7:648-654, R2 violation, no clean inverse); REJECT (b1) carry-2014-period-marginal (§2.7.1-prohibited + biases earn_p10 low)",
    "wage_index_projection": "I_proj(2016),I_proj(2018) = OLS ln(NAWI)~year over trailing decade [2005,2014] extrapolated (== <=2014 trailing-decade geometric-mean wage growth); ss/params.py NAWI series; NEVER realized post-2014 NAWI",
    "leakage_prohibition": "realized post-T* NAWI on the SCORED path PROHIBITED (would leak into gated earn_p10/p50/Dlog-SD/mobility); frame's realized nawi column (steps.py:145-156 aging) admissible only for non-scored seam/taxable_max + report-only alignment; gated path = un-aligned projection (decision 9, §4.8; M2 calibration_disclosure)",
    "re_ranking_law": "PINNED fit-population CDF: rank_t = rhat_agebin(level / I_proj(wave)) -- the inverse of rank_to_level (Qhat_pos/rhat = np.interp on one grid), INTERIOR-EXACT; corner clamps ([ymin,ymax]/[0.001,0.999]) + flat-yval ties are deterministic-but-inexact (candidate5b:250-259, corner_bottom/top); 2018 re-rank is deterministic so chain integrity holds (inexactness = bounded corner rank perturbation); zero -> p0/Q0 regime. REJECT within-frame percentile (references synthetic projected cohort not the PSID fit population, R2 violation, breaks round-trip). R2: certified ranks ARE CellMarginal CDF positions so deployed re-rank must reference the fit-population marginal",
    "secondary_gaps_pinned": {"memory_after_2018": "two rolling generated-biennial-level cols gen_earn_w2/gen_earn_w4 + realized-<=2014 start-lags", "generator_to_substream": "engine.rng.seed_from_generator(rng)->int then SeedSequence([seed,code]) codes {1:gate,2:donor,3:reentry} (m6-engine rng.py; candidate10:239)", "frame_schema": "person_id,age,sex,u_w,realized_earn_2014,realized_earn_2012,earnings,gen_earn_w2,gen_earn_w4; engine refit.py materializes u_w+realized_earn_* at period 0; _merge_period_columns carries", "age_support": "forward law fits [25,64] (EXTENDS transferred gate-1 [25,59], refit.py:802-806) because gated cohorts prime 25-44 + older 45-64 (EARN_COHORTS build_m6_holdout_floors.py:120, no panel age cap); [25,59] is a NON-transferable hyperparameter (conflicts pre-registered surface; would kill earn_zero_rate.older); 8 age_bins {25-29..60-64}; participation-gate/donor-pools/memory/marginal all fit [25,64] pooled <=2014; clip <25->25-29, >64->60-64; <25 & 65+ non-scored; principle: law must cover the scored surface, narrowing to fit support = candidate-informed surface design PROHIBITED"},
    "closes_round2_blocker": true,
    "lock_flip_design_commit_finalizes_to_this_amendment_merge": true
  },
  "scored_run_harness": {
    "amendment_3_section": "2.8 (unblocks the gate_m6 run; designed stop #42 comment 4962773701, registration 4962640241); revised per amendment-3 referee MAJOR REVISION #42 comment 4963629234 (F1 adjudicated + F2-F6 pins + fertility disclosure)",
    "amendment_3d_section": "2.8.10 (the <=T* external-reference input bindings; closes the run lane's SECOND designed pre-scoring stop, #42 registration 4967241464 graded comment 4967433717); resolves finding 11 via the claiming-vintage-freeze branch; revised per amendment-3d referee SPEC-SOUND #42 comment 4968160638 (NAWI rule verified end-to-end; 8 refinement findings landed at revision 10)",
    "amendment_3e_section": "2.8.10.5 (the mortality-rates SHAPE BRIDGE + parameter-dir binding; DOCS-ONLY, closes factory-referee F1/F2 against the merged spec/engine pair, PR #191 comment 4969829131, on the factory's real outputs -- factory 67e7fad graded MERGE-READY build-only). F1 BLOCKING: the pinned seven-band mortality_external_rates shape is rejected by AgeSexMortalityModel.__post_init__ (bands must start at age 0, contiguous through 120, steps.py:57-68) at the run's FIRST phase fit_mortality_model (m6_runner.py:348, before pre-flights/score/write); a bare added 0-24 external row then hits the undefined-fit-cell guard (no <25 PSID exposure; build_exposure_slices drops out-of-band ages). RESOLUTION = option (b), input-side INERT (0,24) projection-coverage pad on BOTH fit inputs, engine untouched: external {central_rate = NCHS (l0-l25)/(T0-T25) ~ 0.000766 male / 0.000451 female, OUTCOME-INERT because aligned_rate = central_rate*(psid_rate/central_rate) = psid_rate cancels it, refit.py:1083-1088} + exposure {event_year = required_interview_year = 2014 (=T*, survives truncation), start_weight = exposure = 1.0, death = 0.0 -> psid_rate = 0 -> <25 hazard = -expm1(0) = 0 exactly}. Verified by execution on the merged refit.py: unpadded raises; padded yields a valid 8-band 0->120 model; the fourteen 25+ cells are byte-identical and invariant to pad value/weight; MORTALITY_BANDS==mf.BANDS guard (registered_m6_inputs.py:238) unchanged; the (0,24) band feeds ONLY apply_mortality as a zero hazard and appears in no scored/disclosed cell (mortality ungated, certifies_nothing_about_mortality_drift). Rejected option (a) [real NCHS <25 level consumed by the fit]: needs an engine no-exposure fallback AND imports an NCHS level, violating the aligned_rate=psid_rate anchor contract (PSID levels, NCHS inert). F2 SHOULD-FIX: assert_pe_us_version reads importlib.metadata but load_ssa_parameters reads YAML from POPULACE_DYNAMICS_PE_US_DIR / a checkout (ss/params.py:187-227), decoupled from the metadata -- a mismatched dir passes the gate. RESOLUTION = factory-side assert_pe_us_param_dir in build_inputs step 1: (_resolve_pe_us(None)/_SSA).resolve() == Path(importlib.metadata.distribution('policyengine-us').locate_file('policyengine_us')).resolve()/parameters/gov/ssa (verified on the real 1.752.2 site-packages install: locate_file == policyengine_us.__file__ parent, and site-packages is non-git so pe_us_revision='unknown'). N1: SOFTEN _era_mapping's docstring 'verified in build()' -> empirical/unasserted (era_map is provenance-only, read by nothing; no assert). Patch lane applies all three to registered_m6_inputs / extract_ssa_claim_ages_2014 + adds the missing fit_mortality_model bridge test.",
    "amendment_3f_section": "2.8.3f (the demographic-seed SEX SOURCE correction; DOCS-AND-PATCH, closes the third registration's crash-2, registered 4971244215 graded #42 comment 4972045579). BUG: 2.8.3's per-field seed-reads list attributed sex to panels.demographic_panel, but the certified demographic_panel schema is {person_id,period,age,sequence,relationship,weight,interview} (panels.py:252-253) -- SEVEN cols, NO sex -- so build_realized_population reading sex from it via _anchor_rows(columns=(age,sex,interview), m6_population.py:186) raised ValueError: anchor source is missing columns ['sex'] (m6_population.py:139) at refit_m6_phase->build_realized_population (m6_runner.py:355), the FIRST real-frame execution of the phase (run 1 masked it: its QRF-import crash at m6_runner.py:345 precedes :355). CANONICAL SOURCE: person sex is ER32000 from the PSID cross-year individual file, read by data.deaths.read_death_records -- the SAME reader 2.8.3 already names for the mortality slices, and the same attach the certified builders use (household_composition.join_demographics household_composition.py:400, disability.attach_sex disability.py:443); marriage MH4 sex (marriage.marriage_history) is marriage-file-scoped, not the demographic universe, so it is not a competing source for the anchor persons. RESOLUTION: build_realized_population takes death_records and joins person sex by person_id before the demographic seed (uniqueness = one coded value per person; full coded-sex coverage over anchor persons, raise on any male/female-less anchor person); demographic_panel seed reads become {person_id,period,age,interview}, sex moves to the death_records read. GUARDS ADDED: m6_schema_audit static full-phase column contract (unit vs committed schemas + integration_psid vs real loaders) + the population fixture rebuilt to the real 7-col demo schema plus a real-schema death-records sibling so the defect class cannot re-hide behind a flattering fixture; sidecar environment_block extended with the fitting-stack (populace-fit/populace-frame) provenance it omitted; run script env prerequisites now name the fitting stack.",
    "amendment_3g_section": "2.8.2g (the MARITAL PROJECTION DOMAIN LAW for sub-START_AGE-at-anchor persons; DOCS-ONLY design amendment, closes the fifth registration's execution failure, registered 4976428384 graded #42 comment 4979269487, forensics round-2 comment 4979437110). BUG: build_anchor_frame (m6_cells.py:122-140) admits every gated-start-wave person with NO age filter so minors are in the anchor; marital_panel_builder overrides start_exposure_year:=anchor_wave (panel_builders.py:187) and demands the certified person_years entry row at that wave (:200-214), but the certified marital person_years begin at birth_year+START_AGE (START_AGE=15, transitions.py:111,260), so any person with anchor_wave<birth+15 has no anchor row and the builder raises (panel_builders.py:209-214). The raise reports missing[:10] (10 ids) but the realized-anchor class is a single uniform 2850 persons (verified read-only: 2568 anchored 2015 / 246 2017 / 36 2019; ages 7-14 at anchor mean 10.8; born 2001-2008; every one py_year_min==birth+15, none with an anchor-wave row). PIN CORRECTION (uncontested): the 2.8.2 marital-builder universe row claimed _valid_persons 'resolves the missing-entry-row case by construction' -- FALSE: _valid_persons (transitions.py:265-277) tests start_exposure_year<=censor_year on the CERTIFIED start_exposure=birth+15 BEFORE the builder overrides start_exposure:=anchor_wave; the entry-row invariant needs person-years-presence-at-anchor (anchor_wave>=birth+15). Corrected universe = attrs INTERSECT anchor INTERSECT {anchor_wave>=birth+START_AGE}. DESIGN DECISION (adjudicated B over A): the 2850 are NOT gated-neutral -- 281 (all born 2001, age 14 at the 2015 anchor, reaching 18 by 2019) carry truth-side 18-29 at-risk person_years at 2019 = 281/3224 of the SEX-POOLED 2019 slice (8.7% persons / 9.0% weight); the actual gated floor cell first_marriage.18-29|female pools 2015-2019 female-only (7832 at-risk person-years) and the class contributes 157 = 2.0% rows / 2.25% weight, all at 2019 (verified read-only); 0 in any gated event; 2569 gated-neutral. A (exclude-and-mark, the 2.8.3a analogue) DIES: symmetric exclusion removes the class's 157 person-years (2.0%) from the truth-side first_marriage.18-29|female denominator -> the frozen v3 floor (e931c886) tolerance moves 0.356->0.355 -> NOT byte-identical (referee-reproduced; magnitude immaterial, ANY nonzero move breaks byte-identity); the only floor-preserving branch (asymmetric, projected-only) scores a projected denominator missing the class person-years the truth side still carries -> identity-guard-rejected (support.py:311-315, fires on real frames) + biases the rate UPWARD. B (seed-at-marital-entry, ADOPTED): for anchor_wave<birth+15 set start_exposure_year=max(anchor_wave, birth+START_AGE)=birth+15 (the certified risk-set entry), read the certified entry row there -> builder is PROJECTED-SIDE ONLY so the frozen floor + every gated denominator stay byte-identical; the 281 enter the PROJECTED 2019 18-29 at-risk support (157 female into the gated first_marriage.18-29|female cell) symmetric with truth; the existing start_exposure<=censor(clip 2022) filter drops the 212 born-2008 (birth+15=2023>2022, never reach 15 in-window); +2638 seeded (281 gated + 2357 report-only). LEAK ADJUDICATION (B, empirical not just structural): the certified entry row at birth+15 is never_married for ALL 2850 (0 exceptions), marriage_duration/years_since_dissolution null/zero; earliest first-marriage EVENT in the class is age 18 (pid 2852062 born 2004, 2022), earliest 'married' person-year age 19 (allow_exact_matches=False shifts married to the following year); 9 of the 11 with n_marriages>0 have a dated event, all strictly after the age-15 seed; the seed is a structural constant carrying zero holdout info (the same pd.isna->never_married entry the core applies, marital.py:100-101) -> leaks nothing. PATCH (pinned, post-ratification): one-line clamp start_exposure_year=max(anchor_wave, birth_year+START_AGE) at panel_builders.py:187 (birth_year already on attrs transitions.py:239-243; START_AGE from data.transitions); assembly.marital_step(:279)/m6_runner unchanged (identical builder signature/return); test_m6_panel_builders.py gains a sub-START_AGE-at-anchor fixture (born 2001 anchored 2015; current fixtures are all born 1980-1982) asserting pre-patch raise / post-patch 2016 never_married seed + holdout_ids membership + entry row at 2016, plus a born-2008 censor-drop case; NO schema-audit/manifest/gates.yaml/runs delta (clamp reads only existing attrs cols birth_year/anchor_wave/censor_year). SCOPE: frozen floors byte-identical; gated cell defs untouched; the certified raise (panel_builders.py:209-214) STAYS (cause removed, invariant satisfied by construction, mirroring 2.8.3a). Edits no gates.yaml cell, moves no threshold, builds no floor, writes no test in the docs PR.",
    "amendment_3h_section": "2.8.2h (the FERTILITY/OPEN-ADDITIONS ROSTER-MATERIALIZATION DOMAIN LAW; DOCS-ONLY design amendment, closes the sixth registration's execution failure, registered 4981073550 graded #42 comment 4984699959, forensics round-3 comment 4984997277). BUG: step-4 fertility draws its maternal-birth schedule from the FRAME-INDEPENDENT 2.8.2 marital at-risk universe (holdout_ids=state.marital_ids, built once per draw with del frame + realized censor, assembly.py:278-293/panel_builders.py:164) but MATERIALIZES the births onto the FRAME-DEPENDENT post-mortality roster (materialize_maternal_births steps.py:381, from apply_fertility steps.py:472, fertility_step assembly.py:315). The schedule is keyed to the realized MH censor (clip to projection_end_year); the roster to simulated wave-loop mortality (apply_mortality steps.py:113, assembly.py:254); nothing reconciles them, so a woman whose simulated death precedes her realized censor stays in the schedule and, when the wave-t fertility RNG draws her a wave-t birth, the parent is absent from the roster -> guard raises (steps.py:412). FORENSICS (deterministic, byte-identical x3 reconstructions, pinned seed-0/draw-0 person side): singleton {782173} at wave 2020 -- a realized SURVIVOR (female, born 1988, anchored 2015, PSID censor 2022) killed by the differential-mortality draw in simulated 2020; per-wave, 2015-2019 materialize 50/55/63/54/34 children with 0 absent parents, only wave 2020 (38 births, 1 absent parent) raises. Hypothesis (a) death-then-drawn CONFIRMED 100% (mortality is the sole roster-removing step; valid_ids is an inner join WITH anchor so the at-risk set is a strict SUBSET of the roster universe -> (c) id/join seam REFUTED; (b) non-mortality removal and (d) 3g interaction refuted, her 3g clamp max(2015,2003) a no-op). Susceptibility class = at-risk fertile women whose simulated death precedes their realized censor and who retain >=1 fertile-age schedulable year while absent = 10 on the seed-0 person side (earliest-absent-fertile-schedulable year {2015:1,2016:1,2017:1,2019:2,2020:1,2021:2,2022:2}; the 2015 member id 5459180 born 1966 would trip first on any draw that drew her a 2015 post-death birth); 456 wave-loop deaths vs 11552 at-risk; live tripwire over the 5 seeds x 2 sides x 20 draws ensemble. PIN GAP: 2.2 step 1 pins 'decedents leave the risk set for all subsequent year-t steps' + the DAG rationale 'no dead person is married, made disabled, or paid' (born-of is the unstated fourth); 2.8.9 pins the at-risk universe frame-independent for SCORING (correct -- gated marital/earnings/disability score on the frame-independent 2.8.2 builder, never the live roster); the code reuses that one frame-independent universe for the MATERIALIZATION step, which 2.2 requires frame-dependent -- no 2.8 law separated the two universes (2.8.2g corrected the marital ENTRY universe; 3h separates the fertility SCORING universe from its MATERIALIZATION universe). STRUCTURAL TELL: every other reconciling step (marital step-3, disability step-5, household step-8) merges its frame-independent projection onto the roster by LEFT MERGE (_merge_period_columns assembly.py:174-192, how=left -> drops absent by construction); step-4 fertility is the SOLE materialize-with-raise (pd.concat behind a hard guard); 3h brings it into line with the merge steps' reconcile-not-raise posture -> LAW not one-hole patch. DOMAIN LAW: scheduled maternal births materialize ONLY for mothers present in the live post-mortality roster at the birth wave; the simulated-mortality roster (2.2 step 1) governs materialization, the frame-independent schedule (2.8.2 builder, realized presence/censor) governs scoring; a scheduled birth whose mother the wave loop removed does NOT materialize and is recorded in the run's report-only reconciliation. RESOLUTION (adjudicated (i) over (ii),(iii); all three keep the frozen floors byte-identical because open additions are report-only 2.1): (i) DROP-WITH-RECONCILIATION, FILTER AFTER THE DRAW (ADOPTED) -- pure function of roster state, fertility RNG stream byte-UNCHANGED (drop is post-draw), report-only births down by the post-death births (exactly 1={782173} on seed-0/draw-0, order ~1 birth/draw), touches no scored/gated surface; (ii) re-draw a replacement DIES on determinism (a re-draw consumes RNG -> shifts every downstream per-person stream + fabricates a birth for a different mother); (iii) keep-raising DIES (a report-only mechanism must not abort a scored pre-artifact run). RNG-ADDRESS DECISION: filter-AFTER-draw (RNG byte-unchanged, only dropped rows differ) over intersect-BEFORE-draw (fewer at-risk rows -> simulate_maternal_births RNG consumption changes -> report-only realizations shift); both floor-inert, the post-draw filter is byte-cheaper and is adopted. CERTIFIED-SCORING INVARIANCE (pinned by code pointer): the scored/certified surfaces score fertility as distributional moments over self-contained frame-independent panels, never over the materialized roster -- simulate_maternal_births (marital.py:288-352) reads ONLY static panel.attrs (start_exposure_year/censor_year/birth_year/sex), never the live roster; candidate-16 internal births + candidate-9 household-composition conditioning fertility (assembly.py:394, a SEPARATE generator(0,FERTILITY) draw that materializes NO roster) are the certified lineage whose censor IS survival -> certified use has no open-population child roster and no independent mortality step, so parent-absent-from-roster cannot arise -> invariant holds VACUOUSLY in certified use; roster is report-only (2.1, 4.8 decision 4) -> the drop provably cannot reach any gated/certified surface; frozen floors m6_holdout_floors_v{1,2,3} + v3 sha e931c886 byte-identical. SIBLING SWEEP (arm-parity): [1 mortality apply_mortality steps.py:113/assembly.py:254 = the removing step]; [2 aging advance_age = in-place, no keying]; [3 marital marital_step->_merge at fertility_step assembly.py:309-314 = left-merge drop-absent]; [4 fertility apply_fertility->materialize_maternal_births steps.py:472/381 = 3h's law, keys parent_person_id]; [5 disability disability_step->_merge assembly.py:372 = left-merge drop-absent]; [6 earnings/7 claiming = in-place]; [8 household household_step->_merge assembly.py:412 = left-merge drop-absent]; [paternal shadow births steps.py:389-391 = NOT materialized (conditioning, deliberately not duplicated)]; [candidate-9 conditioning fertility assembly.py:394 = no roster attach]; [child-ordinal loop.py:329-331 = additive, keyed to POST-materialization roster -> rules out the secondary KeyError: person_generator loop.py:99-104 raises 'no stable RNG ordinal' but newborns are ordinal'd at wave end before the next wave's step-1 mortality]; [scheduled realized openers = LIVE loop-native entrants (loop.py:192-254; frames m6_population.py:65-73 side-split :97-103; the 2017/2019 openers 246+36 counted scheduled_realized_openers m6_runner.py:916-919): pd.concat of fresh realized entrant rows at each period top BEFORE step-1 mortality (loop.py:252), keyed to no roster-removable person, overlap-guarded (loop.py:215-219,:244-250), ordinals pre-seeded (loop.py:214,:234-237) so no KeyError shape -> holds by construction]; [immigrant entry cohorts = DORMANT: would ride the SAME SCHEDULED_ENTRIES_KEY mechanism (NOT PeriodModules assembly.py:426-436), but nothing schedules immigrant frames and the entrants report hardcodes immigrant_cohorts:0 (m6_runner.py:914) -> out of scope, benign-by-construction if wired]; [synthetic_id_allocator steps.py:418 = fresh ids]. Residual report-only household-consistency note FLAGGED not fixed (candidate-9 may compute coresident_spouse for a living person against a mortality-removed spouse -- a merged column value, no materialization, no raise; out of scope for the guard closure). GUARD steps.py:412 STAYS: the reconciliation filter lives in the open-additions CALLER apply_fertility so materialize_maternal_births receives only roster-present parents -> guard becomes unreachable on the lawful path but is retained as the invariant backstop for any un-reconciled caller (a genuine id/join defect, hypothesis (c), which the M6 at-risk universe cannot present but the primitive must still catch). PATCH (pinned, post-ratification, NOT in this PR): in apply_fertility, between the draw simulate_fertility (steps.py:469) and the materialize call (steps.py:472), filter draws.maternal to rows whose parent_person_id is in set(frame['person_id']) (the live post-mortality roster) and record the dropped rows in a report-only roster_absent_births[context.year]={dropped_parent_ids,dropped_count} (alongside the existing birth_store[context.year]=draws steps.py:470); apply_mortality + guard + frame-independent simulate_maternal_births (marital.py:288-352) + holdout_ids (assembly.py:321) unchanged; tests/test_m6_engine_steps.py gains a dead-mother-scheduled-birth fixture (current fixtures use parent_person_id=[10,10] present in the roster, :203/:241) asserting (a) pre-patch materialize raises 'birth parents are absent from the roster', (b) post-patch apply_fertility drops+reconciles+materializes only roster-present-parent children, (c) the simulate_fertility RNG draw is byte-identical vs a roster where the mother is present; REAL-FRAME PROOF TARGET = the 10-person seed-0 susceptibility class, full-window person-side projection completes to 2022 with the drops recorded and counts matching forensics ({782173} dropped at wave 2020), run in the post-ratification patch lane on the run-6 venv. ERRATUM RIDDEN (PR #210 referee NOTE-1, the 3g-implementation referee's deferred one-liner): the m6_schema_audit marital_panel_builder read-set (m6_schema_audit.py:266-268) lists marital.attrs={person_id,censor_year,start_exposure_year,weight} and OMITS birth_year, which the ratified 3g clamp max(anchor_wave,birth_year+START_AGE) (panel_builders.py:187) reads (birth_year on attrs transitions.py:239-243); the 3g-patch lane must add birth_year to that frozenset -- recorded here per the referee's request; 3h itself edits no manifest. SCOPE: frozen floors byte-identical; gated cell defs untouched; the frame-independent fertility schedule + scored moments unchanged (3h touches only which scheduled births MATERIALIZE onto the report-only roster); the certified fertility lineage untouched; the guard stays; the 2.2 order unchanged (3h makes step 4 honor the step-1 decedent exit 2.2 already mandates). Edits no gates.yaml cell, moves no threshold, builds no floor, writes no test in the docs PR.",
    "input_reference_bindings": {
      "blocker": "run_gate_m6_candidate1.py is not self-starting: the --input-factory returning M6HarnessInputs via load_m6_inputs needs three <=T* external references that did not exist; the only committed claiming ref is the 2023 Supplement, and claiming_pmfs_from_reference correctly raises vintage 2023 is post-T* (2014)",
      "claiming": {
        "source": "SSA Annual Statistical Supplement 2014 edition, Table 6.B5.1 (retired-worker awardees by age at entitlement, by sex and entitlement year)",
        "supplement_year": 2014,
        "entitlement_years": "1998-2013 (confirm-at-fetch; each edition Y tabulates 1998..Y-1, contiguous 1998-2022 in the committed 2023 file; admissibility insensitive to 2013-vs-2012 since claiming_pmfs_from_reference needs only supplement_year<=2014 plus >=1 entitlement year <=2014 and the harness snaps nearest-available)",
        "why": "maximal edition with supplement_year<=2014 (the 2015 edition, data through 2014, is rejected); its information is <=2013 so it is leakage-safe on the 2.7.6.3 publication-lag principle",
        "target_file": "data/external/ssa_claim_ages_2014supplement.json; STRUCTURALLY identical (every field claiming._load parses + row shape, claiming.py:147-166) to the committed 2023 file -- NOT literal provenance-key parity: the 2014 provenance additionally carries source_sha256 + retrieval_date (absent from the 2023 provenance)",
        "acquisition": "orchestrator browser-fetch https://www.ssa.gov/policy/docs/statcomps/supplement/2014/6b.html (PDF fallback supplement14.pdf); ssa.gov 403s curl/WebFetch; commit the raw source + record source_url, retrieval_date, fetch_method, source_sha256 in provenance (BUILD-TIME provenance, not the gate); the factory instead asserts the JSON FILE's own sha256 vs a constant hardcoded in registered_m6_inputs (the artifact consumed)",
        "consumption": "REPORT-ONLY (R2 marital-blind; feeds report-only benefit levels; never gated); a HARD blocker only because assemble_m6_inputs materializes the PMFs and rejects any post-2014 vintage BEFORE any refit/score/write (load_m6_raw_inputs reads PSID first; the guard fences the fit/score/write, not the read)",
        "nearest_year_fallback": "HARNESS path = ClaimingSchedule.distribution (steps.py:299-305), nearest-available over the PMF dict claiming_pmfs_from_reference materializes from reference.years()<=T* (refit.py:972-984): for the 2014 file (1998-2013) a projected year >2013 snaps to 2013. CAUTION: module-level claim_age_distribution / draw_claim_ages / _resolve_year are NOT on the M6 path and KeyError for entitlement years 2014-2022 against the 2014 file (their MIN_YEAR/MAX_YEAR=1998/2022 are 2023-file constants, claiming.py:86-87,200-213); drive validation through ClaimingSchedule.distribution / claiming_pmfs_from_reference"
      },
      "ssa_params": {
        "pe_us_revision": "policyengine-us 1.752.2 (deployment_frame CERTIFIED_PIN model_version). ACQUISITION: no git tag 1.752.2 on PolicyEngine/policyengine-us; PyPI has it (released 2026-07-01) -- pip/uv install policyengine-us==1.752.2 and point POPULACE_DYNAMICS_PE_US_DIR at the dir containing policyengine_us/ (site-packages satisfies _SSA). ASSERT importlib.metadata.version(policyengine-us)==CERTIFIED_PIN model_version==1.752.2. PROVENANCE: load_ssa_parameters records pe_us_revision via git log and degrades to unknown for a non-git site-packages dir (ss/params.py:309-318); the artifact carries pe_us_revision=unknown with vintage identity pinned by the asserted version + CERTIFIED_PIN, not a git hash",
        "ssa_params_vintage": 2014,
        "nawi_rule": "realized <=2014 kept; every entry y>2014 REPLACED with I_proj(y)=exp(a+b*y), (a,b)=OLS ln(NAWI)~year over [2005,2014] (byte-identical to fit_projected_wage_index), over the SAME year-key range so max(nawi) is unchanged",
        "why_replace_not_truncate": "indexed_earnings_supply admits a person only if min_nawi<=birth+60<=max_nawi (couple_earnings._admitted) and indexes <=2014 earnings to the age-60-year NAWI; truncation to <=2014 collapses the admitted universe to birth<=1954, emptying the gate-2c modifier axis (likelier: it degenerates to the neutral 1.0 modifier) -- that axis FEEDS THE SCORED PATH OF the gated marital flows (divorce.18-44/first_marriage.18-29|female/remarriage.18-64; modifier multiplies first-marriage hazards directly, marital.py:378, divorce/remarriage via composed occupancy); 2.7.6.3 names AIME indexing a NAWI consumer under the realized-post-T*-NAWI leakage prohibition, and a current checkout realized 2015 NAWI 48098.63 vs I_proj 47252.12 (846 gap) would leak into those gated flows",
        "ordering": "REPLACE only AFTER load_ssa_parameters -- its bend-point cross-check runs on the realized series and re-deriving post-2014 bend points from I_proj would spuriously trip it; use dataclasses.replace on the frozen SSAParameters (no __post_init__)",
        "wage_base_rule": "truncate change-years <=2014 (inert on gated cells: the <=2014 panel credits only <=2014 bases via creditable_history)",
        "bend_points": "derived (nawi[year-2]); inherit the NAWI rule (realized through eligibility year 2016, I_proj beyond); report-only PIA path only",
        "statutory_schedules": "416(l) FRA / PIA factors / 402(q) / 402(w) / max_delayed / aux spouse-survivor constants are birth-year-keyed or statutory -> no calendar vintage -> pass through unchanged",
        "prohibited": ["realized NAWI y>2014 on any refit path", "wage_base change-year >2014 in the bundle", "any pe-us value effective >2014 the refit reads"],
        "cross_check_note": "the 2026 bend-point determination (1286/7749) load_ssa_parameters checks is a load-time self-consistency assertion on the realized series, NOT a fit input or gated cell",
        "alignment_box_vacuous": "2.7.6.3's report-only realized-NAWI alignment box is vacuous under this bundle: no realized post-2014 NAWI in-process; the runner records the alignment layer not collected (build_alignment_displacement(None,None), m6_runner.py:1100) -- the two sections do not conflict"
      },
      "mortality": {
        "vintage": "NCHS United States Life Tables 2010 (NVSR 63-7, released Nov 2014), data/external/nchs_life_tables_2010.json (vintage_year 2010)",
        "mortality_external_vintage": 2010,
        "why_2010": "OUTCOME-INERT: fit_mortality_model computes aligned_rate=central_rate*(psid_rate/central_rate)=psid_rate exactly (refit.py:1083-1088) so the external vintage cannot move any fitted hazard -- untunable to any 2015-2019 observable by construction; it enters only the central_rate>0 guard + the report-only anchor decomposition (external_anchor). Among admissible <=T* vintages the pin is still 2010 (NVSR 63-7, Nov 2014, <=T* on both axes) over 14-yr-older 2000 (NVSR 51-3); 2023 (NVSR 74-6) guard-rejected",
        "external_rates": "columns {lower_age,upper_age,age_band,sex,central_rate}; central_rate=(l_a-l_{b+1})/(T_a-T_{b+1}) from lx/Tx (open [85,120]=l_85/T_85) over 7 MORTALITY_BANDS (25-34,35-44,45-54,55-64,65-74,75-84,85+), sex {female,male}; mirrors build_mortality_floors nchs_band_rates",
        "exposure": "SMALL ADAPTER (neither build_exposure_slices nor mortality_slices emits the pinned shape): re-derive <=2014 PSID person-interval single-year slices from step-4 readers panels.demographic_panel()+deaths.read_death_records() (build_mortality_floors.py:559-561), emit {event_year,required_interview_year,age_band,sex,start_weight,exposure,death}. exposure 0.5 in death slice else 1.0; death 1.0 in death slice; age_band from MORTALITY_BANDS; start_weight=F6 boundary weight. ROW DATING (both fields on EVERY slice): event_year=slice calendar year; required_interview_year=interval CLOSING wave (next_wave from demo wave grid; NOT in read_death_records which has only person_id/sex/death_year, deaths.py:81-91). SYMMETRIC conservative rule (closing-wave for every slice) drops the 2013->2015 interval wholesale (window ends 2011->2013), chosen over anchor-wave-survivors+closing-wave-deaths which would keep 2013-14 exposure without its deaths (downward-biased terminal hazard). DISCLOSURE: start_weight read from a 2015+ interview into a <=T* fit, admissible only because mortality is ungated",
        "consumption": "UNGATED: feeds fit_mortality_model (step-1 age x sex hazard) + report-only mortality/AIME/PIA/household diagnostics; NO death.* in the 11 gated cells (death.85+ report-only, attrition-demoted); per F1 gated flows/earnings score on realized support regardless of simulated death -- touches no gated tolerance"
      },
      "factory": {
        "entry": "registered_m6_inputs:build_inputs -- zero-arg build_inputs() -> M6HarnessInputs (run_gate_m6_candidate1.py --input-factory)",
        "module_and_paths": "module scripts/registered_m6_inputs.py (importable as registered_m6_inputs because scripts/ is sys.path[0] when the runner is invoked from repo root, run_gate_m6_candidate1.py:42); both data/external/... paths repo-root-anchored (DATA=Path(__file__).resolve().parents[1]/data/external, mirroring claiming._ROOT), never CWD-relative",
        "steps": "1) assert importlib.metadata.version(policyengine-us)==CERTIFIED_PIN model_version==1.752.2 + load_ssa_parameters (cross-check on realized); 2) dataclasses.replace NAWI(realized<=2014 + I_proj) + wage_base(<=2014); 3) load_claim_age_reference(DATA/ssa_claim_ages_2014supplement.json) + assert the JSON FILE's sha256 vs a hardcoded constant (raw-source hash is provenance.source_sha256, not the gate); 4) mortality_external_rates(NCHS 2010 band collapse) + mortality_exposure(<=2014 PSID adapter, 2.8.10.3); 5) load_m6_inputs(ssa_params_vintage=2014, mortality_external_vintage=2010, boundary 2014, earnings_seed 5200)",
        "revalidation": "load_m6_inputs re-runs validate_external_vintage x3 + claiming_pmfs_from_reference + prepare_mortality_refit_inputs at assembly (before any refit/score/write; load_m6_raw_inputs reads PSID first); the sha256 gate + three <=T* guards make the acquired claiming source tamper-evident",
        "build_lane_never_runs_real_data": true
      },
      "residual_open_decisions": "none (the mortality-exposure row-dating semantics -- the referee's one under-pinned item -- are now pinned in 2.8.10.3: symmetric closing-wave dating, the re-derivation adapter, the next_wave derivation, the start_weight disclosure)",
      "next": "3d pin -> referee SPEC-SOUND (revise) -> revision 10 landed -> factory build (registered_m6_inputs) merged 67e7fad -> factory-referee MERGE-READY build-only with F1/F2 (PR #191 comment 4969829131) -> amendment 3e (2.8.10.5, revision 11) pins the F1 (0,24) inert shape-bridge pad + F2 param-dir assert + N1 -> referee -> F1/F2/N1 patch to registered_m6_inputs + the missing fit_mortality_model bridge test -> referee -> merge -> THIRD gate_m6 registration (4971244215) -> run FAILED TO EXECUTE, two pre-scoring crashes, one-shot unconsumed (graded 4972045579): crash-1 QRF-import env miss (populace-fit absent, disclosed re-exec fixed it), crash-2 ValueError anchor source missing ['sex'] at m6_population.py:139 -> amendment 3f (2.8.3f) sources the demographic-seed sex canonically from data.deaths + adds the m6_schema_audit full-phase column contract + real-schema fixture + fitting-stack sidecar provenance + run-env docs -> referee (real-frame population construction permitted, no scoring) -> merge -> FOURTH gate_m6 registration (reg-4) DESIGNED ABORT at pre-flight-1 (injected-arm fertility wired, #207 b2a0693) -> FIFTH registration (4976428384) run FAILED TO EXECUTE in seed-1 scoring projection, one-shot unconsumed (graded 4979269487): ValueError no marital entry row at anchor for sub-START_AGE-at-anchor persons (10 of 2850, panel_builders.py:211) -> forensics round-2 (4979437110) -> amendment 3g (2.8.2g, revision 13) seeds the class at birth+START_AGE (option B: frozen v3 floor byte-identical; A dies: any nonzero move of the frozen first_marriage.18-29|female floor cell (class 157/7832 = 2.0%, tol 0.356->0.355) breaks byte-identity) -> referee -> patch -> SIXTH registration -> run"
    },
    "core_adjudication": "year-0 closed panel carries REALIZED histories: builders read each person's realized <= THEIR ANCHOR INTERVIEW marriage/household history (2015 for the bulk; 2017/2019 presence-conditioned openers seed at their later anchor) as the seed (decision 5 reproduction-mode semantics extended from disability_step assembly.py:257-304); the per-person anchor is the true leakage fence (ratified block f6_weight.start_wave + 4.4; seed = realized initial condition, refit stays <=2014, no fitter sees it); projected years constructed from simulated state under the pinned per-field rules; forward/production path out of scope (decision-5 successor gate, support.py FORWARD structural_delta)",
    "builders": {
      "contract": "(frame, context) -> (native_panel, holdout_ids) per assembly.py:59-72; schema-byte-compatible with transitions.build_marital_panel / household_composition.build_household_panel (R1 plumbing); whole-window, certified core run ONCE per draw and cached (the built disability_step pattern)",
      "cached_core_stream_address": "PINNED period-0 deployment address (rng.py:41-44), NOT period-1 first-invocation: reproduces the certified single-period topology and is the address the 2.8.5 internal reference uses (composition.py:643-653, n_periods=0). Per draw k: marital main_rng=generator(0,MARITAL_CORE), gap_rng=child_generator(0,MARITAL_CORE,1); household = the SINGLE composition_rngs_from_registry(registry,0) CompositionRngs (composition.py:532-539); household-conditioning fertility = generator(0,FERTILITY). Bit-exact reproducible (4.9)",
      "marital_universe": "attrs = build_marital_panel(marriage.marriage_history(), death_records, anchor_weight).attrs INTERSECT anchor (F3): identical to the truth-side marital_tables universe; drops MH-absent anchor persons (children, MH-uncovered adults) SYMMETRICALLY (else asymmetric first_marriage denominator via the pd.isna never_married entry path marital.py:100-101); _valid_persons enforces start_exposure<=censor on the CERTIFIED seed (birth+START_AGE) but does NOT resolve the missing-entry-row case (it runs BEFORE the anchor override); person-years presence at the anchor requires anchor_wave>=birth+START_AGE -- CORRECTED by amendment 3g (2.8.2g)",
      "marital_per_field": "attrs person-constant realized; start_exposure_year = realized anchor wave (SEED_WAVE 2015 bulk); censor_year = 2022 clipped to realized presence/death censor; entry person_years row = realized anchor-wave state via the truth side's own build_marital_panel; n_marriages = lifetime MH18 person_attributes writes (transitions.py:239-243, marriage.py:216) = the truth-side value (byte-identical seed), CARRIED-INERT (core never reads it, order re-seeds to 1 marital.py:107,119; _assemble_panel overwrites from simulated episodes+residual marital.py:262-272); episode extension = open episode carries realized current_start, closed at simulated dissolution or emitted intact at censor (marital.py:243-249); changepoint appends = certified marry/dissolve kinds with kind_order tie rule + allow_exact_matches=False (transitions.py:307-352,377-423); duration = year - current_start, years_since = year - dissolution_year (marital.py:188,217 verbatim); no new changepoint kind",
      "household_per_field": "person_waves support = realized holdout support; obs_parent/multigen/cohab/skipgen seed states = realized <= anchor-interview build_household_panel slice; projected years = candidate-9 injected forward evolution (composition.py) consuming the SINGLE period-0 CompositionRngs (composition_rngs_from_registry(registry,0)); transition helpers rebuilt by _add_transitions"
    },
    "year0_slice": "mirror the v3 floor's realized panel EXACTLY: universe/holdout_ids = build_anchor_frame (gated start waves 2015/2017/2019, positive weight); per-field PSID reads = the floor's own sources (panels.demographic_panel, marriage.marriage_history->build_marital_panel, build_household_panel, disability reader, family_earnings_panel incl. realized 2014/2012 + u_w per 2.7.6.5, data.deaths); F6 weight + household_id fixed at the realized anchor wave (StartWaveWeightSnapshot); FLOWS presence set = presence_by_wave, PresenceBasis.START_OF_INTERVAL (opening biennial interview), symmetric via prepare_evaluation_support; EARNINGS support basis (F6) = realized family_earnings_panel person-period row existence at {2016,2018} (present, positive-weight, valid-earnings; earnings_frame), an EXACT_WAVE-class basis on reference years, NOT the odd-wave flow presence sets",
    "earnings_domain_law_2.8.3a_amendment_3b": {
      "blocker": "Sol round-4 (~/PolicyEngine/sol-worktrees/m6-harness-REPORT.md): the closed-panel universe includes persons with NO year-0 earnings state (non-head/spouse anchor members: family_earnings_panel emits rows only for head/spouse family.py:785-857; 2017/2019 openers: no 2014 row from a non-2015 interview), but ForwardEarningsGenerator.materialize_initial_frame RAISES for any initial person absent from realized_earn_2014_by_person / u_w_by_person (forward_earnings.py:882-896), BEFORE any age rule; fit_forward_earnings restricts those maps to realized-2014 anchors (forward_earnings.py:725-736)",
      "adjudication": "from ratified F1 + 4.8 closed-panel: NO synthetic init law needed or permitted; the excluded people are earnings OPEN-ADDITIONS (report-only 4.8), not a gated population",
      "domain": "earnings-module domain = realized-state (closed-panel) population = realized_earn_2014_by_person INTERSECT u_w_by_person (the materialize requirement) = the 2014 realized-earnings cross-section (2.1, 4.8)",
      "marker": "frame column earnings_domain: bool (True iff in domain); for False the wrapper writes earnings=0.0, leaves u_w/realized_earn_2014/2012/gen_earn_w2/w4 absent, seam/report-only earnings = pinned NON-SCORED zero (W2 seam tolerates zeros, 5); count publishes; no fallback earnings law / synthetic u_w / backfilled 2014 anchor. REPORT-ONLY DISTORTION disclosed (finding 3, NO gated-cell impact): the ~21% later earnings-entrants are REAL earners (realized 2016/2018 income) carried at 0.0 throughout, so their report-only AIME/PIA/claiming levels (the 5 W2 seam) are UNDERSTATED -- the family-B report-only benefit tier must not be read at face value for the marked population; AIME/PIA/claiming are report-only, never gated in gate_m6",
      "filter_location": "the EarningsGenerator WRAPPER (not the certified generator), a PER-PERSON check at BOTH entry points (finding 5): (a) initialize->materialize_initial_frame (assembly.py:198-202) materializes state for the in-domain rows only + reassembles the full frame (out-of-domain rows carry marker+earnings=0.0, chain-state cols absent); (b) apply_earnings whose registry path already calls generate per person on a single-row frame (steps.py:213-232), so the domain filter is a PER-PERSON predicate (skip out-of-domain, write non-scored zero), NOT a batch-subframe rewrite; chain-state cols for out-of-domain rows handled by wrapper reassembly. Certified generator sees only in-domain persons -> its requirement satisfied by construction (R1 plumbing, law untouched)",
      "anchor_rule_F4_symmetric": "the chain is 2014-anchored (fit anchors on period==boundary_year==2014), so the earnings-domain wave is UNIFORMLY the 2015 interview (only interview producing a 2014 reference year); NO 2017/2019 earnings-domain entry (openers have realized 2016/2018 but no 2014 -> later earnings-entrants = open additions 4.8/2.1 -> report-only, non-scored marker throughout); symmetric with histories in that each module seeds at the wave producing ITS anchor state (for the 2014-anchored chain that wave is fixed). PER-MODULE open-addition, distinguished from family B (finding 4): these are CLOSED-PANEL members (scored for marital/household/disability flows) reclassified as open-additions FOR EARNINGS ONLY (each module scores its own law's realizable domain) -- a person can be closed-panel for flows and open-addition for earnings; NOT the family-B synthetic-birth/immigrant entrants (no PSID truth for ANY module)",
      "gated_scoring_support": "realized support INTERSECT earnings domain, SYMMETRIC on projection and truth (rate_a); forward chain evaluated (F1) for every realized-present IN-DOMAIN person-period regardless of simulated death; later earnings-entrants excluded from BOTH sides (report-only)",
      "floor_consistency_EMPIRICAL": "point-(6) 'holds by construction' is FALSE: verified vs staged PSID through the frozen floor machinery, of the frozen v3 gated-earnings support (13163 persons at {2016,2018} prime+older) 2722 (~21%: 2199 prime, 590 older) are later earnings-entrants OUTSIDE the 2014-anchored domain -> the frozen v3 floor OVER-INCLUDED open-additions (build_anchor_frame multi-wave-anchor lineage artifact). Holds for non-head/spouse members (no earnings row -> in no cell). Consequence CONSERVATIVE not a leak: closed-panel domain scored vs the frozen full-support tolerance faces ~sqrt(N/N_domain) inflated half-split noise (~1.17x prime, ~1.06x older) -> effective earnings tolerances TIGHTER (harder to pass, no false-PASS). BIND: harness self-check recomputes the domain-restricted earnings floor half-splits, publishes tolerance/earnings-OC delta vs frozen v3 (report-only; frozen tolerances remain the gated contract, applied conservatively). TWO-DIRECTIONAL escalation (finding 2, honoring 4.9 both-directions): route to the floors-ceremony finding rule (adjudication-7/4.9: ceremony finding not silent redesign; possible earnings-floor re-derivation on the domain) if EITHER (a) recomputed closed-panel earnings OC < 0.90 (near-UNPASSABLE, the modal direction) OR (b) the domain-restricted surface trips the 4.9 vacuity guard (near-TAUTOLOGICAL: a domain-restricted gated tolerance at the ln(1.5) cap, or OC ~1 with a near-unfailable cell); both directions publish, neither silently absorbed",
      "self_test_note": "the 2.8.4 byte-identity self-test is CELL-FUNCTION identity on a given support; the domain support-restriction is a documented filter applied on top (its own recompute self-check), not a function change"
    },
    "drift_scoring": {
      "reuse_verbatim": ["build_anchor_frame", "presence_by_wave", "mortality_slices", "marital_tables", "disability_pairs", "earnings_frame", "mortality_cells", "marital_cells", "disability_cells", "earnings_cells", "_rate", "_wquantile", "_score", "harness.moments", "harness.panel.split_panel_by_person", "evaluation.derive_tolerance"],
      "extraction": "floor cell machinery extracted to ONE importable module shared by floor script and scorer; any reimplementation bound by a harness SELF-TEST asserting byte-identical cell values on the realized panel (fail-the-run otherwise)",
      "flow_projected_side": "marital events + at-risk person-years from the frame-INDEPENDENT marital builder (realized presence/censor); disability pairs from realized-support reproduction (assembly.py:257-304) -- both realized-support by construction",
      "gated_earnings_support_ADJUDICATED_F1": "gated earnings cells score on the REALIZED earnings support, NOT the survivor-attrited live frame. Step-1 mortality (apply_mortality steps.py:108-130) drops simulated deaths from every later frame slice -> projected {2016,2018} earnings person-periods are a STRICT SUBSET of realized earnings_frame support on ~every draw; scoring that would abort under the 2.8.3 identity guard (support.py:311-315) OR silently admit not_certified mortality drift into 6 gated cells via survivorship. Adjudicated by the same symmetric-conditioning principle 2.8.1 applies to histories + 4.4 cancellation + 4.7 gate_m4-reproduction-on-realized-support: the forward chain is evaluated for EVERY realized-present person-period regardless of simulated death. BIT-SAFE: apply_earnings draws on person-keyed EARNINGS streams (context.person_generator, steps.py:222-233), so a dead-but-present person's chain consumes ONLY its own entropy, perturbing no survivor's bits. Mortality stays report-only (feeds diagnostics + AIME/claiming/household, not the gated earnings read). Conditions on realized reference-year presence (conditioning-not-leakage); does NOT score mortality's earnings-composition (survivorship) channel = designed reproduction-test scope, same as flows; mortality drift stays not_certified",
      "construction": "per gate seed s in {0..4}: split_panel_by_person on the block's split_unit selects side-A; rate_a = side-A realized presence-conditioned rate; rbar = K=20-draw mean (5200+k) of the engine projecting side-A persons from realized T* state; score per the cell metric (log_ratio | abs_gap_log | abs_gap_corr)",
      "conjunction": "cell clears iff score <= locked block tolerance; seed passes iff EVERY gated cell clears; family A passes iff >= 4 of 5 gate seeds (read from the locked block, computed nowhere)",
      "guards": ["undefined_draw_rule (any undefined gated rate on any draw invalidates)", "regenerated-surface conformance (non-zero across-draw dispersion recorded)"]
    },
    "preflight_1": "candidate-9 re-certification margin on the <=2014-refit REAL panel over gate-seed draws BEFORE any scored phase: simulate_candidate9_injected vs simulate_candidate9_internal_reference, composition_channel_moments over RECERTIFICATION_CHANNEL_SETS, check_candidate9_recertification >=3sigma; holdout-blind (two simulation paths, no holdout cell); failure = DESIGNED ABORT pre-scoring (one-shot not consumed, fuller re-ceremony per 2.6); per-channel margins publish",
    "preflight_2": "verify the certified externally-driven _gate_sign_draw _target_models reconstruction deploys (forward_earnings.py:820-826; reproduces FittedRegimeGatedQRF._gate_draw on engine-supplied uniforms -- today's RegimeGatedQRF exposes _target_models and NO draw_sign, so this is the branch every real candidate-10 gate takes) vs the draw_sign test seam (:815-819; only test doubles define draw_sign) on a SYNTHETIC probe; record which path executed; DESIGNED ABORT if a gate deploys the draw_sign seam. Corrects the prior inversion (harness-referee F1, PR #185 comment 4966859161); restores engine-referee obs 6 (PR #173 comment 4962620806): draw_sign=seam, _target_models=certified",
    "runner_phases": ["refit (refit_m6_components boundary 2014 + from_refit_bundle; RefitProvenance + EARNINGS_SPEC_SHA256 recorded)", "preflight_1 (abort-on-fail)", "preflight_2", "project+score per gate seed (K=20 draws, side-A, v3 floor)", "report_only (shock, not_certified, re-drawn-seed comparison, entrants, alignment displacement)", "assemble + artifacts.write_new(sidecar=True) stamping registration-id + EARNINGS_SPEC_REGISTRATION + floor sha e931c886 + spec sha256s; publishes_regardless"],
    "must_not": ["no gates.yaml read beyond the gate_m6 block's protocol/cells (no tolerance computed, no threshold moved)", "no holdout-informed choice (synthetic frames only until the registered run)", "no realized post-boundary macro read on the scored path (2.7.6.3 fence: I_proj only, never the frame's realized nawi)", "forward-mode inputs stay rejected (EvaluationMode.GATED_REALIZED only; FORWARD rejects realized inputs)"],
    "residual_open_decisions": "two, closed by amendments 3g+3h (2026-07-15): (1) 3g (2.8.2g) -- the marital projection had no domain law for the sub-START_AGE-at-anchor class build_anchor_frame admits; adopted option B (seed-at-marital-entry) so the frozen v3 floor stays byte-identical; (2) 3h (2.8.2h) -- the step-4 fertility/open-additions roster materialization had no domain law separating the frame-independent scoring schedule from the simulated-mortality roster, so a wave-loop-killed mother's scheduled birth tripped the report-only parent-roster guard and aborted the sixth registration (forensics 4984997277); adopted drop-with-reconciliation (materialize only roster-present mothers; drop the dead mother's birth into the report-only reconciliation; filter-after-draw so the fertility RNG stream and every frozen floor stay byte-identical; guard steps.py:412 retained as the invariant backstop)",
    "mechanical_alignments_for_build_lane": ["once-per-draw cached core in marital_step/household_step at the period-0 address (disability_step pattern)", "extract floor cell functions to a shared module (byte-identity by construction, self-tested)", "step-4 fertility stays per-period recompute (apply_fertility steps.py:440-472), births stitched across years -- INERT on every gated cell (children outside holdout_ids + bands; births feed report-only entrant/roster only); the roster MATERIALIZATION of those births is governed by amendment 3h (2.8.2h): materialize only for roster-present mothers, drop-with-reconciliation (filter-after-draw, RNG byte-unchanged), guard steps.py:412 retained -- closes the reg-6 crash, promoted from disclosure to law"]
  },
  "scoring": {
    "estimator": "mean over K=20 draws, numpy.random.default_rng(5200 + k), scored once |ln(rbar/rate_a)|",
    "tolerance": "round(floor mean + k*sd, 3) capped ln(1.5)",
    "mixed_k": {"flow": 3, "stock": 4},
    "gate_seeds": [0, 1, 2, 3, 4],
    "floor_seeds": "0-99",
    "conjunction": "4 of 5 seeds",
    "weight": "realized-T* start-wave weight held fixed across the window (F6), on projection, truth, and floor",
    "floor": "half_vs_half real-vs-real, presence-conditioned, correlation-respecting split, runs/m6_holdout_floors_v1.json (NOT built here)"
  },
  "families": {
    "A_certifiable": "PSID-projected vs PSID-realized dynamics-drift flows (post-demotion, floor-priced)",
    "B_report_only_or_margin": "external alignment audit + alignment displacement + entrant cells (named bridge)",
    "C_binary_optional": "drift-direction fingerprints"
  },
  "registry_frictions": {
    "F1": "exogenous widowhood hazard vs mortality-induced widowhood",
    "F2": "household_composition embeds + re-simulates its own marital core (base_simulator.py:72-77)",
    "F3": "frozen core_seed=5200 has no period axis; engine-level injected generators (plumbing)",
    "F4": "gate-2c earnings axis is person-constant career capacity, not a contemporaneous cycle",
    "F5": "disability_hazard_sim preserves observed support (reproduction, not forward)",
    "F6": "weight semantics over the projection window (start-wave vs per-year calibrated)",
    "F7": "PSID attrition on the truth side (mortality worst)"
  },
  "ceremony_deliverables": ["floor_artifact", "recertification_margin_check", "OC_before_lock_weak_power_pause", "household_id_weight_rule", "claiming_vintage_freeze (PINNED 2.8.10: 2014 Supplement 6.B5.1, supplement_year 2014)"],
  "non_goals": ["behavioral_response", "macro_feedback", "trust_fund_accounting_M7", "rules_on_whole_panel_M8", "new_spec_estimation", "validated_projection_beyond_holdout", "forward_engine_certification_deferred_to_successor_gate"],
  "process_addendum_bindings": {"pass_run_verification": "#42 comment 4948637741", "ladder_search_disclosure": "#42 comment 4948637741 + finding 1 spec-selection"}
}
```
