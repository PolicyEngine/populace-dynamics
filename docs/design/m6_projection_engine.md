# M6 projection engine — design and temporal-holdout gate proposal

- **Design id**: `2026-07-12-m6-projection-engine`
- **Roadmap**: #113 M6 (the projection engine), the last build before #113 M7
  (trust-fund accounting) and #113 M8 (integrated scoring). Workstreams #100
  (W1/W2/W3 seams), ADR-0001 (`docs/adr/0001-populace-axiom-seam-ownership.md`).
- **Status**: DESIGN (draft, revision 2). This document seeds the `gate_m6` lock
  ceremony (design review → floor build → adversarial referee → verification →
  ratify-by-merge → lock). **It edits no `gates.yaml` cell, moves no threshold,
  builds no floor, and writes no test.** The gate block and its floor are
  authored in a later ceremony, not here.
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
- 11 (claiming-table vintage) → §2.2 step 7, §7.
- 12 (biennial boundary peek at `T*`) → §4.2.
- 13 (household-ID persistence + weight carriage) → §2.2 step 8, §9.
- 14 (compute the OC before lock; weak-power pause) → §9.

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
   vintage or its vintage leak is named (finding 11, §7).
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
  `≤2014`** positive earnings, each divided by its wave's wage index (NAWI-
  normalized to a common scale); the only change from the certified object is the
  key `(age_bin, period) → age_bin` and the normalized input. De-index a drawn rank
  `u`: normalized level `ĝ = Qhat_pos_agebin.quantile(u)`, then the nominal
  target-year level is `ĝ × I_proj(target_year)` (§2.7.6.3). This **keeps the
  certified de-index object and mechanism** (a `CellMarginal` quantile — R2-faithful),
  is fully `≤2014`-derivable, is **not** a period-specific marginal (calendar-
  invariant, so §2.7.1-consistent), and has an **exact inverse** for the re-rank
  (§2.7.6.4). `p0` (the zero share) and the participation gate are unchanged; a
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
wage-growth regime and excluding pre-2005 structural breaks — the same trailing-
window logic the certified indexing convention uses; the two scored steps consume
only `I_proj(2016)` and `I_proj(2018)`.

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
  `rhat`** used to de-index in §2.7.6.2 (`candidate5b` `CellMarginal`). This is the
  **exact inverse** of §2.7.6.2 (`Qhat_pos` and `rhat` are inverse plotting-position
  maps on one grid), so rank → level → rank round-trips exactly; a carried zero maps
  to the `p0` / zero-anchor (Q0) regime. **Conditioning-consistency (R2):** the
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
- **Age support.** The earnings refit is fit on ages **25–59** (`refit.py:802-806`;
  gate-1 view `age_range [25,59]`), so the `age_bin` lookup **clips to the nearest
  fitted bin** (`<25 → 25–29`, `>59 → 55–59`); no extrapolation (there is no
  `≤2014` earnings fit outside 25–59). The `<25` and `60+` draws are **non-scored**
  plumbing — the gated earnings-moment surface (§4.5) is the 25–59 range — feeding
  only the seam / AIME.

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
  report-only benefit levels, or the named 2023-Supplement vintage leak.

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
  "revision": 3,
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
    "re_ranking_law": "PINNED fit-population CDF: rank_t = rhat_agebin(level / I_proj(wave)) -- the EXACT inverse of rank_to_level (Qhat_pos/rhat inverse plotting-position maps); zero -> p0/Q0 regime. REJECT within-frame percentile (references synthetic projected cohort not the PSID fit population, R2 violation, breaks round-trip). R2: certified ranks ARE CellMarginal CDF positions so deployed re-rank must reference the fit-population marginal",
    "secondary_gaps_pinned": {"memory_after_2018": "two rolling generated-biennial-level cols gen_earn_w2/gen_earn_w4 + realized-<=2014 start-lags", "generator_to_substream": "engine.rng.seed_from_generator(rng)->int then SeedSequence([seed,code]) codes {1:gate,2:donor,3:reentry} (m6-engine rng.py; candidate10:239)", "frame_schema": "person_id,age,sex,u_w,realized_earn_2014,realized_earn_2012,earnings,gen_earn_w2,gen_earn_w4; engine refit.py materializes u_w+realized_earn_* at period 0; _merge_period_columns carries", "age_support": "clip age_bin to nearest fitted [25,59] (refit.py:802-806; gate-1 age_range [25,59]); <25 & 60+ non-scored plumbing"},
    "closes_round2_blocker": true,
    "lock_flip_design_commit_finalizes_to_this_amendment_merge": true
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
  "ceremony_deliverables": ["floor_artifact", "recertification_margin_check", "OC_before_lock_weak_power_pause", "household_id_weight_rule", "claiming_vintage_freeze"],
  "non_goals": ["behavioral_response", "macro_feedback", "trust_fund_accounting_M7", "rules_on_whole_panel_M8", "new_spec_estimation", "validated_projection_beyond_holdout", "forward_engine_certification_deferred_to_successor_gate"],
  "process_addendum_bindings": {"pass_run_verification": "#42 comment 4948637741", "ladder_search_disclosure": "#42 comment 4948637741 + finding 1 spec-selection"}
}
```
